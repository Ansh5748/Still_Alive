"""Multi-agent pipeline using Gemini (with multi-model fallback) + Indian Kanoon + platform policies.

Key upgrades:
- Model fallback chain to bypass per-model quota errors
- Scene-by-scene analysis (Agent 1)
- Language preservation (output matches transcript language)
- Platform-policy-aware legal engine (Agent 2)
- Mode-only script optimisation (Agent 5) with scene-by-scene BEFORE/AFTER
- Defensive defaults so UI never sees `undefined`
"""
import os
import json
import asyncio
import logging
import re
import uuid
from pathlib import Path
from typing import Dict, Any, List, Callable, Awaitable, Optional
from dotenv import load_dotenv
import google.generativeai as genai

from legal import cross_verify
from policies import policy_for, INDIAN_LEGAL_DIRECTIVE

load_dotenv(Path(__file__).parent / ".env")

log = logging.getLogger(__name__)

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_KEY)

# Fallback chain — each call tries these in order on 429/quota
MODEL_CHAIN = [
    "gemini-1.5-flash-8b", # Smallest, highest quota on free tier
    "gemini-flash-latest",
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

PER_AGENT_TIMEOUT = 120
MAX_RETRIES = 3  # increased for free tier
TRANSIENT_PATTERNS = ("429", "500", "502", "503", "504", "timeout", "ServiceUnavailable", "deadline", "ResourceExhausted", "quota")

# Global lock to prevent parallel agent calls from hitting rate limits too fast
_gen_lock = asyncio.Lock()
# Minimum delay between ANY two agent calls (Free Tier safety)
AGENT_DELAY = 12.0 
last_call_time = 0.0


def _gen_config() -> dict:
    return {
        "temperature": 0.55,
        "top_p": 0.95,
        "max_output_tokens": 6144,
        "response_mime_type": "application/json",
    }


def _extract_json(text: str):
    if not text:
        return None
    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    for opener, closer in [("{", "}"), ("[", "]")]:
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        break
    try:
        return json.loads(text)
    except Exception:
        return None


def _ctx(a: Dict[str, Any]) -> str:
    bits = [
        f"SubjectType: {a.get('subject_type','creator')}",
        f"Platform: {a['platform']}",
        f"Niche: {a['niche']}",
        f"Audience: {a['audience_type']} ({a.get('demographics') or 'general'})",
        f"Intent: {a['intent']}",
        f"Mode: {a['mode']}",
    ]
    if a.get("subject_type") == "brand":
        bits.append(f"Brand: {a.get('brand_name') or 'unspecified'}")
        bits.append(f"CampaignGoal: {a.get('campaign_goal') or 'unspecified'}")
    return " | ".join(bits)


def _channel_brief(ch: Optional[Dict[str, Any]]) -> str:
    if not ch:
        return "No channel data. Reason from generic niche norms."
    if ch.get("error"):
        return f"Channel data limited: {ch.get('error')}. Reason from niche norms."
    parts = [f"Platform: {ch.get('platform')}", f"Channel: {ch.get('channel')}"]
    if ch.get("subscriber_count") is not None:
        parts.append(f"Followers: {ch.get('subscriber_count'):,}")
    bio = ch.get("biography") or ch.get("description") or ""
    if bio:
        parts.append(f"Bio: {bio[:300]}")
    recent = ch.get("recent_videos") or []
    if recent:
        titles = [(r.get("title") or r.get("caption") or "").strip() for r in recent[:10]]
        titles = [t for t in titles if t]
        if titles:
            parts.append("RecentTopContent:\n- " + "\n- ".join(titles[:10]))
    return "\n".join(parts)


def _mode_directive(mode: str) -> str:
    m = (mode or "SAFE").upper()
    if m == "AGGRESSIVE":
        return ("MODE=AGGRESSIVE — maximise virality. Push pattern-interrupt hooks, named callouts, "
                "contrarian takes. Backlash 60-90% acceptable. May tolerate LOW legal risk for reach. "
                "Never soften. Keep CORE meaning + author voice + ORIGINAL LANGUAGE.")
    if m == "CONTROVERSIAL":
        return ("MODE=CONTROVERSIAL — debate-driven. Curiosity hooks, polarising framings, open-loop "
                "questions, strong stances with caveats. Avoid HIGH legal red lines, lean into grey zones. "
                "Keep CORE meaning + author voice + ORIGINAL LANGUAGE.")
    return ("MODE=SAFE — brand-friendly, legally clean, advertiser-safe. Soften polarising claims, add "
            "evidence/sources, neutral tone. Backlash <15%. "
            "Keep CORE meaning + author voice + ORIGINAL LANGUAGE.")


def _language_directive() -> str:
    return ("LANGUAGE: PRESERVE THE EXACT LANGUAGE of the source content. Hindi → respond in Hindi. "
            "Hinglish → keep the Hindi-English mix. Tamil → Tamil. English → English. Never translate "
            "the user's own lines. Analytical labels (tone, intent, etc.) may stay in English.")


async def _gen(system: str, user: str, agent: str) -> str:
    """Generate text by trying each model in MODEL_CHAIN. Returns first success."""
    global last_call_time
    last_err: Optional[Exception] = None
    
    for model_name in MODEL_CHAIN:
        for attempt in range(1, MAX_RETRIES + 1):
            async with _gen_lock:
                # Enforce minimum delay between calls
                elapsed = asyncio.get_event_loop().time() - last_call_time
                if elapsed < AGENT_DELAY:
                    await asyncio.sleep(AGENT_DELAY - elapsed)
                
                try:
                    mdl = genai.GenerativeModel(model_name, system_instruction=system, generation_config=_gen_config())
                    resp = await asyncio.wait_for(
                        asyncio.to_thread(lambda m=mdl: m.generate_content(user)),
                        timeout=PER_AGENT_TIMEOUT,
                    )
                    last_call_time = asyncio.get_event_loop().time()
                    text = getattr(resp, "text", "") or ""
                    if text.strip():
                        log.info(f"[{agent}] ok via {model_name} attempt {attempt}")
                        return text
                except Exception as e:
                    last_call_time = asyncio.get_event_loop().time()
                    last_err = e
                    err = str(e)
                    
                    # If model not found (404), don't retry this model
                    if "404" in err or "not found" in err.lower():
                        log.warning(f"[{agent}] {model_name} not available, skipping.")
                        break
                    
                    transient = any(p.lower() in err.lower() for p in TRANSIENT_PATTERNS) or isinstance(e, asyncio.TimeoutError)
                    log.warning(f"[{agent}] {model_name} attempt {attempt} failed: {err[:140]}")
                    if not transient:
                        break  # try next model
                    
                    # Extract retry delay if present in the error message (e.g., "retry in 40.75s")
                    sleep_time = (5 * attempt) if "429" in err else (2 * attempt)
                    match = re.search(r"retry in ([\d\.]+)s", err)
                    if match:
                        sleep_time = float(match.group(1)) + 1.0
                        log.info(f"[{agent}] Quota reached. Respecting API retry delay: {sleep_time}s")
                    
                    await asyncio.sleep(sleep_time)
        # fall through to next model
    raise last_err if last_err else RuntimeError(f"[{agent}] all models failed")


# ============== AGENT 1 — CONTENT BREAKDOWN (scene by scene) ==============
AGENT1_SYS = """AGENT 1 — CONTENT BREAKDOWN (scene + line level).

Break the content into SCENES (a scene = a contiguous block sharing the same topic/emotion/intent).
For each scene, also pull the most important LINES verbatim.

For EACH scene, output:
- id (S1, S2, ...)
- text: 1-2 representative LINES copied VERBATIM from the source (preserve language)
- topic: 2-5 word topic label (in source language)
- tone: dynamic value — e.g. "neutral", "harsh", "sarcastic", "instructional", "emotional", "aggressive", "directive", "earnest", "playful", "accusatory"
- intent: dynamic value — e.g. "information", "satire", "criticism", "storytelling", "promotion", "fulfilling_request", "warning", "callout"
- entities: SPECIFIC named entities ONLY if present (real people: e.g. "Shahrukh Khan"; brands: "Zerodha"; orgs: "SEBI"; products). Empty if none.
- flags: any entity above that is being TARGETED NEGATIVELY (criticised/insulted/threatened). Empty if none.
- people_named: real person names mentioned
- claims: factual or quasi-factual claims (verifiable). Empty if none.
- numbers_stats: any numbers/%/prices/dates in this scene
- references: books, studies, news outlets, court cases, laws cited
- emotion_score: 0-100 emotional intensity
- audience_relevance: 0-100 alignment with stated audience

NEVER invent. If a field has no real content in the scene, return [] or null.
Output MUST be JSON OBJECT (not array): { "scenes": [ ... ] }
"""


async def agent1_content(a: Dict[str, Any]) -> List[Dict[str, Any]]:
    user = (
        f"CONTEXT: {_ctx(a)}\n{_language_directive()}\n\n"
        f"CHANNEL_GROUNDING:\n{_channel_brief(a.get('channel_context'))}\n\n"
        f"CONTENT:\n{a['content_text']}\n\nReturn JSON now."
    )
    out = await _gen(AGENT1_SYS, user, "agent1")
    data = _extract_json(out) or {}
    scenes = []
    if isinstance(data, dict):
        scenes = data.get("scenes") or data.get("segments") or []
    elif isinstance(data, list):
        scenes = data
    # normalise: ensure each has id + text
    for i, s in enumerate(scenes):
        if not isinstance(s, dict):
            continue
        s.setdefault("id", f"S{i+1}")
        s.setdefault("text", "")
        s.setdefault("topic", "")
        s.setdefault("tone", "")
        s.setdefault("intent", "")
        s.setdefault("entities", [])
        s.setdefault("flags", [])
        s.setdefault("people_named", [])
        s.setdefault("claims", [])
        s.setdefault("numbers_stats", [])
        s.setdefault("references", [])
        s.setdefault("emotion_score", 0)
        s.setdefault("audience_relevance", 0)
    return [s for s in scenes if isinstance(s, dict)]


# ============== AGENT 2 — LEGAL RISK (platform + Indian Kanoon) ==============
AGENT2_SYS_TPL = """AGENT 2 — LEGAL + PLATFORM RISK ENGINE.

You analyse Indian creator/brand content for legal and platform-policy risk.

{indian_legal}

PLATFORM POLICIES (selected platform: {platform}):
{platform_policy}

RULES:
- ONLY flag scenes that have actual LOW / MEDIUM / HIGH risk. SKIP zero-risk scenes entirely.
- Cite REAL acts and sections only. Never cite S66A IT Act (struck down).
- For platform risk, reference the rule type (e.g. "Hate speech", "Misinformation", "Harassment").
- Quote the EXACT risky LINE from the scene (preserve language).

Return JSON:
{{
  "items":[{{
    "segment_id": "S1",
    "risk": "Low|Medium|High",
    "violation_type": "defamation|hate_speech|copyright|misleading_claim|harassment|...",
    "risky_line": "<exact line in source language>",
    "law_name": "Indian Penal Code, 1860",
    "section": "Section 499",
    "platform_rule": "<which platform policy is triggered>",
    "explanation": "In plain language: under <Section> of <Act> and <Platform>'s <Rule>, this is risky because...",
    "prob_report": 0,
    "prob_strike": 0,
    "prob_legal_notice": 0,
    "confidence": "LOW|MEDIUM|HIGH"
  }}]
}}

prob_* are PERCENT integers 0-100. Confidence reflects how strongly the law/policy applies.
If the entire content is clean, return {{"items":[]}}.
"""


async def agent2_legal(a: Dict[str, Any], scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sys_prompt = AGENT2_SYS_TPL.format(
        indian_legal=INDIAN_LEGAL_DIRECTIVE,
        platform=a.get("platform", "youtube"),
        platform_policy=policy_for(a.get("platform", "youtube")),
    )
    seg_text = "\n".join([f"{s.get('id','?')}: {s.get('text','')}" for s in scenes[:14] if s.get("text")])
    user = (
        f"CONTEXT: {_ctx(a)}\n{_language_directive()}\n\n"
        f"SCENES:\n{seg_text}\n\nReturn JSON."
    )
    out = await _gen(sys_prompt, user, "agent2")
    data = _extract_json(out) or {}
    items = data.get("items", []) if isinstance(data, dict) else []
    # Real Indian Kanoon cross-verify
    async def verify_one(item):
        v = await cross_verify(item.get("law_name", ""), item.get("section", ""))
        item["source"] = "Indian Kanoon"
        item["cross_check"] = "Verified" if v["verified"] else "Weak Match"
        item["citations"] = v["citations"]
        # Normalise probabilities to ints
        for k in ("prob_report", "prob_strike", "prob_legal_notice"):
            try:
                item[k] = int(round(float(item.get(k, 0))))
            except Exception:
                item[k] = 0
        return item
    if items:
        items = list(await asyncio.gather(*[verify_one(it) for it in items], return_exceptions=False))
    return items


# ============== AGENT 3 — VIRALITY (per scene) ==============
AGENT3_SYS = """AGENT 3 — VIRALITY + BACKLASH SIMULATOR (scene-level).
For EVERY scene given, simulate audience reaction. Output one item per scene.

Per scene:
- segment_id
- triggered_audience (which sub-tribe reacts hardest)
- virality_score (0-100)
- backlash_probability (0-100)
- retention_score (0-100) — likelihood viewers keep watching this scene
- engagement_type: "comment_storm" | "share" | "save" | "skip" | "rage_dm"
- retention_impact: signed string like "+18%" or "-7%"
- emotional_impact: 0-100
- why: 1-2 sentences in SOURCE LANGUAGE, grounded in channel signal when available

Return JSON: { "items": [...] }
"""


async def agent3_virality(a: Dict[str, Any], scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seg_text = "\n".join([f"{s.get('id','?')}: {s.get('text','')}" for s in scenes[:14]])
    user = (
        f"CONTEXT: {_ctx(a)}\n{_mode_directive(a['mode'])}\n{_language_directive()}\n\n"
        f"CHANNEL_GROUNDING:\n{_channel_brief(a.get('channel_context'))}\n\n"
        f"SCENES:\n{seg_text}\n\nReturn JSON."
    )
    out = await _gen(AGENT3_SYS, user, "agent3")
    data = _extract_json(out) or {}
    items = data.get("items", []) if isinstance(data, dict) else []
    for it in items:
        for k in ("virality_score", "backlash_probability", "retention_score", "emotional_impact"):
            try:
                it[k] = int(round(float(it.get(k, 0))))
            except Exception:
                it[k] = 0
    return items


# ============== AGENT 4 — PERSONAS ==============
AGENT4_SYS = """AGENT 4 — PERSONA FEED.
Predict how each tribe reacts to THIS content, grounded in CHANNEL_GROUNDING (past audience behaviour).
Sample comments must read like real comments on this exact channel — same vocabulary, slang, language mix.

Return JSON OBJECT:
{
  "fans": {"sentiment":"...","sample_comments":["..","..","..","..",".."],"share_likelihood":0},
  "haters": {"sentiment":"...","sample_comments":["..","..",".."],"backlash_likelihood":0},
  "neutral": {"sentiment":"...","sample_comments":["..",".."],"conversion_likelihood":0},
  "influencers": {"reaction":"...","sample_comments":["..",".."]},
  "media": {"narrative":"...","headline_ideas":["..","..",".."]},
  "brands": {"perspective":"...","sponsorship_fit":0}
}
At least 3 sample_comments for fans, 2 for others. PRESERVE SOURCE LANGUAGE in comments."""


async def agent4_personas(a: Dict[str, Any]) -> Dict[str, Any]:
    user = (
        f"CONTEXT: {_ctx(a)}\n{_mode_directive(a['mode'])}\n{_language_directive()}\n\n"
        f"CHANNEL_GROUNDING:\n{_channel_brief(a.get('channel_context'))}\n\n"
        f"CONTENT:\n{a['content_text']}\n\nReturn JSON."
    )
    out = await _gen(AGENT4_SYS, user, "agent4")
    return _extract_json(out) or {}


# ============== AGENT 5 — SCRIPT OPTIMIZATION (mode-only, scene by scene) ==============
AGENT5_SYS = """AGENT 5 — SCRIPT OPTIMIZATION (single mode, scene-by-scene rewrites).

You receive the user's chosen MODE and the per-scene legal/risk flags from Agent 2.
Rewrite ONLY the scenes that have flags or weak hooks — scene by scene.

Rules:
- Output one BEFORE / AFTER pair per scene that needs work.
- BEFORE = the exact risky/weak LINE from the source (preserve language).
- AFTER = your rewrite per the chosen MODE.
- Preserve language, author voice, core meaning, emotional direction.
- Sound human, not AI. No marketing fluff. No "as we all know...".
- For SAFE: remove legal risk + soften polarising bits + add evidence/source.
- For CONTROVERSIAL: keep punch + reframe to avoid HIGH risks + add stronger hook.
- For AGGRESSIVE: maximise virality + keep punch + still avoid HIGH legal risks.

Return JSON:
{
  "scene_rewrites": [
    {"segment_id":"S1","reason":"why this scene needs rewriting","before":"<original line>","after":"<rewritten line, same language>"}
  ],
  "full_script": "<the FULL rewritten script in source language, all scenes stitched, ready to publish>",
  "hook_improvements": ["...","...","..."],
  "retention_suggestions": ["...","...","..."],
  "what_changed": ["preserved core: ...","shifted: ...","kept emotion: ..."]
}
"""


async def agent5_scripts(a: Dict[str, Any], scenes: List[Dict[str, Any]],
                         legal: List[Dict[str, Any]]) -> Dict[str, Any]:
    seg_text = "\n".join([f"{s.get('id','?')}: {s.get('text','')}" for s in scenes[:14]])
    flag_text = "\n".join([
        f"{x.get('segment_id','?')} [{x.get('risk','?')}] {x.get('violation_type','')}: {x.get('risky_line','')}"
        for x in legal[:14]
    ]) or "(no specific flags — focus on hook/retention)"
    user = (
        f"CONTEXT: {_ctx(a)}\n{_mode_directive(a['mode'])}\n{_language_directive()}\n\n"
        f"ORIGINAL_SCRIPT:\n{a['content_text']}\n\n"
        f"SCENES:\n{seg_text}\n\nLEGAL_FLAGS:\n{flag_text}\n\nReturn JSON."
    )
    out = await _gen(AGENT5_SYS, user, "agent5")
    data = _extract_json(out) or {}
    if not isinstance(data, dict):
        data = {}
    data.setdefault("scene_rewrites", [])
    data.setdefault("full_script", "")
    data.setdefault("hook_improvements", [])
    data.setdefault("retention_suggestions", [])
    data.setdefault("what_changed", [])
    data["mode"] = a["mode"]
    return data


# ============== AGENT 6 — AUDIENCE INTELLIGENCE ==============
AGENT6_SYS = """AGENT 6 — AUDIENCE INTELLIGENCE.
Compare THIS content vs the channel's recent top performers AND niche/platform top performers.
PRESERVE SOURCE LANGUAGE in narrative text.

Return JSON:
{
  "loves": ["...","...","..."],
  "ignores": ["...","..."],
  "match_score": 0,
  "content_gaps": ["..."],
  "trending_alignment": ["..."],
  "predicted_outcome": "WILL LIKELY GO LIVE WELL | MIXED | UNDERPERFORM",
  "outcome_reasoning": "..."
}
match_score 0-100. Each list has 3+ items where possible.
"""


async def agent6_audience(a: Dict[str, Any]) -> Dict[str, Any]:
    user = (
        f"CONTEXT: {_ctx(a)}\n{_mode_directive(a['mode'])}\n{_language_directive()}\n\n"
        f"CHANNEL_GROUNDING:\n{_channel_brief(a.get('channel_context'))}\n\n"
        f"CONTENT:\n{a['content_text']}\n\nReturn JSON."
    )
    out = await _gen(AGENT6_SYS, user, "agent6")
    return _extract_json(out) or {}


# ============== AGENT 7 — GROWTH + BRAND DISCOVERY ==============
AGENT7_SYS = """AGENT 7 — GROWTH + DISTRIBUTION + BRAND-FIT DISCOVERY.

OUTPUT MUST BE A JSON OBJECT (not a bare array) with EXACTLY these keys:
{
  "titles": [string x6],
  "hooks": [string x4],
  "thumbnails": [{"concept":string,"text":string} x4],
  "short_clips": [{"timestamp":"MM:SS-MM:SS","why":string} x3],
  "brand_ideas": [{"brand_name":string,"category":string,"placement_idea":string,"match_reason":string,"est_cpm_inr":number} x6],
  "posting_strategy": {"best_time":string,"best_day":string,"platform_specific":string}
}

For brand_ideas: DISCOVER 6 distinct brands across DIFFERENT categories that genuinely fit this content + audience. NEVER default to celebrity placements. Mix D2C, fintech, edtech, gaming, SaaS, F&B, fashion, auto, telecom etc.
Titles in SOURCE LANGUAGE. brand_name, category, placement_idea may be in English (industry standard).
Return the object now. No commentary, no markdown fences.
"""


async def agent7_growth(a: Dict[str, Any]) -> Dict[str, Any]:
    user = (
        f"CONTEXT: {_ctx(a)}\n{_mode_directive(a['mode'])}\n{_language_directive()}\n\n"
        f"CHANNEL_GROUNDING:\n{_channel_brief(a.get('channel_context'))}\n\n"
        f"CONTENT:\n{a['content_text']}\n\nReturn JSON."
    )
    out = await _gen(AGENT7_SYS, user, "agent7")
    data = _extract_json(out)
    if isinstance(data, list):
        if data and isinstance(data[0], dict) and ("brand_name" in data[0] or "category" in data[0]):
            data = {"brand_ideas": data}
        else:
            data = {"titles": [str(x) for x in data if isinstance(x, str)]}
    if not isinstance(data, dict):
        data = {}
    data.setdefault("titles", [])
    data.setdefault("hooks", [])
    data.setdefault("thumbnails", [])
    data.setdefault("short_clips", [])
    data.setdefault("brand_ideas", [])
    data.setdefault("posting_strategy", {})
    return data


# ============== ORCHESTRATOR ==============
async def run_pipeline(
    a: Dict[str, Any],
    on_progress: Optional[Callable[[int, str, Dict[str, Any]], Awaitable[None]]] = None,
) -> Dict[str, Any]:
    """Sequential pipeline. Each agent gets its own retry/fallback chain.
    Saves partial results after each agent so UI streams data in.
    Critical: agent5 depends on agent1 + agent2, so they run first.
    """
    results: Dict[str, Any] = {
        "agent1_segments": [],
        "agent2_legal": [],
        "agent3_virality": [],
        "agent4_personas": {},
        "agent5_scripts": {},
        "agent6_audience": {},
        "agent7_growth": {},
    }
    failures: List[str] = []

    async def step(agent_id: str, key: str, fn: Callable[[], Awaitable[Any]], prog_val: int):
        try:
            results[key] = await fn()
            if on_progress:
                await on_progress(prog_val, f"Completed {agent_id}", dict(results))
        except Exception as e:
            log.error(f"agent failed: {agent_id} - {e}")
            failures.append(f"{agent_id}: {str(e)[:100]}")
            # Provide defensive default based on the key to prevent UI crashes
            if key in ["agent4_personas", "agent5_scripts", "agent6_audience", "agent7_growth"]:
                results[key] = {}
            else:
                results[key] = []
            if on_progress:
                # Still move the progress bar forward even if the agent fails
                await on_progress(prog_val, f"Skipped {agent_id} (Quota limit reached)", dict(results))

    await step("agent1", "agent1_segments", lambda: agent1_content(a), 14)
    scenes = results.get("agent1_segments") or []
    
    # If Agent 1 fails due to quota, preserve the EXACT script by manually chunking it into scenes
    if not scenes and a.get("content_text"):
        log.warning(f"Agent 1 failed. Manually segmenting the FULL script to avoid data loss.")
        lines = [line.strip() for line in a["content_text"].split("\n") if line.strip()]
        # Group lines into chunks of 5 to create "scenes"
        chunk_size = 5
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i + chunk_size]
            scenes.append({
                "id": f"S{len(scenes) + 1}",
                "text": " ".join(chunk_lines),
                "topic": "Content Segment",
                "description": "Script preserved via manual segmentation."
            })
        results["agent1_segments"] = scenes

    await step("agent2", "agent2_legal", lambda: agent2_legal(a, scenes), 28)
    legal = results.get("agent2_legal") or []
    await step("agent3", "agent3_virality", lambda: agent3_virality(a, scenes), 42)
    await step("agent4", "agent4_personas", lambda: agent4_personas(a), 56)
    await step("agent5", "agent5_scripts", lambda: agent5_scripts(a, scenes, legal), 70)
    await step("agent6", "agent6_audience", lambda: agent6_audience(a), 84)
    await step("agent7", "agent7_growth", lambda: agent7_growth(a), 98)
    if failures:
        results["partial_failures"] = failures
    return results
