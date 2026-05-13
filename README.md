# Still Alive — Creator + Brand Intelligence

A multi-agent pipeline for content analysis, legal risk detection, and growth optimization.

## Prerequisites
- Node.js (v16+)
- Python (3.9+)
- MongoDB instance

## Local Setup

### 1. Backend Setup (FastAPI)
Navigate to the backend directory, set up a virtual environment, and start the server:

```powershell
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
uvicorn server:app --reload
```
The backend will be available at `http://localhost:8000`.

### 2. Frontend Setup (React + CRACO)
Navigate to the frontend directory, install dependencies, and start the development server:

```powershell
cd frontend
# If you encounter AJV or dependency errors, run:
# Remove-Item -Recurse -Force node_modules; Remove-Item package-lock.json
# npm install ajv@^8.0.0 ajv-keywords@^5.0.0 --legacy-peer-deps
npm install ajv@^8.0.0 ajv-keywords@^5.0.0 --legacy-peer-deps
npm install --legacy-peer-deps
npm start
```
The frontend will be available at `http://localhost:3000`.

## Project Structure
- `/backend`: FastAPI application, multi-agent logic (Gemini), and MongoDB integration.
- `/frontend`: React dashboard styled with Tailwind CSS and CRACO.

## Deployment

### Render (Native Setup)
1. **Environment**: Select "Python" Runtime.
2. **Build Command**: `bash render_build.sh`
3. **Start Command**: `uvicorn server:app --host 0.0.0.0 --port $PORT`
4. **Environment Variables**: Add `PATH` = `/opt/render/project/src/backend/ffmpeg_bin:$PATH` to ensure Python finds FFmpeg.

### Render (Docker Setup - Recommended)
1. **Environment**: Select "Docker" Runtime.
2. **Build/Start**: Render will automatically use `backend/Dockerfile`.

## Environment Variables
Ensure you have `.env` files in both `backend/` and `frontend/` directories as specified in the project configuration.

**Important**: For transcription to work, `FFmpeg` must be installed on the host machine.
