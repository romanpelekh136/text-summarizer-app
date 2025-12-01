@echo off
echo Starting Text Summarizer App...

:: Start Backend
start "Backend" cmd /k "cd backend && .venv\Scripts\activate && python -m uvicorn main:app --reload --port 8000"

:: Start Frontend
start "Frontend" cmd /k "cd frontend && npm run dev"

echo App started!
echo Frontend: http://localhost:5173
echo Backend: http://localhost:8000
