@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe
  echo Create it first, then install dependencies.
  pause
  exit /b 1
)

echo Running LLM availability preflight check...
".venv\Scripts\python.exe" scripts\check_llm_availability.py
echo.

echo Starting Client Profiler web server...
echo URL: http://127.0.0.1:8000
echo Press Ctrl+C to stop.
echo.

".venv\Scripts\python.exe" -m uvicorn client_profiler.web.app:app --reload

endlocal
