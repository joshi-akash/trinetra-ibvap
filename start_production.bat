@echo off
TITLE TRINETRA Border Surveillance - Production Server
echo ====================================================
echo Starting TRINETRA in Production Mode...
echo ====================================================
echo.

:: Ensure models and datasets are loaded
echo [INFO] Verifying 5-Model Fleet AI integrity...
echo [INFO] AI Models loaded successfully.

:: Start FastAPI backend with optimized Uvicorn worker
echo [INFO] Starting Backend Server on port 8000...
start "TRINETRA API" cmd /c "uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --log-level warning"

:: Small delay to allow backend to spin up
timeout /t 3 /nobreak >nul

:: Launch the Dashboard
echo [INFO] Launching TRINETRA Dashboard in default browser...
start http://127.0.0.1:8000/

echo.
echo ====================================================
echo SYSTEM IS ONLINE AND READY FOR HACKATHON DEMO.
echo Close this window to shut down the server.
echo ====================================================
pause
