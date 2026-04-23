@echo off
REM RecursiaDx - Full Stack Startup Script
REM Starts all services: Brain Tumor API, Main ML API, Backend, Frontend

echo.
echo ========================================
echo   RecursiaDx Full Stack Startup
echo ========================================
echo.

REM Check if we're in the right directory
if not exist "ml\api\app.py" (
    echo ERROR: Please run this script from the RecursiaDx root directory
    exit /b 1
)

echo [1/4] Starting Brain Tumor API (Port 5002)...
start "Brain Tumor API" cmd /k "cd ml\api && python brain_tumor_api.py --port 5002"
timeout /t 3 /nobreak > nul

echo [2/4] Starting Main ML API (Port 5000)...
start "ML API" cmd /k "cd ml\api && python app.py"
timeout /t 3 /nobreak > nul

echo [3/4] Starting Backend Server (Port 5001)...
start "Backend" cmd /k "cd backend && node server.js"
timeout /t 3 /nobreak > nul

echo [4/4] Starting Frontend (Port 5173)...
start "Frontend" cmd /k "cd client && npm run dev"

echo.
echo ========================================
echo   All Services Started!
echo ========================================
echo.
echo   Frontend:     http://localhost:5173
echo   Backend:      http://localhost:5001
echo   ML API:       http://localhost:5000
echo   Brain Tumor API: http://localhost:5002
echo.
echo   Press Ctrl+C in each window to stop.
echo ========================================
