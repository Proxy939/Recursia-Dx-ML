# ============================================================
#  Recursia-Dx-ML  —  Full Stack Startup Script
#  Run this from the project root: .\start.ps1
# ============================================================

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Definition

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   Recursia-Dx-ML  —  Starting All Services" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ── 1. Node.js Backend (Port 5000) ──────────────────────────
Write-Host "[1/5] Starting Node.js Backend on port 5000..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ROOT\backend'; npm run dev" -WindowStyle Normal

Start-Sleep -Seconds 2

# ── 2. React Frontend (Port 5173) ───────────────────────────
Write-Host "[2/5] Starting React Frontend on port 5173..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ROOT\client'; npm run dev" -WindowStyle Normal

Start-Sleep -Seconds 2

# ── 3. ML Gateway (Port 5001) ───────────────────────────────
Write-Host "[3/5] Starting ML Gateway on port 5001..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ROOT\ml'; .\venv\Scripts\Activate.ps1; python api/app.py --port 5001" -WindowStyle Normal

Start-Sleep -Seconds 2

# ── 4. Brain Tumor API (Port 5002) ──────────────────────────
Write-Host "[4/5] Starting Brain Tumor API on port 5002..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ROOT\ml'; .\venv\Scripts\Activate.ps1; python api/brain_tumor_api.py --port 5002" -WindowStyle Normal

Start-Sleep -Seconds 2

# ── 5. Pneumonia Detection API (Port 5003) ──────────────────
Write-Host "[5/5] Starting Pneumonia Detection API on port 5003..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ROOT\ml'; .\venv\Scripts\Activate.ps1; python api/pneumonia_api.py --port 5003" -WindowStyle Normal

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "   All services launched in separate windows!" -ForegroundColor Green
Write-Host "" 
Write-Host "   Frontend  ->  http://localhost:5173" -ForegroundColor White
Write-Host "   Backend   ->  http://localhost:5000" -ForegroundColor White
Write-Host "   ML Gateway->  http://localhost:5001" -ForegroundColor White
Write-Host "   Tumor API ->  http://localhost:5002" -ForegroundColor White
Write-Host "   Pneumonia ->  http://localhost:5003" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Wait ~30 seconds for all ML models to load, then open:" -ForegroundColor Cyan
Write-Host "  http://localhost:5173" -ForegroundColor White
Write-Host ""
