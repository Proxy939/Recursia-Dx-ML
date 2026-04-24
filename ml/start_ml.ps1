# RecursiaDx ML - Clean startup script (no emoji, UTF-8 safe)
# Starts all 3 ML servers: Proxy (5000), Brain Tumor (5002), Pneumonia (5003)

$mlDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $mlDir

# Activate venv
$venvPath = Join-Path $mlDir "venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    Write-Host "[OK] Activating venv..." -ForegroundColor Green
    & $venvPath
} else {
    Write-Host "[WARN] No venv found at $venvPath" -ForegroundColor Yellow
    Write-Host "[INFO] Run: python -m venv venv  then  pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "RecursiaDx ML Stack Starting..." -ForegroundColor Cyan
Write-Host "  Brain Tumor API  -> port 5002" -ForegroundColor White
Write-Host "  Pneumonia API    -> port 5003" -ForegroundColor White
Write-Host "  ML Proxy         -> port 5000" -ForegroundColor White
Write-Host ""

# Start Brain Tumor API in new window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "
    Set-Location '$mlDir';
    & '$mlDir\venv\Scripts\Activate.ps1';
    Write-Host '[Brain Tumor API] Starting on port 5002...' -ForegroundColor Blue;
    python api/brain_tumor_api.py --port 5002
" -WindowStyle Normal

Start-Sleep -Seconds 2

# Start Pneumonia API in new window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "
    Set-Location '$mlDir';
    & '$mlDir\venv\Scripts\Activate.ps1';
    Write-Host '[Pneumonia API] Starting on port 5003...' -ForegroundColor Magenta;
    python api/pneumonia_api.py --port 5003
" -WindowStyle Normal

Start-Sleep -Seconds 2

# Start Main Proxy in this window
Write-Host "[Proxy] Starting ML proxy on port 5000..." -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop all servers" -ForegroundColor Gray
Write-Host ""
python start_server.py --port 5000
