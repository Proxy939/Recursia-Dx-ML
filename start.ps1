# RecursiaDx - One-Click Full Stack Launcher
# Run from project root: .\start.ps1
#
# PORT MAP (must match backend/.env and client/.env):
#   Node Backend  -> 5000   (set in backend/.env as PORT=5000)
#   ML Gateway    -> 5001   (set in backend/.env as ML_API_URL=http://localhost:5001)
#   Brain Tumor   -> 5002
#   Pneumonia     -> 5003
#   Frontend      -> 5173

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$ML   = Join-Path $ROOT "ml"
$BE   = Join-Path $ROOT "backend"
$FE   = Join-Path $ROOT "client"

function Header($msg) {
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host "============================================" -ForegroundColor Cyan
}

function Step($msg)  { Write-Host "[...] $msg" -ForegroundColor Yellow }
function OK($msg)    { Write-Host "[ OK] $msg" -ForegroundColor Green  }
function WARN($msg)  { Write-Host "[!!!] $msg" -ForegroundColor Red    }

# ── Prerequisites ────────────────────────────────────────────
Header "Checking Prerequisites"
if (-not (Get-Command node -ErrorAction SilentlyContinue)) { WARN "Node.js not found"; exit 1 }
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { WARN "Python not found"; exit 1 }
OK "Node $(node --version) | Python $(python --version)"

# ── Check MongoDB ─────────────────────────────────────────────
Header "Checking MongoDB"
$mongoService = Get-Service -Name "MongoDB" -ErrorAction SilentlyContinue
$mongo27017 = netstat -ano 2>$null | Select-String ":27017"
if ($mongoService -and $mongoService.Status -eq 'Running') {
    OK "MongoDB service is running"
} elseif ($mongo27017) {
    OK "MongoDB is listening on port 27017"
} else {
    WARN "MongoDB does not appear to be running on port 27017!"
    WARN "The backend will crash without MongoDB."
    WARN "Start it with: net start MongoDB"
    $ans = Read-Host "Continue anyway? (y/n)"
    if ($ans -ne 'y') { exit 1 }
}

# ── Kill any stale processes on our ports ────────────────────
Header "Clearing Ports (5000, 5001, 5002, 5003)"
@(5000, 5001, 5002, 5003) | ForEach-Object {
    $port = $_
    $pids = netstat -ano 2>$null | Select-String ":$port\s" | ForEach-Object {
        ($_ -split '\s+')[-1]
    } | Where-Object { $_ -match '^\d+$' } | Sort-Object -Unique
    foreach ($p in $pids) {
        try {
            Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
            Write-Host "  Cleared PID $p on port $port" -ForegroundColor DarkGray
        } catch {}
    }
}
Start-Sleep -Seconds 2

# ── Install dependencies (first run only) ────────────────────
Header "Backend Dependencies"
if (-not (Test-Path "$BE\node_modules")) {
    Step "npm install (backend)..."
    Set-Location $BE; npm install --silent
    OK "Backend deps installed"
} else { OK "Backend node_modules exist - skipping" }

Header "Frontend Dependencies"
if (-not (Test-Path "$FE\node_modules")) {
    Step "npm install (client)..."
    Set-Location $FE; npm install --silent
    OK "Frontend deps installed"
} else { OK "Frontend node_modules exist - skipping" }

# ── Python venv ──────────────────────────────────────────────
Header "Python ML Environment"
$venvActivate = "$ML\venv\Scripts\Activate.ps1"
if (-not (Test-Path $venvActivate)) {
    Step "Creating Python venv..."
    Set-Location $ML
    python -m venv venv
    & $venvActivate
    pip install --quiet -r "$ML\requirements.txt"
    OK "Python deps installed"
} else { OK "Python venv exists - skipping" }

# ── Model weights check ──────────────────────────────────────
Header "Checking Model Weights"
$models = @(
    "$ML\models\weights\brain_tumor_efficientnetb3.h5",
    "$ML\pneumonia_detection\pneumonia_detection\models\densenet121_best.pth",
    "$ML\pneumonia_detection\pneumonia_detection\models\efficientnet_b0_best.pth"
)
$missingModels = $false
foreach ($m in $models) {
    if (Test-Path $m) { OK "Found: $(Split-Path $m -Leaf)" }
    else { WARN "MISSING: $(Split-Path $m -Leaf)"; $missingModels = $true }
}
if ($missingModels) {
    WARN "Some model weights are missing. ML analysis will not work."
    $ans = Read-Host "Continue anyway? (y/n)"
    if ($ans -ne 'y') { exit 1 }
}

# ── Environment files check ──────────────────────────────────
Header "Checking Environment Files"
if (-not (Test-Path "$BE\.env")) {
    WARN "backend/.env not found! Creating from example..."
    Copy-Item "$BE\.env.example" "$BE\.env" -ErrorAction SilentlyContinue
    notepad "$BE\.env"
    Read-Host "Press Enter after saving backend/.env"
} else { OK "backend/.env found (Backend=5000, ML_API=5001)" }

# Ensure client/.env always points to the Node backend on 5000
$clientEnvContent = "# Backend API URL - Node backend runs on port 5000`nVITE_API_URL=http://localhost:5000/api`nVITE_SERVER_URL=http://localhost:5000"
Set-Content "$FE\.env" $clientEnvContent -Encoding ASCII
OK "client/.env set (VITE_API_URL=http://localhost:5000/api)"

# ── Launch services ───────────────────────────────────────────
Header "Launching All Services"

# 1. Node Backend — PORT 5000 (reads from backend/.env)
Step "Node Backend (port 5000)..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$BE'; npm run dev"
Start-Sleep -Seconds 3

# 2. ML Gateway — PORT 5001
Step "ML Gateway (port 5001)..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$ML'; & '$ML\venv\Scripts\Activate.ps1'; python api/app.py --port 5001"
Start-Sleep -Seconds 2

# 3. Brain Tumor API — PORT 5002
Step "Brain Tumor API (port 5002)..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$ML'; & '$ML\venv\Scripts\Activate.ps1'; python api/brain_tumor_api.py --port 5002"
Start-Sleep -Seconds 2

# 4. Pneumonia API — PORT 5003
Step "Pneumonia API (port 5003)..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$ML'; & '$ML\venv\Scripts\Activate.ps1'; python api/pneumonia_api.py --port 5003"
Start-Sleep -Seconds 2

# 5. React Frontend — PORT 5173
Step "React Frontend (port 5173)..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$FE'; npm run dev"

# ── Summary ───────────────────────────────────────────────────
Header "All Services Launched!"
Write-Host "  Service       Port   URL" -ForegroundColor White
Write-Host "  ─────────────────────────────────────────────" -ForegroundColor DarkGray
Write-Host "  Frontend    ->  5173   http://localhost:5173" -ForegroundColor Cyan
Write-Host "  Node Backend->  5000   http://localhost:5000/health" -ForegroundColor Green
Write-Host "  ML Gateway  ->  5001   http://localhost:5001/health" -ForegroundColor Yellow
Write-Host "  Brain Tumor ->  5002   http://localhost:5002/health" -ForegroundColor Blue
Write-Host "  Pneumonia   ->  5003   http://localhost:5003/health" -ForegroundColor Magenta
Write-Host ""
Write-Host "  Wait ~30-60 sec for ML models to load, then open:" -ForegroundColor DarkGray
Write-Host "  http://localhost:5173" -ForegroundColor White
Write-Host ""

Start-Sleep -Seconds 10
Start-Process "http://localhost:5173"
