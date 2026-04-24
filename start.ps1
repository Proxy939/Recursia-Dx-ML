# =============================================================
# RecursiaDx - One-Click Full Stack Launcher
# Run this from the project root: .\start.ps1
# =============================================================

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

# ── 0. Check prerequisites ───────────────────────────────────
Header "Checking Prerequisites"

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    WARN "Node.js not found. Install from https://nodejs.org"; exit 1
}
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    WARN "Python not found. Install from https://python.org"; exit 1
}
OK "Node $(node --version) | Python $(python --version)"

# ── 1. Backend deps ──────────────────────────────────────────
Header "Installing Backend Dependencies"
if (-not (Test-Path "$BE\node_modules")) {
    Step "npm install (backend)..."
    Set-Location $BE
    npm install --silent
    OK "Backend deps installed"
} else {
    OK "Backend node_modules already exist - skipping"
}

# ── 2. Frontend deps ─────────────────────────────────────────
Header "Installing Frontend Dependencies"
if (-not (Test-Path "$FE\node_modules")) {
    Step "npm install (client)..."
    Set-Location $FE
    npm install --silent
    OK "Frontend deps installed"
} else {
    OK "Frontend node_modules already exist - skipping"
}

# ── 3. Python venv ───────────────────────────────────────────
Header "Setting Up Python ML Environment"
Set-Location $ML
$venvActivate = "$ML\venv\Scripts\Activate.ps1"

if (-not (Test-Path $venvActivate)) {
    Step "Creating Python virtual environment..."
    python -m venv venv
    OK "venv created"

    Step "Installing Python dependencies (this may take 5-10 min)..."
    & $venvActivate
    pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install --quiet -r "$ML\requirements.txt"
    OK "Python deps installed"
} else {
    OK "Python venv already exists - skipping"
}

# ── 4. Check model weights ───────────────────────────────────
Header "Checking Model Weights"
$models = @(
    "$ML\models\weights\brain_tumor_efficientnetb3.h5",
    "$ML\pneumonia_detection\pneumonia_detection\models\densenet121_best.pth",
    "$ML\pneumonia_detection\pneumonia_detection\models\efficientnet_b0_best.pth"
)
$missingModels = $false
foreach ($m in $models) {
    if (Test-Path $m) {
        OK "Found: $(Split-Path $m -Leaf)"
    } else {
        WARN "MISSING: $m"
        $missingModels = $true
    }
}
if ($missingModels) {
    WARN "Some model weights are missing. ML analysis will not work."
    WARN "Copy the .pth and .h5 files from the original machine."
    Write-Host ""
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne 'y') { exit 1 }
}

# ── 5. Check .env files ──────────────────────────────────────
Header "Checking Environment Files"
if (-not (Test-Path "$BE\.env")) {
    WARN "backend/.env not found!"
    Step "Copying from .env.example..."
    Copy-Item "$BE\.env.example" "$BE\.env"
    WARN "IMPORTANT: Edit backend/.env and set your OPENAI_API_KEY and MONGODB_URI"
    notepad "$BE\.env"
    Read-Host "Press Enter after saving .env to continue"
} else {
    OK "backend/.env found"
}

if (-not (Test-Path "$FE\.env")) {
    Step "Creating client/.env..."
    @"
VITE_API_URL=http://localhost:5001/api
VITE_SERVER_URL=http://localhost:5001
"@ | Set-Content "$FE\.env"
    OK "client/.env created"
} else {
    OK "client/.env found"
}

# ── 6. Launch all services ───────────────────────────────────
Header "Launching All Services"

# Brain Tumor API (port 5002)
Step "Starting Brain Tumor API on port 5002..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "Set-Location '$ML'; `
    & '$ML\venv\Scripts\Activate.ps1'; `
    Write-Host '[Brain Tumor API] port 5002' -ForegroundColor Blue; `
    python api/brain_tumor_api.py --port 5002" `
    -WindowStyle Normal
Start-Sleep -Seconds 2

# Pneumonia API (port 5003)
Step "Starting Pneumonia Detection API on port 5003..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "Set-Location '$ML'; `
    & '$ML\venv\Scripts\Activate.ps1'; `
    Write-Host '[Pneumonia API] port 5003' -ForegroundColor Magenta; `
    python api/pneumonia_api.py --port 5003" `
    -WindowStyle Normal
Start-Sleep -Seconds 2

# ML Proxy (port 5000)
Step "Starting ML Proxy on port 5000..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "Set-Location '$ML'; `
    & '$ML\venv\Scripts\Activate.ps1'; `
    Write-Host '[ML Proxy] port 5000' -ForegroundColor Yellow; `
    python start_server.py --port 5000" `
    -WindowStyle Normal
Start-Sleep -Seconds 3

# Backend (port 5001)
Step "Starting Backend on port 5001..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "Set-Location '$BE'; `
    Write-Host '[Backend] port 5001' -ForegroundColor Green; `
    npm run dev" `
    -WindowStyle Normal
Start-Sleep -Seconds 3

# Frontend (port 5173)
Step "Starting Frontend on port 5173..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "Set-Location '$FE'; `
    Write-Host '[Frontend] port 5173' -ForegroundColor Cyan; `
    npm run dev" `
    -WindowStyle Normal

# ── 7. Done ──────────────────────────────────────────────────
Header "All Services Starting!"
Write-Host ""
Write-Host "  Service          URL" -ForegroundColor White
Write-Host "  ─────────────────────────────────────────" -ForegroundColor DarkGray
Write-Host "  Frontend      -> http://localhost:5173" -ForegroundColor Cyan
Write-Host "  Backend       -> http://localhost:5001" -ForegroundColor Green
Write-Host "  ML Proxy      -> http://localhost:5000/health" -ForegroundColor Yellow
Write-Host "  Brain Tumor   -> http://localhost:5002/health" -ForegroundColor Blue
Write-Host "  Pneumonia     -> http://localhost:5003/health" -ForegroundColor Magenta
Write-Host ""
Write-Host "  ML models take ~30-60 seconds to load after startup." -ForegroundColor DarkGray
Write-Host ""

Start-Sleep -Seconds 8
Start-Process "http://localhost:5173"
