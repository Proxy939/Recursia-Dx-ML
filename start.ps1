# RecursiaDx - One-Click Full Stack Launcher
# Run from project root: .\start.ps1

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

Header "Checking Prerequisites"
if (-not (Get-Command node -ErrorAction SilentlyContinue)) { WARN "Node.js not found"; exit 1 }
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { WARN "Python not found"; exit 1 }
OK "Node $(node --version) | Python $(python --version)"

Header "Installing Backend Dependencies"
if (-not (Test-Path "$BE\node_modules")) {
    Step "npm install (backend)..."
    Set-Location $BE
    npm install --silent
    OK "Backend deps installed"
} else { OK "Backend node_modules already exist - skipping" }

Header "Installing Frontend Dependencies"
if (-not (Test-Path "$FE\node_modules")) {
    Step "npm install (client)..."
    Set-Location $FE
    npm install --silent
    OK "Frontend deps installed"
} else { OK "Frontend node_modules already exist - skipping" }

Header "Setting Up Python ML Environment"
Set-Location $ML
$venvActivate = "$ML\venv\Scripts\Activate.ps1"
if (-not (Test-Path $venvActivate)) {
    Step "Creating Python venv..."
    python -m venv venv
    & $venvActivate
    pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install --quiet -r "$ML\requirements.txt"
    OK "Python deps installed"
} else { OK "Python venv already exists - skipping" }

Header "Checking Model Weights"
$models = @(
    "$ML\models\weights\brain_tumor_efficientnetb3.h5",
    "$ML\pneumonia_detection\pneumonia_detection\models\densenet121_best.pth",
    "$ML\pneumonia_detection\pneumonia_detection\models\efficientnet_b0_best.pth"
)
$missingModels = $false
foreach ($m in $models) {
    if (Test-Path $m) { OK "Found: $(Split-Path $m -Leaf)" }
    else { WARN "MISSING: $m"; $missingModels = $true }
}
if ($missingModels) {
    WARN "Some model weights are missing. ML analysis will not work."
    $ans = Read-Host "Continue anyway? (y/n)"
    if ($ans -ne 'y') { exit 1 }
}

Header "Checking Environment Files"
if (-not (Test-Path "$BE\.env")) {
    Copy-Item "$BE\.env.example" "$BE\.env"
    WARN "Opened backend/.env - fill in OPENAI_API_KEY and MONGODB_URI then save"
    notepad "$BE\.env"
    Read-Host "Press Enter after saving .env"
} else { OK "backend/.env found" }

if (-not (Test-Path "$FE\.env")) {
    "VITE_API_URL=http://localhost:5001/api`nVITE_SERVER_URL=http://localhost:5001" | Set-Content "$FE\.env" -Encoding ASCII
    OK "client/.env created"
} else { OK "client/.env found" }

Header "Launching All Services"

Step "Brain Tumor API (port 5002)..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$ML'; & '$ML\venv\Scripts\Activate.ps1'; python api/brain_tumor_api.py --port 5002"
Start-Sleep -Seconds 2

Step "Pneumonia API (port 5003)..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$ML'; & '$ML\venv\Scripts\Activate.ps1'; python api/pneumonia_api.py --port 5003"
Start-Sleep -Seconds 2

Step "ML Proxy (port 5000)..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$ML'; & '$ML\venv\Scripts\Activate.ps1'; python start_server.py --port 5000"
Start-Sleep -Seconds 3

Step "Backend (port 5001)..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$BE'; npm run dev"
Start-Sleep -Seconds 3

Step "Frontend (port 5173)..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$FE'; npm run dev"

Header "All Services Starting!"
Write-Host "  Frontend   -> http://localhost:5173" -ForegroundColor Cyan
Write-Host "  Backend    -> http://localhost:5001" -ForegroundColor Green
Write-Host "  ML Proxy   -> http://localhost:5000/health" -ForegroundColor Yellow
Write-Host "  BrainTumor -> http://localhost:5002/health" -ForegroundColor Blue
Write-Host "  Pneumonia  -> http://localhost:5003/health" -ForegroundColor Magenta
Write-Host ""
Write-Host "  ML models take ~30-60 sec to load after startup." -ForegroundColor DarkGray

Start-Sleep -Seconds 8
Start-Process "http://localhost:5173"
