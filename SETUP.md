# RecursiaDx — Setup Guide

## ⚡ Quick Start (One Command)

After cloning the repo and placing the model weights (see Step 4 below):

```powershell
# From the project root — installs everything and launches all services
.\start.ps1
```

This script will:
- Install `npm` deps for backend + frontend
- Create Python venv and install ML requirements
- Check for model weights and `.env` files
- Open 5 service windows + the browser automatically

---


## Prerequisites (install on any new PC)

| Tool | Version | Download |
|------|---------|----------|
| Node.js | ≥ 18.0.0 | https://nodejs.org |
| Python | ≥ 3.10 | https://python.org |
| MongoDB | Community | https://mongodb.com/try/download/community |
| Git | Latest | https://git-scm.com |

---

## 1. Clone & Install

```bash
git clone <your-repo-url>
cd Recursia-Dx-ML
```

### Backend
```bash
cd backend
npm install
```

### Frontend
```bash
cd ../client
npm install
```

---

## 2. Environment Variables

### Backend (`backend/.env`)
Copy the example and fill in your keys:
```bash
cp backend/.env.example backend/.env
```

Required values to set:
```env
PORT=5001
MONGODB_URI=mongodb://localhost:27017/recursiadx
JWT_SECRET=any-long-random-string
ENCRYPTION_KEY=exactly-32-characters-here!!!!!
ML_API_URL=http://localhost:5000
OPENAI_API_KEY=sk-proj-your-openai-key-here
```

### Frontend (`client/.env`)
Create `client/.env`:
```env
VITE_API_URL=http://localhost:5001/api
VITE_SERVER_URL=http://localhost:5001
```

---

## 3. Python ML Environment

```bash
cd ml
python -m venv venv

# Windows
.\venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

> ⚠️ **PyTorch install**: If `pip install -r requirements.txt` fails for torch, install it separately first:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
> ```
> Then run `pip install -r requirements.txt` again.

---

## 4. Model Weights

The trained model files are **not in Git** (too large). You need:

| File | Size | Location |
|------|------|----------|
| `brain_tumor_efficientnetb3.h5` | 72 MB | `ml/models/weights/` |
| `densenet121_best.pth` | 29 MB | `ml/pneumonia_detection/pneumonia_detection/models/` |
| `efficientnet_b0_best.pth` | 18 MB | `ml/pneumonia_detection/pneumonia_detection/models/` |

**Get them from:** Share via Google Drive / USB from the original dev machine, or retrain using `ml/train_*.py` scripts.

---

## 5. MongoDB

Make sure MongoDB is running locally:
```bash
# Windows (if installed as service)
net start MongoDB

# Or start manually
mongod --dbpath C:\data\db
```

---

## 6. Start Everything

Open **4 terminals**:

### Terminal 1 — ML Proxy (port 5000)
```bash
cd ml
.\venv\Scripts\activate        # Windows
python api/app.py --port 5000
```

### Terminal 2 — Brain Tumor API (port 5002)
```bash
cd ml
.\venv\Scripts\activate
python api/brain_tumor_api.py --port 5002
```

### Terminal 3 — Pneumonia API (port 5003)
```bash
cd ml
.\venv\Scripts\activate
python api/pneumonia_api.py --port 5003
```

### Terminal 4 — Backend (port 5001)
```bash
cd backend
npm run dev
```

### Terminal 5 — Frontend (port 5173)
```bash
cd client
npm run dev
```

**Or use the PowerShell script** (Windows only):
```powershell
cd ml
.\start_ml.ps1
```

Then open http://localhost:5173

---

## 7. Quick Health Check

After starting everything, verify:
```
http://localhost:5000/health  → ML Proxy
http://localhost:5002/health  → Brain Tumor API
http://localhost:5003/health  → Pneumonia API
http://localhost:5001/api     → Backend
http://localhost:5173         → Frontend
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ML service not available` | Start all 3 Python APIs first |
| `500 on upload` | Check ML services are running on correct ports |
| `DICOM not displaying` | Python + pydicom must be installed; backend auto-converts |
| `OpenAI report fails` | Set `OPENAI_API_KEY` in `backend/.env` |
| `MongoDB connection error` | Start MongoDB service |

---

## ⚠️ Not Cross-Platform (Mac/Linux)

- `start_ml.ps1` is **Windows PowerShell only**
- On Mac/Linux, start each Python service manually (same commands, use `source venv/bin/activate`)
- All other code (Node.js, React) works cross-platform
