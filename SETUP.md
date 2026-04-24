# Recursia-Dx-ML — Setup Guide

A full-stack AI medical imaging platform for **Brain Tumor Detection** and **Pneumonia Detection**.

---

## Prerequisites

| Tool | Version | Download |
|------|---------|----------|
| Node.js | ≥ 18.x | https://nodejs.org |
| Python | ≥ 3.9 | https://python.org |
| MongoDB | 6.x+ | https://www.mongodb.com/try/download/community |
| Git | Any | https://git-scm.com |

---

## 1. Clone the Repository

```bash
git clone https://github.com/Proxy939/Recursia-Dx-ML.git
cd Recursia-Dx-ML
```

---

## 2. Backend Setup (Node.js)

```bash
cd backend
npm install
```

Create `backend/.env` by copying the example:
```bash
copy .env.example .env   # Windows
cp .env.example .env     # Mac/Linux
```

Edit `.env` and fill in:
```env
PORT=5000
NODE_ENV=development
ML_API_URL=http://localhost:5001
MONGODB_URI=mongodb://127.0.0.1:27017/recursiadx
JWT_SECRET=your-secret-key-here
JWT_EXPIRE=7d
JWT_REFRESH_SECRET=your-refresh-secret-here
JWT_REFRESH_EXPIRE=30d
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
```

---

## 3. Frontend Setup (React + Vite)

```bash
cd client
npm install
```

Create `client/.env`:
```env
VITE_API_URL=http://localhost:5000/api
VITE_SERVER_URL=http://localhost:5000
```

---

## 4. ML Setup (Python)

```bash
cd ml
python -m venv venv
```

**Activate the venv:**
```bash
# Windows
.\venv\Scripts\Activate.ps1

# Mac/Linux
source venv/bin/activate
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

> ⚠️ **TensorFlow Note**: `tensorflow==2.15.0` is required for the Brain Tumor model.
> On machines with CUDA GPU, PyTorch will automatically use it for the Pneumonia model.

---

## 5. Model Weights

The trained model weights are stored in Git and will be downloaded automatically with `git clone`.

| Model | File | Location | Size |
|-------|------|----------|------|
| Brain Tumor (EfficientNetB3) | `brain_tumor_efficientnetb3.h5` | `ml/models/weights/` | ~72 MB |
| Pneumonia DenseNet121 | `densenet121_best.pth` | `ml/pneumonia_detection/pneumonia_detection/models/` | ~29 MB |
| Pneumonia EfficientNet-B0 | `efficientnet_b0_best.pth` | `ml/pneumonia_detection/pneumonia_detection/models/` | ~18 MB |

If the files are missing (slow connection / LFS issue), download from the team's shared Google Drive and place them in the paths above.

---

## 6. Start MongoDB

```bash
# Windows (if installed as a service, it may already be running)
net start MongoDB

# Or start manually:
mongod --dbpath C:\data\db
```

---

## 7. Run Everything

### Option A — One Command (Recommended)

From the project root:
```powershell
.\start.ps1
```

This opens **5 separate terminal windows** automatically.

### Option B — Manual (5 separate terminals)

**Terminal 1 — Backend:**
```bash
cd backend
npm run dev
```

**Terminal 2 — Frontend:**
```bash
cd client
npm run dev
```

**Terminal 3 — ML Gateway:**
```bash
cd ml
.\venv\Scripts\Activate.ps1   # Windows
python api/app.py --port 5001
```

**Terminal 4 — Brain Tumor API:**
```bash
cd ml
.\venv\Scripts\Activate.ps1
python api/brain_tumor_api.py --port 5002
```

**Terminal 5 — Pneumonia API:**
```bash
cd ml
.\venv\Scripts\Activate.ps1
python api/pneumonia_api.py --port 5003
```

---

## 8. Access the App

Open your browser at: **http://localhost:5173**

Wait ~30 seconds after starting for all ML models to load into memory.

---

## Port Reference

| Service | Port |
|---------|------|
| React Frontend | 5173 |
| Node.js Backend | 5000 |
| ML Gateway | 5001 |
| Brain Tumor API | 5002 |
| Pneumonia API | 5003 |
| MongoDB | 27017 |

---

## Supported File Formats

| Analysis Type | Supported Formats |
|--------------|------------------|
| Brain Tumor (MRI) | `.jpg`, `.jpeg`, `.png`, `.bmp` |
| Pneumonia (Chest X-ray) | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.dcm` (DICOM) |

---

## Troubleshooting

### "ML Gateway unreachable on port 5001"
→ Make sure `python api/app.py --port 5001` is running in the `ml/` directory with the venv activated.

### "Pneumonia API is not running on port 5003"
→ Start `python api/pneumonia_api.py --port 5003` in a separate terminal.

### "vite: command not found"
→ Run `npm install` inside the `client/` directory.

### Python `ModuleNotFoundError`
→ Make sure the venv is activated before running Python files.

### MongoDB connection error
→ Start MongoDB: `net start MongoDB` (Windows) or `brew services start mongodb-community` (Mac).
