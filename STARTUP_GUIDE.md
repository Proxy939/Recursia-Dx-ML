# RecursiaDx Startup Guide

## Prerequisites Check

Before starting, ensure the ML models are available:

```bash
cd ml
# Check model weights exist
ls models/weights/
```

You should see:
- ✅ `brain_tumor_efficientnetb3.h5` (train with `python train_brain_tumor.py` if missing)
- ✅ `malaria_inceptionv3.pth`
- ✅ `platelet_yolov8.pt`

---

## Starting the Complete System

### Step 1: Start the Brain Tumor API (Port 5002)

```bash
cd ml
python api/brain_tumor_api.py --port 5002
```

**Expected output:**
```
✅ Brain tumor EfficientNetB3 model loaded successfully
🚀 Starting Brain Tumor API on port 5002...
```

**Health check:**
```bash
curl http://localhost:5002/health
```

---

### Step 2: Start the Main ML API (Port 5000)

Open a **new terminal**:

```bash
cd ml
python api/app.py
```

**Expected output:**
```
✅ Brain Tumor API is available
✅ Malaria detection model loaded successfully
✅ Platelet counting model loaded successfully
✅ Server initialization complete
🚀 Starting server on port 5000...
```

**Health check:**
```bash
curl http://localhost:5000/health
```

---

### Step 3: Start the Backend Server (Port 5001)

Open a **new terminal**:

```bash
cd backend
node server.js
```

**Expected output:**
```
🚀 Server started on port 5001
📊 MongoDB connected successfully
```

---

### Step 4: Start the Frontend (Port 5173)

Open a **new terminal**:

```bash
cd client
npm run dev
```

**Expected output:**
```
VITE ready in ...ms
➜  Local:   http://localhost:5173/
```

---

## System Status

| Component | Port | Status Check |
|-----------|------|--------------|
| Brain Tumor API | 5002 | http://localhost:5002/health |
| ML Server | 5000 | http://localhost:5000/health |
| Backend API | 5001 | http://localhost:5001 |
| Frontend | 5173 | http://localhost:5173 |

---

## Important Changes

### ✅ What's New:
1. **Brain Tumor Detection** - EfficientNetB3 for 4-class MRI classification
2. **Separate Brain Tumor API** - Runs on port 5002
3. **Training Script** - `train_brain_tumor.py` for one-click model training

### ❌ What's Removed:
1. GigaPath-AttentionMIL tissue analysis (replaced by EfficientNetB3)
2. Old tumor detection pipeline
3. WSI tiling and heatmap aggregation modules

### ⚠️ Critical:
- **Brain Tumor API MUST be running** for brain MRI analysis to work
- **ML server MUST be running** for malaria/platelet analysis
- Without these servers, uploads will fail with appropriate error messages

---

## Troubleshooting

### Brain Tumor API Won't Start

**Issue:** "Model file not found"
**Solution:**
```bash
cd ml
python train_brain_tumor.py
```

### ML Server Won't Start

**Issue:** "Malaria/Platelet model not found"
**Solution:** Check model files exist in `ml/models/weights/`

### Backend Can't Connect to ML

**Issue:** "ML service is not available"
**Solution:**
1. Check ML server is running: `curl http://localhost:5000/health`
2. Check Brain Tumor API: `curl http://localhost:5002/health`
3. Check firewall settings
4. Restart ML servers

---

## Testing the System

### 1. Test Brain Tumor API Directly:

```bash
curl -X POST http://localhost:5002/analyze \
  -F "image=@/path/to/brain_mri.png"
```

### 2. Test Main ML API:

```bash
curl http://localhost:5000/health
```

### 3. Test Complete Flow:

1. Go to http://localhost:5173
2. Login/Register
3. Upload a sample with images
4. Check that ML analysis runs
5. View results in dashboard

---

## Performance Notes

- **Brain Tumor Analysis:** ~1-3 seconds per image
- **Malaria Detection:** ~2-5 seconds per image
- **Platelet Counting:** ~2-5 seconds per image

---

## Quick Commands

```bash
# Check all services
curl http://localhost:5002/health  # Brain Tumor API
curl http://localhost:5000/health  # ML API
curl http://localhost:5001/        # Backend
curl http://localhost:5173/        # Frontend

# Train brain tumor model
cd ml && python train_brain_tumor.py

# Restart all (PowerShell)
# Press Ctrl+C in each terminal, then restart
```
