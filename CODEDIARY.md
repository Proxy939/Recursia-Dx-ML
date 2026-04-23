# CODEDIARY — RecursiaDx ML Platform

> **Purpose**: This is the living knowledge base of the entire RecursiaDx project.
> It is auto-updated on every `git commit` via a pre-commit hook.
> AI agents and developers should read this file before making any changes to the codebase.
> **Last auto-updated**: 2026-04-24T01:39:00+05:30

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Directory Structure](#directory-structure)
4. [Services & Ports](#services--ports)
5. [ML Models](#ml-models)
6. [API Reference](#api-reference)
7. [Environment Variables](#environment-variables)
8. [Database](#database)
9. [Frontend Components](#frontend-components)
10. [Key Design Decisions](#key-design-decisions)
11. [Changelog](#changelog)

---

## Project Overview

**RecursiaDx** is a medical AI diagnostic platform for clinical pathology analysis.
It supports three types of medical image analysis:

| Analysis Type | Input | Model | Output |
|---|---|---|---|
| Brain MRI | `.jpg/.png` | EfficientNetB3 | Glioma / Meningioma / Pituitary / No Tumor |
| Chest X-ray (Pneumonia) | `.jpg/.png/.dcm` | DenseNet121 + EfficientNet-B0 Ensemble | Normal / Pneumonia + Severity + Heatmap |
| Blood Smear (Malaria) | `.jpg/.png` | InceptionV3 | Infected / Uninfected |
| Blood Smear (Platelet) | `.jpg/.png` | YOLOv8 | Platelet count + bounding boxes |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RecursiaDx Platform                       │
├─────────────────┬───────────────────┬───────────────────────┤
│  Frontend       │  Backend          │  ML Layer             │
│  React + Vite   │  Node.js/Express  │  Python/Flask         │
│  Port: 5173     │  Port: 5000       │                       │
│                 │       │           │  app.py  :5001        │
│                 │       │           │  brain_tumor_api :5002│
│                 │       │           │  pneumonia_api  :5003 │
│                 │  MongoDB          │                       │
│                 │  localhost:27017  │                       │
└─────────────────┴───────────────────┴───────────────────────┘
```

**Request Flow:**
```
User → React Client (5173)
     → Node.js Backend (5000)  [auth, logging, DB]
     → ML Gateway app.py (5001)
     → brain_tumor_api.py (5002)  ← [if image_type = 'tissue']
     → pneumonia_api.py (5003)    ← [if image_type = 'pneumonia']
     → InceptionV3 / YOLOv8      ← [if image_type = 'blood']
```

---

## Directory Structure

```
Recursia-Dx-ML--main/
│
├── CODEDIARY.md              ← This file (auto-updated on commit)
├── README.md
├── STARTUP_GUIDE.md
├── MODEL_SETUP.md
├── SETUP.md
├── start_all.bat             ← Windows startup script
├── start_all.py              ← Cross-platform startup script
├── start_all.sh              ← Linux/macOS startup script
│
├── backend/                  ← Node.js/Express REST API
│   ├── server.js             ← Entry point (port 5000)
│   ├── .env                  ← Secrets (gitignored)
│   ├── .env.example          ← Template for env vars
│   ├── config/
│   │   └── config.js         ← App config (DB, JWT, CORS, etc.)
│   ├── middleware/
│   │   ├── auth.js           ← JWT authentication middleware
│   │   ├── errorHandler.js   ← Global error handling + logging
│   │   ├── upload.js         ← Multer file upload config
│   │   └── validation.js     ← Request body validation
│   ├── models/               ← Mongoose schemas
│   │   ├── User.js           ← User auth model
│   │   ├── Sample.js         ← Medical image sample model
│   │   ├── Analysis.js       ← Analysis results model
│   │   └── Report.js         ← Generated report model
│   ├── routes/
│   │   ├── auth.js           ← POST /api/auth/* (login, register, refresh)
│   │   ├── samples.js        ← POST /api/samples/* (upload, analyze, demo)
│   │   └── reports.js        ← GET/POST /api/reports/*
│   ├── services/
│   │   ├── geminiService.js  ← Google Gemini AI integration
│   │   └── mlService.js      ← ML API communication service
│   ├── utils/
│   │   └── helpers.js        ← Utility functions
│   ├── scripts/              ← DB seeding scripts
│   └── uploads/              ← Runtime uploaded files (gitignored)
│
├── client/                   ← React 18 + Vite frontend
│   ├── index.html
│   ├── vite.config.js        ← Vite config (@/ alias → src/)
│   ├── .env                  ← VITE_API_URL=http://localhost:5000/api
│   ├── src/
│   │   ├── main.jsx          ← React app entry point
│   │   ├── App.jsx           ← Router + layout
│   │   ├── index.css         ← Global styles (Tailwind CSS)
│   │   ├── components/
│   │   │   ├── HomePage.jsx         ← Landing page
│   │   │   ├── LoginPage.jsx        ← Authentication
│   │   │   ├── SignupPage.jsx       ← Registration
│   │   │   ├── AnalysisDashboard.jsx ← Main analysis view
│   │   │   ├── SampleUpload.jsx     ← Image upload component
│   │   │   ├── ResultsReview.jsx    ← View analysis results
│   │   │   ├── ReportGeneration.jsx ← PDF report generation
│   │   │   ├── DashboardSidebar.jsx ← Navigation sidebar
│   │   │   ├── NavigationHeader.jsx ← Top navigation bar
│   │   │   ├── HeatmapViewer.jsx    ← Grad-CAM heatmap view
│   │   │   ├── WSIViewer.jsx        ← Whole slide image viewer
│   │   │   ├── Settings.jsx         ← User settings page
│   │   │   └── ui/                  ← shadcn/ui components
│   │   ├── contexts/
│   │   │   ├── AuthContext.jsx      ← Global auth state
│   │   │   └── ThemeContext.jsx     ← Light/dark mode
│   │   └── services/
│   │       └── api.js               ← Axios API client
│
├── ml/                       ← Python ML microservices
│   ├── requirements.txt      ← torch, tensorflow, flask, etc.
│   ├── .env.example
│   ├── venv/                 ← Python virtual env (gitignored)
│   ├── api/
│   │   ├── app.py            ← ML gateway (port 5001)
│   │   ├── brain_tumor_api.py ← EfficientNetB3 API (port 5002)
│   │   └── pneumonia_api.py   ← DenseNet121+EfficientNet-B0 API (port 5003)
│   ├── models/
│   │   ├── malaria_predictor.py   ← InceptionV3 inference
│   │   ├── platelet_counter.py    ← YOLOv8 inference
│   │   └── weights/
│   │       ├── brain_tumor_efficientnetb3.h5  ← Trained model (72MB)
│   │       └── training_history.json          ← Training metrics
│   ├── train_brain_tumor.py  ← EfficientNetB3 training script
│   ├── calculate_auc.py      ← Post-training AUC evaluation
│   ├── config/
│   │   └── config.py
│   └── utils/
│       ├── data_manager.py
│       └── image_utils.py
│
└── demo_images/              ← Sample images for testing
    ├── cancer.jpg
    ├── maleria.jpeg
    └── non_tumor_sample.jpg
```

---

## Services & Ports

| Service | Port | Technology | Start Command |
|---|---|---|---|
| Frontend | 5173 | React + Vite | `npm run dev` (in `/client`) |
| Backend API | 5000 | Node.js + Express | `npm run dev` (in `/backend`) |
| ML Gateway | 5001 | Python + Flask | `python api/app.py` |
| Brain Tumor API | 5002 | Python + Flask | `python api/brain_tumor_api.py` |
| Pneumonia API | 5003 | Python + Flask | `python api/pneumonia_api.py` |
| MongoDB | 27017 | MongoDB 8.2.7 | Auto (Windows Service) |

---

## ML Models

### 1. Brain Tumor Classifier — EfficientNetB3
- **File**: `ml/models/weights/brain_tumor_efficientnetb3.h5`
- **Size**: 72.53 MB
- **Framework**: TensorFlow 2.15.0 / Keras
- **Classes**: `glioma`, `meningioma`, `notumor`, `pituitary`
- **Input**: 300×300×3 RGB image
- **Dataset**: masoudnickparvar/brain-tumor-mri-dataset (7,023 images)
- **Training**: 2-phase (frozen base → fine-tune top 20 layers)
- **Performance**:
  - Test Accuracy: **89.94%**
  - AUC (One-vs-Rest): **0.9818 (98.18%)**
  - Best Val Accuracy: **94.05%**
- **API**: `POST http://localhost:5002/analyze` with `image` (multipart)
- **Response fields**: `predicted_class`, `confidence_percent`, `all_class_probabilities`, `is_tumor`, `risk_level`

### 2. Pneumonia Detector — DenseNet121 + EfficientNet-B0 Ensemble
- **Weights**: `ml/pneumonia_detection/pneumonia_detection/models/densenet121_best.pth` (30MB) + `efficientnet_b0_best.pth` (19MB)
- **Framework**: PyTorch 2.2
- **Classes**: `Normal`, `Pneumonia` (binary classification)
- **Input**: 224×224 grayscale chest X-ray (CLAHE preprocessed, 3-channel ImageNet normalized)
- **Dataset**: RSNA Pneumonia Detection Challenge 2018 (~26,684 DICOM images)
- **Ensemble Method**: Soft voting (average softmax probabilities)
- **Optimal Threshold**: 0.4579 (from ROC analysis)
- **Performance**:
  - AUC-ROC: **0.8889**
  - Recall (Pneumonia): **0.86** — catches 86% of true pneumonia cases
  - Accuracy: **77.89%** (at optimal threshold)
  - Macro F1: **0.74**
- **Features**: Grad-CAM heatmap, severity scoring (Minimal/Mild/Moderate/Severe), confidence tiering
- **API**: `POST http://localhost:5003/analyze` with `image` (multipart)
- **Response fields**: `predicted_class`, `confidence_percent`, `is_pneumonia`, `per_model`, `severity`, `affected_area_pct`, `heatmap_base64`, `risk_level`

### 3. Malaria Detector — InceptionV3
- **File**: `ml/models/weights/malaria_inceptionv3.pth`
- **Framework**: PyTorch
- **Classes**: `Parasitized`, `Uninfected`
- **API**: Handled directly in `ml/api/app.py` when `image_type = 'blood'`

### 4. Platelet Counter — YOLOv8
- **File**: `ml/models/weights/platelet_yolov8.pt`
- **Framework**: PyTorch (Ultralytics)
- **Output**: Platelet count + bounding boxes
- **API**: Handled directly in `ml/api/app.py` when `image_type = 'blood'`

---

## API Reference

### Backend (Node.js — Port 5000)

#### Auth Routes (`/api/auth`)
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/auth/register` | Register a new user |
| POST | `/api/auth/login` | Login, returns JWT |
| POST | `/api/auth/refresh` | Refresh JWT token |

#### Sample Routes (`/api/samples`)
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/samples/upload` | Upload medical image |
| POST | `/api/samples/analyze` | Run ML analysis on sample |
| POST | `/api/samples/demo-analysis` | Demo analysis (no real upload) |
| GET | `/api/samples` | List all samples for user |
| GET | `/api/samples/:id` | Get single sample |

#### Report Routes (`/api/reports`)
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/reports/generate` | Generate report (uses Gemini AI) |
| GET | `/api/reports` | List all reports |
| GET | `/api/reports/:id` | Get single report |

### ML Gateway (Python — Port 5001)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check for all models |
| POST | `/analyze` | Route to correct model by `image_type` |
| GET | `/models` | List available models |

#### Routing Logic in `app.py`:
```python
if image_type == 'tissue':
    → proxy to brain_tumor_api.py (port 5002)
elif image_type == 'pneumonia':
    → proxy to pneumonia_api.py (port 5003)
elif image_type == 'blood':
    → run malaria_predictor + platelet_counter locally
```

### Brain Tumor API (Python — Port 5002)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check + model info |
| POST | `/analyze` | Classify brain MRI image |
| GET | `/model_info` | Model metadata |

### Pneumonia API (Python — Port 5003)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check + model info |
| POST | `/analyze` | Detect pneumonia from chest X-ray (returns prediction + Grad-CAM heatmap) |
| GET | `/model_info` | Model metadata + ensemble metrics |

---

## Environment Variables

### Backend (`backend/.env`)
```env
PORT=5000
NODE_ENV=development

# Local MongoDB (default)
MONGODB_URI=mongodb://localhost:27017/recursiadx
# Atlas fallback (uncomment for cloud deploy):
# MONGODB_URI=mongodb+srv://ayushpathak16022005_db_user:<password>@cluster0.nzijvyg.mongodb.net/

POSTGRES_URI=postgresql://neondb_owner:<password>@ep-billowing-hat-a4r7msph-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require
DB_NAME=recursiadx

JWT_SECRET=<secret>
JWT_EXPIRE=7d
JWT_REFRESH_SECRET=<secret>
JWT_REFRESH_EXPIRE=30d

ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
MAX_FILE_SIZE=50MB
UPLOAD_PATH=./uploads
```

### Frontend (`client/.env`)
```env
VITE_API_URL=http://localhost:5000/api
```

### ML (`ml/.env.example`)
```env
BRAIN_TUMOR_API_URL=http://localhost:5002
MALARIA_MODEL_PATH=models/weights/malaria_inceptionv3.pth
PLATELET_MODEL_PATH=models/weights/platelet_yolov8.pt
BRAIN_TUMOR_MODEL_PATH=models/weights/brain_tumor_efficientnetb3.h5
```

---

## Database

### MongoDB (Primary — Local)
- **Host**: `mongodb://localhost:27017/recursiadx`
- **Version**: 8.2.7 (installed via winget, runs as Windows service)
- **DB Name**: `recursiadx`
- **Collections**:
  - `users` — User accounts (bcrypt passwords, JWT)
  - `samples` — Uploaded medical image metadata
  - `analyses` — ML analysis results per sample
  - `reports` — AI-generated diagnostic reports

### PostgreSQL (Secondary — Neon Cloud)
- **Provider**: Neon (serverless PostgreSQL)
- **Region**: us-east-1
- **Use**: Secondary/future analytics or relational data

---

## Frontend Components

| Component | File | Purpose |
|---|---|---|
| `HomePage` | `HomePage.jsx` | Landing page with feature showcase |
| `LoginPage` | `LoginPage.jsx` | JWT login form |
| `SignupPage` | `SignupPage.jsx` | User registration |
| `AnalysisDashboard` | `AnalysisDashboard.jsx` | Main workspace after login |
| `SampleUpload` | `SampleUpload.jsx` | Drag-and-drop image upload |
| `ResultsReview` | `ResultsReview.jsx` | View and explore ML results |
| `ReportGeneration` | `ReportGeneration.jsx` | Generate Gemini AI reports |
| `HeatmapViewer` | `HeatmapViewer.jsx` | Grad-CAM attention heatmaps |
| `WSIViewer` | `WSIViewer.jsx` | Whole slide image zoom/pan |
| `DashboardSidebar` | `DashboardSidebar.jsx` | Navigation |
| `Settings` | `Settings.jsx` | User preferences |

**State Management**: React Context API
- `AuthContext` — JWT token, user object, login/logout
- `ThemeContext` — Light/dark mode toggle

**Styling**: Tailwind CSS + shadcn/ui component library

---

## Key Design Decisions

1. **EfficientNetB3 over GigaPath**: Replaced the legacy GigaPath model (closed-source, huge) with EfficientNetB3 trained on open data. Reason: easier deployment, 72MB vs gigabytes, 98.18% AUC.

2. **Microservice isolation**: Brain Tumor API runs on its own port (5002) separately from the ML gateway (5001). This allows it to be started/stopped independently and uses a different Python env (TensorFlow vs PyTorch).

3. **Separate venv for TF**: TensorFlow and PyTorch have conflicting CUDA dependencies. `ml/venv/` is a TensorFlow-only environment, while the base Python install handles PyTorch models.

4. **Local MongoDB by default**: `MONGODB_URI` points to `localhost:27017` for offline development. Atlas connection string is commented out in `.env` for cloud deployment.

5. **Git LFS removed**: `.pth` and `.pt` model weights were tracked via Git LFS but the LFS objects were broken. These files are now in `.gitignore`. Model weights should be manually placed in `ml/models/weights/` or re-trained.

6. **Windows encoding fix**: All ML scripts use `PYTHONIOENCODING=utf-8` and have had emoji characters stripped from `print()` statements to prevent `UnicodeEncodeError` on Windows (cp1252 codec).

---

## Changelog

<!-- AUTO-UPDATED BY PRE-COMMIT HOOK — DO NOT EDIT BELOW THIS LINE MANUALLY -->
<!-- CHANGELOG_START -->

### [2026-04-24 01:38] — Commit (20 file(s) changed)
- **ML**: ml/api/app.py,ml/api/pneumonia_api.py,ml/pneumonia_detection/pneumonia_detection/Medical_Image_Diagnosis_Project_Guide.txt,ml/pneumonia_detection/pneumonia_detection/PROJECT_SUMMARY.txt,ml/pneumonia_detection/pneumonia_detection/README.md,

### [2026-04-24 01:17] — Commit (6 file(s) changed)
- **Backend**: backend/config/config.js,backend/models/Analysis.js,backend/models/Report.js,backend/models/User.js,backend/server.js,
- **Frontend**: client/src/services/api.js,

### [2026-04-24 00:51] — Initial CODEDIARY Created
- Created comprehensive `CODEDIARY.md` documenting the full codebase
- Documents architecture, all services, models, APIs, env vars, DB schema, and components

### [2026-04-24 00:44] — GitHub Private Repo Created
- Created private repository: `Proxy939/Recursia-Dx-ML`
- GitHub CLI (`gh` v2.91.0) installed via winget
- Removed broken Git LFS pointers for `.pth` and `.pt` files from tracking
- Added `ml/models/weights/*.pth` and `*.pt` to `.gitignore`

### [2026-04-24 00:30] — Local MongoDB Setup
- Installed MongoDB Server 8.2.7 (via winget)
- Installed MongoDB Shell 2.8.2 (via winget)
- MongoDB runs as automatic Windows service on port 27017
- `MONGODB_URI` in `backend/.env` switched to `mongodb://localhost:27017/recursiadx`
- Atlas URI preserved as a commented fallback for cloud deployment

### [2026-04-24 00:15] — Environment Files Created
- Created `backend/.env` with all required variables
- Created `client/.env` with `VITE_API_URL=http://localhost:5000/api`
- Added PostgreSQL (Neon) URI to backend `.env`
- Both files protected by root `.gitignore`

### [2026-04-23 23:56] — EfficientNetB3 Integration Verified
- Brain Tumor API started on port 5002
- `/health` endpoint confirmed: `model_loaded: true`
- Glioma test image → `glioma` at 90.88% confidence ✓
- No-tumor test image → `notumor` at 100.0% confidence ✓
- Full proxy chain verified: Node → `app.py:5001` → `brain_tumor_api.py:5002`

### [2026-04-23 23:38] — EfficientNetB3 Model Training Completed
- Trained EfficientNetB3 on `masoudnickparvar/brain-tumor-mri-dataset`
- Dataset: 7,023 images, 4 classes (glioma, meningioma, notumor, pituitary)
- Phase 1: Frozen base, trained classification head (10 epochs)
- Phase 2: Fine-tuned top 20 layers with lower LR (early stop at epoch 17)
- **Final Test Accuracy: 89.94%**
- **AUC Score (OVR): 0.9818**
- Model saved: `ml/models/weights/brain_tumor_efficientnetb3.h5` (72.53 MB)
- Training history saved: `ml/models/weights/training_history.json`

### [2026-04-23 — earlier] — GigaPath Decommissioned / EfficientNetB3 Migration
- Deleted `ml/api/gigapath_api.py` (legacy GigaPath API)
- Deleted `ml/models/weights/gigapath_model.pth` (huge weight file)
- Created `ml/api/brain_tumor_api.py` (EfficientNetB3 Flask API, port 5002)
- Created `ml/train_brain_tumor.py` (full training pipeline with argparse)
- Updated `ml/api/app.py`: removed GigaPath proxy, added Brain Tumor API proxy
- Updated `backend/routes/samples.js`: replaced GigaPath references with Brain Tumor model
- Updated `ml/requirements.txt`: added `tensorflow==2.15.0`, `kagglehub`
- Created Python virtual environment: `ml/venv/` (TensorFlow isolated)
- Added `--data_dir` CLI argument for local dataset path support
- Fixed Windows `UnicodeEncodeError`: removed emojis from print statements, set `PYTHONIOENCODING=utf-8`

<!-- CHANGELOG_END -->
