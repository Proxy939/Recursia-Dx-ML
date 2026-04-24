# RecursiaDx AI Diagnostic Platform

AI-powered digital pathology platform providing automated brain tumor detection and pneumonia detection, featuring Explainable AI (Grad-CAM) and AI-generated clinical reports.

## Overview

RecursiaDx is a comprehensive medical image analysis platform that integrates state-of-the-art machine learning models for clinical diagnostics:
- **Brain Tumor Analysis**: 4-class MRI classification (Glioma, Meningioma, Pituitary, No Tumor) using EfficientNetB3.
- **Pneumonia Detection**: Chest X-ray binary classification (Normal, Pneumonia) using a powerful ensemble of DenseNet121 + EfficientNet-B0.
- **Explainable AI (XAI)**: Generates Grad-CAM heatmaps for both brain MRIs and chest X-rays to visualize the regions of interest the models focused on.
- **AI Report Generation**: Integrates Google Gemini AI for synthesizing complex ML predictions and affected area metrics into professional, human-readable clinical summaries.

## Key Features

✅ **Multi-Modal Clinical Diagnostics**
- Brain tumor MRI classification with spatial severity mapping.
- Pneumonia chest X-ray detection via high-accuracy model ensembles.

✅ **Explainable AI (XAI)**
- Real-time Grad-CAM heatmap generation.
- Dynamic calculation of affected area percentages.
- Visual attention maps side-by-side with original scans.

✅ **AI-Powered Workflows & Professional Reporting**
- 5-step clinical workflow (Upload → Analysis → Dashboard → Review → Report).
- Dynamic report generation with Gemini AI interpretation.
- Severity scoring and confidence tiering.

✅ **Interactive UI**
- Fully responsive dashboard with real-time status tracking.
- Interactive imaging viewers (Original + Grad-CAM overlay).
- Dark/Light theme support.

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | React 18 + Vite + Tailwind CSS |
| **Backend** | Node.js 18+ + Express + MongoDB |
| **ML Gateway** | Python Flask (Proxy and pre-processing) |
| **ML Inference** | TensorFlow/Keras (EfficientNetB3) + PyTorch (DenseNet/EfficientNet) |
| **AI Integration** | Google Gemini 2.5 Flash |

## Quick Start

### Prerequisites
- Node.js 18+ and npm
- Python 3.10+ with pip
- MongoDB (local or Atlas)
- **Git LFS** (required for model weights)

### Installation

1. **Install Git LFS** (required to pull large model weights)
   ```bash
   # Windows (with Git for Windows)
   git lfs install
   
   # macOS
   brew install git-lfs && git lfs install
   
   # Ubuntu/Debian
   sudo apt install git-lfs && git lfs install
   ```

2. **Clone repository**
   ```bash
   git clone https://github.com/Proxy939/Recursia-Dx-ML.git
   cd Recursia-Dx-ML
   ```

3. **Backend setup**
   ```bash
   cd backend
   npm install
   cp .env.example .env  # Configure MongoDB URI and Gemini API key
   ```

4. **Frontend setup**
   ```bash
   cd client
   npm install
   ```

5. **ML setup**
   ```bash
   cd ml
   pip install -r requirements.txt
   ```

### Running the Application

To run the application, you need to spin up the 4 different microservices:

```bash
# Terminal 1 - Brain Tumor API (TensorFlow)
cd ml
python api/brain_tumor_api.py --port 5002

# Terminal 2 - Pneumonia API (PyTorch)
cd ml
python api/pneumonia_api.py --port 5003

# Terminal 3 - ML Gateway (Flask Router)
cd ml
python api/app.py --port 5001

# Terminal 4 - Node.js Backend
cd backend
node server.js

# Terminal 5 - React Frontend
cd client
npm run dev
```

Access the application at `http://localhost:5173`

## Environment Variables

### Backend (.env)
```env
MONGODB_URI=mongodb://localhost:27017/recursiadx
PORT=5000
GEMINI_API_KEY=your_gemini_api_key_here
ML_SERVICE_URL=http://localhost:5001
```

### ML Gateway Configuration
The ML Gateway (`app.py`) internally routes traffic to:
- `BRAIN_TUMOR_API_URL=http://localhost:5002`
- `PNEUMONIA_API_URL=http://localhost:5003`

## Project Structure

```
RecursiaDx/
├── backend/          # Node.js API server (MongoDB, Gemini integrations)
├── client/           # React frontend
├── ml/               # Python Machine Learning microservices
│   ├── api/
│   │   ├── app.py               # Main ML Gateway proxy (Port 5001)
│   │   ├── brain_tumor_api.py   # TF/Keras inference + Grad-CAM (Port 5002)
│   │   └── pneumonia_api.py     # PyTorch ensemble + Grad-CAM (Port 5003)
│   └── pneumonia_detection/     # Custom PyTorch model package
└── test_images/      # Sample MRIs and X-rays for testing
```

## Model Information

| Model | Task | Framework | Metric |
|-------|------|--------------|----------|
| **EfficientNetB3** | Brain Tumor Detection (4-class) | TensorFlow | ~99% Accuracy |
| **DenseNet121 + EfficientNet-B0** | Pneumonia Detection (Binary ensemble) | PyTorch | 0.88 AUC |

## License

This project is for educational and research purposes.

---

**Status**: ✅ Production Ready | 🔄 Active Development

For issues or questions, please open a GitHub issue.
