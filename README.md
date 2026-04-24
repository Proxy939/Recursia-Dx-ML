# RecursiaDx


AI-powered digital pathology platform for automated brain tumor detection, malaria detection, and platelet counting with AI-generated clinical reports.

## Overview

RecursiaDx is a comprehensive medical image analysis platform that integrates state-of-the-art machine learning models for:
- **Brain Tumor Analysis**: 4-class MRI classification using EfficientNetB3 (Glioma, Meningioma, Pituitary, No Tumor)
- **Malaria Detection**: Blood smear analysis using InceptionV3
- **Platelet Counting**: Automated platelet detection using YOLOv11
- **AI Report Generation**: Gemini AI-powered clinical summary generation

## Key Features

✅ **Multi-Modal Analysis**
- Brain tumor MRI classification (EfficientNetB3 Transfer Learning)
- Malaria parasite detection (Transfer Learning)
- Platelet counting (YOLO object detection)

✅ **AI-Powered Workflows**
- 5-step clinical workflow (Upload → Analysis → Dashboard → Review → Report)
- Real-time ML inference with interactive visualizations
- Gemini AI-generated clinical summaries and recommendations

✅ **Professional Reporting**
- Dynamic report generation with AI interpretation
- Morphological findings analysis
- Clinical recommendations
- HIPAA-compliant data handling

✅ **Interactive UI**
- Dark/Light theme support
- Sample type adaptation (Blood vs. MRI)
- Real-time status tracking
- Demo mode for testing

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | React 18 + Vite + Tailwind CSS |
| **Backend** | Node.js 18+ + Express + MongoDB |
| **ML Models** | TensorFlow/Keras (EfficientNetB3) + PyTorch (InceptionV3) + YOLOv11 |
| **AI Integration** | Google Gemini 2.5 Flash |
| **Database** | MongoDB Atlas |

## Quick Start

### Prerequisites
- Node.js 18+ and npm
- Python 3.10+ with pip
- MongoDB (local or Atlas)
- **Git LFS** (required for model weights)
- Google Gemini API key (optional, for AI reports)

### Installation

1. **Install Git LFS** (required for model weights)
   ```bash
   # Windows (with Git for Windows)
   git lfs install
   
   # macOS
   brew install git-lfs && git lfs install
   
   # Ubuntu/Debian
   sudo apt install git-lfs && git lfs install
   ```

2. **Clone repository** (models download automatically via LFS)
   ```bash
   git clone https://github.com/AyushX1602/Recursia-Dx-ML-.git
   cd Recursia-Dx-ML-
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
   # Train the brain tumor model (one-time):
   python train_brain_tumor.py
   ```

### Running the Application

**Option 1: Cross-platform Python script (Recommended)**
```bash
python start_all.py
```

**Option 2: Platform-specific scripts**
```bash
# Windows
.\start_all.bat

# macOS/Linux
chmod +x start_all.sh
./start_all.sh
```

**Option 3: Manual start (all platforms)**
```bash
# Terminal 1 - Brain Tumor API
cd ml
python api/brain_tumor_api.py --port 5002

# Terminal 2 - Main ML API
cd ml
python api/app.py

# Terminal 3 - Backend
cd backend
node server.js

# Terminal 4 - Frontend
cd client
npm run dev
```

Access the application at `http://localhost:5173`

## Environment Variables

### Backend (.env)
```env
MONGODB_URI=mongodb://localhost:27017/recursiadx
PORT=5001
GEMINI_API_KEY=your_gemini_api_key_here  # Optional
ML_SERVICE_URL=http://localhost:5000
BRAIN_TUMOR_API_URL=http://localhost:5002
```

### ML (.env)
```env
BRAIN_TUMOR_MODEL_PATH=models/weights/brain_tumor_efficientnetb3.h5
MALARIA_MODEL_PATH=models/weights/malaria_inceptionv3.pth
PLATELET_MODEL_PATH=models/weights/platelet_yolov8.pt
```

## Workflow Steps

1. **Sample Upload** - Upload MRI/blood images
2. **Analysis** - ML models process the images
3. **Dashboard** - View results and visualizations
4. **Technician Review** - Approve or request re-analysis
5. **Report Generation** - Generate AI-powered clinical reports

## Project Structure

```
RecursiaDx/
├── backend/          # Node.js API server
│   ├── routes/      # API endpoints
│   ├── models/      # MongoDB schemas
│   ├── services/    # Gemini integration
│   └── server.js    # Entry point
├── client/          # React frontend
│   └── src/
│       ├── components/   # UI components
│       └── lib/         # Utilities
├── ml/              # ML services
│   ├── api/
│   │   ├── app.py              # Malaria/Platelet API (port 5000)
│   │   └── brain_tumor_api.py  # Brain Tumor API (port 5002)
│   ├── models/
│   │   ├── malaria_predictor.py
│   │   ├── platelet_counter.py
│   │   └── weights/            # Model weight files
│   └── train_brain_tumor.py    # Training script
└── test/            # Test images

```

## Model Information

| Model | Task | Architecture | Accuracy |
|-------|------|--------------|----------|
| EfficientNetB3 | Brain Tumor Detection (4-class) | Transfer Learning (ImageNet) | ~99% |
| InceptionV3 | Malaria Detection | Transfer Learning | ~95% |
| YOLOv11n | Platelet Counting | Object Detection | ~90% |

## API Endpoints

### Backend API (Port 5001)
- `POST /api/samples/upload` - Upload sample images
- `POST /api/samples/demo-analysis` - Demo mode analysis
- `POST /api/reports/generate/:id` - Generate report
- `POST /api/reports/generate-full/:id` - Generate with Gemini

### ML APIs
- **Brain Tumor**: `http://localhost:5002/analyze` (EfficientNetB3)
- **Blood**: `http://localhost:5000/analyze` (Malaria + Platelet)

## Gemini Integration

The platform uses Google Gemini 2.5 Flash for:
- Clinical summary generation
- Result interpretation
- Morphological findings description
- Clinical recommendations
- Diagnostic conclusions

**Without Gemini API key**: System falls back to rule-based summaries.

## Documentation

- [Startup Guide](STARTUP_GUIDE.md)
- [Model Setup](MODEL_SETUP.md)
- [ML Setup](ml/README.md)
- [Backend API](backend/README.md)

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is for educational and research purposes.

## Acknowledgments

- EfficientNet by Google Research
- Gemini AI by Google
- Open-source ML communities

---

**Status**: ✅ Production Ready | 🔄 Active Development

For issues or questions, please open a GitHub issue.
