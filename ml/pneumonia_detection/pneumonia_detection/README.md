# PneumoAI — Chest X-Ray Pneumonia Detection

An AI-powered pneumonia detection system using an ensemble of DenseNet121 and EfficientNet-B0, trained on the RSNA Pneumonia Detection Challenge dataset.

## Features
- **Ensemble model** (DenseNet121 + EfficientNet-B0 soft voting)
- **Grad-CAM** visual attention heatmaps
- **LIME** superpixel explainability
- **Severity scoring** from Grad-CAM area analysis
- **DICOM file support** (real clinical format)
- **PDF diagnostic report** download
- **Prediction history** (SQLite dashboard)
- **Batch processing** for multiple images

## Project Structure
```
pneumonia_detection/
├── kaggle/
│   └── train_ensemble_v2.py    # Kaggle training script
├── models/                      # Place trained weights here
│   ├── densenet121_best.pth
│   ├── efficientnet_b0_best.pth
│   └── norm_stats.json
├── metrics/                     # Place training outputs here
│   ├── test_metrics.json
│   ├── training_metrics.csv
│   └── figures/
├── src/
│   ├── model.py
│   ├── inference.py
│   ├── gradcam.py
│   ├── lime_explainer.py
│   ├── database.py
│   └── report.py
└── app.py
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train models on Kaggle
1. Go to kaggle.com → New Notebook
2. Add dataset: `rsna-pneumonia-detection-challenge`
3. Enable GPU (T4 x2)
4. Run `kaggle/train_ensemble_v2.py`
5. Download `/kaggle/working/output/`

### 3. Place outputs
```
models/densenet121_best.pth
models/efficientnet_b0_best.pth
models/norm_stats.json
metrics/test_metrics.json
metrics/training_metrics.csv
metrics/figures/*.png
```

### 4. Run the app
```bash
streamlit run app.py
```

## Model Architecture
| Model | Pre-training | Fine-tuning |
|-------|-------------|-------------|
| DenseNet121 | ImageNet | Phase 1: head, Phase 2: full |
| EfficientNet-B0 | ImageNet | Phase 1: head, Phase 2: full |
| Ensemble | — | Soft voting (average probabilities) |

## Disclaimers
- For educational and demonstration purposes only
- NOT intended for clinical use or medical diagnosis
- Always consult a qualified radiologist
