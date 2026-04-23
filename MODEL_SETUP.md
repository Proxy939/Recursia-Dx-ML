# Model Setup Guide

## Quick Start

After cloning the repository, the model files need to be set up:

### Option 1: Models Already Included (Platelet Only)
The platelet detection models are included via Git LFS:
- ✅ `Platelates/runs/detect/bccd_blood_cells/weights/best.pt`
- ✅ `Platelates/yolo11n.pt`
- ✅ `Platelates/yolov8s.pt`

### Option 2: Train / Download Missing Models

The Brain Tumor and Malaria models are NOT included in the repository due to size constraints.

#### Required Models

| Model | Size | Location | Status |
|-------|------|----------|--------|
| EfficientNetB3 (Brain Tumor) | ~50MB | `ml/models/weights/brain_tumor_efficientnetb3.h5` | ❌ Train with `train_brain_tumor.py` |
| Malaria (InceptionV3) | ~85MB | `ml/models/weights/malaria_inceptionv3.pth` | ❌ Not included |
| Platelet (YOLOv8) | ~22MB each | `ml/models/weights/platelet_yolov8.pt` | ✅ Included via LFS |

---

## Setup Instructions

### 1. Install Git LFS (if not already installed)

```bash
# Windows (using Chocolatey)
choco install git-lfs

# macOS
brew install git-lfs

# Linux (Debian/Ubuntu)
sudo apt-get install git-lfs

# Initialize LFS
git lfs install
```

### 2. Clone with LFS

```bash
git clone https://github.com/AyushX1602/Recursia-Dx-ML-.git
cd Recursia-Dx-ML-
git lfs pull  # Download LFS files
```

### 3. Train the Brain Tumor Model

```bash
cd ml
pip install -r requirements.txt
python train_brain_tumor.py
```

This will:
1. Download the brain tumor MRI dataset from Kaggle (7023 images)
2. Train EfficientNetB3 with transfer learning + fine-tuning
3. Save the model to `ml/models/weights/brain_tumor_efficientnetb3.h5`
4. Save training history to `ml/models/weights/training_history.json`

**Expected accuracy: 98-99%**

### 4. Download / Obtain Malaria Model

**Contact the repository owner** to obtain:
- `malaria_inceptionv3.pth` (Malaria model)

Place it in `ml/models/weights/`.

---

## Verification

Run this to verify models are in place:

```bash
# Check Brain Tumor model
ls ml/models/weights/brain_tumor_efficientnetb3.h5

# Check Malaria model
ls ml/models/weights/malaria_inceptionv3.pth

# Check Platelet models
ls ml/models/weights/platelet_yolov8.pt
```

---

## Alternative: Train Your Own Models

### Brain Tumor Model
```bash
cd ml
python train_brain_tumor.py
```
See `ml/train_brain_tumor.py` for details.

### Malaria Model
See `Malaria-Disease-Detection-Using-Transfer-Learning/README.md` for training instructions.

### Platelet Model
See `Platelates/README.md` for training instructions.

---

## Troubleshooting

### "Model not found" error
- Ensure model files are in `ml/models/weights/`
- Check file names match exactly (case-sensitive)
- Verify Git LFS is installed and initialized

### Git LFS not downloading files
```bash
git lfs install
git lfs pull
```

### Brain Tumor training fails
- Ensure you have TensorFlow 2.15.0 installed
- Ensure you have a Kaggle account configured for kagglehub
- Check GPU memory (at least 4GB recommended)
