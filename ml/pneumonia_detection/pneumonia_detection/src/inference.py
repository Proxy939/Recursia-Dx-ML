"""
inference.py — Preprocessing and ensemble inference pipeline.
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
from PIL import Image
import pydicom

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224

_normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def load_image(file_obj) -> np.ndarray:
    """
    Load an uploaded image (PNG/JPG or DICOM).
    Returns uint8 grayscale numpy array at IMAGE_SIZE x IMAGE_SIZE.
    """
    fname = getattr(file_obj, "filename", "") or getattr(file_obj, "name", "")
    if fname.lower().endswith(".dcm"):
        ds = pydicom.dcmread(file_obj)
        arr = ds.pixel_array.astype(np.float32)
        if arr.max() > 0:
            arr = (arr / arr.max() * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    else:
        pil = Image.open(file_obj).convert("L")
        arr = np.array(pil)

    # CLAHE — same as training
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    arr = clahe.apply(arr)

    # Resize
    arr = cv2.resize(arr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LANCZOS4)
    return arr  # uint8 [0,255], shape (224,224)


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convert uint8 grayscale array → normalized 3-channel tensor."""
    pil_rgb = Image.fromarray(arr, mode="L").convert("RGB")
    t = T.ToTensor()(pil_rgb)   # [3,224,224] float [0,1]
    t = _normalize(t)
    return t.unsqueeze(0)       # [1,3,224,224]


@torch.no_grad()
def predict_ensemble(tensor: torch.Tensor, densenet, efficientnet,
                     device: str, threshold: float = 0.5):
    """
    Run ensemble inference.
    Returns dict with keys:
      - label (str): 'Normal' or 'Pneumonia'
      - confidence (float): 0-100
      - prob_pneumonia (float): raw probability
      - dn_prob (float): DenseNet121 pneumonia prob
      - en_prob (float): EfficientNet-B0 pneumonia prob
    """
    tensor = tensor.to(device)

    dn_out = densenet(tensor)
    en_out = efficientnet(tensor)

    dn_prob = F.softmax(dn_out, dim=1)[0, 1].item()
    en_prob = F.softmax(en_out, dim=1)[0, 1].item()
    ens_prob = (dn_prob + en_prob) / 2.0

    label = "Pneumonia" if ens_prob >= threshold else "Normal"
    confidence = ens_prob * 100 if label == "Pneumonia" else (1 - ens_prob) * 100

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "prob_pneumonia": round(ens_prob, 4),
        "dn_prob": round(dn_prob, 4),
        "en_prob": round(en_prob, 4),
    }


def confidence_tier(result: dict) -> tuple:
    """
    Returns (color, message) based on confidence.
    color: 'green' | 'yellow' | 'red'
    """
    conf = result["confidence"]
    label = result["label"]
    if conf >= 85:
        return "green", f"High confidence: {label}"
    elif conf >= 70:
        return "yellow", f"Moderate confidence — interpret with caution"
    else:
        return "red", "Low confidence — recommend radiologist review"


def estimate_severity(heatmap_norm: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Use Grad-CAM heatmap to estimate severity.
    heatmap_norm: float32 array [0,1], same size as image.
    Returns dict with severity label and percentage.
    """
    active_frac = float(np.mean(heatmap_norm > threshold))
    pct = round(active_frac * 100, 1)
    if pct < 10:
        sev = "Minimal / Normal"
    elif pct < 20:
        sev = "Mild"
    elif pct < 35:
        sev = "Moderate"
    else:
        sev = "Severe"
    return {"severity": sev, "affected_pct": pct}
