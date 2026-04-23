"""
model.py — Model architecture definitions.
Matches exactly what train_ensemble_v2.py saved.
"""

import torch
import torch.nn as nn
import torchvision.models as models


def build_densenet121():
    m = models.densenet121(weights=None)
    nf = m.classifier.in_features
    m.classifier = nn.Sequential(
        nn.Linear(nf, 512), nn.ReLU(), nn.BatchNorm1d(512),
        nn.Dropout(0.3), nn.Linear(512, 2))
    return m


def build_efficientnet_b0():
    m = models.efficientnet_b0(weights=None)
    nf = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(nf, 512), nn.ReLU(),
        nn.BatchNorm1d(512), nn.Dropout(0.2), nn.Linear(512, 2))
    return m


def load_ensemble(models_dir: str, device: str = "cpu"):
    """Load both models from saved weights. Returns (densenet, efficientnet)."""
    import os
    dn = build_densenet121()
    dn.load_state_dict(torch.load(
        os.path.join(models_dir, "densenet121_best.pth"),
        map_location=device))
    dn.to(device).eval()

    en = build_efficientnet_b0()
    en.load_state_dict(torch.load(
        os.path.join(models_dir, "efficientnet_b0_best.pth"),
        map_location=device))
    en.to(device).eval()

    return dn, en
