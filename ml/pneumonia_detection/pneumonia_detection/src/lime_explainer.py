"""
lime_explainer.py — LIME superpixel explanation for chest X-ray predictions.
"""

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
_normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def _arr_to_tensor(rgb_uint8: np.ndarray) -> torch.Tensor:
    """Convert HxWx3 uint8 RGB array → normalized tensor [1,3,224,224]."""
    pil = Image.fromarray(rgb_uint8)
    t = T.ToTensor()(pil)
    t = _normalize(t)
    return t.unsqueeze(0)


def run_lime(gray_uint8: np.ndarray, densenet, efficientnet,
             device: str, num_samples: int = 400,
             num_features: int = 8) -> np.ndarray:
    """
    Run LIME on the image using ensemble predictor.
    Returns float32 mask [224,224] with positive region values.
    num_samples: fewer = faster but less stable (400 is reasonable)
    """
    try:
        from lime import lime_image
        from skimage.segmentation import mark_boundaries
    except ImportError:
        raise ImportError("Install lime: pip install lime")

    # LIME expects HxWx3 RGB
    rgb = np.stack([gray_uint8, gray_uint8, gray_uint8], axis=-1)

    def predictor(images: np.ndarray) -> np.ndarray:
        """images: [N, H, W, 3] uint8 RGB"""
        batch = []
        for img in images:
            t = _arr_to_tensor(img.astype(np.uint8))
            batch.append(t)
        batch = torch.cat(batch, dim=0).to(device)

        with torch.no_grad():
            dn = F.softmax(densenet(batch), dim=1).cpu().numpy()
            en = F.softmax(efficientnet(batch), dim=1).cpu().numpy()
        return (dn + en) / 2.0   # [N, 2]

    explainer = lime_image.LimeImageExplainer(random_state=42)
    explanation = explainer.explain_instance(
        rgb,
        predictor,
        top_labels=2,
        hide_color=0,
        num_samples=num_samples,
        random_seed=42
    )

    # Get mask for class 1 (Pneumonia)
    _, mask = explanation.get_image_and_mask(
        label=1,
        positive_only=True,
        num_features=num_features,
        hide_rest=False
    )
    return mask.astype(np.float32)


def overlay_lime(gray_uint8: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Overlay LIME mask on original image.
    Returns RGB uint8 [224,224,3].
    """
    import cv2
    base_rgb = cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2RGB)
    green_overlay = base_rgb.copy()
    green_overlay[mask > 0] = [0, 220, 80]
    result = cv2.addWeighted(base_rgb, 0.6, green_overlay, 0.4, 0)
    # Draw contour
    mask_u8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 100), 1)
    return result
