"""
gradcam.py — Grad-CAM for DenseNet121 and EfficientNet-B0.
"""

import numpy as np
import torch
import cv2


class GradCAM:
    def __init__(self, model, model_name: str, device: str = "cpu"):
        self.model = model
        self.model_name = model_name
        self.device = device
        self._fwd = None
        self._bwd = None
        self._handles = []
        self._register_hooks()

    def _register_hooks(self):
        # Remove any old hooks
        for h in self._handles:
            h.remove()
        self._handles.clear()

        if self.model_name == "densenet121":
            # Last conv block before global pool
            target = self.model.features.denseblock4.denselayer16.conv2
        else:  # efficientnet_b0
            target = self.model.features[8][0]   # last MBConv block's conv

        def fwd_hook(_, __, output):
            self._fwd = output.detach()

        def bwd_hook(_, __, grad_output):
            self._bwd = grad_output[0].detach()

        self._handles.append(target.register_forward_hook(fwd_hook))
        self._handles.append(target.register_full_backward_hook(bwd_hook))

    def generate(self, gray_uint8: np.ndarray, class_idx: int = 1) -> np.ndarray:
        """
        Run Grad-CAM on a raw uint8 grayscale image.
        Returns float32 heatmap [0,1] sized 224x224.
        """
        from src.inference import to_tensor

        tensor = to_tensor(gray_uint8).to(self.device)
        tensor.requires_grad_(True)

        self.model.eval()
        out = self.model(tensor)
        self.model.zero_grad()
        out[0, class_idx].backward()

        if self._bwd is None or self._fwd is None:
            # Fallback: return blank heatmap
            return np.zeros((224, 224), dtype=np.float32)

        weights = self._bwd.mean(dim=(2, 3), keepdim=True)   # [1,C,1,1]
        cam = (weights * self._fwd).sum(dim=1).squeeze()      # [H,W]
        cam = torch.relu(cam).cpu().detach().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)
        return cam.astype(np.float32)


def overlay_heatmap(gray_uint8: np.ndarray, cam: np.ndarray,
                    alpha: float = 0.45) -> np.ndarray:
    """
    Overlay jet colormap heatmap on original grayscale X-ray.
    Returns RGB uint8 [224,224,3].
    """
    base_rgb = cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2RGB)
    heat_uint8 = (cam * 255).astype(np.uint8)
    colormap = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(base_rgb, 1 - alpha, colormap, alpha, 0)
    return blended
