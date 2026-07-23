"""
Grad-CAM explainability (Phase 7).

Implements Gradient-weighted Class Activation Mapping from scratch using
forward/backward hooks — no external CAM dependency required. Produces a
heatmap that highlights the image regions most responsible for the
"fake" prediction, then overlays it on the original face.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization" (ICCV 2017).
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ml.models.base import BaseDetector


class GradCAM:
    def __init__(self, model: BaseDetector, target_layer=None):
        self.model = model.eval()
        # In-place ReLUs create autograd view+inplace conflicts with backward
        # hooks; disable in-place so Grad-CAM works for any backbone.
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = False
        self.target_layer = target_layer or model.target_layer()
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._fwd = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def remove(self) -> None:
        """Detach hooks — always call when done to avoid leaks."""
        self._fwd.remove()
        self._bwd.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """
        Args:
            input_tensor: preprocessed image tensor of shape (1, 3, H, W).
            class_idx: target class (defaults to the predicted class).
        Returns:
            A (H, W) heatmap in [0, 1].
        """
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        logits[0, class_idx].backward(retain_graph=True)

        # Global-average-pool the gradients → channel importance weights.
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)       # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)   # (1, 1, h, w)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:],
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def overlay_heatmap(
    rgb_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a [0,1] heatmap over an RGB image and return an RGB overlay."""
    h, w = rgb_image.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h))
    colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(alpha * colored + (1 - alpha) * rgb_image)
    return overlay
