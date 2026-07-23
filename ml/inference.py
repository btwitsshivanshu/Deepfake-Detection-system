"""
Unified inference engine (Phase 6/7 → consumed by the FastAPI backend).

Provides high-level `predict_image` and `predict_video` functions that:
  1. preprocess input (face crop / frame sampling),
  2. run the selected model,
  3. compute confidence,
  4. generate a Grad-CAM heatmap and save the overlay,
  5. return a serializable result dict.

Models are cached in-process so weights load only once.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch

from ml.config import CONFIG, IDX_TO_CLASS
from ml.dataset.face_extraction import (
    extract_face,
    extract_faces_from_video,
    load_image_rgb,
)
from ml.dataset.transforms import eval_transforms
from ml.explain.gradcam import GradCAM, overlay_heatmap
from ml.models.factory import load_model
from ml.utils.seed import get_device

_MODEL_CACHE: dict[str, torch.nn.Module] = {}
_TRANSFORM = eval_transforms(CONFIG.image_size)
_DEVICE = get_device(CONFIG.device)


def _get_model(architecture: str, weights_path: str):
    key = f"{architecture}:{weights_path}"
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_model(
            architecture, weights_path, num_classes=CONFIG.num_classes, device=_DEVICE
        )
    return _MODEL_CACHE[key]


def _to_tensor(face_rgb: np.ndarray) -> torch.Tensor:
    return _TRANSFORM(image=face_rgb)["image"].unsqueeze(0).to(_DEVICE)


def _infer_face(model, face_rgb: np.ndarray) -> tuple[int, float]:
    tensor = _to_tensor(face_rgb)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    idx = int(probs.argmax().item())
    return idx, float(probs[idx].item())


def predict_image(
    image_path: str | Path,
    architecture: str,
    weights_path: str,
    heatmap_dir: str | Path,
) -> dict:
    """Run detection on a single image and produce a Grad-CAM overlay."""
    t0 = time.time()
    model = _get_model(architecture, weights_path)

    image = load_image_rgb(image_path)
    face = extract_face(image, out_size=CONFIG.image_size)
    if face is None:  # fall back to whole image if no face detected
        face = cv2.resize(image, (CONFIG.image_size, CONFIG.image_size))

    idx, confidence = _infer_face(model, face)
    fake_prob = confidence if idx == 1 else 1.0 - confidence

    # Grad-CAM overlay
    tensor = _to_tensor(face)
    with GradCAM(model) as cam:
        heatmap = cam.generate(tensor, class_idx=1)
    overlay = overlay_heatmap(face, heatmap)

    Path(heatmap_dir).mkdir(parents=True, exist_ok=True)
    heatmap_path = Path(heatmap_dir) / f"heatmap_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(str(heatmap_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return {
        "verdict": IDX_TO_CLASS[idx],
        "confidence": round(confidence, 4),
        "fake_probability": round(float(fake_prob), 4),
        "frames_analyzed": 1,
        "inference_ms": int((time.time() - t0) * 1000),
        "heatmap_path": str(heatmap_path),
        "frames": [{"frame_index": 0, "fake_probability": round(float(fake_prob), 4),
                    "heatmap_path": str(heatmap_path)}],
    }


def predict_video(
    video_path: str | Path,
    architecture: str,
    weights_path: str,
    heatmap_dir: str | Path,
    num_frames: int = CONFIG.video_frames,
    agg: str = CONFIG.frame_agg,
) -> dict:
    """Sample frames, run per-frame detection and aggregate to a video verdict."""
    t0 = time.time()
    model = _get_model(architecture, weights_path)
    faces = extract_faces_from_video(video_path, num_frames=num_frames,
                                     out_size=CONFIG.image_size)
    if not faces:
        raise ValueError("No frames could be extracted from the video.")

    Path(heatmap_dir).mkdir(parents=True, exist_ok=True)
    frame_results, fake_probs = [], []
    key_heatmap = None
    best_fake = -1.0

    for i, face in enumerate(faces):
        idx, conf = _infer_face(model, face)
        fp = conf if idx == 1 else 1.0 - conf
        fake_probs.append(fp)

        # keep a heatmap for the most-fake frame (most informative)
        if fp > best_fake:
            best_fake = fp
            tensor = _to_tensor(face)
            with GradCAM(model) as cam:
                heatmap = cam.generate(tensor, class_idx=1)
            overlay = overlay_heatmap(face, heatmap)
            key_heatmap = Path(heatmap_dir) / f"heatmap_{uuid.uuid4().hex}.jpg"
            cv2.imwrite(str(key_heatmap), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        frame_results.append({"frame_index": i, "fake_probability": round(float(fp), 4)})

    fake_probs_np = np.array(fake_probs)
    if agg == "max":
        video_fake = float(fake_probs_np.max())
    elif agg == "vote":
        video_fake = float((fake_probs_np > 0.5).mean())
    else:  # mean
        video_fake = float(fake_probs_np.mean())

    verdict = "fake" if video_fake > 0.5 else "real"
    confidence = video_fake if verdict == "fake" else 1.0 - video_fake

    return {
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "fake_probability": round(video_fake, 4),
        "frames_analyzed": len(faces),
        "inference_ms": int((time.time() - t0) * 1000),
        "heatmap_path": str(key_heatmap) if key_heatmap else None,
        "frames": frame_results,
    }
