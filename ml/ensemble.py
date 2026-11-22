"""
Ensemble inference (Phase 4): three specialised detectors + a learned combiner.

  xception  GAN-generated faces       (face crop, 224 px)
  sbi       face swaps / face edits   (face crop, 256 px, EfficientNet-B4 trained
                                       with Self-Blended Images)
  clip      AI-generated images       (whole image, CLIP ViT-L/14 + linear head)

The combiner (gradient-boosted trees, ml/checkpoints/combiner.pkl) turns the
three fake-probabilities into one. It was fitted only on validation data.

The verdict uses the same real / uncertain / fake bands as the single models,
and when an image is judged fake we report which detector fired hardest as
the likely kind of fake.
"""
from __future__ import annotations

import pickle
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from ml.config import CONFIG
from ml.dataset.face_extraction import extract_face, load_image_rgb, sample_frames
from ml.dataset.screenshot import crop_screenshot
from ml.dataset.transforms import eval_transforms
from ml.explain.gradcam import GradCAM, overlay_heatmap
from ml.models.factory import load_model
from ml.utils.seed import get_device

_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = {
    "xception": _ROOT / "ml/checkpoints/xception_best.pth",
    "sbi": _ROOT / "ml/checkpoints/efficientnet_b4_sbi_best.pth",
    "clip_head": _ROOT / "ml/checkpoints/clip_vitl14_head.pt",
    "combiner": _ROOT / "ml/checkpoints/combiner.pkl",
}
FAKE_TYPE = {
    "xception": "GAN-generated face",
    "sbi": "Face swap / face edit",
    "clip": "AI-generated image",
}

_lock = threading.Lock()      # Grad-CAM hooks + shared models: one request at a time
_state: dict = {}


def _load():
    if _state:
        return _state
    device = get_device(CONFIG.device)
    import open_clip
    clip, _, pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    head = torch.load(WEIGHTS["clip_head"], map_location=device)
    comb = pickle.loads(WEIGHTS["combiner"].read_bytes())
    _state.update(
        device=device,
        xception=load_model("xception", str(WEIGHTS["xception"]), device=device),
        sbi=load_model("efficientnet_b4", str(WEIGHTS["sbi"]), device=device),
        t224=eval_transforms(224), t256=eval_transforms(256),
        clip=clip.to(device).eval().half() if device.type == "cuda" else clip.to(device).eval(),
        clip_pre=pre, clip_w=head["weight"].to(device), clip_b=head["bias"].to(device),
        combiner=comb["model"], order=comb["detectors"],
    )
    return _state


@torch.no_grad()
def _face_prob(model, transform, face: np.ndarray, device) -> float:
    x = transform(image=face)["image"].unsqueeze(0).to(device)
    return float(torch.softmax(model(x).float(), 1)[0, 1])


@torch.no_grad()
def _clip_prob(image: np.ndarray, s) -> float:
    x = s["clip_pre"](Image.fromarray(image)).unsqueeze(0).to(s["device"])
    if s["device"].type == "cuda":
        x = x.half()
    f = torch.nn.functional.normalize(s["clip"].encode_image(x).float(), dim=-1)
    return float(torch.sigmoid(f @ s["clip_w"].T + s["clip_b"]))


def _score(image: np.ndarray):
    """Return (scores dict, face crop 256 px or None)."""
    s = _load()
    face = extract_face(image, out_size=256, face_ratio=CONFIG.infer_face_ratio)
    crop = face if face is not None else cv2.resize(image, (256, 256))
    scores = {
        "xception": _face_prob(s["xception"], s["t224"], cv2.resize(crop, (224, 224)), s["device"]),
        "sbi": _face_prob(s["sbi"], s["t256"], crop, s["device"]),
        "clip": _clip_prob(image, s),
    }
    return scores, face


def _combine(scores: dict) -> float:
    s = _load()
    x = np.array([[scores[d] for d in s["order"]]])
    return float(s["combiner"].predict_proba(x)[0, 1])


def _verdict(p: float) -> tuple[str, float]:
    if p >= CONFIG.uncertain_high:
        return "fake", p
    if p <= CONFIG.uncertain_low:
        return "real", 1.0 - p
    return "uncertain", max(p, 1.0 - p)


def _fake_type(verdict: str, scores: dict) -> str | None:
    if verdict == "real":
        return None
    return FAKE_TYPE[max(scores, key=scores.get)]


def _heatmap(face: np.ndarray | None, scores: dict, heatmap_dir) -> str | None:
    """Grad-CAM from whichever face model is more suspicious (faces only)."""
    if face is None:
        return None
    s = _load()
    use_sbi = scores["sbi"] >= scores["xception"]
    model = s["sbi"] if use_sbi else s["xception"]
    img = face if use_sbi else cv2.resize(face, (224, 224))
    t = s["t256"] if use_sbi else s["t224"]
    tensor = t(image=img)["image"].unsqueeze(0).to(s["device"])
    with GradCAM(model) as cam:
        hm = cam.generate(tensor, class_idx=1)
    overlay = overlay_heatmap(img, hm)
    Path(heatmap_dir).mkdir(parents=True, exist_ok=True)
    out = Path(heatmap_dir) / f"heatmap_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(str(out), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return str(out)


def predict_image_ensemble(image_path, heatmap_dir) -> dict:
    t0 = time.time()
    with _lock:
        image = load_image_rgb(image_path)
        image, ss_box = crop_screenshot(image)       # phone screenshot -> analyse the photo only
        scores, face = _score(image)
        p = _combine(scores)
        verdict, conf = _verdict(p)
        heat = _heatmap(face, scores, heatmap_dir)
    return {
        "verdict": verdict,
        "confidence": round(conf, 4),
        "fake_probability": round(p, 4),
        "frames_analyzed": 1,
        "inference_ms": int((time.time() - t0) * 1000),
        "heatmap_path": heat,
        "fake_type": _fake_type(verdict, scores),
        "detector_scores": {k: round(v, 4) for k, v in scores.items()},
        "screenshot_crop": list(ss_box) if ss_box else None,
        "frames": [{"frame_index": 0, "fake_probability": round(p, 4), "heatmap_path": heat}],
    }


def predict_video_ensemble(video_path, heatmap_dir, num_frames: int = CONFIG.video_frames) -> dict:
    t0 = time.time()
    with _lock:
        frames = sample_frames(video_path, num_frames)
        if not frames:
            raise ValueError("No frames could be extracted from the video.")
        per, probs, worst = [], [], (-1.0, None, None)
        for i, fr in enumerate(frames):
            scores, face = _score(fr)
            p = _combine(scores)
            per.append(scores)
            probs.append(p)
            if p > worst[0]:
                worst = (p, face, scores)
        p = float(np.mean(probs))
        mean_scores = {k: float(np.mean([s[k] for s in per])) for k in per[0]}
        verdict, conf = _verdict(p)
        heat = _heatmap(worst[1], worst[2], heatmap_dir)
    return {
        "verdict": verdict,
        "confidence": round(conf, 4),
        "fake_probability": round(p, 4),
        "frames_analyzed": len(frames),
        "inference_ms": int((time.time() - t0) * 1000),
        "heatmap_path": heat,
        "fake_type": _fake_type(verdict, mean_scores),
        "detector_scores": {k: round(v, 4) for k, v in mean_scores.items()},
        "frames": [{"frame_index": i, "fake_probability": round(float(q), 4)} for i, q in enumerate(probs)],
    }
