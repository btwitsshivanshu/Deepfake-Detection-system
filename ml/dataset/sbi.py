"""
Self-Blended Images (SBI) — Shiohara & Yamasaki, CVPR 2022.

Builds a pseudo face-swap from a single REAL face: a slightly altered copy of
the face (colour / sharpness / scale / position) is blended back onto the
original through a soft, randomly deformed face mask. The result carries the
artefacts every face swap has — blending boundary, colour mismatch, slight
geometric misalignment — without any identity change. A detector trained on
real-vs-SBI learns those generic artefacts and therefore generalises to swap
methods it never saw.

Our variant builds the mask from YuNet's face box + 5 landmarks (the paper
used 81 dlib landmarks).
"""
from __future__ import annotations

import albumentations as A
import cv2
import numpy as np

# Changes applied to the *source* copy before blending (paper: "source-target
# augmentation" — colour + frequency + slight resize/translate).
_SOURCE_COLOR = A.Compose([
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    A.OneOf([A.Sharpen(alpha=(0.2, 0.5), p=1.0),
             A.GaussianBlur(blur_limit=(3, 5), p=1.0),
             A.Downscale(scale_range=(0.5, 0.9), p=1.0)], p=0.5),
])


def _source_geometry(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    s = rng.uniform(0.95, 1.05)
    tx, ty = rng.uniform(-0.03, 0.03) * w, rng.uniform(-0.03, 0.03) * h
    M = np.float32([[s, 0, (1 - s) * w / 2 + tx], [0, s, (1 - s) * h / 2 + ty]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def _face_mask(shape, box, lm, rng: np.random.Generator) -> np.ndarray:
    """Soft (0..1) elliptical face mask, randomly sized, deformed and blurred."""
    h, w = shape[:2]
    x, y, bw, bh = box
    if lm is not None:
        cx = float(np.mean(lm[:, 0]))
        cy = float(np.mean(lm[:, 1]))                 # ~ nose level
    else:
        cx, cy = x + bw / 2, y + bh * 0.55
    cy += rng.uniform(-0.05, 0.08) * bh
    ax = bw * rng.uniform(0.36, 0.52)
    ay = bh * rng.uniform(0.45, 0.62)
    mask = np.zeros((h, w), np.float32)
    cv2.ellipse(mask, (int(cx), int(cy)), (max(2, int(ax)), max(2, int(ay))),
                rng.uniform(-12, 12), 0, 360, 1.0, -1)

    # smooth random elastic deformation of the mask outline
    g = 6
    dx = cv2.resize(rng.uniform(-1, 1, (g, g)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(rng.uniform(-1, 1, (g, g)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    amp = rng.uniform(0.02, 0.06) * bw
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mask = cv2.remap(mask, gx + dx * amp, gy + dy * amp, cv2.INTER_LINEAR)

    # random erosion/dilation then blur -> soft boundary of random width
    k = int(rng.integers(1, 6)) * 2 + 1
    kernel = np.ones((k, k), np.uint8)
    if rng.random() < 0.5:
        mask = cv2.erode(mask, kernel)
    else:
        mask = cv2.dilate(mask, kernel)
    b = int(rng.integers(2, 16)) * 2 + 1
    mask = cv2.GaussianBlur(mask, (b, b), 0)
    return np.clip(mask, 0, 1)


def self_blend(img: np.ndarray, box, lm, rng: np.random.Generator) -> np.ndarray:
    """Return an SBI pseudo-fake of an RGB uint8 face image."""
    src = _SOURCE_COLOR(image=img)["image"]
    src = _source_geometry(src, rng)
    mask = _face_mask(img.shape, box, lm, rng)
    alpha = rng.choice([0.25, 0.5, 0.75, 1.0, 1.0, 1.0])
    m = (mask * alpha)[..., None]
    out = src.astype(np.float32) * m + img.astype(np.float32) * (1 - m)
    return np.clip(out, 0, 255).astype(np.uint8)
