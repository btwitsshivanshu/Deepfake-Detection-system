"""
Face & frame extraction (Phase 5 — Frame Extraction / Preprocessing).

- extract_face:   detect and crop the largest face from an image (OpenCV Haar / DNN).
- sample_frames:  uniformly sample N frames from a video.
- extract_faces_from_video: sample frames then crop faces for inference.

OpenCV's Haar cascade is used to keep dependencies light and CPU-friendly.
For higher accuracy you can swap in an MTCNN/RetinaFace detector behind the
same function signatures.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Haar cascade ships with OpenCV — no extra download needed.
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_FACE_CASCADE = cv2.CascadeClassifier(_CASCADE_PATH)


def _largest_face(faces: np.ndarray) -> tuple[int, int, int, int] | None:
    if len(faces) == 0:
        return None
    # pick the box with the largest area
    return max(faces, key=lambda b: b[2] * b[3])


def extract_face(
    image: np.ndarray,
    margin: float = 0.25,
    out_size: int = 224,
) -> np.ndarray | None:
    """
    Detect the largest face and return an RGB crop of size (out_size, out_size).
    Returns None if no face is found (caller may fall back to full frame).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    box = _largest_face(faces)
    if box is None:
        return None

    x, y, w, h = box
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(image.shape[1], x + w + mx), min(image.shape[0], y + h + my)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (out_size, out_size))


def sample_frames(video_path: str | Path, num_frames: int = 20) -> list[np.ndarray]:
    """Uniformly sample `num_frames` RGB frames across the whole video."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    idxs = np.linspace(0, total - 1, num=min(num_frames, total), dtype=int)
    frames: list[np.ndarray] = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def extract_faces_from_video(
    video_path: str | Path,
    num_frames: int = 20,
    out_size: int = 224,
) -> list[np.ndarray]:
    """Sample frames from a video and return the cropped face of each frame."""
    faces: list[np.ndarray] = []
    for frame in sample_frames(video_path, num_frames):
        face = extract_face(frame, out_size=out_size)
        faces.append(face if face is not None else cv2.resize(frame, (out_size, out_size)))
    return faces


def load_image_rgb(path: str | Path) -> np.ndarray:
    """Read an image from disk as RGB (raises if unreadable)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
