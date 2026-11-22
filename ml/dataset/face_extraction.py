"""
Face & frame extraction (Phase 5 — Frame Extraction / Preprocessing).

- extract_face:   detect and crop the largest face from an image (OpenCV Haar / DNN).
- sample_frames:  uniformly sample N frames from a video.
- extract_faces_from_video: sample frames then crop faces for inference.

Detector: OpenCV YuNet (small CNN, ~230 KB ONNX in ml/weights/) — far better
than Haar on small, angled, partly occluded and low-quality faces, and it also
returns 5 landmarks (eyes, nose, mouth corners). Falls back to the Haar
cascade if the YuNet model file is missing.
"""
from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

# Haar cascade ships with OpenCV — fallback only.
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_FACE_CASCADE = cv2.CascadeClassifier(_CASCADE_PATH)

_YUNET_PATH = Path(__file__).resolve().parents[1] / "weights" / "face_detection_yunet_2023mar.onnx"
_YUNET_MAX_SIDE = 1280          # downscale huge photos before detection (speed)
_YUNET_SCORE = 0.7
_tls = threading.local()        # FaceDetectorYN is not thread-safe -> one per thread


def _yunet():
    det = getattr(_tls, "yunet", None)
    if det is None and _YUNET_PATH.exists():
        det = cv2.FaceDetectorYN.create(str(_YUNET_PATH), "", (320, 320),
                                        score_threshold=_YUNET_SCORE)
        _tls.yunet = det
    return det


def detect_largest_face(image: np.ndarray):
    """
    Return (box, landmarks) for the largest face in an RGB image, or (None, None).
    box = (x, y, w, h) ints; landmarks = (5, 2) float array (right eye, left eye,
    nose tip, right mouth corner, left mouth corner) or None with Haar fallback.
    """
    det = _yunet()
    if det is not None:
        H, W = image.shape[:2]
        s = min(1.0, _YUNET_MAX_SIDE / max(H, W))
        img = image if s == 1.0 else cv2.resize(image, (round(W * s), round(H * s)),
                                                interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        det.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, faces = det.detect(bgr)
        if faces is None or len(faces) == 0:
            return None, None
        f = max(faces, key=lambda r: r[2] * r[3]) / s
        x, y, w, h = (int(round(v)) for v in f[:4])
        x, y = max(0, x), max(0, y)
        w, h = min(w, W - x), min(h, H - y)
        if w <= 1 or h <= 1:
            return None, None
        return (x, y, w, h), f[4:14].reshape(5, 2)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    box = _largest_face(faces)
    return (tuple(int(v) for v in box), None) if box is not None else (None, None)


def _largest_face(faces: np.ndarray) -> tuple[int, int, int, int] | None:
    if len(faces) == 0:
        return None
    # pick the box with the largest area
    return max(faces, key=lambda b: b[2] * b[3])


def extract_face(
    image: np.ndarray,
    margin: float = 0.25,
    out_size: int = 224,
    face_ratio: float | None = None,
) -> np.ndarray | None:
    """
    Detect the largest face and return an RGB crop of size (out_size, out_size).
    Returns None if no face is found (caller may fall back to full frame).

    face_ratio: if set, ignore `margin` and take a square crop in which the face
    box spans this fraction of the crop width (e.g. 0.47 matches the framing of
    the face datasets the models were trained on). The square is shifted /
    shrunk to stay inside the image, so an already-close-up photo is used whole.
    """
    box, _ = detect_largest_face(image)
    if box is None:
        return None

    x, y, w, h = box
    if face_ratio:
        H, W = image.shape[:2]
        side = min(int(round(max(w, h) / face_ratio)), H, W)
        cx, cy = x + w / 2, y + h / 2
        x0 = int(min(max(0, cx - side / 2), W - side))
        y0 = int(min(max(0, cy - side / 2), H - side))
        crop = image[y0:y0 + side, x0:x0 + side]
        return cv2.resize(crop, (out_size, out_size)) if crop.size else None

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
    face_ratio: float | None = None,
) -> list[np.ndarray]:
    """Sample frames from a video and return the cropped face of each frame."""
    faces: list[np.ndarray] = []
    for frame in sample_frames(video_path, num_frames):
        face = extract_face(frame, out_size=out_size, face_ratio=face_ratio)
        faces.append(face if face is not None else cv2.resize(frame, (out_size, out_size)))
    return faces


def load_image_rgb(path: str | Path) -> np.ndarray:
    """Read an image from disk as RGB (raises if unreadable)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
