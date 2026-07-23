"""
Dataset preprocessing pipeline (Phase 4/5).

Converts raw benchmark datasets (FaceForensics++, Celeb-DF, DFDC) — which are
distributed as videos + labels — into a flat, model-ready image folder:

    datasets/processed/real/*.jpg
    datasets/processed/fake/*.jpg

Usage:
    python -m ml.dataset.preprocess \
        --videos-dir datasets/raw/faceforensics \
        --labels datasets/raw/faceforensics/labels.csv \
        --out datasets/processed --frames 15
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from ml.dataset.face_extraction import extract_faces_from_video
from ml.utils.logger import get_logger

logger = get_logger("preprocess")


def _load_labels(csv_path: Path) -> dict[str, str]:
    """CSV with columns: filename,label  (label in {real,fake})."""
    mapping: dict[str, str] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            mapping[row["filename"]] = row["label"].strip().lower()
    return mapping


def preprocess(videos_dir: Path, labels_csv: Path, out_dir: Path, frames: int) -> None:
    labels = _load_labels(labels_csv)
    (out_dir / "real").mkdir(parents=True, exist_ok=True)
    (out_dir / "fake").mkdir(parents=True, exist_ok=True)

    saved = 0
    for video in sorted(videos_dir.rglob("*")):
        if video.suffix.lower() not in {".mp4", ".avi", ".mov"}:
            continue
        label = labels.get(video.name)
        if label not in {"real", "fake"}:
            logger.warning("No/invalid label for %s — skipped", video.name)
            continue

        faces = extract_faces_from_video(video, num_frames=frames)
        for i, face in enumerate(faces):
            bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            out = out_dir / label / f"{video.stem}_f{i:03d}.jpg"
            cv2.imwrite(str(out), bgr)
            saved += 1
        logger.info("Processed %s (%d faces)", video.name, len(faces))

    logger.info("Done. Saved %d face crops to %s", saved, out_dir)


def main() -> None:
    p = argparse.ArgumentParser(description="Preprocess deepfake video datasets")
    p.add_argument("--videos-dir", required=True, type=Path)
    p.add_argument("--labels", required=True, type=Path)
    p.add_argument("--out", default=Path("datasets/processed"), type=Path)
    p.add_argument("--frames", default=15, type=int)
    args = p.parse_args()
    preprocess(args.videos_dir, args.labels, args.out, args.frames)


if __name__ == "__main__":
    main()
