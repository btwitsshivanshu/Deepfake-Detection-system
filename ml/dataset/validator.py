"""
Dataset validator (Phase 4).

Sanity-checks a processed dataset folder before training:
- both classes present
- no corrupt/unreadable images
- reports class balance

Usage:  python -m ml.dataset.validator --root datasets/processed
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from ml.config import CLASS_TO_IDX
from ml.utils.logger import get_logger

logger = get_logger("validator")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def validate(root: Path) -> bool:
    ok = True
    counts: dict[str, int] = {}
    corrupt: list[str] = []

    for cls in CLASS_TO_IDX:
        folder = root / cls
        if not folder.exists():
            logger.error("Missing class folder: %s", folder)
            ok = False
            counts[cls] = 0
            continue
        files = [f for f in folder.rglob("*") if f.suffix.lower() in IMG_EXTS]
        counts[cls] = len(files)
        for f in files:
            if cv2.imread(str(f)) is None:
                corrupt.append(str(f))

    logger.info("Class distribution: %s", counts)
    if corrupt:
        ok = False
        logger.error("Found %d corrupt images (first 5): %s", len(corrupt), corrupt[:5])

    if min(counts.values(), default=0) == 0:
        ok = False
        logger.error("At least one class has zero samples.")

    total = sum(counts.values())
    if total:
        balance = {k: round(v / total, 3) for k, v in counts.items()}
        logger.info("Class balance ratio: %s", balance)

    logger.info("Validation %s", "PASSED ✅" if ok else "FAILED ❌")
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description="Validate processed dataset")
    p.add_argument("--root", default=Path("datasets/processed"), type=Path)
    args = p.parse_args()
    raise SystemExit(0 if validate(args.root) else 1)


if __name__ == "__main__":
    main()
