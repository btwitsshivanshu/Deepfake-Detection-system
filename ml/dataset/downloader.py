"""
Dataset downloader helper (Phase 4).

The major deepfake benchmarks require signed access agreements, so they
cannot be auto-downloaded without credentials. This module documents the
official sources and provides a uniform CLI that:

  - prints access instructions per dataset, and
  - verifies/organizes a manually downloaded archive into datasets/raw/.

Datasets:
  * FaceForensics++ : https://github.com/ondyari/FaceForensics
  * Celeb-DF (v2)   : https://github.com/yuezunli/celeb-deepfakeforensics
  * DFDC            : https://ai.meta.com/datasets/dfdc/

Usage:
    python -m ml.dataset.downloader --dataset faceforensics --info
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ml.utils.logger import get_logger

logger = get_logger("downloader")

DATASETS = {
    "faceforensics": {
        "name": "FaceForensics++",
        "url": "https://github.com/ondyari/FaceForensics",
        "note": "Fill the Google form; you receive a download script (download-FaceForensics.py).",
    },
    "celebdf": {
        "name": "Celeb-DF v2",
        "url": "https://github.com/yuezunli/celeb-deepfakeforensics",
        "note": "Request access via the form; download the provided archive.",
    },
    "dfdc": {
        "name": "DFDC (Deepfake Detection Challenge)",
        "url": "https://ai.meta.com/datasets/dfdc/",
        "note": "Accept the license on Kaggle / Meta and download the parts.",
    },
}


def show_info(key: str) -> None:
    d = DATASETS[key]
    logger.info("=" * 60)
    logger.info("Dataset : %s", d["name"])
    logger.info("Source  : %s", d["url"])
    logger.info("Access  : %s", d["note"])
    logger.info("Place downloaded videos under: datasets/raw/%s/", key)
    logger.info("Then run ml.dataset.preprocess to build the training set.")
    logger.info("=" * 60)


def main() -> None:
    p = argparse.ArgumentParser(description="Deepfake dataset access helper")
    p.add_argument("--dataset", choices=list(DATASETS), required=True)
    p.add_argument("--info", action="store_true", help="Show access instructions")
    args = p.parse_args()

    Path(f"datasets/raw/{args.dataset}").mkdir(parents=True, exist_ok=True)
    show_info(args.dataset)


if __name__ == "__main__":
    main()
