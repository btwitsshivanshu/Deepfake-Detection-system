"""
Deepfake image dataset (Phase 5 — Dataset loader & splitting).

Expects a directory layout produced by the preprocessing pipeline:

    datasets/processed/
        real/   *.jpg
        fake/   *.jpg

`build_dataloaders` performs a stratified train/val/test split and wraps
each split in a DataLoader with the appropriate transforms.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from ml.config import CLASS_TO_IDX
from ml.dataset.face_extraction import load_image_rgb
from ml.dataset.transforms import train_transforms, eval_transforms

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class DeepfakeImageDataset(Dataset):
    """A list of (image_path, label) samples with a given transform."""

    def __init__(self, samples: list[tuple[Path, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = load_image_rgb(path)
        augmented = self.transform(image=image)
        return augmented["image"], label


def _gather_samples(root: str | Path) -> list[tuple[Path, int]]:
    root = Path(root)
    samples: list[tuple[Path, int]] = []
    for cls, idx in CLASS_TO_IDX.items():
        folder = root / cls
        if not folder.exists():
            continue
        for f in folder.rglob("*"):
            if f.suffix.lower() in IMG_EXTS:
                samples.append((f, idx))
    if not samples:
        raise FileNotFoundError(
            f"No images found under {root}. Expected real/ and fake/ subfolders."
        )
    return samples


def build_dataloaders(
    root: str | Path,
    image_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 4,
    seed: int = 42,
    max_per_class: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader) with stratified splits."""
    samples = _gather_samples(root)
    if max_per_class:
        samples = _capped(samples, max_per_class, seed)
    labels = [lbl for _, lbl in samples]

    train_val, test = train_test_split(
        samples, test_size=test_split, stratify=labels, random_state=seed
    )
    tv_labels = [lbl for _, lbl in train_val]
    rel_val = val_split / (1.0 - test_split)
    train, val = train_test_split(
        train_val, test_size=rel_val, stratify=tv_labels, random_state=seed
    )

    train_ds = DeepfakeImageDataset(train, train_transforms(image_size))
    val_ds = DeepfakeImageDataset(val, eval_transforms(image_size))
    test_ds = DeepfakeImageDataset(test, eval_transforms(image_size))

    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                  persistent_workers=num_workers > 0,
                  prefetch_factor=4 if num_workers > 0 else None)
    return (
        DataLoader(train_ds, shuffle=True, **common),
        DataLoader(val_ds, shuffle=False, **common),
        DataLoader(test_ds, shuffle=False, **common),
    )


# --- Fixed held-out split (shared by every later training / evaluation script) ---
def _capped(samples, max_per_class, seed):
    """Per-class cap, identical to build_dataloaders(max_per_class=...)."""
    rng = random.Random(seed)
    out = []
    for cls_idx in sorted({lbl for _, lbl in samples}):
        cls = [s for s in samples if s[1] == cls_idx]
        rng.shuffle(cls)
        out.extend(cls[:max_per_class])
    return out


def previous_split(all_samples, max_per_class, val_split, test_split, seed):
    samples = _capped(all_samples, max_per_class, seed) if max_per_class else all_samples
    labels = [l for _, l in samples]
    train_val, test = train_test_split(samples, test_size=test_split,
                                       stratify=labels, random_state=seed)
    tv_labels = [l for _, l in train_val]
    _, val = train_test_split(train_val, test_size=val_split / (1.0 - test_split),
                              stratify=tv_labels, random_state=seed)
    return val, test
