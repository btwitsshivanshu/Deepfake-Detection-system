"""
Image transforms & augmentation (Phase 5 — Data Augmentation).

Uses Albumentations for training augmentation and deterministic
normalization for validation/inference. ImageNet stats are used
because all backbones are ImageNet-pretrained.
"""
from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transforms(image_size: int = 224) -> A.Compose:
    """Augmentations that preserve forensic artifacts while adding robustness."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=(-0.05, 0.05), scale=(0.9, 1.1), rotate=(-10, 10), p=0.4),
        A.OneOf([
            A.GaussNoise(std_range=(0.01, 0.02)),            # ~ var 5-25 on a 0-255 scale
            A.ImageCompression(quality_range=(60, 100)),     # JPEG artifacts
        ], p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=12, val_shift_limit=8, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def eval_transforms(image_size: int = 224) -> A.Compose:
    """Deterministic transform for validation, testing and inference."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
