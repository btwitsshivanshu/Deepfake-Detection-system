"""
CLIP-based AI-generated image detector (UnivFD approach, Ojha et al. CVPR 2023).

A frozen CLIP ViT-L/14 turns every image into a 768-d feature vector; only a
small linear classifier on top is trained. Because CLIP's features are general
(trained on 400M image-text pairs) the detector transfers to unseen generators
far better than a CNN trained from scratch.

Training data:
  - OpenFake subset (real LAION/Pexels photos vs 58 generators incl. Imagen,
    nano-banana (Gemini), GPT-image, Flux, Midjourney, SD, DALL-E)
  - a few thousand Hemg + 140k-faces images (GAN faces), for breadth.
Each training image is embedded twice: as-is and after a random real-world
degradation (JPEG / downscale / blur), so the classifier tolerates
screenshots & WhatsApp recompression.

Usage:
    python -m ml.training.clip_train
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import albumentations as A
import numpy as np
import open_clip
import torch
import torch.nn as nn
from PIL import Image

from ml.config import CONFIG, PATHS
from ml.dataset.dataset import _gather_samples, previous_split
from ml.utils.logger import get_logger
from ml.utils.metrics import compute_metrics

FEAT_DIR = Path("datasets/clip_feats")
DEGRADE = A.Compose([
    A.OneOf([A.ImageCompression(quality_range=(30, 90), p=1.0),
             A.Downscale(scale_range=(0.4, 0.8), p=1.0),
             A.GaussianBlur(blur_limit=(3, 5), p=1.0)], p=1.0),
    A.ImageCompression(quality_range=(60, 95), p=0.5),
])


def load_clip(device):
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    model = model.to(device).eval().half()
    return model, preprocess


@torch.no_grad()
def embed(model, preprocess, paths, device, degrade=False, bs=64, logger=None, tag=""):
    feats, t0 = [], time.time()
    for i in range(0, len(paths), bs):
        ims = []
        for p in paths[i:i + bs]:
            im = Image.open(p).convert("RGB")
            if degrade:
                im = Image.fromarray(DEGRADE(image=np.asarray(im))["image"])
            ims.append(preprocess(im))
        x = torch.stack(ims).to(device).half()
        f = model.encode_image(x).float()
        feats.append(torch.nn.functional.normalize(f, dim=-1).cpu())
        if logger and (i // bs) % 20 == 0:
            logger.info("  embed %s %d/%d (%.0fs)", tag, min(i + bs, len(paths)), len(paths), time.time() - t0)
    return torch.cat(feats).numpy()


def cached(name, fn):
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    f = FEAT_DIR / f"{name}.npy"
    if f.exists():
        return np.load(f)
    arr = fn()
    np.save(f, arr)
    return arr


def openfake_split(root: Path, split: str):
    out = []
    for cls, lbl in (("real", 0), ("fake", 1)):
        out += [(str(p), lbl) for p in sorted((root / split / cls).glob("*.jpg"))]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Train the CLIP AI-image detector")
    p.add_argument("--openfake", default="datasets/openfake")
    p.add_argument("--hemg", default="datasets/hemg")
    p.add_argument("--archive", default="D:/archive/real_vs_fake/real-vs-fake/train")
    p.add_argument("--face-extra", type=int, default=2000, help="per face dataset")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--out", default="ml/checkpoints/clip_vitl14_head.pt")
    args = p.parse_args()

    PATHS.ensure()
    logger = get_logger("clip", PATHS.logs / "clip_train.log")
    device = torch.device("cuda")
    rng = random.Random(CONFIG.seed)

    of = Path(args.openfake)
    train = openfake_split(of, "train")
    val = openfake_split(of, "validation")
    test = openfake_split(of, "test")

    # GAN-face breadth (from training pools only)
    hemg_all = _gather_samples(args.hemg)
    h_val, h_test = previous_split(hemg_all, 15000, CONFIG.val_split, CONFIG.test_split, CONFIG.seed)
    held = {str(s[0]) for s in h_val} | {str(s[0]) for s in h_test}
    hemg_pool = [(str(s[0]), s[1]) for s in hemg_all if str(s[0]) not in held]
    arch_pool = [(str(p), l) for l, c in ((0, "real"), (1, "fake"))
                 for p in sorted(Path(args.archive, c).iterdir())]
    for pool in (hemg_pool, arch_pool):
        rng.shuffle(pool)
        train += [s for s in pool if s[1] == 0][: args.face_extra // 2]
        train += [s for s in pool if s[1] == 1][: args.face_extra // 2]
    logger.info("Images: train=%d val=%d test=%d", len(train), len(val), len(test))

    model, preprocess = load_clip(device)
    tp, ty = [s[0] for s in train], np.array([s[1] for s in train])
    Xtr = np.concatenate([
        cached(f"train_clean_{len(tp)}", lambda: embed(model, preprocess, tp, device, logger=logger, tag="train")),
        cached(f"train_degr_{len(tp)}", lambda: embed(model, preprocess, tp, device, True, logger=logger, tag="train-degraded")),
    ])
    ytr = np.concatenate([ty, ty])
    Xva = cached(f"val_{len(val)}", lambda: embed(model, preprocess, [s[0] for s in val], device, logger=logger, tag="val"))
    yva = np.array([s[1] for s in val])
    Xte = cached(f"test_{len(test)}", lambda: embed(model, preprocess, [s[0] for s in test], device, logger=logger, tag="test"))
    yte = np.array([s[1] for s in test])

    # linear head (logistic regression) trained with AdamW; pick weight decay on val
    def fit(wd):
        torch.manual_seed(CONFIG.seed)
        head = nn.Linear(Xtr.shape[1], 1).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=wd)
        X = torch.tensor(Xtr, device=device); y = torch.tensor(ytr, dtype=torch.float32, device=device)
        for _ in range(args.epochs):
            perm = torch.randperm(len(X), device=device)
            for i in range(0, len(X), 512):
                idx = perm[i:i + 512]
                loss = nn.functional.binary_cross_entropy_with_logits(head(X[idx]).squeeze(1), y[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        return head

    def score(head, X):
        with torch.no_grad():
            return torch.sigmoid(head(torch.tensor(X, device=device)).squeeze(1)).cpu().numpy()

    best = None
    for wd in (0.0, 1e-4, 1e-3, 1e-2):
        head = fit(wd)
        pv = score(head, Xva)
        m = compute_metrics(yva.tolist(), (pv > 0.5).astype(int).tolist(), pv.tolist())
        logger.info("wd=%g  val acc=%.4f auc=%.4f", wd, m.accuracy, m.roc_auc)
        if best is None or m.roc_auc > best[1].roc_auc:
            best = (head, m, wd)
    head, vm, wd = best
    pt = score(head, Xte)
    tm = compute_metrics(yte.tolist(), (pt > 0.5).astype(int).tolist(), pt.tolist())
    logger.info("BEST wd=%g | val acc=%.4f auc=%.4f | TEST acc=%.4f auc=%.4f",
                wd, vm.accuracy, vm.roc_auc, tm.accuracy, tm.roc_auc)

    # per-generator accuracy on the OpenFake test split
    import csv
    man = {r["path"].replace("\\", "/"): r["model"] for r in
           csv.DictReader(open(of / "manifest.csv", encoding="utf-8"))}
    per = {}
    for (path, lbl), prob in zip(test, pt):
        key = man.get(str(Path(path).relative_to(of)).replace("\\", "/"), "?") if lbl == 1 else "REAL photos"
        per.setdefault(key, []).append(int((prob > 0.5) == lbl))
    rows = sorted(((k, np.mean(v), len(v)) for k, v in per.items()), key=lambda r: -r[2])
    for k, a, n in rows[:30]:
        logger.info("  %-40s acc=%.2f (n=%d)", k, a, n)

    torch.save({"weight": head.weight.detach().cpu(), "bias": head.bias.detach().cpu(),
                "clip": "ViT-L-14/openai", "val": vm.to_dict(), "test": tm.to_dict()}, args.out)
    (PATHS.reports / "clip_test_metrics.json").write_text(json.dumps(
        {"val": vm.to_dict(), "test": tm.to_dict(),
         "per_generator": {k: {"acc": float(a), "n": n} for k, a, n in rows}}, indent=2))
    print(f"\nSaved CLIP head: {args.out}")


if __name__ == "__main__":
    main()
