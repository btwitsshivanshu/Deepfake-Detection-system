"""
Phase 0 benchmark: one fixed test suite for every fake type, scored by every
detector, in clean and "WhatsApp-quality" (downscaled + JPEG q50) versions.

Test sets (never used for training):
  hemg       Hemg held-out test split            real vs GAN-style fakes
  140k       140k-faces official test split      real FFHQ vs StyleGAN
  openfake   OpenFake official test split        real photos vs 58 diffusion/AR generators
  dff_swap   DeepFakeFace wiki vs insight        real celebs vs InsightFace face swaps

Detectors: xception (GAN-face model in the app), sbi (face-swap model),
clip (AI-image detector). Scores are cached per (detector, set, quality) in
ml/reports/bench_scores.npz so the combiner can be fitted on them.

Usage:
    python -m ml.evaluation.benchmark --detectors xception sbi clip
"""
from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from ml.config import CONFIG, PATHS
from ml.dataset.dataset import _gather_samples, previous_split
from ml.dataset.face_extraction import extract_face, load_image_rgb
from ml.dataset.transforms import IMAGENET_MEAN, IMAGENET_STD
from ml.models.factory import load_model
from ml.utils.metrics import compute_metrics

SCORES = Path("ml/reports/bench_scores.npz")


def test_sets(n_per_class: int, seed: int = 7) -> dict[str, list[tuple[str, int]]]:
    rng = random.Random(seed)

    def readable(paths):
        out = []
        for p in paths:
            if len(out) == n_per_class:
                break
            if cv2.imread(str(p)) is not None:      # skip corrupt files
                out.append(p)
        return out

    def pick(reals, fakes):
        rng.shuffle(reals); rng.shuffle(fakes)
        return ([(str(p), 0) for p in readable(reals)] +
                [(str(p), 1) for p in readable(fakes)])

    sets = {}
    hemg_all = _gather_samples("datasets/hemg")
    _, h_test = previous_split(hemg_all, 15000, CONFIG.val_split, CONFIG.test_split, CONFIG.seed)
    sets["hemg"] = pick([s[0] for s in h_test if s[1] == 0], [s[0] for s in h_test if s[1] == 1])
    a = Path("D:/archive/real_vs_fake/real-vs-fake/test")
    if a.exists():
        sets["140k"] = pick(sorted((a / "real").iterdir()), sorted((a / "fake").iterdir()))
    of = Path("datasets/openfake/test")
    if of.exists():
        sets["openfake"] = pick(sorted((of / "real").glob("*.jpg")), sorted((of / "fake").glob("*.jpg")))
    d = Path("datasets/dff")
    if (d / "wiki").exists() and (d / "insight").exists():
        sets["dff_swap"] = pick(sorted((d / "wiki").rglob("*.jpg")), sorted((d / "insight").rglob("*.jpg")))
    return sets


def whatsapp(img: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic 'shared on a phone' degradation: downscale + JPEG q50."""
    h, w = img.shape[:2]
    s = 0.5 if max(h, w) > 300 else 0.75
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 50])
    return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


class FaceDetector:
    def __init__(self, arch, weights, size, device):
        self.m = load_model(arch, weights, device=device)
        self.size, self.device = size, device
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, imgs):
        crops = []
        for im in imgs:
            f = extract_face(im, out_size=self.size, face_ratio=CONFIG.infer_face_ratio)
            crops.append(f if f is not None else cv2.resize(im, (self.size, self.size)))
        x = torch.tensor(np.stack(crops), device=self.device).permute(0, 3, 1, 2).float() / 255
        x = (x - self.mean) / self.std
        with torch.autocast("cuda"):
            return torch.softmax(self.m(x).float(), 1)[:, 1].cpu().numpy()


class ClipDetector:
    def __init__(self, head_path, device):
        from ml.training.clip_train import load_clip
        self.m, self.pre = load_clip(device)
        h = torch.load(head_path, map_location=device)
        self.w, self.b, self.device = h["weight"].to(device), h["bias"].to(device), device

    @torch.no_grad()
    def __call__(self, imgs):
        x = torch.stack([self.pre(Image.fromarray(im)) for im in imgs]).to(self.device).half()
        f = torch.nn.functional.normalize(self.m.encode_image(x).float(), dim=-1)
        return torch.sigmoid(f @ self.w.T + self.b).squeeze(1).cpu().numpy()


DETECTORS = {
    "xception": lambda d: FaceDetector("xception", "ml/checkpoints/xception_best.pth", 224, d),
    "sbi": lambda d: FaceDetector("efficientnet_b4", "ml/checkpoints/efficientnet_b4_sbi_best.pth", 256, d),
    "clip": lambda d: ClipDetector("ml/checkpoints/clip_vitl14_head.pt", d),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--detectors", nargs="+", default=["xception", "sbi", "clip"])
    p.add_argument("--n", type=int, default=500, help="images per class per set")
    args = p.parse_args()

    device = torch.device("cuda")
    sets = test_sets(args.n)
    store = dict(np.load(SCORES, allow_pickle=True)) if SCORES.exists() else {}
    report = {}
    for det_name in args.detectors:
        det = DETECTORS[det_name](device)
        for set_name, items in sets.items():
            labels = np.array([l for _, l in items])
            for quality in ("clean", "whatsapp"):
                probs = []
                for i in range(0, len(items), 32):
                    ims = [load_image_rgb(pth) for pth, _ in items[i:i + 32]]
                    if quality == "whatsapp":
                        ims = [whatsapp(im, i + k) for k, im in enumerate(ims)]
                    probs.append(det(ims))
                probs = np.concatenate(probs)
                store[f"{det_name}|{set_name}|{quality}"] = probs
                store[f"labels|{set_name}"] = labels
                m = compute_metrics(labels.tolist(), (probs > 0.5).astype(int).tolist(), probs.tolist())
                report[f"{det_name}|{set_name}|{quality}"] = m.to_dict()
                print(f"{det_name:9s} {set_name:9s} {quality:9s} acc={m.accuracy:.3f} auc={m.roc_auc:.3f} "
                      f"(n={len(items)})", flush=True)
        del det
        torch.cuda.empty_cache()
    np.savez(SCORES, **store)
    out = PATHS.reports / "benchmark.json"
    old = json.loads(out.read_text()) if out.exists() else {}
    old.update(report)
    out.write_text(json.dumps(old, indent=2))


if __name__ == "__main__":
    main()
