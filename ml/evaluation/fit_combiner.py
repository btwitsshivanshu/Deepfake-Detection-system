"""
Fit the ensemble combiner (Phase 4).

Inputs : fake-probabilities from the 3 detectors (xception = GAN faces,
         sbi = face swaps, clip = AI-generated images).
Output : one calibrated fake probability (small gradient-boosted tree model).

Fitted ONLY on validation data, disjoint from the Phase-0 benchmark test
images:
  hemg_val      Hemg held-out validation split
  140k_val      140k-faces official 'valid' split
  openfake_val  OpenFake official validation split
  dff_val       DeepFakeFace wiki/insight images NOT picked by the benchmark
Each set is scored clean and WhatsApp-degraded.

Then the fitted combiner is evaluated on the benchmark's cached TEST scores.

Usage:
    python -m ml.evaluation.fit_combiner
"""
from __future__ import annotations

import json
import pickle
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from ml.config import CONFIG, PATHS
from ml.dataset.dataset import _gather_samples, previous_split
from ml.dataset.face_extraction import load_image_rgb
from ml.evaluation.benchmark import DETECTORS, SCORES, test_sets, whatsapp

DETS = ["xception", "sbi", "clip"]
OUT = Path("ml/checkpoints/combiner.pkl")
VAL_SCORES = Path("ml/reports/combiner_val_scores.npz")


def val_sets(n: int, seed: int = 11) -> dict[str, list[tuple[str, int]]]:
    rng = random.Random(seed)
    bench = {p for items in test_sets(500).values() for p, _ in items}   # never reuse test images

    def pick(reals, fakes):
        out = []
        for paths, lbl in ((reals, 0), (fakes, 1)):
            paths = [p for p in paths if str(p) not in bench]
            rng.shuffle(paths)
            got = 0
            for p in paths:
                if got == n:
                    break
                if cv2.imread(str(p)) is not None:
                    out.append((str(p), lbl)); got += 1
        return out

    hemg_all = _gather_samples("datasets/hemg")
    h_val, _ = previous_split(hemg_all, 15000, CONFIG.val_split, CONFIG.test_split, CONFIG.seed)
    a = Path("D:/archive/real_vs_fake/real-vs-fake/valid")
    of = Path("datasets/openfake/validation")
    d = Path("datasets/dff")
    return {
        "hemg_val": pick([s[0] for s in h_val if s[1] == 0], [s[0] for s in h_val if s[1] == 1]),
        "140k_val": pick(sorted((a / "real").iterdir()), sorted((a / "fake").iterdir())),
        "openfake_val": pick(sorted((of / "real").glob("*.jpg")), sorted((of / "fake").glob("*.jpg"))),
        "dff_val": pick(sorted((d / "wiki").rglob("*.jpg")), sorted((d / "insight").rglob("*.jpg"))),
    }


def score_sets(sets) -> dict[str, np.ndarray]:
    device = torch.device("cuda")
    store = dict(np.load(VAL_SCORES)) if VAL_SCORES.exists() else {}
    for det in DETS:
        if all(f"{det}|{s}|{q}" in store for s in sets for q in ("clean", "whatsapp")):
            continue
        model = DETECTORS[det](device)
        for s, items in sets.items():
            store[f"labels|{s}"] = np.array([l for _, l in items])
            for q in ("clean", "whatsapp"):
                probs = []
                for i in range(0, len(items), 32):
                    ims = [load_image_rgb(p) for p, _ in items[i:i + 32]]
                    if q == "whatsapp":
                        ims = [whatsapp(im, 0) for im in ims]
                    probs.append(model(ims))
                store[f"{det}|{s}|{q}"] = np.concatenate(probs)
                print(f"scored {det:9s} {s:13s} {q}", flush=True)
        del model
        torch.cuda.empty_cache()
        np.savez(VAL_SCORES, **store)
    return store


def stack(store, names):
    X, y, g = [], [], []
    for s in names:
        for q in ("clean", "whatsapp"):
            X.append(np.stack([store[f"{d}|{s}|{q}"] for d in DETS], 1))
            y.append(store[f"labels|{s}"])
            g += [f"{s}|{q}"] * len(y[-1])
    return np.concatenate(X), np.concatenate(y), np.array(g)


def main() -> None:
    sets = val_sets(500)
    for k, v in sets.items():
        print(f"{k}: {len(v)} images")
    store = score_sets(sets)
    X, y, _ = stack(store, list(sets))
    clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                         random_state=0).fit(X, y)
    OUT.write_bytes(pickle.dumps({"model": clf, "detectors": DETS}))
    print(f"combiner saved -> {OUT}")

    # honest evaluation on the benchmark TEST scores
    test = dict(np.load(SCORES))
    Xt, yt, gt = stack(test, ["hemg", "140k", "openfake", "dff_swap"])
    pt = clf.predict_proba(Xt)[:, 1]
    report = {}
    print(f"\n{'TEST set':20s} {'old xception':>13s} {'ENSEMBLE':>9s}  auc")
    for grp in dict.fromkeys(gt):
        m = gt == grp
        a_old, a_new = accuracy_score(yt[m], Xt[m, 0] > .5), accuracy_score(yt[m], pt[m] > .5)
        auc = roc_auc_score(yt[m], pt[m])
        report[grp] = {"old_acc": a_old, "acc": a_new, "auc": auc}
        print(f"{grp:20s} {a_old:13.1%} {a_new:9.1%}  {auc:.3f}")
    a_old, a_new, auc = accuracy_score(yt, Xt[:, 0] > .5), accuracy_score(yt, pt > .5), roc_auc_score(yt, pt)
    real = yt == 0
    unc = (pt > CONFIG.uncertain_low) & (pt < CONFIG.uncertain_high)
    report["ALL"] = {"old_acc": a_old, "acc": a_new, "auc": auc,
                     "false_alarm_old": float(np.mean(Xt[real, 0] > .5)),
                     "false_alarm": float(np.mean(pt[real] > .5)),
                     "uncertain_share": float(unc.mean()),
                     "acc_when_decided": float(accuracy_score(yt[~unc], pt[~unc] > .5))}
    print(f"{'ALL':20s} {a_old:13.1%} {a_new:9.1%}  {auc:.3f}")
    print(json.dumps(report["ALL"], indent=2))
    (PATHS.reports / "ensemble_test_metrics.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
