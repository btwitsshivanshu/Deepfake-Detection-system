"""
Screenshot benchmark: does cropping the photo out of a phone screenshot help?

For each benchmark test set, every image is rendered inside a synthetic phone
screenshot (status bar, header, captions, buttons; half of them then
JPEG-compressed as if shared on WhatsApp). The full ensemble is run
  (a) on the whole screenshot, and
  (b) on the region returned by find_photo_region()  [what the app now does].

Usage:
    python -m ml.evaluation.screenshot_eval --n 250
"""
from __future__ import annotations

import argparse
import json
import pickle

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from ml.config import PATHS
from ml.dataset.face_extraction import load_image_rgb
from ml.dataset.screenshot import crop_screenshot, make_fake_screenshot
from ml.evaluation.benchmark import DETECTORS, test_sets

DETS = ["xception", "sbi", "clip"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=250, help="images per class per set")
    args = p.parse_args()

    sets = test_sets(500)
    rng = np.random.default_rng(3)
    shots, crops, labels, groups, found = [], [], [], [], []
    for name, items in sets.items():
        sel = [it for it in items if it[1] == 0][: args.n] + [it for it in items if it[1] == 1][: args.n]
        for path, lbl in sel:
            ss, _ = make_fake_screenshot(load_image_rgb(path), rng)
            if rng.random() < 0.5:
                ok, enc = cv2.imencode(".jpg", cv2.cvtColor(ss, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
                ss = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            crop, box = crop_screenshot(ss)
            shots.append(ss); crops.append(crop); labels.append(lbl); groups.append(name); found.append(box is not None)
    labels, groups, found = np.array(labels), np.array(groups), np.array(found)
    print(f"{len(shots)} screenshots, photo region found in {found.mean():.1%}", flush=True)

    device = torch.device("cuda")
    scores = {"whole": {}, "crop": {}}
    for det in DETS:
        model = DETECTORS[det](device)
        for mode, imgs in (("whole", shots), ("crop", crops)):
            scores[mode][det] = np.concatenate([model(imgs[i:i + 32]) for i in range(0, len(imgs), 32)])
        print(f"scored {det}", flush=True)
        del model
        torch.cuda.empty_cache()

    comb = pickle.load(open("ml/checkpoints/combiner.pkl", "rb"))["model"]
    report = {}
    print(f"\n{'set':10s} {'whole screenshot':>17s} {'cropped (app)':>14s}")
    for g in list(dict.fromkeys(groups)) + ["ALL"]:
        m = np.ones(len(labels), bool) if g == "ALL" else groups == g
        row = {}
        for mode in ("whole", "crop"):
            X = np.stack([scores[mode][d][m] for d in DETS], 1)
            pr = comb.predict_proba(X)[:, 1]
            row[mode] = {"acc": float(accuracy_score(labels[m], pr > .5)),
                         "auc": float(roc_auc_score(labels[m], pr))}
        report[g] = row
        print(f"{g:10s} {row['whole']['acc']:9.1%} (auc {row['whole']['auc']:.2f}) "
              f"{row['crop']['acc']:7.1%} (auc {row['crop']['auc']:.2f})")
    report["found_rate"] = float(found.mean())
    (PATHS.reports / "screenshot_eval.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
