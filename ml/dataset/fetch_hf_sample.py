"""
Fetch a small balanced real/fake face sample from a public HuggingFace dataset
(Hemg/deepfake-and-real-images) into datasets/processed/{real,fake}.

No credentials required — uses the public HF datasets-server rows API.
Label mapping in that dataset: 0 = Fake, 1 = Real.

Usage:
    python -m ml.dataset.fetch_hf_sample --per-class 300 --out datasets/processed
"""
from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

DATASET = "Hemg/deepfake-and-real-images"
ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    f"?dataset={DATASET}&config=default&split=train&offset={{offset}}&length={{length}}"
)
LABEL_DIR = {0: "fake", 1: "real"}
PAGE = 100
UA = {"User-Agent": "Mozilla/5.0 (deepfake-finalyear dataset fetch)"}


def _get_json(url: str):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as r:
        import json
        return json.load(r)


def _download(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=90) as r:
            data = r.read()
        if len(data) < 512:  # too small to be a real jpg
            return False
        dest.write_bytes(data)
        return True
    except Exception:
        return False


def fetch(per_class: int, out_dir: Path, start_offset: int = 0,
          only_label: int | None = None) -> None:
    counts = {0: 0, 1: 0}
    targets = {only_label} if only_label is not None else set(LABEL_DIR)
    for lbl, name in LABEL_DIR.items():
        (out_dir / name).mkdir(parents=True, exist_ok=True)

    offset = start_offset
    pages_without_progress = 0
    while any(counts[l] < per_class for l in targets):
        url = ROWS_URL.format(offset=offset, length=PAGE)
        try:
            data = _get_json(url)
        except Exception as e:
            print(f"[warn] page at offset {offset} failed: {e}; retrying once")
            time.sleep(2)
            try:
                data = _get_json(url)
            except Exception:
                offset += PAGE
                continue

        rows = data.get("rows", [])
        if not rows:
            print("[info] no more rows; stopping")
            break

        progressed = False
        for item in rows:
            row = item["row"]
            lbl = int(row["label"])
            if lbl not in targets or counts[lbl] >= per_class:
                continue
            src = row["image"].get("src") if isinstance(row["image"], dict) else None
            if not src:
                continue
            name = LABEL_DIR[lbl]
            dest = out_dir / name / f"hf_{lbl}_{counts[lbl]:04d}.jpg"
            if _download(src, dest):
                counts[lbl] += 1
                progressed = True

        print(f"offset={offset:>6}  real={counts[1]:>4}  fake={counts[0]:>4}")
        offset += PAGE
        pages_without_progress = 0 if progressed else pages_without_progress + 1
        if pages_without_progress >= 5:
            print("[warn] no progress for several pages; stopping early")
            break

    print(f"\nDone. real={counts[1]} fake={counts[0]} -> {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch small HF real/fake face sample")
    p.add_argument("--per-class", type=int, default=300)
    p.add_argument("--out", type=Path, default=Path("datasets/processed"))
    p.add_argument("--start-offset", type=int, default=0)
    p.add_argument("--only-label", type=int, default=None,
                   help="0=Fake, 1=Real; omit for both")
    args = p.parse_args()
    fetch(args.per_class, args.out, args.start_offset, args.only_label)


if __name__ == "__main__":
    main()
