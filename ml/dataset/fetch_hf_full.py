"""
Download the FULL Hemg/deepfake-and-real-images dataset (~190k images, ~1.8 GB
of parquet) and extract it into an ImageFolder layout: <out>/{real,fake}/*.jpg

Unlike fetch_hf_sample.py (row API, 100 rows per request), this pulls the
parquet shards directly, which is orders of magnitude faster.
Label mapping in that dataset: 0 = Fake, 1 = Real.

Usage:
    python -m ml.dataset.fetch_hf_full --out datasets/hemg
    python -m ml.dataset.fetch_hf_full --out datasets/hemg --size 224   # resize on extract
    python -m ml.dataset.fetch_hf_full --skip-download                  # re-extract only
"""
from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from PIL import Image

DATASET = "Hemg/deepfake-and-real-images"
LABEL_DIR = {0: "fake", 1: "real"}


def download(raw_dir: Path, retries: int = 8, workers: int = 4) -> list[Path]:
    """Download shards, resuming partial files and retrying on dropped connections."""
    for attempt in range(1, retries + 1):
        try:
            snapshot_download(
                repo_id=DATASET,
                repo_type="dataset",
                local_dir=str(raw_dir),
                allow_patterns=["data/*.parquet"],
                max_workers=workers,
            )
            break
        except Exception as e:
            if attempt == retries:
                raise
            wait = min(5 * attempt, 30)
            print(f"[warn] download attempt {attempt} failed ({e.__class__.__name__}: {e}); "
                  f"retrying in {wait}s", flush=True)
            time.sleep(wait)
    return sorted((raw_dir / "data").glob("*.parquet"))


def extract(shards: list[Path], out_dir: Path, size: int | None) -> dict[str, int]:
    counts = {name: 0 for name in LABEL_DIR.values()}
    for name in LABEL_DIR.values():
        (out_dir / name).mkdir(parents=True, exist_ok=True)

    for shard_idx, shard in enumerate(shards):
        pf = pq.ParquetFile(shard)
        print(f"[shard {shard_idx + 1}/{len(shards)}] {shard.name}: "
              f"{pf.metadata.num_rows} rows", flush=True)
        for batch in pf.iter_batches(batch_size=512, columns=["image", "label"]):
            images = batch.column("image").to_pylist()
            labels = batch.column("label").to_pylist()
            for img, lbl in zip(images, labels):
                name = LABEL_DIR[int(lbl)]
                dest = out_dir / name / f"hemg_s{shard_idx}_{counts[name]:06d}.jpg"
                counts[name] += 1
                if dest.exists():
                    continue
                data = img["bytes"] if isinstance(img, dict) else img
                if not data:
                    continue
                try:
                    if size is None:
                        dest.write_bytes(data)
                    else:
                        im = Image.open(io.BytesIO(data)).convert("RGB")
                        im = im.resize((size, size), Image.BICUBIC)
                        im.save(dest, "JPEG", quality=95)
                except Exception as e:  # corrupt image: skip it
                    print(f"[warn] skipped {dest.name}: {e}")
        print(f"  running totals: {counts}", flush=True)
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch full Hemg deepfake dataset")
    p.add_argument("--raw", default="datasets/raw/hemg")
    p.add_argument("--out", default="datasets/hemg")
    p.add_argument("--size", type=int, default=None,
                   help="resize to SIZExSIZE on extract (default: keep original)")
    p.add_argument("--skip-download", action="store_true")
    args = p.parse_args()

    raw_dir, out_dir = Path(args.raw), Path(args.out)
    shards = (sorted((raw_dir / "data").glob("*.parquet")) if args.skip_download
              else download(raw_dir))
    if not shards:
        raise SystemExit(f"No parquet shards found under {raw_dir / 'data'}")
    counts = extract(shards, out_dir, args.size)
    print(f"\nDone. {sum(counts.values())} images -> {out_dir}  {counts}")


if __name__ == "__main__":
    main()
