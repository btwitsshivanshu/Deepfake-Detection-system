"""
Stream a balanced subset of OpenFake (ComplexDataLab/OpenFake, CC-BY-NC-4.0)
for training a CLIP-based AI-image detector.

The full dataset is ~3.4 TB, so we never download whole files: we read
individual parquet row groups (~78 images each) over HTTP, from many
different shards for generator diversity, and immediately shrink every image.

Shortcut removal: every image (real and fake alike) is resized so its short
side is 256 px and re-saved as JPEG q95. Originals differ in format/size
(real = mostly JPEG, fake = mostly PNG 1024px); without this the detector can
learn "PNG/large = fake" instead of real generator traces.

Resumable: finished (split, file, row_group) units are recorded in
<out>/manifest.csv and skipped on restart.

Usage:
    python -m ml.dataset.fetch_openfake --out datasets/openfake \
        --train 30000 --val 2000 --test 4000 --workers 6
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from PIL import Image

BASE = "datasets/ComplexDataLab/OpenFake/core/"
SHORT_SIDE = 256
ROWS_PER_RG = 78  # approx; used only to decide how many row groups to fetch

_lock = threading.Lock()


def _shrink(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = im.size
    s = SHORT_SIDE / min(w, h)
    if s < 1:
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    return im


def _plan(fs, split: str, n_images: int, seed: int) -> list[tuple[str, str, int]]:
    """Pick row groups spread over as many shard files as possible."""
    files = sorted(fs.glob(BASE + f"{split}-*.parquet"))
    rng = random.Random(seed)
    with fs.open(files[0], "rb") as fh:          # row-group size differs per split
        md = pq.ParquetFile(fh).metadata
        rows_per_rg = max(1, md.num_rows // md.num_row_groups)
    n_rg = -(-n_images // rows_per_rg)
    rng.shuffle(files)
    units = []
    i = 0
    while len(units) < n_rg:
        f = files[i % len(files)]
        # row-group index chosen without reading the footer; clamped later
        units.append((split, f, rng.randrange(0, 50)))
        i += 1
    return units


def _fetch(unit, out: Path, retries: int = 6):
    split, f, rg = unit
    for attempt in range(1, retries + 1):
        try:
            fs = HfFileSystem(skip_instance_cache=True)
            with fs.open(f, "rb", block_size=16 * 2**20) as fh:
                pf = pq.ParquetFile(fh)
                rg = min(rg, pf.metadata.num_row_groups - 1)
                tb = pf.read_row_group(rg, columns=["image", "label", "model"])
            rows, nbytes = [], 0
            stem = Path(f).stem
            for k, (img, lbl, model) in enumerate(zip(tb.column("image").to_pylist(),
                                                     tb.column("label").to_pylist(),
                                                     tb.column("model").to_pylist())):
                data = img["bytes"] if isinstance(img, dict) else img
                if not data:
                    continue
                nbytes += len(data)
                try:
                    im = _shrink(data)
                except Exception:
                    continue
                cls = "fake" if lbl == "fake" else "real"
                dest = out / split / cls / f"{stem}_rg{rg}_{k}.jpg"
                dest.parent.mkdir(parents=True, exist_ok=True)
                im.save(dest, "JPEG", quality=95)
                rows.append((split, cls, model, str(dest.relative_to(out)), f, rg))
            return unit, rows, nbytes
        except Exception as e:
            if attempt == retries:
                print(f"[fail] {Path(f).name} rg{rg}: {e}", flush=True)
                return unit, [], 0
            time.sleep(15 * attempt)   # hotspot drops: wait it out


def main() -> None:
    p = argparse.ArgumentParser(description="Stream an OpenFake subset")
    p.add_argument("--out", default="datasets/openfake")
    p.add_argument("--train", type=int, default=30000)
    p.add_argument("--val", type=int, default=2000)
    p.add_argument("--test", type=int, default=4000)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.csv"
    done_units: set[tuple[str, str, int]] = set()
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                done_units.add((r["split"], r["file"], int(r["rg_requested"])))

    fs = HfFileSystem()
    units = (_plan(fs, "test", args.test, args.seed)
             + _plan(fs, "validation", args.val, args.seed)
             + _plan(fs, "train", args.train, args.seed))
    units = list(dict.fromkeys(units))
    todo = [u for u in units if u not in done_units]
    print(f"row groups planned={len(units)} already done={len(units) - len(todo)} "
          f"to fetch={len(todo)}", flush=True)

    new = not manifest.exists()
    fh = manifest.open("a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if new:
        w.writerow(["split", "label", "model", "path", "file", "rg_requested"])
        fh.flush()

    t0, total_bytes, total_imgs, n_done = time.time(), 0, 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_fetch, u, out): u for u in todo}
        for fut in as_completed(futs):
            unit, rows, nbytes = fut.result()
            with _lock:
                if rows:
                    for r in rows:
                        # record the *requested* rg so resume matches the plan
                        w.writerow([r[0], r[1], r[2], r[3], r[4], unit[2]])
                    fh.flush()
                n_done += 1
                total_bytes += nbytes
                total_imgs += len(rows)
                el = time.time() - t0
                rate = total_bytes / 1e6 / max(el, 1)
                eta = (len(todo) - n_done) * (el / n_done) / 60
                print(f"[{n_done}/{len(todo)}] +{len(rows)} imgs ({unit[0]}) | total {total_imgs} imgs, "
                      f"{total_bytes/1e9:.1f} GB, {rate:.1f} MB/s | ETA {eta:.0f} min", flush=True)
    fh.close()
    print(f"\nDone. {total_imgs} new images -> {out}")


if __name__ == "__main__":
    main()
