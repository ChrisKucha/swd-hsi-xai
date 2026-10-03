#!/usr/bin/env python3
"""
make_demo_data.py — build the tiny demo dataset shipped with this repository.

The full model-ready dataset (per-cell hyperspectral shards) is large and is not
distributed here. This script carves a balanced subset out of the full shard set
so that anyone who clones the repository can run the pipeline end to end as a
functional demo. It samples cells per sensor, split, and class, shrinks each
cell spatially, and stores it as float16, so the demo folder remains small
enough for GitHub while still containing hundreds of samples.

IMPORTANT
---------
The demo subset is for verifying that the code runs, not for reproducing the
paper's reported numbers. Spatial resolution and sample counts are reduced far
below the real experiment. Full-data reproduction requires the complete shard
set (see README).

Run this ONCE on the machine that holds the full shards, from the repository
root, then commit the generated data_demo/ folder:

    python scripts/make_demo_data.py \
        --source-manifest /path/to/full/shards/manifest.csv \
        --out data_demo/shards

The full-shard manifest.csv is produced by swd_detection/create_shards.py.
"""
from __future__ import annotations
import argparse
import csv
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

# Manifest columns written by create_shards.py (kept identical so the demo
# manifest is a drop-in for ShardedBlueberryDataset / run_all.py).
FIELDS = [
    "shard_path", "sensor", "split", "label", "class_name",
    "ripeness", "file_stem", "day", "cell_idx", "cell_h", "cell_w", "n_bands",
]


def compact_label(value: str) -> str:
    """Normalize labels for concise, stable demo filenames."""
    value = str(value).strip()
    return "Ripe" if value.lower() in {"completelyripe", "completely"} else value


def slug(value: str) -> str:
    """Small filename-safe token for demo shard names."""
    return compact_label(value).lower().replace(" ", "_")


def nn_resize_spatial(cell: np.ndarray, hw: int) -> np.ndarray:
    """Nearest-neighbor spatial downsample of a (B, H, W) cell to (B, hw, hw).

    Nearest sampling keeps the script dependency-free (numpy only). The demo does
    not need faithful spatial content, only a valid, small input tensor.
    """
    B, H, W = cell.shape
    ri = np.linspace(0, H - 1, hw).round().astype(int)
    ci = np.linspace(0, W - 1, hw).round().astype(int)
    return cell[:, ri][:, :, ci]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-manifest", required=True,
                    help="Path to the FULL shard manifest.csv produced by create_shards.py")
    ap.add_argument("--out", default="data_demo/shards",
                    help="Output shard directory (default: data_demo/shards)")
    ap.add_argument("--n-train", type=int, default=48,
                    help="Cells per (sensor, class) for the train split (default 48)")
    ap.add_argument("--n-val", type=int, default=16,
                    help="Cells per (sensor, class) for the val split (default 16)")
    ap.add_argument("--n-test", type=int, default=16,
                    help="Cells per (sensor, class) for the test split (default 16)")
    ap.add_argument("--hw", type=int, default=24,
                    help="Spatial size of each demo cell in pixels (default 24)")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"],
                    help="Stored dtype for demo cells (default float16)")
    ap.add_argument("--seed", type=int, default=42, help="Sampling seed (default 42)")
    args = ap.parse_args()

    random.seed(args.seed)
    out_root = Path(args.out)
    per_split = {"train": args.n_train, "val": args.n_val, "test": args.n_test}

    with open(args.source_manifest, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"No rows found in {args.source_manifest}")

    # Group by (sensor, split, class) and sample a few from each group.
    groups: dict = defaultdict(list)
    for r in rows:
        groups[(r["sensor"], r["split"], r["label"])].append(r)

    selected = []
    for (sensor, split, label), grp in sorted(groups.items()):
        n = per_split.get(split, 0)
        if n <= 0 or not grp:
            continue
        random.shuffle(grp)
        selected.extend(grp[:n])

    if not selected:
        raise SystemExit("Nothing selected. Check that the manifest has the "
                         "expected sensor/split/label columns.")

    out_rows = []
    for r in selected:
        src = Path(r["shard_path"])
        try:
            cell = np.load(src)                       # (B, H, W) float32
        except FileNotFoundError:
            print(f"  [skip] missing shard: {src}")
            continue
        cell = nn_resize_spatial(cell, args.hw).astype(args.dtype)

        rel_dir = out_root / r["sensor"] / r["split"]
        rel_dir.mkdir(parents=True, exist_ok=True)
        demo_id = f"demo_{len(out_rows):03d}"
        ripeness = compact_label(r.get("ripeness", ""))
        name = (
            f"{slug(r['sensor'])}_{slug(r['split'])}_{slug(r.get('class_name', r['label']))}_"
            f"{slug(ripeness)}_d{int(r.get('day', 0)):02d}_"
            f"c{int(r.get('cell_idx', 0)):02d}_{demo_id[-3:]}.npy"
        )
        np.save(str(rel_dir / name), cell)

        out_rows.append({
            "shard_path": (Path(args.out) / r["sensor"] / r["split"] / name).as_posix(),
            "sensor":     r["sensor"],
            "split":      r["split"],
            "label":      r["label"],
            "class_name": r.get("class_name", ""),
            "ripeness":   ripeness,
            "file_stem":  demo_id,
            "day":        r.get("day", ""),
            "cell_idx":   r.get("cell_idx", ""),
            "cell_h":     args.hw,
            "cell_w":     args.hw,
            "n_bands":    cell.shape[0],
        })

    manifest_path = out_root / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out_rows)

    total_mb = sum(os.path.getsize(os.path.join(dp, f))
                   for dp, _, fs in os.walk(out_root) for f in fs) / 1e6
    print(f"\nWrote {len(out_rows)} demo shards to {out_root}")
    print(f"Manifest: {manifest_path}")
    print(f"Demo folder size: {total_mb:.1f} MB")
    print("Now commit the data_demo/ folder to the repository.")


if __name__ == "__main__":
    main()
