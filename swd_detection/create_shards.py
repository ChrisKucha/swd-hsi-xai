"""
create_shards.py — Pre-process blueberry HSI boards into per-cell shards.

Run this ONCE on the Linux training machine before starting any training run.
It reads every board .npy file from the data paths in config.py, extracts
all 36 cells, normalizes each, and saves them as individual small .npy files.

After sharding, __getitem__ becomes a single np.load on a ~200 KB file
instead of loading a ~35 MB cube and extracting a cell.  This eliminates
RAM pressure (no preloading needed) while keeping per-sample I/O tiny.

Output layout
-------------
{SHARD_DIR}/
    manifest.csv          ← index of every shard (path, split, label, ripeness, sensor)
    nir/
        train/
            Infested__Ripe__board_001__Day1__c00.npy   # (B, H, W) float32
            Infested__Ripe__board_001__Day1__c01.npy
            ...
        val/
        test/
    vnir/
        train/ ...
        val/   ...
        test/  ...

Usage
-----
    # From the swd_detection/ directory on the Linux machine:
    python create_shards.py

    # Custom shard directory:
    SWD_SHARD_DIR=/data/blueberry_shards python create_shards.py

    # Only one sensor:
    python create_shards.py --sensor nir

    # Dry run (print counts, don't write files):
    python create_shards.py --dry-run
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

# ── Ensure swd_detection/ is on the path ─────────────────────────────────────
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config as CFG
from data.discovery import discover_and_split
from data.dataset   import load_cube, normalize_cube, extract_cell

# ── Shard directory ───────────────────────────────────────────────────────────
DEFAULT_SHARD_DIR = Path(
    os.environ.get("SWD_SHARD_DIR",
                   getattr(CFG, "SHARD_DIR", str(_HERE / "shards")))
)


def _safe_stem(board: dict) -> str:
    """Build a filename-safe identifier from board metadata."""
    cls  = board["class_name"].replace(" ", "_")
    rip  = board["ripeness"].replace(" ", "_")
    stem = board["file_stem"].replace(" ", "_")
    day  = board["day"]
    return f"{cls}__{rip}__{stem}__Day{day}"


def shard_sensor(
    sensor:    str,
    shard_dir: Path,
    dry_run:   bool = False,
    require_all_classes: bool = True,
) -> list:
    """
    Process all boards for one sensor.

    Returns a list of manifest rows (dicts) for every shard written.
    """
    print(f"\n{'='*60}")
    print(f"  Sensor: {sensor.upper()}")
    print(f"{'='*60}")

    paths   = CFG.PATHS[sensor]
    boards  = discover_and_split(sensor=sensor, paths=paths)

    if not boards:
        print(f"  [WARN] No boards found for sensor '{sensor}'.  Check PATHS in config.py.")
        return []

    if require_all_classes:
        required_classes = set(getattr(CFG, "CLASS_NAMES", ["Infested", "Healthy"]))
        present_classes = {b["class_name"] for b in boards}
        missing_classes = sorted(required_classes - present_classes)
        if missing_classes:
            configured = ", ".join(
                f"{cls}={paths.get(cls, '<not configured>')!r}"
                for cls in sorted(required_classes)
            )
            raise RuntimeError(
                f"Refusing to create {sensor.upper()} shards with missing class(es): "
                f"{', '.join(missing_classes)}. Configured paths: {configured}. "
                "Set the missing SWD_*_HEALTHY/SWD_*_INFECTED path(s), or rerun with "
                "--allow-single-class only for debugging."
            )

    manifest_rows = []
    total_cells   = 0
    skipped       = 0
    t0            = time.time()

    for board in boards:
        split     = board["split"]           # train / val / test
        safe_id   = _safe_stem(board)
        out_dir   = shard_dir / sensor / split

        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)

        # Load and normalize the full board cube once
        try:
            cube, _, _ = normalize_cube(
                load_cube(board["path"], sensor)
            )
        except Exception as exc:
            print(f"  [ERROR] Could not load {board['path']}: {exc}")
            skipped += 1
            continue

        # Infer cell dimensions from this cube
        H, W, B = cube.shape
        nr, nc  = CFG.BERRY_GRID_ROWS, CFG.BERRY_GRID_COLS
        row_e   = np.linspace(0, H, nr + 1, dtype=int)
        col_e   = np.linspace(0, W, nc + 1, dtype=int)
        cell_h  = int(row_e[1] - row_e[0])
        cell_w  = int(col_e[1] - col_e[0])

        # Extract and save each of the 36 cells
        for ci in range(CFG.BERRY_GRID_ROWS * CFG.BERRY_GRID_COLS):
            cell      = extract_cell(cube, ci, cell_h, cell_w)  # (B, H, W)
            shard_name = f"{safe_id}__c{ci:02d}.npy"
            shard_path = out_dir / shard_name

            if not dry_run:
                np.save(str(shard_path), cell)

            manifest_rows.append({
                "shard_path": str(shard_path),
                "sensor":     sensor,
                "split":      split,
                "label":      board["label"],
                "class_name": board["class_name"],
                "ripeness":   board["ripeness"],
                "file_stem":  board["file_stem"],
                "day":        board["day"],
                "cell_idx":   ci,
                "cell_h":     cell_h,
                "cell_w":     cell_w,
                "n_bands":    B,
            })
            total_cells += 1

        del cube   # free memory before next board

    elapsed = time.time() - t0
    print(f"  {len(boards) - skipped} boards processed, "
          f"{total_cells} shards {'(dry-run)' if dry_run else 'written'}  "
          f"[{elapsed:.1f}s, {skipped} skipped]")
    return manifest_rows


def write_manifest(rows: list, shard_dir: Path) -> Path:
    """Write combined manifest CSV for all sensors."""
    if not rows:
        return None
    manifest_path = shard_dir / "manifest.csv"
    fieldnames    = list(rows[0].keys())
    with open(manifest_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Manifest written → {manifest_path}  ({len(rows)} rows)")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Create per-cell shards for SWD training.")
    parser.add_argument(
        "--sensor", choices=["nir", "vnir", "both"], default="both",
        help="Which sensor(s) to shard (default: both)",
    )
    parser.add_argument(
        "--shard-dir", type=Path, default=DEFAULT_SHARD_DIR,
        help=f"Root output directory for shards (default: {DEFAULT_SHARD_DIR})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print statistics without writing any files.",
    )
    parser.add_argument(
        "--allow-single-class", action="store_true",
        help="Allow sharding when only one class is discovered. Use only for debugging.",
    )
    args = parser.parse_args()

    shard_dir = args.shard_dir
    if not args.dry_run:
        shard_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nShard directory: {shard_dir}")

    sensors = ["nir", "vnir"] if args.sensor == "both" else [args.sensor]

    all_rows = []
    for sensor in sensors:
        rows = shard_sensor(
            sensor,
            shard_dir,
            dry_run=args.dry_run,
            require_all_classes=not args.allow_single_class,
        )
        all_rows.extend(rows)

    if not args.dry_run and all_rows:
        write_manifest(all_rows, shard_dir)

    # Summary by split
    from collections import Counter
    split_counts = Counter(r["split"] for r in all_rows)
    sensor_counts = Counter(r["sensor"] for r in all_rows)
    print("\n  Summary by split:")
    for split, cnt in sorted(split_counts.items()):
        print(f"    {split:6s}: {cnt:6d} shards")
    print("  Summary by sensor:")
    for sensor, cnt in sorted(sensor_counts.items()):
        print(f"    {sensor:6s}: {cnt:6d} shards")
    print("\nDone.  You can now run training with sharded data.\n")


if __name__ == "__main__":
    main()
