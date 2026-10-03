"""
create_segmented_shards.py — Build berry-mask segmented per-cell shards.

This is a segmented variant of create_shards.py. It keeps the baseline shards
untouched and writes a separate shard set where pixels outside the segmented
berry mask are zeroed:

    segmented_cell = normalized_cell * berry_mask

The saved cell shape is still (bands, H, W), so the existing training code can
use these shards without model changes.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config as CFG
from data.discovery import discover_and_split
from data.dataset import extract_cell, load_cube, normalize_cube

_SEG_DIR = _HERE.parent / "segmentation"
if str(_SEG_DIR) not in sys.path:
    sys.path.append(str(_SEG_DIR))

from segmentation import segment_board, visualize_board


DEFAULT_SEGMENTED_SHARD_DIR = Path(
    os.environ.get(
        "SWD_SEGMENTED_SHARD_DIR",
        "/media/kuchalab/New Volume/swd_detection_segmented_shards",
    )
)


def _safe_stem(board: dict) -> str:
    cls = board["class_name"].replace(" ", "_")
    rip = board["ripeness"].replace(" ", "_")
    stem = board["file_stem"].replace(" ", "_")
    day = board["day"]
    return f"{cls}__{rip}__{stem}__Day{day}"


def _cell_bounds(h: int, w: int, cell_idx: int):
    row_e = np.linspace(0, h, CFG.BERRY_GRID_ROWS + 1, dtype=int)
    col_e = np.linspace(0, w, CFG.BERRY_GRID_COLS + 1, dtype=int)
    r, c = divmod(cell_idx, CFG.BERRY_GRID_COLS)
    return row_e[r], row_e[r + 1], col_e[c], col_e[c + 1]


def _mask_cell(mask: np.ndarray, cell_idx: int, cell_h: int, cell_w: int) -> np.ndarray:
    h, w = mask.shape
    r0, r1, c0, c1 = _cell_bounds(h, w, cell_idx)
    cell_mask = mask[r0:r1, c0:c1].astype(bool)

    if cell_mask.shape != (cell_h, cell_w):
        from skimage.transform import resize as sk_resize
        cell_mask = sk_resize(
            cell_mask.astype(np.float32),
            (cell_h, cell_w),
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ) >= 0.5

    return cell_mask


def _mask_quality(cell_mask: np.ndarray) -> dict:
    mask_pixels = int(cell_mask.sum())
    cell_area = int(cell_mask.size)
    area_frac = float(mask_pixels / max(cell_area, 1))
    return {
        "mask_pixel_count": mask_pixels,
        "cell_area": cell_area,
        "mask_area_fraction": f"{area_frac:.6f}",
        "mask_detected": int(mask_pixels > 0),
    }


def _write_preview(raw_cube, intensity, masks, final_mask, board, preview_dir: Path, dry_run: bool):
    if dry_run:
        return
    preview_dir.mkdir(parents=True, exist_ok=True)
    safe_id = _safe_stem(board)
    out_path = preview_dir / f"{board['sensor']}__{safe_id}.png"
    title = f"{board['sensor'].upper()} {board['class_name']} {board['ripeness']} Day{board['day']}"
    visualize_board(raw_cube, intensity, masks, final_mask, title=title, save_path=str(out_path))


def shard_sensor_segmented(
    sensor: str,
    shard_dir: Path,
    dry_run: bool = False,
    require_all_classes: bool = True,
    save_masks: bool = True,
    preview_limit: int = 12,
    max_boards: int = None,
) -> list:
    print(f"\n{'=' * 60}")
    print(f"  Segmented sensor: {sensor.upper()}")
    print(f"{'=' * 60}")

    paths = CFG.PATHS[sensor]
    boards = discover_and_split(sensor=sensor, paths=paths)
    if not boards:
        print(f"  [WARN] No boards found for sensor '{sensor}'. Check PATHS in config.py.")
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
                f"Refusing to create {sensor.upper()} segmented shards with missing class(es): "
                f"{', '.join(missing_classes)}. Configured paths: {configured}."
            )

    if max_boards is not None:
        boards = boards[:max_boards]
        print(f"  [TEST] Limiting to first {len(boards)} board-scan(s).")

    manifest_rows = []
    total_cells = 0
    skipped = 0
    preview_count = 0
    t0 = time.time()

    board_iter = tqdm(boards, desc=f"Segment {sensor}", unit="board", dynamic_ncols=True)
    for board in board_iter:
        board_iter.set_postfix(
            split=board.get("split"),
            cls=board.get("class_name"),
            rip=board.get("ripeness"),
        )
        split = board["split"]
        safe_id = _safe_stem(board)
        out_dir = shard_dir / sensor / split
        mask_dir = shard_dir / "masks" / sensor / split
        preview_dir = shard_dir / "previews" / sensor

        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            if save_masks:
                mask_dir.mkdir(parents=True, exist_ok=True)

        try:
            raw_cube = load_cube(board["path"], sensor)
            final_mask, berry_masks, intensity = segment_board(
                raw_cube,
                sensor,
                board.get("ripeness"),
            )
            norm_cube, _, _ = normalize_cube(raw_cube)
        except Exception as exc:
            print(f"  [ERROR] Could not segment {board['path']}: {exc}")
            skipped += 1
            continue

        if preview_count < preview_limit:
            try:
                _write_preview(raw_cube, intensity, berry_masks, final_mask, board, preview_dir, dry_run)
                preview_count += 1
            except Exception as exc:
                print(f"  [WARN] Preview failed for {board['path']}: {exc}")

        h, w, n_bands = norm_cube.shape
        row_e = np.linspace(0, h, CFG.BERRY_GRID_ROWS + 1, dtype=int)
        col_e = np.linspace(0, w, CFG.BERRY_GRID_COLS + 1, dtype=int)
        cell_h = int(row_e[1] - row_e[0])
        cell_w = int(col_e[1] - col_e[0])

        for ci in range(CFG.BERRY_GRID_ROWS * CFG.BERRY_GRID_COLS):
            cell = extract_cell(norm_cube, ci, cell_h, cell_w)
            cell_mask = _mask_cell(berry_masks[ci], ci, cell_h, cell_w)
            segmented_cell = cell * cell_mask[None, :, :].astype(np.float32)

            shard_name = f"{safe_id}__c{ci:02d}.npy"
            shard_path = out_dir / shard_name
            mask_name = f"{safe_id}__c{ci:02d}_mask.npy"
            mask_path = mask_dir / mask_name

            if not dry_run:
                np.save(str(shard_path), segmented_cell.astype(np.float32))
                if save_masks:
                    np.save(str(mask_path), cell_mask.astype(np.uint8))

            quality = _mask_quality(cell_mask)
            manifest_rows.append({
                "shard_path": str(shard_path),
                "mask_path": str(mask_path) if save_masks else "",
                "sensor": sensor,
                "split": split,
                "label": board["label"],
                "class_name": board["class_name"],
                "ripeness": board["ripeness"],
                "file_stem": board["file_stem"],
                "day": board["day"],
                "cell_idx": ci,
                "cell_h": cell_h,
                "cell_w": cell_w,
                "n_bands": n_bands,
                **quality,
            })
            total_cells += 1

        del raw_cube, norm_cube, final_mask, berry_masks, intensity

    elapsed = time.time() - t0
    print(f"  {len(boards) - skipped} boards processed, "
          f"{total_cells} segmented shards {'(dry-run)' if dry_run else 'written'} "
          f"[{elapsed:.1f}s, {skipped} skipped]")
    return manifest_rows


def write_manifest(rows: list, shard_dir: Path) -> Path:
    if not rows:
        return None
    manifest_path = shard_dir / "manifest.csv"
    fieldnames = list(rows[0].keys())
    with open(manifest_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Manifest written -> {manifest_path} ({len(rows)} rows)")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Create segmented per-cell shards for SWD training.")
    parser.add_argument("--sensor", choices=["nir", "vnir", "both"], default="both")
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SEGMENTED_SHARD_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-single-class", action="store_true")
    parser.add_argument("--no-save-masks", action="store_true",
                        help="Do not save separate mask .npy files.")
    parser.add_argument("--preview-limit", type=int, default=12,
                        help="Number of diagnostic board previews to save per sensor.")
    parser.add_argument("--max-boards", type=int, default=None,
                        help="Optional test limit per sensor.")
    args = parser.parse_args()

    shard_dir = args.shard_dir
    if not args.dry_run:
        shard_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSegmented shard directory: {shard_dir}")

    sensors = ["nir", "vnir"] if args.sensor == "both" else [args.sensor]
    all_rows = []
    for sensor in sensors:
        rows = shard_sensor_segmented(
            sensor=sensor,
            shard_dir=shard_dir,
            dry_run=args.dry_run,
            require_all_classes=not args.allow_single_class,
            save_masks=not args.no_save_masks,
            preview_limit=args.preview_limit,
            max_boards=args.max_boards,
        )
        all_rows.extend(rows)

    if not args.dry_run and all_rows:
        write_manifest(all_rows, shard_dir)

    from collections import Counter
    print("\n  Summary by split:")
    for split, cnt in sorted(Counter(r["split"] for r in all_rows).items()):
        print(f"    {split:6s}: {cnt:6d} shards")
    print("  Summary by sensor:")
    for sensor, cnt in sorted(Counter(r["sensor"] for r in all_rows).items()):
        print(f"    {sensor:6s}: {cnt:6d} shards")
    print("\nDone. Use this shard directory with run_all.py --shard_dir.\n")


if __name__ == "__main__":
    main()
