"""
discovery.py — Board discovery and board-level train/val/test splitting.

Board identity
--------------
Each physical board of 36 berries is scanned once per day for 6 days.
Board identity = (class_name, ripeness, filename_stem).
ALL 6 days of the same board are assigned to the SAME split to prevent
data leakage (the model must never see the same physical berries in both
training and test).

Directory structure (second approach)
--------------------------------------
{class}_npy_files_combine_{sensor}/
    Day{N}/
        {Ripeness}/
            board_001.npy
            board_002.npy
            ...

Public API
----------
    discover_boards(sensor, class_name, days_filter, rip_filter) → list[dict]
    assign_splits(boards, train_frac, val_frac, seed)            → list[dict]
    discover_and_split(sensor, classes, train_frac, val_frac, seed,
                       days_filter, rip_filter)                   → list[dict]
"""

import os
import re
import random
from collections import defaultdict
from pathlib import Path

# ── Ripeness alias map ────────────────────────────────────────────────────────
_RIPENESS_ALIASES = [
    ("completelyripe", "Ripe"),
    ("completely",     "Ripe"),
    ("midripe",        "Midripe"),
    ("unripe",         "Unripe"),
    ("mid",            "Midripe"),
    ("ripe",           "Ripe"),
]


def infer_ripeness(folder_name: str) -> str:
    key = folder_name.lower().replace("_", "").replace(" ", "")
    for alias, canonical in _RIPENESS_ALIASES:
        if alias in key:
            return canonical
    return "Unknown"


def infer_day(path_str: str) -> int:
    m = re.search(r"[Dd]ay\s*(\d+)", path_str)
    return int(m.group(1)) if m else -1


def _natural_sort_key(p: Path):
    return [int(x) if x.isdigit() else x.lower()
            for x in re.split(r"(\d+)", p.name)]


def normalize_file_stem(stem: str, ripeness: str = None) -> str:
    """
    Normalize known sensor-specific filename variants before board matching.

    VNIR Day 3 Unripe files use stems like "unriperipe_1_emptyname", while
    the paired NIR files use "unripe_1_emptyname". Treat these as the same
    physical board identity so NIR/VNIR splits remain paired.
    """
    normalized = stem
    if ripeness == "Unripe" and normalized.lower().startswith("unriperipe_"):
        normalized = "unripe_" + normalized.split("_", 1)[1]
    return normalized


# ── Board discovery ───────────────────────────────────────────────────────────

def discover_boards(
    sensor: str,
    class_name: str,
    root: str,
    days_filter=None,
    rip_filter=None,
) -> list:
    """
    Walk one class root directory and return a list of board dicts.

    Each dict contains:
        path        : str  — absolute path to the .npy file
        sensor      : str  — "nir" or "vnir"
        class_name  : str  — "Healthy" or "Infested"
        label       : int  — 1 = Healthy, 0 = Infested
        ripeness    : str  — "Ripe" | "Midripe" | "Unripe"
        day         : int  — 1 … 6
        file_stem   : str  — filename without extension (used for board identity)
        board_id    : str  — unique board identity key (class+ripeness+stem)
    """
    root = Path(root)
    if not root.exists():
        print(f"  [WARN] Path not found, skipping: {root}")
        return []

    label  = 1 if class_name == "Healthy" else 0
    boards = []

    for day_dir in sorted(root.iterdir()):
        if not day_dir.is_dir():
            continue
        day = infer_day(day_dir.name)
        if day < 1:
            continue
        if days_filter and day not in days_filter:
            continue

        for rip_dir in sorted(day_dir.iterdir()):
            if not rip_dir.is_dir():
                continue
            ripeness = infer_ripeness(rip_dir.name)
            if ripeness == "Unknown":
                continue
            if rip_filter and ripeness not in rip_filter:
                continue

            npy_files = sorted(
                [f for f in rip_dir.iterdir() if f.suffix.lower() == ".npy"],
                key=_natural_sort_key,
            )
            for fpath in npy_files:
                stem     = normalize_file_stem(fpath.stem, ripeness)
                board_id = f"{class_name}__{ripeness}__{stem}"
                boards.append(dict(
                    path       = str(fpath),
                    sensor     = sensor,
                    class_name = class_name,
                    label      = label,
                    ripeness   = ripeness,
                    day        = day,
                    file_stem  = stem,
                    board_id   = board_id,
                ))

    return boards


# ── Board-level split ─────────────────────────────────────────────────────────

def assign_splits(
    boards: list,
    train_frac: float = 0.60,
    val_frac:   float = 0.20,
    seed:       int   = 42,
) -> list:
    """
    Assign each board to 'train', 'val', or 'test'.

    The split is performed at the BOARD IDENTITY level:
        board_id = class_name + ripeness + filename_stem

    All 6 daily scans of the same physical board always land in the
    same partition — this is the key guarantee against data leakage.

    Stratification is by (class_name, ripeness) to keep class and
    ripeness proportions balanced across splits.

    Parameters
    ----------
    boards     : list of board dicts (output of discover_boards)
    train_frac : fraction of unique boards for training (default 0.60)
    val_frac   : fraction for validation (default 0.20)
    seed       : random seed for reproducibility

    Returns
    -------
    boards with 'split' key added: 'train' | 'val' | 'test'
    """
    rng = random.Random(seed)

    # Group unique board identities by stratum (class, ripeness)
    strata: dict = defaultdict(set)
    for b in boards:
        stratum = (b["class_name"], b["ripeness"])
        strata[stratum].add(b["board_id"])

    # Assign board_ids to splits within each stratum
    board_id_to_split: dict = {}
    for stratum, board_ids in strata.items():
        ids = sorted(board_ids)
        rng.shuffle(ids)
        n          = len(ids)
        n_train    = max(1, round(n * train_frac))
        n_val      = max(1, round(n * val_frac))
        # Ensure at least 1 board in test
        n_train    = min(n_train, n - 2)
        n_val      = min(n_val,   n - n_train - 1)

        for i, bid in enumerate(ids):
            if i < n_train:
                board_id_to_split[bid] = "train"
            elif i < n_train + n_val:
                board_id_to_split[bid] = "val"
            else:
                board_id_to_split[bid] = "test"

    # Tag every board dict
    for b in boards:
        b["split"] = board_id_to_split.get(b["board_id"], "train")

    return boards


# ── Summary helpers ───────────────────────────────────────────────────────────

def split_summary(boards: list) -> dict:
    """
    Return a nested dict: split → class → ripeness → count.
    Prints a human-readable table.
    """
    from collections import Counter
    summary = defaultdict(lambda: defaultdict(Counter))
    board_ids_seen = defaultdict(set)

    for b in boards:
        split   = b["split"]
        cls     = b["class_name"]
        rip     = b["ripeness"]
        bid     = b["board_id"]
        # Count unique boards (not individual day-scans)
        if bid not in board_ids_seen[split]:
            board_ids_seen[split].add(bid)
            summary[split][cls][rip] += 1

    print("\n  Board-level split summary (unique physical boards):")
    print(f"  {'Split':<8} {'Class':<12} {'Ripe':>16} "
          f"{'Midripe':>10} {'Unripe':>8} {'Total':>7}")
    print("  " + "-" * 65)
    for split in ["train", "val", "test"]:
        for cls in ["Healthy", "Infested"]:
            rip_counts = summary[split][cls]
            total = sum(rip_counts.values())
            if total == 0:
                continue
            print(f"  {split:<8} {cls:<12} "
                  f"{rip_counts['Ripe']:>16} "
                  f"{rip_counts['Midripe']:>10} "
                  f"{rip_counts['Unripe']:>8} "
                  f"{total:>7}")
    print()
    return dict(summary)


def file_summary(boards: list):
    """Print count of individual .npy files (board × day) per split."""
    from collections import Counter
    counts = Counter(b["split"] for b in boards)
    berries = {s: counts[s] * 36 for s in counts}
    print("  File-level counts (boards × 6 days × 36 berries per board):")
    for split in ["train", "val", "test"]:
        n = counts.get(split, 0)
        print(f"    {split:<6}: {n:>5} board-scans  →  {berries.get(split,0):>7} berry samples")
    print()


# ── Convenience wrapper ───────────────────────────────────────────────────────

def discover_and_split(
    sensor: str,
    paths:  dict,
    classes: list      = None,
    train_frac: float  = 0.60,
    val_frac:   float  = 0.20,
    seed: int          = 42,
    days_filter        = None,
    rip_filter         = None,
    verbose: bool      = True,
) -> list:
    """
    Discover boards for all requested classes, assign splits, print summary.

    Parameters
    ----------
    sensor      : "nir" or "vnir"
    paths       : dict  {class_name: root_directory}
                  e.g. {"Healthy": "Z:\\...", "Infested": "Z:\\..."}
    classes     : list of class names to include (default: all keys in paths)
    train_frac  : fraction of unique boards for training
    val_frac    : fraction for validation
    seed        : random seed
    days_filter : list of ints e.g. [1,2,3] or None for all
    rip_filter  : list of ripeness strings or None for all
    verbose     : print discovery and split summary

    Returns
    -------
    list of board dicts with 'split' key
    """
    if classes is None:
        classes = list(paths.keys())

    all_boards = []
    for cls in classes:
        root = paths.get(cls)
        if not root:
            print(f"  [WARN] No path configured for sensor={sensor} class={cls}")
            continue
        bds = discover_boards(sensor, cls, root, days_filter, rip_filter)
        if verbose:
            print(f"  Discovered {len(bds):>5} board-scans  "
                  f"sensor={sensor.upper()}  class={cls}")
        all_boards.extend(bds)

    if not all_boards:
        print(f"  [ERROR] No boards found for sensor={sensor}. "
              f"Check PATHS in config.py.")
        return []

    all_boards = assign_splits(all_boards, train_frac, val_frac, seed)

    if verbose:
        split_summary(all_boards)
        file_summary(all_boards)

    return all_boards
