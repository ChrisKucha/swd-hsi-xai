"""
dataset.py — PyTorch Dataset for the SWD blueberry HSI pipeline.

Cell extraction
---------------
Each board (.npy file) is divided into a 6×6 grid of cells after cropping.
One cell = one berry sample.  No segmentation is applied — the full cell
region (including any dark border around the berry) is fed to the model.
This avoids the segmentation uncertainty entirely and lets the 3D-CNN learn
the berry appearance from context.

Label modes
-----------
"binary"     : returns (tensor, class_label)         — Healthy=1, Infested=0
"multi_task" : returns (tensor, class_label, rip_label)
               rip_label: Ripe=0, Midripe=1, Unripe=2

Normalization
-------------
Per-board z-score: mean and std computed over the entire cropped cube (all
pixels, all bands) and applied to the extracted cell tensor.  This removes
board-level illumination differences while preserving spectral shape.

Speed note
----------
Each board file contains 36 cells. BlueberryDataset
use a lazy per-instance cache: the first __getitem__ call for a given board
loads the cube from disk and stores it in self._cube_cache (or
the per-board cube cache). All subsequent
cells from the same board read from RAM.  This means one disk I/O per board
per epoch instead of one per sample — 36× fewer reads without pre-loading
the entire dataset at startup.

For maximum speed, copy the .npy files to the Linux machine's local SSD
and update config.py PATHS to point there.  See copy_data_local.py.

Usage
-----
    from data.discovery import discover_and_split
    from data.dataset   import BlueberryDataset
    from torch.utils.data import DataLoader

    boards = discover_and_split(sensor="nir", paths=PATHS["nir"])
    train_ds = BlueberryDataset(
        [b for b in boards if b["split"] == "train"],
        sensor="nir", label_mode="binary"
    )
    loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# ── Ensure swd_detection/ is at the front of sys.path ────────────────────────
# Guarantees swd_detection/config.py is found before pytorch/config.py
_HERE = Path(__file__).parent.parent   # swd_detection/
_SWD_DIR = str(_HERE.resolve())
if _SWD_DIR in sys.path:
    sys.path.remove(_SWD_DIR)
sys.path.insert(0, _SWD_DIR)

from config import (
    NUM_CLASSES,
    BERRY_GRID_ROWS, BERRY_GRID_COLS,
    NIR_TRIM_TOP,  NIR_TRIM_BOTTOM,  NIR_TRIM_LEFT,  NIR_TRIM_RIGHT,
    VNIR_TRIM_TOP, VNIR_TRIM_BOTTOM, VNIR_TRIM_LEFT, VNIR_TRIM_RIGHT,
    VNIR_FLIP_HORIZONTAL,
    NIR_SELECTED_BANDS, VNIR_SELECTED_BANDS,
    CELL_H, CELL_W,
    RIPENESS_NAMES,
)

# ── Ripeness index map ────────────────────────────────────────────────────────
RIPENESS_TO_IDX = {r: i for i, r in enumerate(RIPENESS_NAMES)}


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _crop(raw: np.ndarray, sensor: str) -> np.ndarray:
    if sensor == "nir":
        top, bot, left, right = (NIR_TRIM_TOP, NIR_TRIM_BOTTOM,
                                  NIR_TRIM_LEFT, NIR_TRIM_RIGHT)
    else:
        top, bot, left, right = (VNIR_TRIM_TOP, VNIR_TRIM_BOTTOM,
                                  VNIR_TRIM_LEFT, VNIR_TRIM_RIGHT)
    return raw[
        top  : (None if bot   == 0 else -bot),
        left : (None if right == 0 else -right),
        :,
    ]


def _select_bands(cube: np.ndarray, sensor: str) -> np.ndarray:
    sel = NIR_SELECTED_BANDS if sensor == "nir" else VNIR_SELECTED_BANDS
    return cube if sel is None else cube[..., sel]


def load_cube(
    path: str,
    sensor: str,
    selected_bands: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Load one .npy board, crop, optionally flip (VNIR), select bands.
    Returns float32 array of shape (H, W, B).

    Parameters
    ----------
    path           : path to .npy file
    sensor         : "nir" | "vnir"
    selected_bands : optional integer array of band indices to keep AFTER the
                     standard band selection.  Used by run_multispectral.py to
                     simulate a K-band multispectral camera.
    """
    raw  = np.load(path)
    cube = _crop(raw, sensor).astype(np.float32)
    del raw
    if sensor == "vnir" and VNIR_FLIP_HORIZONTAL:
        cube = cube[:, ::-1, :].copy()
    cube = _select_bands(cube, sensor)
    if selected_bands is not None:
        cube = cube[..., selected_bands]
    return cube


def normalize_cube(cube: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Per-board z-score normalization over all pixels and bands.
    Returns (normalized_cube, mean, std).
    """
    mu  = float(cube.mean())
    std = float(cube.std())
    if std < 1e-6:
        std = 1e-6
    return (cube - mu) / std, mu, std


def extract_cell(
    cube:     np.ndarray,
    cell_idx: int,
    cell_h:   Optional[int] = None,
    cell_w:   Optional[int] = None,
) -> np.ndarray:
    """
    Extract one grid cell from a (H, W, B) cube.

    Parameters
    ----------
    cube     : (H, W, B) float32 array
    cell_idx : 0-based index, row-major (0 = top-left, 35 = bottom-right)
    cell_h   : target height for resize (None = no resize)
    cell_w   : target width  for resize (None = no resize)

    Returns
    -------
    cell : (B, cell_h, cell_w) float32 tensor-ready array
    """
    H, W, B = cube.shape
    nr, nc  = BERRY_GRID_ROWS, BERRY_GRID_COLS
    row_e   = np.linspace(0, H, nr + 1, dtype=int)
    col_e   = np.linspace(0, W, nc + 1, dtype=int)

    r, c    = divmod(cell_idx, nc)
    r0, r1  = row_e[r], row_e[r + 1]
    c0, c1  = col_e[c], col_e[c + 1]
    cell    = cube[r0:r1, c0:c1, :]   # (ch, cw, B)

    if cell_h is not None and cell_w is not None:
        ch, cw = cell.shape[:2]
        if ch != cell_h or cw != cell_w:
            from skimage.transform import resize as sk_resize
            cell = sk_resize(cell, (cell_h, cell_w, B),
                             order=1, preserve_range=True).astype(np.float32)

    # (H, W, B) → (B, H, W)  — PyTorch channels-first convention
    return cell.transpose(2, 0, 1)


# ── Dataset ───────────────────────────────────────────────────────────────────

class BlueberryDataset(Dataset):
    """
    PyTorch Dataset that yields one grid cell per __getitem__ call.

    Each board cube is loaded from disk on first access and cached in RAM for
    the remainder of training.  This gives 36× fewer disk reads than loading
    per sample without pre-loading the entire dataset at startup.

    Parameters
    ----------
    boards      : list of board dicts (output of discovery.discover_and_split)
    sensor      : "nir" | "vnir"
    label_mode  : "binary" | "multi_task"
    transform   : optional callable applied to the (B, H, W) float32 tensor
    cell_h      : target cell height (None = infer from first board)
    cell_w      : target cell width  (None = infer from first board)
    normalize   : if True, apply per-board z-score normalization

    Returns (label_mode == "binary")
    ---------------------------------
    tensor : (B, H_cell, W_cell) float32
    label  : int64 scalar  (1 = Healthy, 0 = Infested)

    Returns (label_mode == "multi_task")
    -------------------------------------
    tensor    : (B, H_cell, W_cell) float32
    cls_label : int64 scalar
    rip_label : int64 scalar (0=Ripe, 1=Midripe, 2=Unripe)
    """

    def __init__(
        self,
        boards:          List[dict],
        sensor:          str,
        label_mode:      str  = "binary",
        transform              = None,
        cell_h:          Optional[int] = None,
        cell_w:          Optional[int] = None,
        normalize:       bool = True,
        selected_bands:  Optional[np.ndarray] = None,
    ):
        if label_mode not in ("binary", "multi_task"):
            raise ValueError(f"label_mode must be 'binary' or 'multi_task', got {label_mode!r}")

        self.boards          = boards
        self.sensor          = sensor.lower()
        self.label_mode      = label_mode
        self.transform       = transform
        self.normalize       = normalize
        self.n_cells         = BERRY_GRID_ROWS * BERRY_GRID_COLS   # 36
        self._selected_bands = (np.asarray(selected_bands, dtype=np.intp)
                                if selected_bands is not None else None)

        # Lazy per-instance cache: populated on first __getitem__ for each board
        self._cube_cache: dict = {}

        # Infer or fix cell dimensions
        self._cell_h = cell_h or CELL_H
        self._cell_w = cell_w or CELL_W
        if self._cell_h is None or self._cell_w is None:
            self._cell_h, self._cell_w = self._infer_cell_size()

        # Build flat index: each entry = (board_dict, cell_idx)
        self._index: List[Tuple[dict, int]] = []
        for board in boards:
            for ci in range(self.n_cells):
                self._index.append((board, ci))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _infer_cell_size(self) -> Tuple[int, int]:
        """Load the first board to determine cell dimensions."""
        if not self.boards:
            raise RuntimeError("boards list is empty — cannot infer cell size.")
        cube = load_cube(self.boards[0]["path"], self.sensor)
        H, W, _ = cube.shape
        nr, nc  = BERRY_GRID_ROWS, BERRY_GRID_COLS
        row_e   = np.linspace(0, H, nr + 1, dtype=int)
        col_e   = np.linspace(0, W, nc + 1, dtype=int)
        cell_h  = int(row_e[1] - row_e[0])
        cell_w  = int(col_e[1] - col_e[0])
        print(f"  [Dataset] Inferred cell size: {cell_h} × {cell_w} px  "
              f"(from {Path(self.boards[0]['path']).name})")
        return cell_h, cell_w

    @property
    def cell_h(self) -> int:
        return self._cell_h

    @property
    def cell_w(self) -> int:
        return self._cell_w

    @property
    def n_bands(self) -> int:
        """Number of spectral bands (inferred from first board)."""
        if not hasattr(self, "_n_bands"):
            cube = load_cube(self.boards[0]["path"], self.sensor)
            self._n_bands = cube.shape[2]
        return self._n_bands

    def _get_cube(self, path: str) -> np.ndarray:
        """
        Return cube from RAM cache, loading from disk only on first access.
        One disk read per board per process lifetime — 36× fewer than per-sample.
        """
        if path not in self._cube_cache:
            cube = load_cube(path, self.sensor, self._selected_bands)
            if self.normalize:
                cube, _, _ = normalize_cube(cube)
            self._cube_cache[path] = cube
        return self._cube_cache[path]

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        board, cell_idx = self._index[idx]

        cube = self._get_cube(board["path"])
        cell = extract_cell(cube, cell_idx, self._cell_h, self._cell_w)
        # cell shape: (B, H_cell, W_cell)

        tensor = torch.from_numpy(cell)   # float32

        if self.transform is not None:
            tensor = self.transform(tensor)

        cls_label = torch.tensor(board["label"],   dtype=torch.long)
        rip_label = torch.tensor(
            RIPENESS_TO_IDX.get(board["ripeness"], 0), dtype=torch.long
        )

        if self.label_mode == "binary":
            return tensor, cls_label
        else:   # multi_task
            return tensor, cls_label, rip_label

    def class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for weighted loss.
        Returns a (NUM_CLASSES,) float32 tensor.
        """
        from collections import Counter
        counts = Counter(b["label"] for b, _ in self._index)
        total  = sum(counts.values())
        weights = torch.ones(NUM_CLASSES, dtype=torch.float32)
        for cls, cnt in counts.items():
            weights[cls] = total / (NUM_CLASSES * cnt)
        return weights

    def ripeness_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for ripeness labels (multi-task)."""
        from collections import Counter
        counts = Counter(
            RIPENESS_TO_IDX.get(b["ripeness"], 0)
            for b, _ in self._index
        )
        total  = sum(counts.values())
        n_rip  = len(RIPENESS_NAMES)
        weights = torch.zeros(n_rip, dtype=torch.float32)
        for rip, cnt in counts.items():
            weights[rip] = total / (n_rip * cnt)
        return weights


# ── Sharded datasets (fastest: load pre-extracted per-cell .npy shards) ───────

class ShardedBlueberryDataset(Dataset):
    """
    Fast dataset that loads pre-extracted per-cell shards instead of full board
    cubes.  Each shard is a tiny (B, H, W) float32 .npy file written by
    create_shards.py.  __getitem__ is a single np.load on a ~200 KB file —
    no cube caching, no RAM preloading, DataLoader workers parallelize freely.

    Parameters
    ----------
    rows           : list of manifest row dicts filtered to the desired
                     split / sensor / ripeness (from manifest.csv)
    label_mode     : "binary" | "multi_task"
    transform      : optional callable on the (B, H, W) float32 tensor
    selected_bands : optional int array to sub-select bands after loading
                     (used by the multispectral cascade)
    """

    def __init__(
        self,
        rows:           list,
        label_mode:     str  = "binary",
        transform             = None,
        selected_bands: Optional[np.ndarray] = None,
    ):
        if label_mode not in ("binary", "multi_task"):
            raise ValueError(f"label_mode must be 'binary' or 'multi_task', got {label_mode!r}")
        self._rows           = rows
        self.label_mode      = label_mode
        self.transform       = transform
        self._selected_bands = (np.asarray(selected_bands, dtype=np.intp)
                                if selected_bands is not None else None)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int):
        row  = self._rows[idx]
        cell = np.load(row["shard_path"])          # (B, H, W), often float16 for demo shards
        if self._selected_bands is not None:
            cell = cell[self._selected_bands]
        tensor = torch.from_numpy(cell.copy()).float()
        if self.transform is not None:
            tensor = self.transform(tensor)
        cls_label = torch.tensor(int(row["label"]),                        dtype=torch.long)
        rip_label = torch.tensor(RIPENESS_TO_IDX.get(row["ripeness"], 0),  dtype=torch.long)
        if self.label_mode == "binary":
            return tensor, cls_label
        return tensor, cls_label, rip_label

    @property
    def cell_h(self) -> int:
        return int(self._rows[0]["cell_h"]) if self._rows else 0

    @property
    def cell_w(self) -> int:
        return int(self._rows[0]["cell_w"]) if self._rows else 0

    @property
    def n_bands(self) -> int:
        if not self._rows:
            return 0
        n = int(self._rows[0]["n_bands"])
        return len(self._selected_bands) if self._selected_bands is not None else n

    def class_weights(self) -> torch.Tensor:
        from collections import Counter
        counts  = Counter(int(r["label"]) for r in self._rows)
        total   = sum(counts.values())
        weights = torch.ones(NUM_CLASSES, dtype=torch.float32)
        for cls, cnt in counts.items():
            weights[cls] = total / (NUM_CLASSES * cnt)
        return weights

    def ripeness_weights(self) -> torch.Tensor:
        from collections import Counter
        counts  = Counter(RIPENESS_TO_IDX.get(r["ripeness"], 0) for r in self._rows)
        total   = sum(counts.values())
        n_rip   = len(RIPENESS_NAMES)
        weights = torch.zeros(n_rip, dtype=torch.float32)
        for rip, cnt in counts.items():
            weights[rip] = total / (n_rip * cnt)
        return weights
