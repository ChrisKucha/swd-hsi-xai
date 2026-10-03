"""
segmentation.py — Robust cell-wise segmentation for 6×6 blueberry boards.

Drop-in replacement for segmentation.py.  Improvements:

  1. Multi-view PCA intensity
       Three spectral sub-range averages are projected onto the first principal
       component of their 3-channel covariance matrix.  This automatically
       maximizes berry–background contrast regardless of ripeness stage.

  2. Ripeness-aware band selection
       Band fractions for each spectral view are tuned per ripeness stage
       (Unripe / Midripe / Ripe) and per sensor (NIR / VNIR).

  3. Circularity-constrained blob selection
       After connected-component labeling, blobs are ranked by circularity
       (4π·area / perimeter²) rather than just area.  This avoids selecting
       merged or background blobs that happen to be large.

  4. 2-Means secondary fallback
       A simple iterative 1-D 2-means threshold replaces the raw percentile
       fallback.  It is more robust when the cell histogram is not bimodal.

  5. Cross-sensor consensus (fused mode)
       NIR and VNIR masks are compared per cell via IoU.  High agreement
       keeps the NIR mask; low agreement keeps whichever mask is larger.

  6. Debug visualization helpers
       visualize_board() saves a 2×3 diagnostic PNG per board.
       false_color_rgb() produces a display-ready RGB thumbnail.

Public API (identical signatures to segmentation.py):
    load_and_segment(board, mode)  → cube, final_mask, berry_masks, mean, std
    segment_board(cube, sensor, ripeness=None)  → final_mask, berry_masks, intensity
    build_intensity(cube, b_start, b_end)       → intensity  (legacy, kept for compat)
"""

import os
import sys
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.filters import threshold_otsu
from skimage.measure import label as sk_label, regionprops
from skimage.draw import disk
from skimage.transform import resize

# ── Try to import constants from parent config; fall back to inline defaults ──
try:
    _parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    from config import (
        BERRY_GRID_ROWS, BERRY_GRID_COLS,
        NIR_SEG_BAND_START, NIR_SEG_BAND_END,
        VNIR_SEG_BAND_START, VNIR_SEG_BAND_END,
        NIR_SEG_MIN_REL_AREA, NIR_SEG_MAX_REL_AREA,
        VNIR_SEG_MIN_REL_AREA, VNIR_SEG_MAX_REL_AREA,
        NIR_TRIM_TOP, NIR_TRIM_BOTTOM, NIR_TRIM_LEFT, NIR_TRIM_RIGHT,
        VNIR_TRIM_TOP, VNIR_TRIM_BOTTOM, VNIR_TRIM_LEFT, VNIR_TRIM_RIGHT,
        NIR_SELECTED_BANDS, VNIR_SELECTED_BANDS,
        SEG_BLUR_SIGMA, CELLWISE_PERCENTILE,
        CELLWISE_CLOSE_RADIUS, CELLWISE_FALLBACK_RADIUS_SCALE,
        VNIR_FLIP_HORIZONTAL,
    )
    from utils import crop_cube, select_bands
    _CONFIG_SOURCE = "parent config.py"
except ImportError:
    # ── Inline defaults (mirrors config.py values) ─────────────────────────
    BERRY_GRID_ROWS, BERRY_GRID_COLS = 6, 6
    NIR_SEG_BAND_START,  NIR_SEG_BAND_END  = 20, 120
    VNIR_SEG_BAND_START, VNIR_SEG_BAND_END = 80, 180
    NIR_SEG_MIN_REL_AREA,  NIR_SEG_MAX_REL_AREA  = 0.01, 0.40
    VNIR_SEG_MIN_REL_AREA, VNIR_SEG_MAX_REL_AREA = 0.01, 0.40
    NIR_TRIM_TOP,  NIR_TRIM_BOTTOM,  NIR_TRIM_LEFT,  NIR_TRIM_RIGHT  = 0, 60, 40, 100
    VNIR_TRIM_TOP, VNIR_TRIM_BOTTOM, VNIR_TRIM_LEFT, VNIR_TRIM_RIGHT = 0, 60, 165, 100
    NIR_SELECTED_BANDS  = None
    VNIR_SELECTED_BANDS = None
    SEG_BLUR_SIGMA                 = 1.5
    CELLWISE_PERCENTILE            = 90.0
    CELLWISE_CLOSE_RADIUS          = 2
    CELLWISE_FALLBACK_RADIUS_SCALE = 0.25
    VNIR_FLIP_HORIZONTAL           = True

    def crop_cube(cube, top, bot, left, right):
        return cube[top:(-bot or None), left:(-right or None), :]

    def select_bands(cube, sel):
        return cube if sel is None else cube[..., sel]

    _CONFIG_SOURCE = "inline defaults"


# ─────────────────────────────────────────────────────────────────────────────
#  Ripeness-aware spectral view fractions
#
#  Each entry is a list of three (lo_frac, hi_frac) pairs, one per spectral
#  view.  Fractions are applied to the actual band count B at runtime so they
#  work regardless of camera model or band-selection subset.
#
#  Rationale:
#    Unripe berries are green → strong reflectance at short visible wavelengths
#    (low VNIR indices) and lower NIR water absorption.
#    Ripe berries have high anthocyanin + water content → best
#    contrast at red-NIR wavelengths.
# ─────────────────────────────────────────────────────────────────────────────

_RIPENESS_FRACS = {
    # ── VNIR (400–1000 nm, 600 nm range) ─────────────────────────────────────
    # High-contrast region for blueberry vs black board is the red-edge / NIR
    # transition: ~700–1000 nm = fractions 0.50–1.00.
    # At 400–700 nm the board is still very dark AND berries absorb strongly
    # (anthocyanins), so there is little berry–board contrast.
    # Using the upper half of VNIR maximizes berry brightness relative to board.
    "vnir": {
        "unripe":         [(0.48, 0.70), (0.62, 0.82), (0.75, 1.00)],
        "midripe":        [(0.45, 0.68), (0.60, 0.80), (0.72, 1.00)],
        "completelyripe": [(0.42, 0.65), (0.57, 0.78), (0.70, 1.00)],
        "default":        [(0.45, 0.68), (0.60, 0.80), (0.72, 1.00)],
    },
    # ── NIR (900–1700 nm, 800 nm range) ──────────────────────────────────────
    # High-contrast region is 900–1100 nm = fractions 0.00–0.25.
    # Beyond ~1100 nm, blueberry reflectance drops sharply due to water
    # absorption (1200 nm and 1450 nm troughs).  The black board is near-zero
    # at ALL wavelengths, so berry–board contrast is only meaningful where
    # berries are still bright.  Keeping all three views inside 0.00–0.32
    # ensures the PCA has strong contrast to work with.
    "nir": {
        "unripe":         [(0.00, 0.20), (0.05, 0.25), (0.08, 0.30)],
        "midripe":        [(0.00, 0.20), (0.05, 0.25), (0.08, 0.30)],
        "completelyripe": [(0.00, 0.18), (0.04, 0.22), (0.06, 0.28)],
        "default":        [(0.00, 0.20), (0.05, 0.25), (0.08, 0.30)],
    },
}


def _get_band_fracs(sensor, ripeness):
    """Return list of 3 (lo_frac, hi_frac) tuples for the given sensor/ripeness."""
    sensor  = (sensor or "nir").lower()
    table   = _RIPENESS_FRACS.get(sensor, _RIPENESS_FRACS["nir"])
    if not ripeness:
        return table["default"]
    rip_key = ripeness.lower().replace("_", "").replace(" ", "")
    for key in table:
        if key in rip_key or rip_key in key:
            return table[key]
    return table["default"]


# ─────────────────────────────────────────────────────────────────────────────
#  Intensity image builders
# ─────────────────────────────────────────────────────────────────────────────

def build_intensity(cube, b_start, b_end):
    """
    Average a fixed band range and Gaussian-blur → grayscale intensity.
    Kept for backwards compatibility.  New code should prefer
    build_multi_view_intensity().
    """
    H, W, B = cube.shape
    b_start = max(0, min(b_start, B - 1))
    b_end   = max(b_start + 1, min(b_end, B))
    intensity = cube[:, :, b_start:b_end].mean(axis=2).astype(np.float32)
    return gaussian_filter(intensity, sigma=SEG_BLUR_SIGMA)


def build_multi_view_intensity(cube, sensor, ripeness=None):
    """
    Contrast-maximized intensity image via 3-view spectral PCA.

    Algorithm
    ---------
    1. Compute three sub-range band averages (views) using ripeness-aware
       spectral fractions.
    2. Stack the three views into an (N_pixels × 3) matrix and center it.
    3. Project onto the first principal component (highest-variance direction).
    4. Use image corners as background reference to ensure berries are bright
       (flip sign if PCA inverted the polarity).
    5. Apply Gaussian blur.

    Falls back to a plain mean of the three views if the covariance matrix
    is singular (degenerate / constant cube).

    Parameters
    ----------
    cube     : (H, W, B) float32 hyperspectral array (already cropped)
    sensor   : "nir" or "vnir"
    ripeness : optional string — "Unripe", "Midripe", "Ripe"

    Returns
    -------
    intensity : (H, W) float32, Gaussian-blurred, berries bright
    """
    H, W, B = cube.shape
    fracs = _get_band_fracs(sensor, ripeness)

    views = []
    for lo_f, hi_f in fracs:
        b0 = max(0, int(lo_f * B))
        b1 = max(b0 + 1, min(int(hi_f * B), B))
        views.append(cube[:, :, b0:b1].mean(axis=2).astype(np.float32))

    # ── PCA projection ──────────────────────────────────────────────────────
    stack   = np.stack([v.ravel() for v in views], axis=1)   # (N_pix, 3)
    mu      = stack.mean(axis=0)
    stack_c = stack - mu
    cov     = np.cov(stack_c.T)

    try:
        eigvals, eigvecs = np.linalg.eigh(cov)               # ascending order
        pc1      = stack_c @ eigvecs[:, -1]
        intensity = pc1.reshape(H, W).astype(np.float32)
    except np.linalg.LinAlgError:
        intensity = np.mean(views, axis=0).astype(np.float32)

    # ── Polarity fix — cell centers should be brighter than cell borders ──
    #
    # The earlier whole-image corner/center check can fail because the central
    # image region still contains much more black board than berry.  Here we
    # exploit the fixed 6x6 board layout: each berry is near the center of its
    # cell, while the cell border is mostly black board.
    row_e = np.linspace(0, H, BERRY_GRID_ROWS + 1, dtype=int)
    col_e = np.linspace(0, W, BERRY_GRID_COLS + 1, dtype=int)
    center_vals = []
    border_vals = []
    for r in range(BERRY_GRID_ROWS):
        for c in range(BERRY_GRID_COLS):
            r0, r1 = row_e[r], row_e[r + 1]
            c0, c1 = col_e[c], col_e[c + 1]
            cell = intensity[r0:r1, c0:c1]
            ch, cw = cell.shape
            rr0, rr1 = max(0, int(0.30 * ch)), max(1, int(0.70 * ch))
            cc0, cc1 = max(0, int(0.30 * cw)), max(1, int(0.70 * cw))
            center_vals.append(cell[rr0:rr1, cc0:cc1].mean())
            border = np.ones(cell.shape, dtype=bool)
            border[rr0:rr1, cc0:cc1] = False
            border_vals.append(cell[border].mean())

    # Board should be darker than berries.  If cell borders are brighter than
    # cell centers, the PCA sign is inverted.
    if np.mean(border_vals) > np.mean(center_vals):
        intensity = -intensity

    return gaussian_filter(intensity, sigma=SEG_BLUR_SIGMA)


# ─────────────────────────────────────────────────────────────────────────────
#  Skimage API compatibility wrappers
# ─────────────────────────────────────────────────────────────────────────────

def _remove_small_objects_compat(bw, size):
    try:
        return morphology.remove_small_objects(bw, max_size=size)
    except TypeError:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            return morphology.remove_small_objects(bw, min_size=size)


def _remove_small_holes_compat(bw, size):
    try:
        return morphology.remove_small_holes(bw, max_size=size)
    except TypeError:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            return morphology.remove_small_holes(bw, area_threshold=size)


# ─────────────────────────────────────────────────────────────────────────────
#  Thresholding helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cell_2means_threshold(cell_flat):
    """
    Iterative 1-D 2-means threshold (no sklearn dependency).
    Converges in ~10 iterations; robust when histogram is not bimodal.
    Returns the threshold value separating low/high pixels.
    """
    vals = cell_flat.astype(np.float32)
    thr  = float(vals.mean())
    for _ in range(20):
        lo = vals[vals <= thr]
        hi = vals[vals >  thr]
        if lo.size == 0 or hi.size == 0:
            break
        new_thr = (float(lo.mean()) + float(hi.mean())) / 2.0
        if abs(new_thr - thr) < 1e-7:
            break
        thr = new_thr
    return thr


# ─────────────────────────────────────────────────────────────────────────────
#  Blob selection with circularity constraint
# ─────────────────────────────────────────────────────────────────────────────

def _circularity(region):
    """4π·area / perimeter²  →  1.0 for a perfect circle, 0 for degenerate."""
    p = region.perimeter
    return (4.0 * np.pi * region.area) / (p * p) if p > 0 else 0.0


def _best_berry_blob(regs, cell_area, min_rel, max_rel, min_circ=0.35):
    """
    Select the best connected component to represent one berry.

    Priority order:
      1. Regions within area bounds AND circularity ≥ min_circ  → most circular.
      2. Regions within area bounds only                         → largest.
      3. No region in bounds                                     → closest to
         expected relative area of 0.15 (empirical berry-to-cell ratio).

    Parameters
    ----------
    regs      : list of skimage regionprops objects
    cell_area : int, total pixel area of the cell
    min_rel, max_rel : float, acceptable area fraction range
    min_circ  : float, minimum circularity for priority-1 selection
    """
    in_bounds = [rg for rg in regs
                 if min_rel <= rg.area / cell_area <= max_rel]

    if in_bounds:
        target = cell_area * 0.15
        scored = []
        for rg in in_bounds:
            circ = _circularity(rg)
            area_score = 1.0 - min(1.0, abs(rg.area - target) / max(target, 1.0))
            score = (2.0 * circ) + area_score
            scored.append((score, circ, rg.area, rg))
        circular = [item for item in scored if item[1] >= min_circ]
        if circular:
            return max(circular, key=lambda x: x[0])[3]
        return max(scored, key=lambda x: x[0])[3]

    # Nothing in bounds: reject this threshold attempt.  Returning an
    # out-of-bounds blob can swallow the black board and contaminate spectra.
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Cell-wise segmentation
# ─────────────────────────────────────────────────────────────────────────────

def _apply_threshold(cell, thr, cell_area, min_rel, max_rel, r0, c0, H, W):
    """
    Threshold a cell, clean up morphologically, find best blob.
    Returns an (H, W) bool mask or None if no blob found.
    """
    bw = cell > thr
    bw = _remove_small_objects_compat(bw, max(1, int(cell_area * min_rel)))
    bw = _remove_small_holes_compat(bw,   max(1, int(cell_area * min_rel)))
    bw = morphology.closing(bw, morphology.disk(CELLWISE_CLOSE_RADIUS))
    lab  = sk_label(bw)
    regs = regionprops(lab)
    if not regs:
        return None
    best = _best_berry_blob(regs, cell_area, min_rel, max_rel)
    if best is None:
        return None
    m = np.zeros((H, W), dtype=bool)
    m[best.coords[:, 0] + r0, best.coords[:, 1] + c0] = True
    return m


def _segment_cellwise(intensity, n_rows, n_cols, min_rel, max_rel):
    """
    Segment one berry per grid cell using a cascaded thresholding strategy.

    Attempt order per cell
    ----------------------
    1. Otsu threshold        → circularity-constrained blob selection
    2. 2-Means threshold     → circularity-constrained blob selection
    3. 90th-percentile thr.  → largest blob (as last-resort before disk)
    4. Brightest-pixel disk  → guaranteed non-empty fallback
    """
    H, W = intensity.shape
    row_e = np.linspace(0, H, n_rows + 1, dtype=int)
    col_e = np.linspace(0, W, n_cols + 1, dtype=int)
    masks = []

    for r in range(n_rows):
        for c in range(n_cols):
            r0, r1 = row_e[r], row_e[r + 1]
            c0, c1 = col_e[c], col_e[c + 1]
            cell   = intensity[r0:r1, c0:c1]
            ch, cw = cell.shape
            area   = max(1, ch * cw)
            flat   = cell.ravel()
            m      = None

            # ── Attempt 1: Otsu ────────────────────────────────────────────
            try:
                m = _apply_threshold(cell, threshold_otsu(cell),
                                     area, min_rel, max_rel, r0, c0, H, W)
            except ValueError:
                pass

            # ── Attempt 2: 2-Means ────────────────────────────────────────
            if m is None:
                thr2 = _cell_2means_threshold(flat)
                m = _apply_threshold(cell, thr2, area, min_rel, max_rel, r0, c0, H, W)

            # ── Attempt 3: 90th-percentile ────────────────────────────────
            if m is None:
                thr_p = np.percentile(flat, CELLWISE_PERCENTILE)
                m = _apply_threshold(cell, thr_p, area, min_rel, max_rel, r0, c0, H, W)

            # ── Attempt 4: Brightest-pixel disk ───────────────────────────
            if m is None:
                py, px = np.unravel_index(int(np.argmax(cell)), cell.shape)
                rad    = CELLWISE_FALLBACK_RADIUS_SCALE * min(ch, cw)
                rr, cc = disk((py + r0, px + c0), rad, shape=(H, W))
                m = np.zeros((H, W), dtype=bool)
                m[rr, cc] = True

            masks.append(m)

    return masks


# ─────────────────────────────────────────────────────────────────────────────
#  Cross-sensor consensus (fused mode)
# ─────────────────────────────────────────────────────────────────────────────

def _consensus_masks(bm_nir, bm_vnir):
    """
    Per-cell consensus between NIR and VNIR berry masks.

    IoU ≥ 0.40 : good agreement → keep NIR mask (typically higher contrast).
    IoU <  0.40 : disagreement  → keep the larger of the two masks.
    """
    out = []
    for m_n, m_v in zip(bm_nir, bm_vnir):
        inter = int((m_n & m_v).sum())
        union = int((m_n | m_v).sum())
        iou   = inter / union if union > 0 else 0.0
        if iou >= 0.40:
            out.append(m_n)
        else:
            out.append(m_n if m_n.sum() >= m_v.sum() else m_v)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Public: segment_board
# ─────────────────────────────────────────────────────────────────────────────

def segment_board(cube, sensor, ripeness=None):
    """
    Segment a cropped hyperspectral cube into 36 berry masks.

    Parameters
    ----------
    cube     : (H, W, B) float32 array
    sensor   : "nir" or "vnir"
    ripeness : optional str  — kept for API compatibility, not used for intensity

    Returns
    -------
    final_mask  : (H, W) bool  — union of all 36 berry masks
    berry_masks : list[36] of (H, W) bool  — one mask per grid cell
    intensity   : (H, W) float32  — band-averaged intensity image used
    """
    H, W, B = cube.shape

    if sensor == "nir":
        mn, mx = NIR_SEG_MIN_REL_AREA, NIR_SEG_MAX_REL_AREA
    else:
        mn, mx = VNIR_SEG_MIN_REL_AREA, VNIR_SEG_MAX_REL_AREA

    intensity = build_multi_view_intensity(cube, sensor, ripeness)
    masks     = _segment_cellwise(intensity, BERRY_GRID_ROWS, BERRY_GRID_COLS,
                                  mn, mx)
    final = np.zeros(intensity.shape, dtype=bool)
    for msk in masks:
        final |= msk

    return final, masks, intensity


# ─────────────────────────────────────────────────────────────────────────────
#  Public: load_and_segment  (drop-in replacement for segmentation.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_and_segment(board, mode):
    """
    Load a .npy board file, crop, and segment.

    Parameters
    ----------
    board : dict with keys: nir, vnir, ripeness (optional), day (optional)
    mode  : "nir" | "vnir" | "fused"

    Returns
    -------
    cube        : (H, W, B) float32
    final_mask  : (H, W) bool
    berry_masks : list[36] of (H, W) bool
    mean        : float  — foreground mean reflectance
    std         : float  — foreground std  reflectance
    """
    ripeness = board.get("ripeness", None)

    if mode == "nir":
        raw  = np.load(board["nir"])
        crop = crop_cube(raw, NIR_TRIM_TOP, NIR_TRIM_BOTTOM,
                         NIR_TRIM_LEFT, NIR_TRIM_RIGHT)
        cube = select_bands(crop, NIR_SELECTED_BANDS).astype(np.float32)
        fm, bm, _ = segment_board(cube, "nir", ripeness)

    elif mode == "vnir":
        raw  = np.load(board["vnir"])
        crop = crop_cube(raw, VNIR_TRIM_TOP, VNIR_TRIM_BOTTOM,
                         VNIR_TRIM_LEFT, VNIR_TRIM_RIGHT)
        if VNIR_FLIP_HORIZONTAL:
            crop = crop[:, ::-1, :].copy()
        cube = select_bands(crop, VNIR_SELECTED_BANDS).astype(np.float32)
        fm, bm, _ = segment_board(cube, "vnir", ripeness)

    else:  # fused
        nr   = np.load(board["nir"])
        nc   = crop_cube(nr, NIR_TRIM_TOP,  NIR_TRIM_BOTTOM,
                         NIR_TRIM_LEFT, NIR_TRIM_RIGHT)
        vr   = np.load(board["vnir"])
        vc   = crop_cube(vr, VNIR_TRIM_TOP, VNIR_TRIM_BOTTOM,
                         VNIR_TRIM_LEFT, VNIR_TRIM_RIGHT)
        if VNIR_FLIP_HORIZONTAL:
            vc = vc[:, ::-1, :].copy()
        if vc.shape[:2] != nc.shape[:2]:
            vc = resize(vc, (*nc.shape[:2], vc.shape[2]),
                        order=1, preserve_range=True).astype(np.float32)
        nir_c  = select_bands(nc.astype(np.float32), NIR_SELECTED_BANDS)
        vnir_c = select_bands(vc.astype(np.float32), VNIR_SELECTED_BANDS)
        cube   = np.concatenate([vnir_c, nir_c], axis=-1)

        fm_nir,  bm_nir,  _ = segment_board(nir_c,  "nir",  ripeness)
        fm_vnir, bm_vnir, _ = segment_board(vnir_c, "vnir", ripeness)
        bm = _consensus_masks(bm_nir, bm_vnir)
        fm = np.zeros(nir_c.shape[:2], dtype=bool)
        for msk in bm:
            fm |= msk

    fg   = cube[fm]
    mean = float(fg.mean()) if fg.size else float(cube.mean())
    std  = float(fg.std())  if fg.size else float(cube.std())
    if std < 1e-6:
        std = 1e-6

    return cube, fm, bm, mean, std


# ─────────────────────────────────────────────────────────────────────────────
#  Debug visualization helpers
# ─────────────────────────────────────────────────────────────────────────────

def false_color_rgb(cube):
    """
    Generate a false-color RGB thumbnail from a (H, W, B) hyperspectral cube.
    Selects bands at 25 %, 50 %, 75 % of the spectral axis and scales to [0, 1].
    """
    B    = cube.shape[2]
    idxs = [max(0, B // 4), max(0, B // 2), max(0, 3 * B // 4)]
    rgb  = np.stack([cube[:, :, i] for i in idxs], axis=2).astype(np.float32)
    lo, hi = rgb.min(), rgb.max()
    return (rgb - lo) / (hi - lo + 1e-8)


def _draw_grid(ax, H, W, n_rows=6, n_cols=6, color="yellow", lw=0.5):
    for i in range(1, n_rows):
        ax.axhline(H * i / n_rows, color=color, lw=lw)
    for j in range(1, n_cols):
        ax.axvline(W * j / n_cols, color=color, lw=lw)


def visualize_board(cube, intensity, berry_masks, final_mask,
                    title="", save_path=None):
    """
    Save a 2×3 diagnostic figure for one board.

    Layout
    ------
    Row 0: False-color RGB  |  Multi-view PCA intensity  |  Area-fraction heatmap
    Row 1: Final mask overlay |  Per-berry colored masks  |  Circularity heatmap

    Parameters
    ----------
    cube        : (H, W, B) float32
    intensity   : (H, W) float32 — output of build_multi_view_intensity
    berry_masks : list[36] of (H, W) bool
    final_mask  : (H, W) bool
    title       : str, suptitle for the figure
    save_path   : str or None — if given, figure is saved here (directory is created)

    Returns
    -------
    quality : (6, 6) float array — per-cell mask area fraction
    circs   : (6, 6) float array — per-cell best-blob circularity
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import mark_boundaries

    H, W, _ = cube.shape
    nr, nc  = BERRY_GRID_ROWS, BERRY_GRID_COLS
    row_e   = np.linspace(0, H, nr + 1, dtype=int)
    col_e   = np.linspace(0, W, nc + 1, dtype=int)

    # ── Per-cell metrics ──────────────────────────────────────────────────
    quality = np.zeros((nr, nc), dtype=float)
    circs   = np.zeros((nr, nc), dtype=float)
    for idx, msk in enumerate(berry_masks):
        r, c   = divmod(idx, nc)
        r0, r1 = row_e[r], row_e[r + 1]
        c0, c1 = col_e[c], col_e[c + 1]
        cell_area = max(1, (r1 - r0) * (c1 - c0))
        quality[r, c] = msk[r0:r1, c0:c1].sum() / cell_area
        lab  = sk_label(msk[r0:r1, c0:c1])
        regs = regionprops(lab)
        if regs:
            best = max(regs, key=lambda rg: rg.area)
            circs[r, c] = _circularity(best)

    # ── Images ────────────────────────────────────────────────────────────
    rgb      = false_color_rgb(cube)
    norm_int = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)
    overlay  = mark_boundaries(
        np.stack([norm_int] * 3, axis=2),
        final_mask.astype(int), color=(1.0, 0.4, 0.0))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(title, fontsize=11, fontweight="bold")

    # (0,0) False-color RGB
    axes[0, 0].imshow(rgb)
    _draw_grid(axes[0, 0], H, W)
    axes[0, 0].set_title("False-color RGB (bands 25/50/75 %)")
    axes[0, 0].axis("off")

    # (0,1) Multi-view PCA intensity
    axes[0, 1].imshow(norm_int, cmap="gray")
    _draw_grid(axes[0, 1], H, W)
    axes[0, 1].set_title("Multi-view PCA intensity")
    axes[0, 1].axis("off")

    # (0,2) Per-cell area fraction heatmap
    im0 = axes[0, 2].imshow(quality, vmin=0.0, vmax=0.35,
                             cmap="RdYlGn", interpolation="nearest")
    plt.colorbar(im0, ax=axes[0, 2], fraction=0.046, pad=0.04)
    axes[0, 2].set_title("Cell mask area fraction\n(green=good, red=empty/overflow)")
    axes[0, 2].set_xticks(range(nc))
    axes[0, 2].set_xticklabels([f"C{i}" for i in range(nc)], fontsize=7)
    axes[0, 2].set_yticks(range(nr))
    axes[0, 2].set_yticklabels([f"R{i}" for i in range(nr)], fontsize=7)

    # (1,0) Final mask overlay
    axes[1, 0].imshow(overlay)
    _draw_grid(axes[1, 0], H, W)
    axes[1, 0].set_title("Final mask overlay (orange border)")
    axes[1, 0].axis("off")

    # (1,1) Per-berry colored masks
    colored = np.zeros((H, W, 3), dtype=np.float32)
    cmap_b   = plt.cm.get_cmap("tab20", 36)
    for idx, msk in enumerate(berry_masks):
        colored[msk] = cmap_b(idx)[:3]
    axes[1, 1].imshow(rgb * 0.45 + colored * 0.55)
    _draw_grid(axes[1, 1], H, W)
    axes[1, 1].set_title("Per-berry colored masks (36 colors)")
    axes[1, 1].axis("off")

    # (1,2) Circularity heatmap
    im1 = axes[1, 2].imshow(circs, vmin=0.0, vmax=1.0,
                             cmap="RdYlGn", interpolation="nearest")
    plt.colorbar(im1, ax=axes[1, 2], fraction=0.046, pad=0.04)
    axes[1, 2].set_title("Best-blob circularity per cell\n(1.0 = perfect circle)")
    axes[1, 2].set_xticks(range(nc))
    axes[1, 2].set_xticklabels([f"C{i}" for i in range(nc)], fontsize=7)
    axes[1, 2].set_yticks(range(nr))
    axes[1, 2].set_yticklabels([f"R{i}" for i in range(nr)], fontsize=7)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
    return quality, circs
