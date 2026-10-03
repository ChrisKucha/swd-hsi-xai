"""
explain.py — Integrated Gradients spectral attribution for the SWD pipeline.

  Integrated Gradients  (requires captum)
     Attributes the prediction to each input spectral band by integrating
     gradients from a zero baseline.  After averaging |attribution| over the
     spatial dimensions this gives a per-wavelength importance curve:
     WHICH wavelengths drive the Infested / Healthy decision.


Outputs saved per experiment
----------------------------
  {run_name}_spectral_importance.png — IG per-band attribution curves
  {run_name}_spectral_importance.csv — numerical band importances

Public API
----------
    from swd_detection.explain import run_explain

    run_explain(
        model       = model,
        test_loader = test_loader,
        output_dir  = "outputs/nir/cnn3d/stage_agnostic/all",
        run_name    = "nir__cnn3d__stage_agnostic__all",
        sensor      = "nir",
        model_name  = "cnn3d",
        n_samples   = 16,
        n_ig_steps  = 50,
        class_names = ["Infested", "Healthy"],
    )
"""

import csv
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")   # non-interactive — safe for server / subprocess
import matplotlib.pyplot as plt

# ── Ensure swd_detection/ is at the front of sys.path ─────────────────────
_HERE    = Path(__file__).parent
_SWD_DIR = str(_HERE.resolve())
if _SWD_DIR in sys.path:
    sys.path.remove(_SWD_DIR)
sys.path.insert(0, _SWD_DIR)

try:
    from config import CLASS_NAMES, WAVELENGTHS
except ImportError:
    CLASS_NAMES = ["Infested", "Healthy"]
    WAVELENGTHS = {}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap(model: nn.Module) -> nn.Module:
    """Strip DataParallel wrapper if present."""
    return model.module if isinstance(model, nn.DataParallel) else model


def _band_wavelengths(sensor: str, n_bands: int) -> np.ndarray:
    """
    Map band indices to wavelength centers (nm).

    Exact wavelength arrays from config.py are used when their length matches
    the current band count. If a selected-band run changes the number of bands,
    fall back to interpolating across the configured wavelength centers.
    """
    def sensor_wavelengths(name: str, count: int) -> np.ndarray:
        wl = np.asarray(WAVELENGTHS[name], dtype=np.float32)
        if wl.size == count:
            return wl
        return np.interp(
            np.linspace(0, wl.size - 1, count),
            np.arange(wl.size),
            wl,
        ).astype(np.float32)

    if sensor == "nir":
        return sensor_wavelengths("nir", n_bands)
    return sensor_wavelengths("vnir", n_bands)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Integrated Gradients — spectral band importance  (requires captum)
# ─────────────────────────────────────────────────────────────────────────────

def _captum_available() -> bool:
    try:
        import captum  # noqa: F401
        return True
    except ImportError:
        warnings.warn(
            "captum is not installed — Integrated Gradients will be skipped.\n"
            "Install with:  pip install captum --break-system-packages",
            UserWarning, stacklevel=3,
        )
        return False


def spectral_ig(
    model:     nn.Module,
    x:         torch.Tensor,
    class_idx: Optional[int] = None,
    n_steps:   int = 50,
    internal_batch_size: int = 4,
) -> Optional[np.ndarray]:
    """
    Per-band attribution via Integrated Gradients.

    Integrates gradients from a zero baseline to the actual input along
    `n_steps` evenly-spaced steps, attributing the prediction score to each
    (band, H, W) input element.  Spatial dimensions are averaged out to yield
    a (n_bands,) importance profile.

    Parameters
    ----------
    model     : unwrapped nn.Module (eval mode)
    x         : (1, n_bands, H, W) on the model's device
    class_idx : target class (None → argmax)
    n_steps   : integration steps (default 50; higher = more accurate)
    internal_batch_size : number of IG scaled samples to forward at once.
                          Keeps Captum from materialising all n_steps on GPU.

    Returns
    -------
    band_attr : (n_bands,) float32 numpy, normalized [0, 1]
                None if captum is not installed.
    """
    if not _captum_available():
        return None

    from captum.attr import IntegratedGradients

    model.eval()

    def _forward(inp):
        out = model(inp)
        return out[0] if isinstance(out, (tuple, list)) else out

    if class_idx is None:
        with torch.no_grad():
            class_idx = int(_forward(x).argmax(dim=1).item())

    ig       = IntegratedGradients(_forward)
    baseline = torch.zeros_like(x)
    attr     = ig.attribute(
        x,
        baselines=baseline,
        target=class_idx,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
    )
    # attr: (1, n_bands, H, W)
    arr = attr.detach().cpu().numpy().squeeze(0)  # (n_bands, H, W)

    # Mean |attribution| over spatial dims → (n_bands,)
    band_attr = np.abs(arr).mean(axis=(1, 2))
    if band_attr.max() > 1e-8:
        band_attr = band_attr / band_attr.max()
    return band_attr.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Figure-saving helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_spectral_importance(
    band_attr_per_class: Dict[int, np.ndarray],   # {class_idx: (n_bands,)}
    output_dir:  str,
    run_name:    str,
    sensor:      str,
    n_bands:     int,
    class_names: List[str],
):
    wavelengths = _band_wavelengths(sensor, n_bands)
    colors     = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(11, 4))

    for ci, attr in sorted(band_attr_per_class.items()):
        label = class_names[ci] if ci < len(class_names) else f"Class {ci}"
        ax.plot(wavelengths, attr, label=label,
                color=colors[ci % len(colors)], linewidth=1.6, alpha=0.9)

    ax.set_xlabel("Wavelength (nm)", fontsize=11)
    ax.set_ylabel("Normalized |attribution|", fontsize=11)
    ax.set_title(
        f"Spectral Band Importance — Integrated Gradients\n{run_name}", fontsize=10
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    fig_path = os.path.join(output_dir, f"{run_name}_spectral_importance.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Explain] Spectral importance  → {fig_path}")

    # CSV
    csv_path = os.path.join(output_dir, f"{run_name}_spectral_importance.csv")
    sorted_classes = sorted(band_attr_per_class.keys())
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        header = ["band_idx", "wavelength_nm"] + [
            class_names[ci] if ci < len(class_names) else f"class_{ci}"
            for ci in sorted_classes
        ]
        writer.writerow(header)
        for bi, wl in enumerate(wavelengths):
            row = [bi, f"{wl:.1f}"] + [
                f"{band_attr_per_class[ci][bi]:.6f}" for ci in sorted_classes
            ]
            writer.writerow(row)
    print(f"  [Explain] Spectral CSV          → {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_explain(
    model:       nn.Module,
    test_loader,
    output_dir:  str,
    run_name:    str,
    sensor:      str,
    model_name:  str,
    n_samples:   int = 16,
    n_ig_steps:  int = 50,
    class_names: Optional[List[str]] = None,
    device:      Optional[torch.device] = None,
    split_name:  str = "test",
) -> None:
    """
    Run all applicable explainability methods for one trained model and save
    all figures and CSVs to output_dir.

    Parameters
    ----------
    model        : trained nn.Module (DataParallel-wrapped or bare)
    test_loader  : DataLoader for the split to explain
    output_dir   : directory to write outputs into
    run_name     : filename prefix (same as used in train / evaluate)
    sensor       : "nir" | "vnir"
    model_name   : "cnn3d" | "cnn3d_transformer"
    n_samples    : how many test cells to include in spatial visualizations
    n_ig_steps   : integration steps for Integrated Gradients (50 is sufficient
                   for relative band rankings; use 200+ for publication figures)
    class_names  : override default class names
    device       : torch.device (auto-detected if None)
    """
    if class_names is None:
        class_names = CLASS_NAMES
    os.makedirs(output_dir, exist_ok=True)

    m = _unwrap(model)
    m.eval()

    if device is None:
        try:
            device = next(m.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    # ── Collect samples ───────────────────────────────────────────────────────
    print(f"\n  [Explain] Collecting up to {n_samples} {split_name} samples …")
    cells:       List[torch.Tensor] = []
    true_labels: List[int]          = []
    pred_labels: List[int]          = []

    collect_bar = tqdm(test_loader, desc="  Collecting samples", unit="batch",
                       dynamic_ncols=True)
    with torch.no_grad():
        for batch in collect_bar:
            x_batch   = batch[0].to(device)
            lbl_batch = batch[1].to(device)
            out = model(x_batch)
            if isinstance(out, (tuple, list)):
                out = out[0]
            preds = out.argmax(dim=1)
            for i in range(x_batch.size(0)):
                cells.append(x_batch[i].cpu())
                true_labels.append(int(lbl_batch[i].item()))
                pred_labels.append(int(preds[i].item()))
                if len(cells) >= n_samples:
                    break
            if len(cells) >= n_samples:
                break

    n_collected = len(cells)
    if n_collected == 0:
        print(f"  [Explain] No {split_name} samples found — skipping.")
        return

    n_bands = cells[0].shape[0]
    cell_h  = cells[0].shape[1]
    cell_w  = cells[0].shape[2]
    print(f"  [Explain] {n_collected} samples  "
          f"({n_bands} bands, {cell_h}×{cell_w} px)")

    # ── Integrated Gradients — spectral band importance ──────────────────────
    print("  [Explain] Running Integrated Gradients …")
    attr_sums:   Dict[int, np.ndarray] = {}
    attr_counts: Dict[int, int]        = {}
    ig_ok = True

    ig_steps = min(n_ig_steps, 16) if "transformer" in model_name.lower() else n_ig_steps
    if ig_steps != n_ig_steps:
        print(f"  [Explain] Reducing IG steps from {n_ig_steps} to {ig_steps} for {model_name}.")

    for cell, true_lbl, pred_lbl in tqdm(
            zip(cells, true_labels, pred_labels), total=n_collected,
            desc="  IG bands", unit="sample", dynamic_ncols=True):
        x    = cell.unsqueeze(0).to(device)
        try:
            attr = spectral_ig(
                m,
                x,
                class_idx=pred_lbl,
                n_steps=ig_steps,
                internal_batch_size=1,
            )
        except torch.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            warnings.warn(
                "Integrated Gradients ran out of GPU memory; skipping IG "
                "spectral importance for this run.",
                RuntimeWarning,
            )
            ig_ok = False
            break
        if attr is None:
            ig_ok = False
            break
        if pred_lbl not in attr_sums:
            attr_sums[pred_lbl]   = np.zeros(n_bands, dtype=np.float64)
            attr_counts[pred_lbl] = 0
        attr_sums[pred_lbl]   += attr.astype(np.float64)
        attr_counts[pred_lbl] += 1

    if ig_ok and attr_sums:
        mean_attr = {
            ci: (attr_sums[ci] / max(attr_counts[ci], 1)).astype(np.float32)
            for ci in attr_sums
        }
        _save_spectral_importance(mean_attr, output_dir, run_name,
                                  sensor, n_bands, class_names)

    print(f"\n  [Explain] Done.  Outputs in: {output_dir}")
