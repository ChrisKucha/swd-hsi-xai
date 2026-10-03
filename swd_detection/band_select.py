"""
band_select.py — Select diagnostic wavelengths from IG spectral importance.

Reads the {run_name}_spectral_importance.csv produced by explain.py and
selects the top-K bands under two strategies:

  "infested"  — top bands by attribution for the Infested class specifically.
                Best for maximizing disease-detection sensitivity.
  "averaged"  — top bands by mean attribution across all classes.
                More balanced; useful when both healthy and infested
                spectral signatures matter.

Outputs
-------
  {run_name}_band_selection.json
      Selected band indices (and corresponding wavelengths) for every
      combination of strategy × K.  This is the input consumed by
      run_multispectral.py.

  {run_name}_band_selection.png
      Spectral importance curve with vertical markers showing which bands
      are picked at each K value for both strategies.

Usage
-----
    python band_select.py \\
        --csv  outputs/nir/cnn3d/stage_agnostic/all/nir__cnn3d__stage_agnostic__all_spectral_importance.csv \\
        --sensor nir \\
        --sweep 5,10,15,20,25,30

    # The JSON is written next to the CSV:
    #   .../nir__cnn3d__stage_agnostic__all_band_selection.json
"""

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── path fix ──────────────────────────────────────────────────────────────────
_HERE    = Path(__file__).parent
_SWD_DIR = str(_HERE.resolve())
if _SWD_DIR in sys.path:
    sys.path.remove(_SWD_DIR)
sys.path.insert(0, _SWD_DIR)

try:
    from config import MS_BAND_COUNTS, WAVELENGTHS
except ImportError:
    MS_BAND_COUNTS = [5, 10, 15, 20, 25, 30]
    WAVELENGTHS    = {}


# ─────────────────────────────────────────────────────────────────────────────
# Core selection logic
# ─────────────────────────────────────────────────────────────────────────────

def load_importance_csv(csv_path: str) -> Dict[str, np.ndarray]:
    """
    Read a spectral_importance.csv and return a dict of
    {column_name: float32 array of length n_bands}.

    Expected columns: band_idx, wavelength_nm, <class0>, <class1>, …
    """
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        rows   = list(reader)

    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")

    fixed  = {"band_idx", "wavelength_nm"}
    cols   = [c for c in rows[0].keys() if c not in fixed]
    result = {c: np.array([float(r[c]) for r in rows], dtype=np.float32)
              for c in cols}
    result["wavelength_nm"] = np.array(
        [float(r["wavelength_nm"]) for r in rows], dtype=np.float32
    )
    return result


def select_bands(
    importance:     Dict[str, np.ndarray],
    sweep:          List[int],
    infested_key:   str = "Infested",
    strategies:     List[str] = ("infested", "averaged"),
) -> Dict[str, Dict[int, List[int]]]:
    """
    Select the top-K band indices for each strategy and each K in sweep.

    Parameters
    ----------
    importance    : output of load_importance_csv
    sweep         : list of K values, e.g. [5, 10, 15, 20, 25, 30]
    infested_key  : column name for the Infested class attribution
    strategies    : which strategies to compute

    Returns
    -------
    selections : {strategy: {K: [sorted band indices]}}
    """
    # Determine attribution arrays
    class_keys = [k for k in importance if k != "wavelength_nm"]

    # Infested attribution — fall back to first class if key not found
    inf_key = infested_key if infested_key in importance else class_keys[0]
    inf_attr = importance[inf_key]

    # Averaged attribution across all classes
    avg_attr = np.mean(
        np.stack([importance[k] for k in class_keys], axis=0), axis=0
    )

    attr_map = {
        "infested": inf_attr,
        "averaged": avg_attr,
    }

    selections: Dict[str, Dict[int, List[int]]] = {}
    for strategy in strategies:
        attr = attr_map[strategy]
        selections[strategy] = {}
        for k in sweep:
            k_clamped = min(k, len(attr))
            # argsort descending, take top k
            top_idx   = np.argsort(attr)[::-1][:k_clamped]
            # Return sorted by band index (ascending) for readability
            selections[strategy][k] = sorted(top_idx.tolist())

    return selections


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

_STRATEGY_COLORS = {
    "infested": "#e74c3c",   # red
    "averaged": "#2980b9",   # blue
}

_MARKER_COLORS = [
    "#f39c12", "#27ae60", "#8e44ad",
    "#16a085", "#d35400", "#2c3e50",
]


def plot_band_selection(
    importance:  Dict[str, np.ndarray],
    selections:  Dict[str, Dict[int, List[int]]],
    sweep:       List[int],
    sensor:      str,
    run_name:    str,
    out_path:    str,
):
    """
    Two-panel figure:
      Top row    — full spectral importance curves for both strategies with
                   vertical lines marking selected bands at the largest K.
      Bottom row — zoom panels showing which bands are added at each K step
                   (cumulative) for each strategy.
    """
    wavelengths = importance["wavelength_nm"]
    class_keys  = [k for k in importance if k != "wavelength_nm"]
    strategies  = list(selections.keys())
    max_k       = max(sweep)

    fig, axes = plt.subplots(
        len(strategies), 1,
        figsize=(13, 4 * len(strategies)),
        sharex=True,
    )
    if len(strategies) == 1:
        axes = [axes]

    for ax, strategy in zip(axes, strategies):
        attr   = importance.get(
            "Infested" if strategy == "infested" else class_keys[0],
            np.mean(np.stack([importance[k] for k in class_keys]), axis=0)
        )
        if strategy == "averaged":
            attr = np.mean(
                np.stack([importance[k] for k in class_keys], axis=0), axis=0
            )

        color = _STRATEGY_COLORS.get(strategy, "black")
        label  = ("Top bands — Infested class"
                  if strategy == "infested"
                  else "Top bands — averaged classes")

        ax.plot(wavelengths, attr, color=color, linewidth=1.5,
                alpha=0.85, label=label)
        ax.fill_between(wavelengths, attr, alpha=0.12, color=color)

        # Mark selected bands for each K with vertical lines
        prev_set: set = set()
        for ki, k in enumerate(sweep):
            current_set = set(selections[strategy][k])
            new_bands   = sorted(current_set - prev_set)
            mk_color   = _MARKER_COLORS[ki % len(_MARKER_COLORS)]
            for bi in new_bands:
                ax.axvline(wavelengths[bi], color=mk_color,
                           linewidth=0.8, alpha=0.7, linestyle="--")
            # Invisible line for legend entry
            ax.axvline(wavelengths[new_bands[0]] if new_bands else wavelengths[0],
                       color=mk_color, linewidth=1.5, linestyle="--",
                       label=f"K={k} ({len(current_set)} bands)",
                       alpha=0.0)
            # Visible legend proxy
            ax.plot([], [], color=mk_color, linewidth=1.5,
                    linestyle="--", label=f"K={k}")
            prev_set = current_set

        ax.set_ylabel("Normalized |attribution|", fontsize=10)
        title_suffix = ("Infested class" if strategy == "infested"
                        else "Averaged classes")
        ax.set_title(
            f"Band Selection — {title_suffix}  |  {run_name}", fontsize=10
        )
        ax.legend(fontsize=7, ncol=4, loc="upper right")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Wavelength (nm)", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Band selection plot → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Save / load JSON
# ─────────────────────────────────────────────────────────────────────────────

def save_selection_json(
    selections:  Dict[str, Dict[int, List[int]]],
    wavelengths: np.ndarray,
    sensor:      str,
    source_csv:  str,
    run_name:    str,
    out_path:    str,
):
    """Serialise selections to JSON, including wavelength values for reference."""
    payload = {
        "sensor":     sensor,
        "source_csv": source_csv,
        "run_name":   run_name,
        "strategies": {},
    }
    for strategy, k_dict in selections.items():
        payload["strategies"][strategy] = {}
        for k, band_indices in k_dict.items():
            payload["strategies"][strategy][str(k)] = {
                "band_indices":  band_indices,
                "wavelengths_nm": [round(float(wavelengths[i]), 1)
                                   for i in band_indices],
            }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  Band selection JSON → {out_path}")


def load_selection_json(json_path: str) -> Dict:
    """Load a band_selection.json produced by this script."""
    with open(json_path) as fh:
        return json.load(fh)


def get_band_indices(
    json_data:  Dict,
    strategy:   str,
    k:          int,
) -> List[int]:
    """
    Convenience accessor for the band indices for a given strategy and K.

    Parameters
    ----------
    json_data : output of load_selection_json
    strategy  : "infested" | "averaged"
    k         : number of bands

    Returns
    -------
    list of band indices (integers)
    """
    return json_data["strategies"][strategy][str(k)]["band_indices"]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Select top-K diagnostic bands from spectral importance CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--csv", required=True,
        help="Path to {run_name}_spectral_importance.csv from explain.py",
    )
    p.add_argument(
        "--sensor", default="nir", choices=["nir", "vnir"],
        help="Sensor used to generate the importance CSV (for wavelength labeling)",
    )
    p.add_argument(
        "--sweep", default=",".join(str(k) for k in MS_BAND_COUNTS),
        help="Comma-separated list of band counts to sweep "
             f"(default: {MS_BAND_COUNTS})",
    )
    p.add_argument(
        "--infested_col", default="Infested",
        help="Column name for the Infested class in the CSV (default: Infested)",
    )
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        p.error(f"CSV not found: {csv_path}")

    sweep    = [int(x.strip()) for x in args.sweep.split(",") if x.strip()]
    run_name = csv_path.stem.replace("_spectral_importance", "")
    out_dir  = str(csv_path.parent)

    print(f"\nBand selection")
    print(f"  CSV     : {csv_path}")
    print(f"  Sensor  : {args.sensor}")
    print(f"  Sweep   : {sweep}")

    # Load importance data
    importance = load_importance_csv(str(csv_path))
    wavelengths = importance["wavelength_nm"]
    n_bands     = len(wavelengths)
    print(f"  Bands   : {n_bands}")

    # Select bands
    selections = select_bands(
        importance,
        sweep=sweep,
        infested_key=args.infested_col,
    )

    # Print summary
    for strategy, k_dict in selections.items():
        print(f"\n  Strategy: {strategy}")
        for k, indices in k_dict.items():
            wls = [f"{wavelengths[i]:.0f}" for i in indices]
            print(f"    K={k:>2d}  bands={indices}  wavelengths={wls} nm")

    # Save JSON
    json_path = os.path.join(out_dir, f"{run_name}_band_selection.json")
    save_selection_json(
        selections, wavelengths, args.sensor,
        source_csv=str(csv_path),
        run_name=run_name,
        out_path=json_path,
    )

    # Save plot
    png_path = os.path.join(out_dir, f"{run_name}_band_selection.png")
    plot_band_selection(
        importance, selections, sweep,
        sensor=args.sensor, run_name=run_name, out_path=png_path,
    )

    print(f"\nDone.  Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
