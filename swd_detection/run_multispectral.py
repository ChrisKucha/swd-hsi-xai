"""
run_multispectral.py — Multispectral band-sweep experiments.

Reads the band_selection.json produced by band_select.py and trains a model
for every combination of (strategy, K) where strategy ∈ {infested, averaged}
and K ∈ the configured sweep (default: 5, 10, 15, 20, 25, 30).

Each run uses exactly K spectral bands, selected by their IG attribution rank,
simulating a K-band multispectral camera.  Training, evaluation, and
explainability all use the same pipeline as run_all.py.

After all runs a comparison figure is saved:
  {output_root}/multispectral_comparison.png
    Accuracy vs. band count, one curve per strategy + the full hyperspectral
    baseline for reference.

  {output_root}/multispectral_comparison.csv
    Tabular results for every (strategy, K) combination.

Prerequisite
------------
  1.  Run the hyperspectral experiment and its explainability step:
          python run_all.py --sensor nir --model cnn3d --mode stage_agnostic

  2.  Run band selection on the importance CSV:
          python band_select.py \\
              --csv outputs/nir/cnn3d/stage_agnostic/all/\\
                    nir__cnn3d__stage_agnostic__all_spectral_importance.csv \\
              --sensor nir

  3.  Then run this script:
          python run_multispectral.py \\
              --sensor nir --model cnn3d --mode stage_agnostic \\
              --selection_json outputs/nir/cnn3d/stage_agnostic/all/\\
                               nir__cnn3d__stage_agnostic__all_band_selection.json

Usage
-----
    python run_multispectral.py \\
        --sensor   nir \\
        --model    cnn3d \\
        --mode     stage_agnostic \\
        --selection_json  <path_to_band_selection.json> \\
        [--sweep   5,10,15,20,25,30] \\
        [--strategy  averaged]        # "infested" | "averaged" | "both"
        [--baseline_json  <path>]     # optional: hyperspectral summary.json
        [--shard_dir      <path>]     # optional: directory containing manifest.csv
        [--output_dir     <path>]     # optional: separate output root
        [--demo_run]
        [--skip_existing]
"""

import argparse
import csv
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Ensure swd_detection/ is at the front of sys.path ─────────────────────
_HERE    = Path(__file__).parent
_SWD_DIR = str(_HERE.resolve())
if _SWD_DIR in sys.path:
    sys.path.remove(_SWD_DIR)
sys.path.insert(0, _SWD_DIR)

import config as CFG
from data.discovery  import discover_and_split
from data.dataset    import (
    BlueberryDataset,
    ShardedBlueberryDataset,
    load_cube,
)
from data.transforms import get_train_transform, get_val_transform
from models          import build_model
from train           import train
from evaluate        import evaluate
from band_select     import load_selection_json, get_band_indices


def _set_training_seed(seed: int) -> None:
    """Set training randomness without changing the manifest-defined split."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_manifest(shard_dir) -> list:
    """Load manifest.csv from shard_dir for sharded multispectral runs."""
    manifest_path = Path(shard_dir) / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.csv not found in {shard_dir}")
    with open(manifest_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    print(f"  [Shards] Manifest loaded: {len(rows)} rows from {manifest_path}")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory (mirrors run_all.py but injects selected_bands)
# ─────────────────────────────────────────────────────────────────────────────

def _filter_manifest(rows, sensor: str, split: str, rip_filter):
    return [
        r for r in rows
        if r.get("sensor") == sensor
        and r.get("split") == split
        and (rip_filter is None or r.get("ripeness") == rip_filter)
    ]


def _make_loaders(
    sensor:         str,
    boards_nir,
    boards_vnir,
    split:          str,
    rip_filter,
    label_mode:     str,
    batch_size:     int,
    smoke_test:     bool,
    cell_h:         int,
    cell_w:         int,
    nir_bands:      np.ndarray,
    vnir_bands:     np.ndarray,
    manifest_rows=None,
):
    n_workers = 0 if os.name == "nt" else int(getattr(CFG, "NUM_WORKERS", 4))

    transform = get_train_transform() if split == "train" else get_val_transform()

    if manifest_rows is not None:
        bands = nir_bands if sensor == "nir" else vnir_bands
        rows = _filter_manifest(manifest_rows, sensor, split, rip_filter)
        ds = ShardedBlueberryDataset(
            rows,
            label_mode=label_mode,
            transform=transform,
            selected_bands=bands,
        )
    else:
        subset_nir = [b for b in boards_nir if b["split"] == split
                      and (rip_filter is None or b["ripeness"] == rip_filter)]
        subset_vnir = [b for b in boards_vnir if b["split"] == split
                       and (rip_filter is None or b["ripeness"] == rip_filter)]

        subset = subset_nir if sensor == "nir" else subset_vnir
        bands = nir_bands if sensor == "nir" else vnir_bands
        ds = BlueberryDataset(
            subset, sensor=sensor,
            label_mode=label_mode,
            transform=transform,
            cell_h=cell_h, cell_w=cell_w,
            selected_bands=bands,
        )

    if smoke_test:
        n = min(len(ds), 100)
        ds, _ = torch.utils.data.random_split(ds, [n, len(ds) - n])

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=n_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train" and len(ds) > batch_size),
        persistent_workers=(n_workers > 0),
        prefetch_factor=(4 if n_workers > 0 else None),
    )
    print(f"    {split:<6}: {len(ds):>6} samples  →  {len(loader):>4} batches")
    return loader, ds


# ─────────────────────────────────────────────────────────────────────────────
# Single (strategy, K) experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_one(
    sensor:       str,
    model_name:   str,
    train_mode:   str,
    rip_filter,
    boards_nir,
    boards_vnir,
    nir_sel:      np.ndarray,   # selected NIR band indices
    vnir_sel:     np.ndarray,   # selected VNIR band indices
    k:            int,
    strategy:     str,
    output_root:  Path,
    smoke_test:   bool,
    skip_existing: bool,
    resume_existing: bool = True,
    manifest_rows=None,
    seed: int = None,
) -> dict:
    label_mode = "binary"
    rip_tag  = rip_filter or "all"
    run_name = (f"{sensor}__{model_name}__{train_mode}__{rip_tag}"
                f"__ms_{strategy}_k{k}")
    out_dir  = output_root / "multispectral" / strategy / f"k{k:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if skip_existing and (out_dir / f"{run_name}_summary.json").exists():
        print(f"  [SKIP] {run_name}")
        with open(out_dir / f"{run_name}_summary.json") as fh:
            return json.load(fh)

    print(f"\n{'='*70}")
    print(f"  MULTISPECTRAL  strategy={strategy}  K={k}")
    print(f"  run: {run_name}")
    print(f"{'='*70}")

    # ── Cell size ────────────────────────────────────────────────────────────
    _src_boards = boards_nir if sensor == "nir" else boards_vnir
    _src_sensor = "nir"      if sensor == "nir" else "vnir"
    _rip_boards = [b for b in _src_boards
                   if rip_filter is None or b["ripeness"] == rip_filter]
    if not _rip_boards:
        print("  [WARN] No boards found — skipping.")
        return {}

    if manifest_rows is not None:
        first_rows = _filter_manifest(manifest_rows, _src_sensor, "train", rip_filter)
        if not first_rows:
            print("  [WARN] No shard rows found — skipping.")
            return {}
        cell_h = int(first_rows[0]["cell_h"])
        cell_w = int(first_rows[0]["cell_w"])
    else:
        # Cell size is determined from the FULL cube (not band-selected)
        _first_cube = load_cube(_rip_boards[0]["path"], _src_sensor)
        _H, _W, _   = _first_cube.shape
        _row_e      = np.linspace(0, _H, CFG.BERRY_GRID_ROWS + 1, dtype=int)
        _col_e      = np.linspace(0, _W, CFG.BERRY_GRID_COLS + 1, dtype=int)
        cell_h      = int(_row_e[1] - _row_e[0])
        cell_w      = int(_col_e[1] - _col_e[0])
    print(f"  Cell size: {cell_h}×{cell_w}")

    # ── DataLoaders ──────────────────────────────────────────────────────────
    print("\n  Building DataLoaders …")
    train_loader, train_ds = _make_loaders(
        sensor, boards_nir, boards_vnir, "train",
        rip_filter, label_mode, CFG.BATCH_SIZE, smoke_test,
        cell_h, cell_w, nir_sel, vnir_sel, manifest_rows=manifest_rows)
    val_loader,   _        = _make_loaders(
        sensor, boards_nir, boards_vnir, "val",
        rip_filter, label_mode, CFG.BATCH_SIZE, smoke_test,
        cell_h, cell_w, nir_sel, vnir_sel, manifest_rows=manifest_rows)
    test_loader,  _        = _make_loaders(
        sensor, boards_nir, boards_vnir, "test",
        rip_filter, label_mode, CFG.BATCH_SIZE * 2, smoke_test,
        cell_h, cell_w, nir_sel, vnir_sel, manifest_rows=manifest_rows)

    if len(train_loader) == 0 or len(val_loader) == 0:
        print("  [WARN] Empty loaders — skipping.")
        return {}

    # ── Band counts after selection ──────────────────────────────────────────
    n_nir_bands  = len(nir_sel)  if nir_sel  is not None else 0
    n_vnir_bands = len(vnir_sel) if vnir_sel is not None else 0
    n_bands = n_vnir_bands if sensor == "vnir" else n_nir_bands

    # ── Model ────────────────────────────────────────────────────────────────
    print(f"\n  Building model ({n_bands} bands) …")
    kwargs = dict(num_classes=CFG.NUM_CLASSES, dropout=CFG.DROPOUT,
                  label_mode=label_mode)

    kwargs.update(n_bands=n_bands, cell_h=cell_h, cell_w=cell_w)
    if model_name == "cnn3d_transformer":
        kwargs.update(n_layers=CFG.TRANSFORMER_N_LAYERS,
                      n_heads=CFG.TRANSFORMER_N_HEADS)

    model    = build_model(model_name, **kwargs)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")

    # ── Train ────────────────────────────────────────────────────────────────
    train_cfg = dict(
        num_epochs             = 3 if smoke_test else CFG.NUM_EPOCHS,
        learning_rate          = CFG.LEARNING_RATE,
        weight_decay           = CFG.WEIGHT_DECAY,
        dropout                = CFG.DROPOUT,
        early_stop_patience    = CFG.EARLY_STOP_PATIENCE,
        lr_patience            = CFG.LR_PATIENCE,
        use_amp                = CFG.USE_AMP,
        label_smoothing        = CFG.LABEL_SMOOTHING,
        label_mode             = label_mode,
        resume_existing        = resume_existing,
    )
    print("\n  Training …")
    t0           = time.time()
    train_result = train(model, train_loader, val_loader,
                         train_cfg, str(out_dir), run_name)
    elapsed = time.time() - t0

    # ── Evaluate ─────────────────────────────────────────────────────────────
    best_ckpt = out_dir / f"{run_name}_best.pt"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location="cpu"))

    print("\n  Evaluating …")
    eval_result = evaluate(
        model, test_loader,
        output_dir=str(out_dir),
        run_name=run_name,
        label_mode=label_mode,
        ripeness_names=CFG.RIPENESS_NAMES,
        class_names=CFG.CLASS_NAMES,
    )

    summary = {
        "run_name":       run_name,
        "seed":           seed,
        "sensor":         sensor,
        "model":          model_name,
        "train_mode":     train_mode,
        "ripeness_filter": rip_tag,
        "strategy":       strategy,
        "k_bands":        k,
        "n_params":       n_params,
        "epochs_trained": train_result["epochs_trained"],
        "best_val_loss":  train_result["best_val_loss"],
        "best_val_acc":   float(train_result["best_val_acc"]),
        "train_time_min": round(elapsed / 60, 2),
        **{kk: v for kk, v in eval_result.items() if not isinstance(v, dict)},
    }
    with open(out_dir / f"{run_name}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Comparison plot
# ─────────────────────────────────────────────────────────────────────────────

def _plot_comparison(
    results:        list,          # list of summary dicts
    baseline_acc:   float,         # hyperspectral test accuracy
    baseline_bands: int,           # total bands in hyperspectral model
    output_path:    str,
    run_label:      str,
):
    """Accuracy-vs-band-count curve, one line per strategy."""
    from collections import defaultdict

    by_strategy = defaultdict(lambda: {"k": [], "acc": []})
    for r in results:
        s = r.get("strategy", "unknown")
        by_strategy[s]["k"].append(r["k_bands"])
        by_strategy[s]["acc"].append(r.get("test_accuracy", r.get("best_val_acc", 0)))

    fig, ax = plt.subplots(figsize=(9, 5))

    colors = {"infested": "#e74c3c", "averaged": "#2980b9"}
    markers = {"infested": "o",       "averaged": "s"}

    for strategy, data in sorted(by_strategy.items()):
        k_arr   = np.array(data["k"])
        acc_arr = np.array(data["acc"])
        order   = np.argsort(k_arr)
        color  = colors.get(strategy, "gray")
        marker  = markers.get(strategy, "^")
        label   = ("Infested-class selection"
                   if strategy == "infested" else "Averaged-class selection")
        ax.plot(k_arr[order], acc_arr[order] * 100,
                color=color, marker=marker, linewidth=2,
                markersize=7, label=label)

    # Hyperspectral baseline
    if baseline_acc is not None:
        ax.axhline(baseline_acc * 100, color="black", linewidth=1.5,
                   linestyle="--", label=f"Hyperspectral baseline ({baseline_bands} bands)")
        ax.text(results[0]["k_bands"] if results else 5,
                baseline_acc * 100 + 0.5,
                f"{baseline_acc*100:.1f}%", fontsize=8, color="black")

    ax.set_xlabel("Number of selected bands (K)", fontsize=12)
    ax.set_ylabel("Test accuracy (%)", fontsize=12)
    ax.set_title(f"Multispectral Sweep — Accuracy vs. Band Count\n{run_label}",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks(sorted({r["k_bands"] for r in results}))

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Comparison plot → {output_path}")


def _save_comparison_csv(results: list, csv_path: str):
    if not results:
        return
    fields = ["strategy", "k_bands", "best_val_acc",
              "test_accuracy", "epochs_trained", "train_time_min", "run_name"]
    fields = [f for f in fields if any(f in r for r in results)]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"  Comparison CSV   → {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    train_seed = int(getattr(args, "seed", CFG.SEED))
    _set_training_seed(train_seed)

    selection = load_selection_json(args.selection_json)
    sensor     = selection["sensor"]
    run_name_hs = selection["run_name"]     # hyperspectral run name

    # Override sensor / model / mode from CLI if provided
    if args.sensor:
        sensor = args.sensor
    model_name = args.model
    train_mode = args.mode

    sweep = [int(x.strip()) for x in args.sweep.split(",") if x.strip()]

    strategies = (["infested", "averaged"] if args.strategy == "both"
                  else [args.strategy])

    output_root = Path(args.output_dir) if args.output_dir else Path(CFG.OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = _load_manifest(args.shard_dir) if args.shard_dir else None

    print("\n" + "="*70)
    print("  SWD Detection — Multispectral Band Sweep")
    print("="*70)
    print(f"  Sensor    : {sensor}")
    print(f"  Model     : {model_name}")
    print(f"  Mode      : {train_mode}")
    print(f"  Strategies: {strategies}")
    print(f"  Sweep     : {sweep}")
    print(f"  Seed      : {train_seed}")
    print(f"  Demo      : {args.smoke_test}")
    print("="*70)

    # ── Discover boards ───────────────────────────────────────────────────────
    print("\n  Discovering NIR boards …")
    boards_nir = discover_and_split(
        sensor="nir", paths=CFG.PATHS["nir"],
        train_frac=CFG.TRAIN_FRAC, val_frac=CFG.VAL_FRAC,
        seed=CFG.SEED, verbose=True,
    )
    print("\n  Discovering VNIR boards …")
    boards_vnir = discover_and_split(
        sensor="vnir", paths=CFG.PATHS["vnir"],
        train_frac=CFG.TRAIN_FRAC, val_frac=CFG.VAL_FRAC,
        seed=CFG.SEED, verbose=True,
    )

    # ── Hyperspectral baseline accuracy ──────────────────────────────────────
    baseline_acc   = 0.0
    baseline_bands = 224   # default
    if args.baseline_json and Path(args.baseline_json).exists():
        with open(args.baseline_json) as fh:
            bl = json.load(fh)
        baseline_acc   = float(bl.get("test_accuracy", bl.get("best_val_acc", 0)))
        baseline_bands = int(bl.get("n_bands", 224))
        print(f"\n  Hyperspectral baseline: {baseline_acc*100:.1f}% "
              f"({baseline_bands} bands)")

    # ── Run sweep ─────────────────────────────────────────────────────────────
    rip_filter   = None   # stage_agnostic
    all_summaries = []

    sweep_combos = [(s, k) for s in strategies for k in sweep]
    sweep_bar    = tqdm(sweep_combos, desc="MS sweep", unit="run",
                        dynamic_ncols=True)

    for strategy, k in sweep_bar:
        sweep_bar.set_description(f"MS sweep  strategy={strategy}  K={k}")
        try:
            # Retrieve selected band indices
            band_indices = get_band_indices(selection, strategy, k)
            band_arr     = np.array(band_indices, dtype=np.intp)

            nir_sel  = band_arr if sensor == "nir"  else None
            vnir_sel = band_arr if sensor == "vnir" else None

            summary = run_one(
                sensor=sensor,
                model_name=model_name,
                train_mode=train_mode,
                rip_filter=rip_filter,
                boards_nir=boards_nir,
                boards_vnir=boards_vnir,
                nir_sel=nir_sel,
                vnir_sel=vnir_sel,
                k=k,
                strategy=strategy,
                output_root=output_root,
                smoke_test=args.smoke_test,
                skip_existing=args.skip_existing,
                resume_existing=not args.no_resume,
                manifest_rows=manifest_rows,
                seed=train_seed,
            )
            if summary:
                all_summaries.append(summary)
        except Exception:
            tqdm.write(f"\n  [ERROR] strategy={strategy} K={k} failed:")
            traceback.print_exc()

    # ── Comparison outputs ────────────────────────────────────────────────────
    if all_summaries:
        ms_dir = output_root / "multispectral"
        ms_dir.mkdir(exist_ok=True)

        hs_label = f"{sensor}__{model_name}__{train_mode}__all"
        _plot_comparison(
            all_summaries,
            baseline_acc=baseline_acc,
            baseline_bands=baseline_bands,
            output_path=str(ms_dir / "multispectral_comparison.png"),
            run_label=hs_label,
        )
        _save_comparison_csv(
            all_summaries,
            csv_path=str(ms_dir / "multispectral_comparison.csv"),
        )

    print("\n  Multispectral sweep complete.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Multispectral band-sweep experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--sensor",  default=None,
                   choices=["nir", "vnir"],
                   help="Override sensor from band_selection.json")
    p.add_argument("--model",   default="cnn3d",
                   choices=["cnn3d", "cnn3d_transformer"],
                   help="Model architecture to use")
    p.add_argument("--mode",    default="stage_agnostic",
                   choices=["stage_agnostic", "per_stage"],
                   help="Training mode")
    p.add_argument("--ripeness", default=None,
                   choices=["Ripe", "Midripe", "Unripe"],
                   help="Ripeness stage to train when --mode per_stage")
    p.add_argument("--selection_json", required=True,
                   help="Path to band_selection.json from band_select.py")
    p.add_argument("--sweep",
                   default=",".join(str(k) for k in CFG.MS_BAND_COUNTS),
                   help="Comma-separated band counts to sweep")
    p.add_argument("--strategy", default="averaged",
                   choices=["infested", "averaged", "both"],
                   help="Band-selection strategy to use")
    p.add_argument("--baseline_json", default=None,
                   help="Path to {run_name}_summary.json for hyperspectral baseline accuracy")
    p.add_argument("--shard_dir", default=None,
                   help="Path to shards directory containing manifest.csv")
    p.add_argument("--output_dir", default=None,
                   help="Output root for this multispectral run")
    p.add_argument("--seed", type=int, default=getattr(CFG, "SEED", 42),
                   help="Training random seed. The sharded manifest split is unchanged.")
    p.add_argument("--demo_run", dest="smoke_test", action="store_true",
                   help="Demo run: 3 epochs on a reduced sample")
    p.add_argument("--smoke_test", dest="smoke_test", action="store_true",
                   help=argparse.SUPPRESS)
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip runs that already have a summary.json")
    p.add_argument("--no_resume", action="store_true",
                   help="Do not resume from an existing *_last.pt checkpoint")
    main(p.parse_args())
