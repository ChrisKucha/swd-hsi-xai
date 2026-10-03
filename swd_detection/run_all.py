"""
run_all.py — Orchestrator for all SWD detection experiments.

Loops over every combination of:
    sensor       : nir | vnir
    model        : cnn3d | cnn3d_transformer
    training mode: stage_agnostic | per_stage (Ripe, Midripe, Unripe)

For each combination:
  1. Discovers and splits boards (board-level, no leakage)
  2. Builds train/val/test DataLoaders
  3. Instantiates the model
  4. Trains with early stopping
  5. Evaluates on the test set
  6. Saves all artifacts to outputs/{sensor}/{model}/{mode}/

After all runs, calls compare_runs() to produce a summary table and bar chart.

Usage
-----
  # Run everything (all sensors × all models × all modes):
  python run_all.py

  # Demo run (3 epochs, reduced dataset):
  python run_all.py --demo_run

  # Single combination:
  python run_all.py --sensor nir --model cnn3d --mode stage_agnostic

  # Skip already-completed runs:
  python run_all.py --skip_existing
"""

import argparse
import json
import os
import random
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

# ── Ensure swd_detection/ is at the front of sys.path ────────────────────────
# swd_detection/config.py must be found before pytorch/config.py
_HERE = Path(__file__).parent
_SWD_DIR = str(_HERE.resolve())
if _SWD_DIR in sys.path:
    sys.path.remove(_SWD_DIR)
sys.path.insert(0, _SWD_DIR)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

import config as CFG
from data.discovery import discover_and_split
from data.dataset   import BlueberryDataset, ShardedBlueberryDataset
from data.transforms import get_train_transform, get_val_transform
from models          import build_model
from train           import train
from evaluate        import evaluate, compare_runs
from explain         import run_explain
from band_select     import (load_importance_csv, select_bands,
                             save_selection_json, plot_band_selection)
from run_multispectral import run_one as ms_run_one, _plot_comparison, _save_comparison_csv

# ── Experiment matrix ────────────────────────────────────────────────────────
ALL_SENSORS = ["nir", "vnir"]
ALL_MODELS  = ["cnn3d", "cnn3d_transformer"]
ALL_MODES   = ["stage_agnostic", "per_stage"]
RIPENESS_STAGES = ["Ripe", "Midripe", "Unripe"]


def _set_training_seed(seed: int) -> None:
    """Set training randomness without changing the manifest-defined split."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Shard manifest helper ─────────────────────────────────────────────────────

def _load_manifest(shard_dir) -> list:
    """
    Load manifest.csv from shard_dir.  Returns list of row dicts, or None if
    the manifest does not exist (signals: fall back to cube-based loading).
    """
    import csv
    manifest_path = Path(shard_dir) / "manifest.csv"
    if not manifest_path.exists():
        return None
    with open(manifest_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        if row.get("ripeness", "").lower() in {"completelyripe", "completely"}:
            row["ripeness"] = "Ripe"
    print(f"  [Shards] Manifest loaded: {len(rows)} rows from {manifest_path}")
    return rows


def _filter_manifest(rows, sensor, split, rip_filter):
    """Return manifest rows matching sensor / split / ripeness filter."""
    return [
        r for r in rows
        if r["sensor"] == sensor
        and r["split"]  == split
        and (rip_filter is None or r["ripeness"] == rip_filter)
    ]


def _labels_for_sampler(ds):
    """Return class labels without loading tensors when available."""
    if isinstance(ds, torch.utils.data.Subset):
        base_labels = _labels_for_sampler(ds.dataset)
        if base_labels is None:
            return None
        return [base_labels[i] for i in ds.indices]
    if hasattr(ds, "_rows"):
        return [int(r["label"]) for r in ds._rows]
    if hasattr(ds, "_index"):
        return [int(board["label"]) for board, _ in ds._index]
    return None


def _balanced_train_sampler(ds):
    labels = _labels_for_sampler(ds)
    if not labels:
        return None, None
    counts = Counter(labels)
    weights = torch.as_tensor([1.0 / counts[label] for label in labels], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler, counts


# ── DataLoader factory ────────────────────────────────────────────────────────

def _make_loaders(
    sensor:     str,
    boards_nir,
    boards_vnir,
    split:      str,
    rip_filter,
    label_mode: str,
    batch_size: int,
    smoke_test: bool,
    cell_h:     int = None,
    cell_w:     int = None,
    nir_selected_bands=None,    # pre-selected NIR band indices (selected-band models)
    vnir_selected_bands=None,   # pre-selected VNIR band indices (selected-band models)
    manifest_rows=None,         # pre-loaded manifest rows (sharded mode)
    balanced_train_sampler: bool = False,
) -> DataLoader:
    """Build one DataLoader for a given split, sensor, and ripeness filter.

    If manifest_rows is provided (i.e. shards exist), uses ShardedBlueberryDataset
    each __getitem__ is a single tiny np.load, no RAM
    pressure.  Falls back to lazy cube loading when manifest_rows is None.
    """
    n_workers = 0 if os.name == "nt" else int(getattr(CFG, "NUM_WORKERS", 4))
    transform = get_train_transform() if split == "train" else get_val_transform()

    # ── Sharded path ──────────────────────────────────────────────────────────
    if manifest_rows is not None:
        rows = _filter_manifest(manifest_rows, sensor, split, rip_filter)
        ds = ShardedBlueberryDataset(
            rows,
            label_mode=label_mode,
            transform=transform,
            selected_bands=(nir_selected_bands if sensor == "nir"
                            else vnir_selected_bands),
        )

    # ── Cube-based fallback ───────────────────────────────────────────────────
    else:
        subset = [b for b in (boards_nir if sensor != "vnir" else boards_vnir)
                  if b["split"] == split
                  and (rip_filter is None or b["ripeness"] == rip_filter)]
        ds = BlueberryDataset(
            subset, sensor=sensor,
            label_mode=label_mode,
            transform=transform,
            cell_h=cell_h,
            cell_w=cell_w,
        )

    if smoke_test:
        # Use a small random subset for fast validation
        n = min(len(ds), 200)
        ds, _ = torch.utils.data.random_split(ds, [n, len(ds) - n])

    sampler = None
    sampler_counts = None
    if split == "train" and balanced_train_sampler:
        sampler, sampler_counts = _balanced_train_sampler(ds)
        if sampler is None:
            print("    [WARN] Balanced sampler requested but labels were unavailable.")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train" and sampler is None),
        sampler=sampler,
        num_workers=n_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train" and len(ds) > batch_size),
        persistent_workers=(n_workers > 0),
        prefetch_factor=(4 if n_workers > 0 else None),
    )
    print(f"    {split:<6}: {len(ds):>6} samples  →  {len(loader):>4} batches  "
          f"(workers={n_workers})")
    if sampler_counts is not None:
        print(f"           balanced train sampler over labels: {dict(sorted(sampler_counts.items()))}")
    return loader, ds


def _get_n_bands(sensor, boards_nir, boards_vnir):
    """Determine band counts by loading the first board of each sensor."""
    import numpy as np
    from data.dataset import load_cube
    if sensor == "nir":
        b = next((b for b in boards_nir), None)
        if b:
            nir_bands = load_cube(b["path"], "nir").shape[2]
        else:
            nir_bands = 224
    else:
        nir_bands = 0

    if sensor == "vnir":
        b = next((b for b in boards_vnir), None)
        if b:
            vnir_bands = load_cube(b["path"], "vnir").shape[2]
        else:
            vnir_bands = 224
    else:
        vnir_bands = 0

    return nir_bands, vnir_bands


# ── Model factory ─────────────────────────────────────────────────────────────

def _build(
    model_name: str,
    sensor:     str,
    nir_bands:  int,
    vnir_bands: int,
    cell_h:     int,
    cell_w:     int,
    label_mode: str,
    smoke_test: bool,
    nir_selected_bands=None,
    vnir_selected_bands=None,
):
    """Instantiate the requested model with the correct band counts."""
    kwargs = dict(
        num_classes=CFG.NUM_CLASSES,
        dropout=CFG.DROPOUT,
        label_mode=label_mode,
    )

    n_bands = vnir_bands if sensor == "vnir" else nir_bands
    if sensor == "nir" and nir_selected_bands is not None:
        n_bands = len(nir_selected_bands)
    elif sensor == "vnir" and vnir_selected_bands is not None:
        n_bands = len(vnir_selected_bands)

    kwargs.update(
        n_bands=n_bands,
        cell_h=cell_h,
        cell_w=cell_w,
    )
    if model_name == "cnn3d_transformer":
        kwargs.update(
            n_layers=CFG.TRANSFORMER_N_LAYERS,
            n_heads=CFG.TRANSFORMER_N_HEADS,
        )

    return build_model(model_name, **kwargs)


# ── Multispectral cascade ─────────────────────────────────────────────────────

def _run_multispectral_cascade(
    sensor:       str,
    model_name:   str,
    train_mode:   str,
    rip_filter,
    boards_nir,
    boards_vnir,
    out_dir:      Path,
    run_name:     str,
    output_root:  Path,
    smoke_test:   bool,
    sweep=None,
    manifest_rows=None,
):
    """
    Automatically cascade from hyperspectral results into a multispectral sweep.

    Steps
    -----
    1. Locate the spectral_importance.csv written by run_explain().
    2. Run band selection (both strategies × all K in sweep).
    3. For every (strategy, K) pair train + evaluate a new model using only
       those K bands — simulating a K-band multispectral camera.
    4. Plot accuracy-vs-bands comparison curve and save a CSV summary.
    """
    if sweep is None:
        sweep = CFG.MS_BAND_COUNTS

    imp_csv = out_dir / f"{run_name}_spectral_importance.csv"
    if not imp_csv.exists():
        print(f"  [MS-CASCADE] No spectral importance CSV found at {imp_csv} — "
              f"skipping multispectral cascade.")
        return

    print(f"\n  {'='*66}")
    print(f"  MULTISPECTRAL CASCADE  |  {run_name}")
    print(f"  {'='*66}")

    # ── Band selection ────────────────────────────────────────────────────────
    importance  = load_importance_csv(str(imp_csv))
    wavelengths = importance["wavelength_nm"]

    strategies = list(getattr(CFG, "MS_STRATEGIES", ["averaged"]))
    selections  = select_bands(importance, sweep=sweep, strategies=strategies)

    json_path = str(out_dir / f"{run_name}_band_selection.json")
    save_selection_json(
        selections, wavelengths,
        sensor=sensor,
        source_csv=str(imp_csv),
        run_name=run_name,
        out_path=json_path,
    )

    png_path = str(out_dir / f"{run_name}_band_selection.png")
    plot_band_selection(
        importance, selections, sweep,
        sensor=sensor, run_name=run_name, out_path=png_path,
    )

    # Hyperspectral baseline band count
    n_bands_total = len(wavelengths)

    # ── Multispectral sweep ───────────────────────────────────────────────────
    import numpy as _np

    ms_results: list = []
    strategies = list(selections.keys())

    for strategy in strategies:
        for k in sweep:
            band_indices = selections[strategy][k]
            band_arr = _np.array(band_indices, dtype=_np.int64)

            # nir_sel / vnir_sel follow the same convention as run_multispectral.py:
            #   nir sensor  → nir_sel=band_arr,  vnir_sel=None
            #   vnir sensor → nir_sel=None,       vnir_sel=band_arr
            nir_sel  = band_arr if sensor == "nir"  else None
            vnir_sel = band_arr if sensor == "vnir" else None

            result = ms_run_one(
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
                output_root=output_root,   # run_one appends /multispectral/{strategy}/k{K}
                smoke_test=smoke_test,
                skip_existing=True,
                manifest_rows=manifest_rows,
            )
            if result is not None:
                ms_results.append(result)

    # ── Comparison plot + CSV ────────────────────────────────────────────────
    if ms_results:
        comp_dir = output_root / "multispectral"
        comp_dir.mkdir(parents=True, exist_ok=True)

        # Recover hyperspectral baseline accuracy from the summary JSON
        baseline_acc = None
        summary_path = out_dir / f"{run_name}_summary.json"
        if summary_path.exists():
            with open(summary_path) as _f:
                summary_data = json.load(_f)
                baseline_acc = summary_data.get(
                    "test_accuracy",
                    summary_data.get("accuracy", summary_data.get("best_val_acc")),
                )

        _plot_comparison(
            results=ms_results,
            baseline_acc=baseline_acc,
            baseline_bands=n_bands_total,
            output_path=str(comp_dir / f"{run_name}_ms_comparison.png"),
            run_label=run_name,
        )
        _save_comparison_csv(
            results=ms_results,
            csv_path=str(comp_dir / f"{run_name}_ms_comparison.csv"),
        )
        print(f"\n  [MS-CASCADE] Done.  Results in: {comp_dir}")


# ── Single experiment ─────────────────────────────────────────────────────────

def run_experiment(
    sensor:      str,
    model_name:  str,
    train_mode:  str,
    rip_filter,           # None = stage_agnostic; str = specific ripeness
    boards_nir,
    boards_vnir,
    output_root: Path,
    smoke_test:  bool,
    skip_existing: bool,
    nir_selected_bands=None,   # pre-selected NIR band indices (selected-band models)
    vnir_selected_bands=None,  # pre-selected VNIR band indices (selected-band models)
    manifest_rows=None,        # sharded manifest rows (None = cube-based loading)
    balanced_train_sampler: bool = False,
    resume_existing: bool = True,
    seed: int = None,
):
    label_mode = "binary"

    rip_tag  = rip_filter or "all"
    run_name = f"{sensor}__{model_name}__{train_mode}__{rip_tag}"
    out_dir  = output_root / sensor / model_name / train_mode / rip_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already done
    summary_path = out_dir / f"{run_name}_summary.json"
    if skip_existing and summary_path.exists():
        print(f"  [SKIP] {run_name} — already completed.")
        with open(summary_path) as f:
            return json.load(f)

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT : {run_name}")
    print(f"  Output dir : {out_dir}")
    print(f"{'='*70}")

    # ── Infer fixed cell size from first training board ───────────────────────
    # Done ONCE here so all three loaders use identical cell dimensions,
    # preventing shape mismatches when boards have different raw spatial sizes.
    print("\n  Inferring cell size …")
    from data.dataset import load_cube
    import numpy as np
    cell_h = cell_w = None
    if manifest_rows is not None:
        _sensor = "nir" if sensor == "nir" else "vnir"
        _rows = [r for r in manifest_rows
                 if r.get("sensor") == _sensor
                 and (rip_filter is None or r.get("ripeness") == rip_filter)]
        if _rows:
            cell_h = int(_rows[0]["cell_h"]); cell_w = int(_rows[0]["cell_w"])
    if cell_h is None:
        _src_boards = boards_nir if sensor == "nir" else boards_vnir
        _src_sensor = "nir" if sensor == "nir" else "vnir"
        _rip_boards = [b for b in _src_boards
                       if rip_filter is None or b["ripeness"] == rip_filter]
        if not _rip_boards:
            print("  [WARN] No boards or manifest rows for this filter — skipping.")
            return
        _first_cube = load_cube(_rip_boards[0]["path"], _src_sensor)
        _H, _W, _   = _first_cube.shape
        _row_e      = np.linspace(0, _H, CFG.BERRY_GRID_ROWS + 1, dtype=int)
        _col_e      = np.linspace(0, _W, CFG.BERRY_GRID_COLS + 1, dtype=int)
        cell_h      = int(_row_e[1] - _row_e[0])
        cell_w      = int(_col_e[1] - _col_e[0])
    print(f"  Fixed cell size: {cell_h} × {cell_w} px  "
          f"(all boards resized to this)")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    print("\n  Building DataLoaders …")
    _loader_kw = dict(
        nir_selected_bands=nir_selected_bands,
        vnir_selected_bands=vnir_selected_bands,
        manifest_rows=manifest_rows,
        balanced_train_sampler=balanced_train_sampler,
    )
    train_loader, train_ds = _make_loaders(
        sensor, boards_nir, boards_vnir, "train",
        rip_filter, label_mode, CFG.BATCH_SIZE, smoke_test,
        cell_h=cell_h, cell_w=cell_w, **_loader_kw)
    val_loader,   _        = _make_loaders(
        sensor, boards_nir, boards_vnir, "val",
        rip_filter, label_mode, CFG.BATCH_SIZE, smoke_test,
        cell_h=cell_h, cell_w=cell_w, **_loader_kw)
    test_loader,  _        = _make_loaders(
        sensor, boards_nir, boards_vnir, "test",
        rip_filter, label_mode, getattr(CFG, "EVAL_BATCH_SIZE", CFG.BATCH_SIZE), smoke_test,
        cell_h=cell_h, cell_w=cell_w, **_loader_kw)

    if len(train_loader) == 0 or len(val_loader) == 0:
        print("  [WARN] Empty train or val loader — skipping experiment.")
        return

    # ── Band counts ───────────────────────────────────────────────────────────
    nir_bands, vnir_bands = _get_n_bands(sensor, boards_nir, boards_vnir)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n  Building model …")
    model = _build(model_name, sensor, nir_bands, vnir_bands,
                   cell_h, cell_w, label_mode, smoke_test,
                   nir_selected_bands=nir_selected_bands,
                   vnir_selected_bands=vnir_selected_bands)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ── Training config ───────────────────────────────────────────────────────
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
        max_grad_norm          = CFG.MAX_GRAD_NORM,
        resume_existing        = resume_existing,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n  Training …")
    t0       = time.time()
    train_result = train(model, train_loader, val_loader,
                         train_cfg, str(out_dir), run_name)
    elapsed  = time.time() - t0
    print(f"\n  Training complete in {elapsed/60:.1f} min  "
          f"({train_result['epochs_trained']} epochs)")
    print(f"  Best val loss: {train_result['best_val_loss']:.4f}  "
          f"Best val acc: {train_result['best_val_acc']:.4f}")

    # ── Load best weights ─────────────────────────────────────────────────────
    best_ckpt = out_dir / f"{run_name}_best.pt"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location="cpu")
        model.load_state_dict(state)
        print("  Best checkpoint loaded for evaluation.")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n  Evaluating on test set …")
    eval_result = evaluate(
        model, test_loader,
        output_dir=str(out_dir),
        run_name=run_name,
        label_mode=label_mode,
        ripeness_names=CFG.RIPENESS_NAMES,
        class_names=CFG.CLASS_NAMES,
    )

    # ── Explainability ────────────────────────────────────────────────────────
    print("\n  Running explainability analysis …")
    try:
        run_explain(
            model       = model,
            test_loader = test_loader,
            output_dir  = str(out_dir),
            run_name    = run_name,
            sensor      = sensor,
            model_name  = model_name,
            n_samples   = 8 if smoke_test else 32,
            n_ig_steps  = 10 if smoke_test else 50,
            class_names = CFG.CLASS_NAMES,
        )
    except Exception:
        print("  [WARN] Explainability failed (non-fatal):")
        traceback.print_exc()

    # Merge train + eval results into one summary
    summary = {
        "run_name":       run_name,
        "seed":           seed,
        "sensor":         sensor,
        "model":          model_name,
        "train_mode":     train_mode,
        "ripeness_filter": rip_tag,
        "label_mode":     label_mode,
        "n_params":       n_params,
        "epochs_trained": train_result["epochs_trained"],
        "best_val_loss":  train_result["best_val_loss"],
        "best_val_acc":   float(train_result["best_val_acc"]),
        "train_time_min": round(elapsed / 60, 2),
        **{k: v for k, v in eval_result.items() if not isinstance(v, dict)},
    }
    with open(out_dir / f"{run_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary saved → {out_dir / f'{run_name}_summary.json'}")
    return summary




# ── Main orchestrator ─────────────────────────────────────────────────────────

def main(args):
    train_seed = int(getattr(args, "seed", CFG.SEED))
    _set_training_seed(train_seed)

    output_root = Path(getattr(args, "output_dir", None) or CFG.OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    sensors    = [args.sensor]    if args.sensor    else ALL_SENSORS
    models     = [args.model]     if args.model     else ALL_MODELS
    excluded_models = set(getattr(args, "exclude_model", []) or [])
    models = [m for m in models if m not in excluded_models]
    modes      = [args.mode]      if args.mode      else ALL_MODES
    requested_ripeness = getattr(args, "ripeness", None)

    # ── Shard manifest (optional) ─────────────────────────────────────────────
    manifest_rows = None
    shard_dir     = getattr(args, "shard_dir", None)
    if shard_dir:
        manifest_rows = _load_manifest(shard_dir)
        if manifest_rows is None:
            print(f"  [Shards] manifest.csv not found in {shard_dir}.")
            print(f"    Run:  python create_shards.py --shard-dir {shard_dir}")
            print(f"    Falling back to cube-based loading.")
        else:
            print(f"  [Shards] Using sharded data from {shard_dir}  "
                  f"({len(manifest_rows)} cells)")

    # ── Discover boards once for NIR and VNIR ─────────────────────────────────
    print("\n" + "="*70)
    print("  SWD Detection — Experiment Runner")
    print("="*70)
    print(f"  Sensors : {sensors}")
    print(f"  Models  : {models}")
    print(f"  Modes   : {modes}")
    print(f"  Seed    : {train_seed}")
    print(f"  Demo    : {args.smoke_test}")
    print("="*70)

    if manifest_rows is not None:
        print("\n  Sharded manifest detected; skipping raw-board discovery.")
        boards_nir = []
        boards_vnir = []
    else:
        print("\n  Discovering NIR boards …")
        boards_nir = discover_and_split(
            sensor="nir",
            paths=CFG.PATHS["nir"],
            train_frac=CFG.TRAIN_FRAC,
            val_frac=CFG.VAL_FRAC,
            seed=CFG.SEED,
            verbose=True,
        )

        print("\n  Discovering VNIR boards …")
        boards_vnir = discover_and_split(
            sensor="vnir",
            paths=CFG.PATHS["vnir"],
            train_frac=CFG.TRAIN_FRAC,
            val_frac=CFG.VAL_FRAC,
            seed=CFG.SEED,
            verbose=True,
        )

    # ── Build flat experiment list for progress tracking ──────────────────────
    experiments = []
    for sensor in sensors:
        for model_name in models:
            for mode in modes:
                rip_filters = [None] if mode == "stage_agnostic" else RIPENESS_STAGES
                if mode == "per_stage" and requested_ripeness:
                    rip_filters = [requested_ripeness]
                for rip_filter in rip_filters:
                    experiments.append((sensor, model_name, mode, rip_filter))

    # ── Run experiments ───────────────────────────────────────────────────────
    all_summaries = []
    exp_bar = tqdm(experiments, desc="Experiments", unit="run", dynamic_ncols=True)

    for sensor in sensors:
        for model_name in models:
            for mode in modes:
                rip_filters = (
                    [None] if mode == "stage_agnostic"
                    else RIPENESS_STAGES
                )
                if mode == "per_stage" and requested_ripeness:
                    rip_filters = [requested_ripeness]
                for rip_filter in rip_filters:
                    exp_bar.set_description(
                        f"{sensor}/{model_name}/{mode}/{rip_filter or 'all'}"
                    )
                    tqdm.write(f"\n  sensor={sensor}  model={model_name}  "
                               f"mode={mode}  ripeness={rip_filter or 'all'}")
                    try:
                        summary = run_experiment(
                            sensor=sensor,
                            model_name=model_name,
                            train_mode=mode,
                            rip_filter=rip_filter,
                            boards_nir=boards_nir,
                            boards_vnir=boards_vnir,
                            output_root=output_root,
                            smoke_test=args.smoke_test,
                            skip_existing=args.skip_existing,
                            nir_selected_bands=None,
                            vnir_selected_bands=None,
                            manifest_rows=manifest_rows,
                            balanced_train_sampler=args.balanced_train_sampler,
                            resume_existing=not args.no_resume,
                            seed=train_seed,
                        )
                        if summary:
                            all_summaries.append(summary)
                    except Exception:
                        tqdm.write(f"\n  [ERROR] Experiment failed:")
                        traceback.print_exc()
                    finally:
                        exp_bar.update(1)

    # ── Final comparison table ────────────────────────────────────────────────
    if all_summaries:
        print("\n" + "="*70)
        print("  Generating comparison table …")
        compare_runs(str(output_root), str(output_root))
        print("="*70)

    # ── Multispectral cascade — stage-agnostic/all model per sensor ───────────
    if all_summaries and not args.skip_multispectral:
        print("\n" + "="*70)
        print("  Multispectral cascade — running stage_agnostic/all models per sensor …")
        print("="*70)

        # Group summaries by sensor, then restrict the cascade to stage_agnostic/all
        # runs so selected multispectral bands target all maturity stages together.
        # Falls back to best_val_acc if test_acc is not present.
        def _acc(s):
            return float(s.get("test_accuracy",
                           s.get("accuracy",
                           s.get("best_val_acc", 0.0))))

        summaries_by_sensor = {}
        for s in all_summaries:
            summaries_by_sensor.setdefault(s["sensor"], []).append(s)

        for sensor_name, sensor_summaries in summaries_by_sensor.items():
            stage_agnostic_all = [
                s for s in sensor_summaries
                if s.get("train_mode") == "stage_agnostic"
                and s.get("ripeness_filter", "all") == "all"
            ]
            if not stage_agnostic_all:
                print(f"\n  [MS-CASCADE] No stage_agnostic/all summary for sensor={sensor_name}; skipping.")
                continue

            for summary in sorted(stage_agnostic_all, key=lambda s: s.get("model", "")):
                print(f"\n  Stage-agnostic/all model for sensor={sensor_name}:")
                print(f"    run_name : {summary['run_name']}")
                print(f"    model    : {summary['model']}")
                print(f"    mode     : {summary['train_mode']}")
                print(f"    accuracy : {_acc(summary):.4f}")

                rip_tag = summary.get("ripeness_filter", "all")
                summary_out_dir = (output_root / summary["sensor"]
                                   / summary["model"] / summary["train_mode"] / rip_tag)
                try:
                    _run_multispectral_cascade(
                        sensor      = summary["sensor"],
                        model_name  = summary["model"],
                        train_mode  = summary["train_mode"],
                        rip_filter  = None if rip_tag == "all" else rip_tag,
                        boards_nir  = boards_nir,
                        boards_vnir = boards_vnir,
                        out_dir     = summary_out_dir,
                        run_name    = summary["run_name"],
                        output_root = output_root,
                        smoke_test  = args.smoke_test,
                        manifest_rows = manifest_rows,
                    )
                except Exception:
                    tqdm.write(f"  [WARN] Cascade failed for {summary['run_name']}:")
                    traceback.print_exc()

        print(f"\n  All done.  Results in: {output_root}")
        print("="*70)



# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Run all SWD detection experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--sensor", choices=ALL_SENSORS,  default=None,
                   help="Single sensor to run (default: all)")
    p.add_argument("--model",  choices=ALL_MODELS,   default=None,
                   help="Single model to run (default: all)")
    p.add_argument("--exclude_model", choices=ALL_MODELS, action="append", default=[],
                   help="Model to skip. Can be passed multiple times.")
    p.add_argument("--mode",   choices=ALL_MODES,    default=None,
                   help="Single training mode (default: all)")
    p.add_argument("--ripeness", choices=RIPENESS_STAGES, default=None,
                   help="Single ripeness stage for --mode per_stage")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Override output directory for this run")
    p.add_argument("--seed", type=int, default=getattr(CFG, "SEED", 42),
                   help="Training random seed. The sharded manifest split is unchanged.")
    p.add_argument("--balanced_train_sampler", action="store_true",
                   help="Use inverse-frequency weighted sampling for the training loader")
    p.add_argument("--demo_run", dest="smoke_test", action="store_true",
                   help="Demo run: 3 epochs with reduced attribution analysis")
    p.add_argument("--smoke_test", dest="smoke_test", action="store_true",
                   help=argparse.SUPPRESS)
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip runs that already have a summary.json")
    p.add_argument("--no_resume", action="store_true",
                   help="Do not resume from an incomplete *_last.pt checkpoint")
    p.add_argument("--skip_multispectral", action="store_true",
                   help="Skip the multispectral cascade (runs on best model per sensor by default)")
    p.add_argument("--shard_dir", type=str, default=getattr(CFG, "SHARD_DIR", None),
                   help="Path to shards directory (containing manifest.csv).  "
                        "If set, uses fast ShardedDataset instead of loading full cubes.  "
                        "Create shards first with:  python create_shards.py --shard-dir DIR")
    main(p.parse_args())
