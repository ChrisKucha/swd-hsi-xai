"""
run_shap.py — SHAP spectral importance for trained full-spectrum models.

This script loads an existing full-spectrum checkpoint, computes SHAP
attributions on a small balanced sample, aggregates attribution magnitude per
wavelength, and writes band-selection files compatible with run_multispectral.py.

Example
-------
  python run_shap.py \\
    --shard_dir "/media/kuchalab/New Volume/swd_detection_shards" \\
    --sensor nir --model cnn3d --mode stage_agnostic --stage all \\
    --background_samples 16 --explain_samples 48 --top_k 30
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

_HERE = Path(__file__).parent
_SWD_DIR = str(_HERE.resolve())
if _SWD_DIR in sys.path:
    sys.path.remove(_SWD_DIR)
sys.path.insert(0, _SWD_DIR)

import config as CFG
from band_select import plot_band_selection, save_selection_json, select_bands
from data.dataset import ShardedBlueberryDataset
from data.transforms import get_val_transform
from models import build_model


def _load_manifest(shard_dir: str) -> List[dict]:
    manifest_path = Path(shard_dir) / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.csv not found in {shard_dir}")
    with open(manifest_path, newline="") as fh:
        return list(csv.DictReader(fh))


def _filter_rows(
    rows: Iterable[dict],
    sensor: str,
    split: str,
    ripeness: str | None = None,
) -> List[dict]:
    return [
        r for r in rows
        if r.get("sensor") == sensor
        and r.get("split") == split
        and r.get("ripeness")
        and (ripeness is None or r.get("ripeness") == ripeness)
    ]


def _balanced_indices(rows: List[dict], n: int, seed: int) -> List[int]:
    """Return indices balanced across class and ripeness as much as possible."""
    rng = np.random.default_rng(seed)
    groups: Dict[tuple, List[int]] = {}
    for idx, row in enumerate(rows):
        key = (row.get("label"), row.get("ripeness"))
        groups.setdefault(key, []).append(idx)

    keys = sorted(groups)
    for key in keys:
        rng.shuffle(groups[key])

    selected: List[int] = []
    cursor = 0
    while len(selected) < n and keys:
        key = keys[cursor % len(keys)]
        if groups[key]:
            selected.append(groups[key].pop())
        elif all(not groups[k] for k in keys):
            break
        cursor += 1
    return selected


def _build_model(model_name: str, sensor: str, n_bands: int, cell_h: int, cell_w: int):
    kwargs = dict(
        num_classes=CFG.NUM_CLASSES,
        dropout=CFG.DROPOUT,
        label_mode="binary",
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


def _wavelengths(sensor: str, n_bands: int) -> np.ndarray:
    wl = np.asarray(CFG.WAVELENGTHS[sensor], dtype=np.float32)
    if wl.size == n_bands:
        return wl
    return np.interp(
        np.linspace(0, wl.size - 1, n_bands),
        np.arange(wl.size),
        wl,
    ).astype(np.float32)


def _stack_samples(
    ds: ShardedBlueberryDataset,
    indices: List[int],
    spatial_size=None,
) -> torch.Tensor:
    xs = [ds[i][0] for i in indices]
    x = torch.stack(xs, dim=0).float()
    if spatial_size is not None:
        x = F.interpolate(
            x,
            size=spatial_size,
            mode="bilinear",
            align_corners=False,
        )
    return x


def _normalize_shap_values(shap_values, n_classes: int) -> List[np.ndarray]:
    """Convert SHAP output variants into one array per class."""
    if isinstance(shap_values, torch.Tensor):
        shap_values = shap_values.detach().cpu().numpy()

    if isinstance(shap_values, list):
        arrays = []
        for value in shap_values:
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            arrays.append(np.asarray(value))
        return arrays[:n_classes]

    arr = np.asarray(shap_values)
    if arr.ndim == 5 and arr.shape[-1] == n_classes:
        return [arr[..., ci] for ci in range(n_classes)]
    if arr.ndim == 5 and arr.shape[0] == n_classes:
        return [arr[ci] for ci in range(n_classes)]
    if arr.ndim == 4:
        return [arr]
    raise ValueError(f"Unsupported SHAP output shape: {arr.shape}")


def _save_shap_importance_csv(
    out_path: Path,
    wavelengths: np.ndarray,
    class_importance: Dict[str, np.ndarray],
) -> None:
    class_names = list(class_importance)
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["band_idx", "wavelength_nm", *class_names])
        for bi, wl in enumerate(wavelengths):
            writer.writerow(
                [bi, f"{float(wl):.2f}"]
                + [f"{float(class_importance[name][bi]):.8f}" for name in class_names]
            )
    print(f"  SHAP spectral importance CSV -> {out_path}")


def _save_shap_plot(
    out_path: Path,
    wavelengths: np.ndarray,
    class_importance: Dict[str, np.ndarray],
    run_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, values in class_importance.items():
        ax.plot(wavelengths, values, linewidth=1.5, label=name)
    avg = np.mean(np.stack(list(class_importance.values()), axis=0), axis=0)
    ax.plot(wavelengths, avg, color="black", linewidth=2.0, linestyle="--",
            label="Averaged")
    ax.set_title(f"SHAP Spectral Importance - {run_name}", fontsize=10)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Mean |SHAP value|")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  SHAP spectral importance plot -> {out_path}")


def run_shap(args) -> None:
    try:
        import shap
    except ImportError as exc:
        raise SystemExit(
            "shap is not installed. Install it in this environment with:\n"
            "  pip install shap"
        ) from exc

    if args.model not in ("cnn3d", "cnn3d_transformer"):
        raise ValueError("run_shap.py currently supports cnn3d and cnn3d_transformer.")

    if getattr(args, "disable_cudnn", False):
        torch.backends.cudnn.enabled = False
        print("  [SHAP] cuDNN disabled to reduce peak GPU workspace memory.")

    mode = getattr(args, "mode", "stage_agnostic")
    requested_stage = getattr(args, "stage", "all")

    if mode == "stage_agnostic":
        stage = "all"
        ripeness_filter = None
    else:
        if requested_stage == "all":
            raise ValueError("--mode per_stage requires --stage Ripe, Midripe, or Unripe")
        stage = requested_stage
        ripeness_filter = stage

    rows = _load_manifest(args.shard_dir)
    explain_split = getattr(args, "explain_split", "test")
    train_rows = _filter_rows(rows, args.sensor, "train", ripeness_filter)
    explain_rows = _filter_rows(rows, args.sensor, explain_split, ripeness_filter)
    if not train_rows or not explain_rows:
        raise ValueError(
            f"No train/{explain_split} shard rows found for sensor={args.sensor}, "
            f"mode={mode}, stage={stage}"
        )

    cell_h = int(train_rows[0]["cell_h"])
    cell_w = int(train_rows[0]["cell_w"])
    n_bands = int(train_rows[0]["n_bands"])
    run_name = f"{args.sensor}__{args.model}__{mode}__{stage}"
    output_run_name = run_name if explain_split == "test" else f"{run_name}_{explain_split}"
    outputs_root = Path(getattr(args, "outputs_dir", CFG.OUTPUT_DIR))
    out_dir = outputs_root / args.sensor / args.model / mode / stage
    ckpt_path = out_dir / f"{run_name}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    selection_json = out_dir / f"{output_run_name}_shap_band_selection.json"
    if getattr(args, "skip_existing", False) and selection_json.exists():
        print(f"\nSHAP spectral importance")
        print(f"  [SKIP] Existing SHAP band selection found: {selection_json}")
        return

    print("\nSHAP spectral importance")
    print(f"  Run        : {run_name}")
    print(f"  Explain set: {explain_split}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Samples    : background={args.background_samples}, explain={args.explain_samples}")
    sampling_seed = int(getattr(args, "seed", CFG.SEED))
    print(f"  Seed       : sampling={sampling_seed}, explain_sampling={sampling_seed + 1}")
    spatial_size = None
    if args.spatial_h and args.spatial_w:
        spatial_size = (args.spatial_h, args.spatial_w)
    print(f"  Cell       : {cell_h}x{cell_w}, bands={n_bands}")
    if spatial_size is not None:
        print(f"  SHAP input : downsampled to {spatial_size[0]}x{spatial_size[1]}")

    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = _build_model(args.model, args.sensor, n_bands, cell_h, cell_w)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    ds_train = ShardedBlueberryDataset(train_rows, label_mode="binary", transform=get_val_transform())
    ds_explain = ShardedBlueberryDataset(explain_rows, label_mode="binary", transform=get_val_transform())

    bg_idx = _balanced_indices(train_rows, args.background_samples, sampling_seed)
    ex_idx = _balanced_indices(explain_rows, args.explain_samples, sampling_seed + 1)
    background = _stack_samples(ds_train, bg_idx, spatial_size=spatial_size).to(device)

    explainer = shap.GradientExplainer(model, background)

    class_sums = [np.zeros(n_bands, dtype=np.float64) for _ in range(CFG.NUM_CLASSES)]
    count = 0
    for start in tqdm(range(0, len(ex_idx), args.batch_size), desc="SHAP", unit="batch"):
        batch_idx = ex_idx[start:start + args.batch_size]
        x = _stack_samples(ds_explain, batch_idx, spatial_size=spatial_size).to(device)
        shap_values = explainer.shap_values(x)
        per_class = _normalize_shap_values(shap_values, CFG.NUM_CLASSES)
        for ci, values in enumerate(per_class):
            # values: (N, bands, H, W)
            band_values = np.abs(values).mean(axis=(0, 2, 3))
            class_sums[ci] += band_values * values.shape[0]
        count += x.shape[0]
        del x, shap_values, per_class
        if device.type == "cuda":
            torch.cuda.empty_cache()

    class_importance = {}
    for ci, name in enumerate(CFG.CLASS_NAMES[:CFG.NUM_CLASSES]):
        values = (class_sums[ci] / max(count, 1)).astype(np.float32)
        max_val = float(values.max())
        if max_val > 0:
            values = values / max_val
        class_importance[name] = values

    wavelengths = _wavelengths(args.sensor, n_bands)
    shap_csv = out_dir / f"{output_run_name}_shap_spectral_importance.csv"
    shap_png = out_dir / f"{output_run_name}_shap_spectral_importance.png"
    _save_shap_importance_csv(shap_csv, wavelengths, class_importance)
    _save_shap_plot(shap_png, wavelengths, class_importance, output_run_name)

    importance = {name: values for name, values in class_importance.items()}
    importance["wavelength_nm"] = wavelengths
    sweep = [int(x.strip()) for x in args.sweep.split(",") if x.strip()]
    strategies = list(getattr(CFG, "MS_STRATEGIES", ["averaged"]))
    selections = select_bands(importance, sweep=sweep, strategies=strategies)

    selection_png = out_dir / f"{output_run_name}_shap_band_selection.png"
    save_selection_json(
        selections,
        wavelengths,
        sensor=args.sensor,
        source_csv=str(shap_csv),
        run_name=output_run_name,
        out_path=str(selection_json),
    )
    plot_band_selection(
        importance,
        selections,
        sweep=sweep,
        sensor=args.sensor,
        run_name=f"{output_run_name} SHAP",
        out_path=str(selection_png),
    )
    print("\nDone.")


def main():
    p = argparse.ArgumentParser(description="Run SHAP on a trained full-spectrum model.")
    p.add_argument("--shard_dir", required=True, help="Path to shard directory containing manifest.csv")
    p.add_argument("--sensor", required=True, choices=["nir", "vnir"])
    p.add_argument("--model", required=True, choices=["cnn3d", "cnn3d_transformer"])
    p.add_argument("--mode", default="stage_agnostic", choices=["stage_agnostic", "per_stage"])
    p.add_argument("--stage", default="all", choices=["all", *CFG.RIPENESS_STAGES])
    p.add_argument("--explain_split", default="test", choices=["train", "val", "test"],
                   help="Dataset split used for explained samples. Use val for wavelength selection.")
    p.add_argument("--background_samples", type=int, default=16)
    p.add_argument("--explain_samples", type=int, default=48)
    p.add_argument("--seed", type=int, default=CFG.SEED,
                   help="Seed used only for SHAP background/explain sample selection.")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--sweep", default=",".join(str(k) for k in CFG.MS_BAND_COUNTS))
    p.add_argument("--device", default=None, help="Optional torch device, e.g. cuda:0 or cpu")
    p.add_argument("--outputs_dir", default=CFG.OUTPUT_DIR,
                   help="Root output directory containing trained checkpoints")
    p.add_argument("--spatial_h", type=int, default=None,
                   help="Optional SHAP-only spatial downsample height")
    p.add_argument("--spatial_w", type=int, default=None,
                   help="Optional SHAP-only spatial downsample width")
    p.add_argument("--disable_cudnn", action="store_true",
                   help="Disable cuDNN during SHAP to avoid high-memory convolution workspaces.")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip this run if the SHAP band-selection JSON already exists")
    args = p.parse_args()
    run_shap(args)


if __name__ == "__main__":
    main()
