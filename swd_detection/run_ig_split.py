"""
run_ig_split.py - Integrated Gradients wavelength selection on a chosen split.

This is used for validation-based band selection: train the full-spectrum
model, explain validation samples, select top wavelengths, then evaluate the
selected-wavelength models on the test set.
"""

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = Path(__file__).parent
_SWD_DIR = str(_HERE.resolve())
if _SWD_DIR in sys.path:
    sys.path.remove(_SWD_DIR)
sys.path.insert(0, _SWD_DIR)

import config as CFG
from band_select import load_importance_csv, plot_band_selection, save_selection_json, select_bands
from data.dataset import ShardedBlueberryDataset
from data.transforms import get_val_transform
from explain import run_explain
from models import build_model


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _balanced_rows(rows: List[dict], n: int, seed: int) -> List[dict]:
    """Return rows balanced across label and ripeness as much as possible."""
    rng = np.random.default_rng(seed)
    groups: Dict[tuple, List[dict]] = {}
    for row in rows:
        key = (row.get("label"), row.get("ripeness"))
        groups.setdefault(key, []).append(row)

    keys = sorted(groups)
    for key in keys:
        rng.shuffle(groups[key])

    selected: List[dict] = []
    cursor = 0
    while len(selected) < n and keys:
        key = keys[cursor % len(keys)]
        if groups[key]:
            selected.append(groups[key].pop())
        elif all(not groups[k] for k in keys):
            break
        cursor += 1
    return selected


def _build_model(model_name: str, n_bands: int, cell_h: int, cell_w: int):
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


def run_ig_split(args) -> None:
    if args.model not in ("cnn3d", "cnn3d_transformer"):
        raise ValueError("run_ig_split.py supports cnn3d and cnn3d_transformer.")

    _set_seed(int(args.seed))

    mode = args.mode
    if mode == "stage_agnostic":
        stage = "all"
        ripeness_filter = None
    else:
        if args.stage == "all":
            raise ValueError("--mode per_stage requires a ripeness --stage")
        stage = args.stage
        ripeness_filter = stage

    run_name = f"{args.sensor}__{args.model}__{mode}__{stage}"
    output_run_name = run_name if args.split == "test" else f"{run_name}_{args.split}"

    outputs_root = Path(args.outputs_dir)
    out_dir = outputs_root / args.sensor / args.model / mode / stage
    ckpt_path = out_dir / f"{run_name}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    selection_json = out_dir / f"{output_run_name}_band_selection.json"
    if args.skip_existing and selection_json.exists():
        print(f"\nIG split explanation")
        print(f"  [SKIP] Existing IG band selection found: {selection_json}")
        return

    rows = _load_manifest(args.shard_dir)
    explain_rows = _filter_rows(rows, args.sensor, args.split, ripeness_filter)
    if not explain_rows:
        raise ValueError(
            f"No {args.split} shard rows found for sensor={args.sensor}, "
            f"mode={mode}, stage={stage}"
        )

    selected_rows = _balanced_rows(explain_rows, int(args.n_samples), int(args.seed) + 101)
    if not selected_rows:
        raise ValueError("No rows selected for IG explanation.")

    cell_h = int(selected_rows[0]["cell_h"])
    cell_w = int(selected_rows[0]["cell_w"])
    n_bands = int(selected_rows[0]["n_bands"])

    print("\nIG split explanation")
    print(f"  Run        : {run_name}")
    print(f"  Explain set: {args.split}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Samples    : {len(selected_rows)}")
    print(f"  Cell       : {cell_h}x{cell_w}, bands={n_bands}")

    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = _build_model(args.model, n_bands, cell_h, cell_w)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    ds = ShardedBlueberryDataset(selected_rows, label_mode="binary", transform=get_val_transform())
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    run_explain(
        model=model,
        test_loader=loader,
        output_dir=str(out_dir),
        run_name=output_run_name,
        sensor=args.sensor,
        model_name=args.model,
        n_samples=len(selected_rows),
        n_ig_steps=int(args.n_ig_steps),
        class_names=CFG.CLASS_NAMES,
        device=device,
        split_name=args.split,
    )

    csv_path = out_dir / f"{output_run_name}_spectral_importance.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"IG spectral importance CSV was not created: {csv_path}")

    importance = load_importance_csv(str(csv_path))
    wavelengths = importance["wavelength_nm"]
    sweep = [int(x.strip()) for x in args.sweep.split(",") if x.strip()]
    strategies = list(getattr(CFG, "MS_STRATEGIES", ["averaged"]))
    selections = select_bands(importance, sweep=sweep, strategies=strategies)

    save_selection_json(
        selections,
        wavelengths,
        sensor=args.sensor,
        source_csv=str(csv_path),
        run_name=output_run_name,
        out_path=str(selection_json),
    )
    plot_band_selection(
        importance,
        selections,
        sweep=sweep,
        sensor=args.sensor,
        run_name=output_run_name,
        out_path=str(out_dir / f"{output_run_name}_band_selection.png"),
    )
    print("\nDone.")


def main():
    p = argparse.ArgumentParser(description="Run IG on a trained model using a chosen split.")
    p.add_argument("--shard_dir", required=True, help="Path to shard directory containing manifest.csv")
    p.add_argument("--outputs_dir", default=CFG.OUTPUT_DIR,
                   help="Root output directory containing trained checkpoints")
    p.add_argument("--sensor", required=True, choices=["nir", "vnir"])
    p.add_argument("--model", required=True, choices=["cnn3d", "cnn3d_transformer"])
    p.add_argument("--mode", default="stage_agnostic", choices=["stage_agnostic", "per_stage"])
    p.add_argument("--stage", default="all", choices=["all", *CFG.RIPENESS_STAGES])
    p.add_argument("--split", default="val", choices=["train", "val", "test"],
                   help="Dataset split to explain. Use val for wavelength selection.")
    p.add_argument("--n_samples", type=int, default=32)
    p.add_argument("--n_ig_steps", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--sweep", default=",".join(str(k) for k in CFG.MS_BAND_COUNTS))
    p.add_argument("--device", default=None, help="Optional torch device, e.g. cuda:0 or cpu")
    p.add_argument("--seed", type=int, default=getattr(CFG, "SEED", 42))
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip if the split-specific IG band-selection JSON already exists")
    args = p.parse_args()
    run_ig_split(args)


if __name__ == "__main__":
    main()
