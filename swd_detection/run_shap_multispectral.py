"""
run_shap_multispectral.py — End-to-end SHAP band selection and multispectral training.

For each requested sensor/model stage_agnostic/all checkpoint:
  1. Run SHAP spectral importance if the SHAP band-selection JSON is missing.
  2. Train multispectral models from the SHAP-selected bands.

Outputs
-------
SHAP importance files are written beside the full-spectrum model:
  outputs/{sensor}/{model}/stage_agnostic/all/*_shap_*.*

SHAP-selected multispectral model outputs are written separately:
  outputs/shap_selected/multispectral/{strategy}/k{K}/
"""

import argparse
import csv
import gc
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

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
from band_select import get_band_indices, load_selection_json
from data.discovery import discover_and_split
from run_multispectral import run_one
from run_shap import run_shap


def _load_manifest(shard_dir: str) -> list:
    manifest_path = Path(shard_dir) / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.csv not found in {shard_dir}")
    with open(manifest_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    print(f"  [Shards] Manifest loaded: {len(rows)} rows from {manifest_path}")
    return rows


def _selection_json(sensor: str, model: str) -> Path:
    run_name = f"{sensor}__{model}__stage_agnostic__all"
    return (
        Path(CFG.OUTPUT_DIR)
        / sensor / model / "stage_agnostic" / "all"
        / f"{run_name}_shap_band_selection.json"
    )


def _summary_json(sensor: str, model: str) -> Path:
    run_name = f"{sensor}__{model}__stage_agnostic__all"
    return (
        Path(CFG.OUTPUT_DIR)
        / sensor / model / "stage_agnostic" / "all"
        / f"{run_name}_summary.json"
    )


def _run_shap_if_needed(args, sensor: str, model: str) -> None:
    selection_json = _selection_json(sensor, model)
    if args.skip_existing and selection_json.exists():
        print(f"  [SHAP SKIP] {selection_json.name} already exists.")
        return

    shap_args = argparse.Namespace(
        shard_dir=args.shard_dir,
        sensor=sensor,
        model=model,
        background_samples=args.background_samples,
        explain_samples=args.explain_samples,
        batch_size=args.shap_batch_size,
        sweep=args.sweep,
        device=args.device,
        spatial_h=args.spatial_h,
        spatial_w=args.spatial_w,
    )
    run_shap(shap_args)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(args) -> None:
    manifest_rows = _load_manifest(args.shard_dir)

    sensors = args.sensors.split(",")
    models = args.models.split(",")
    sweep = [int(x.strip()) for x in args.sweep.split(",") if x.strip()]
    strategies = (
        ["infested", "averaged"] if args.strategy == "both"
        else [args.strategy]
    )

    print("\nDiscovering NIR boards ...")
    boards_nir = discover_and_split(
        sensor="nir",
        paths=CFG.PATHS["nir"],
        train_frac=CFG.TRAIN_FRAC,
        val_frac=CFG.VAL_FRAC,
        seed=CFG.SEED,
        verbose=True,
    )
    print("\nDiscovering VNIR boards ...")
    boards_vnir = discover_and_split(
        sensor="vnir",
        paths=CFG.PATHS["vnir"],
        train_frac=CFG.TRAIN_FRAC,
        val_frac=CFG.VAL_FRAC,
        seed=CFG.SEED,
        verbose=True,
    )

    output_root = Path(CFG.OUTPUT_DIR) / "shap_selected"
    output_root.mkdir(parents=True, exist_ok=True)

    jobs = [(sensor, model) for sensor in sensors for model in models]
    for sensor, model in tqdm(jobs, desc="SHAP+MS jobs", unit="model"):
        try:
            print("\n" + "=" * 70)
            print(f"  SHAP-selected multispectral | sensor={sensor} model={model}")
            print("=" * 70)

            _run_shap_if_needed(args, sensor, model)

            selection_path = _selection_json(sensor, model)
            if not selection_path.exists():
                print(f"  [WARN] Missing SHAP selection JSON: {selection_path}")
                continue
            selection = load_selection_json(str(selection_path))

            for strategy in strategies:
                for k in sweep:
                    band_indices = get_band_indices(selection, strategy, k)
                    band_arr = np.array(band_indices, dtype=np.intp)
                    nir_sel = band_arr if sensor == "nir" else None
                    vnir_sel = band_arr if sensor == "vnir" else None

                    run_one(
                        sensor=sensor,
                        model_name=model,
                        train_mode="stage_agnostic",
                        rip_filter=None,
                        boards_nir=boards_nir,
                        boards_vnir=boards_vnir,
                        nir_sel=nir_sel,
                        vnir_sel=vnir_sel,
                        k=k,
                        strategy=f"shap_{strategy}",
                        output_root=output_root,
                        smoke_test=args.smoke_test,
                        skip_existing=args.skip_existing,
                        manifest_rows=manifest_rows,
                    )

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        except Exception:
            print(f"\n  [ERROR] Failed SHAP+MS job sensor={sensor} model={model}:")
            traceback.print_exc()

    print(f"\nDone. SHAP-selected multispectral outputs are in: {output_root}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Run SHAP wavelength selection and train SHAP-selected multispectral models."
    )
    p.add_argument("--shard_dir", required=True)
    p.add_argument("--sensors", default="nir,vnir")
    p.add_argument("--models", default="cnn3d,cnn3d_transformer")
    p.add_argument("--background_samples", type=int, default=16)
    p.add_argument("--explain_samples", type=int, default=48)
    p.add_argument("--shap_batch_size", type=int, default=1)
    p.add_argument("--sweep", default=",".join(str(k) for k in CFG.MS_BAND_COUNTS))
    p.add_argument("--strategy", default="averaged", choices=["infested", "averaged", "both"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--spatial_h", type=int, default=None,
                   help="Optional SHAP-only spatial downsample height")
    p.add_argument("--spatial_w", type=int, default=None,
                   help="Optional SHAP-only spatial downsample width")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--demo_run", dest="smoke_test", action="store_true")
    p.add_argument("--smoke_test", dest="smoke_test", action="store_true",
                   help=argparse.SUPPRESS)
    main(p.parse_args())
