"""
run_combined_attribution_multispectral.py - train models from combined IG/SHAP bands.

This script combines selected bands from multiple attribution-selection JSONs
for a source model, then trains one or more target multispectral models using
the union or intersection of those bands.

Example:
    python run_combined_attribution_multispectral.py \
      --shard_dir "/media/kuchalab/New Volume/swd_detection_shards" \
      --sensor vnir \
      --source_model cnn3d_transformer \
      --train_models cnn3d_transformer \
      --methods ig,shap \
      --strategy averaged \
      --k 30 \
      --combine union \
      --skip_existing
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import config as CFG
from data.discovery import discover_and_split
from run_multispectral import run_one


def _load_manifest(shard_dir: str) -> list:
    manifest_path = Path(shard_dir) / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.csv not found in {shard_dir}")
    with manifest_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    print(f"  [Shards] Manifest loaded: {len(rows)} rows from {manifest_path}")
    return rows


def _selection_json(sensor: str, model: str, method: str) -> Path:
    run_name = f"{sensor}__{model}__stage_agnostic__all"
    suffix = "_shap_band_selection.json" if method == "shap" else "_band_selection.json"
    return (
        Path(CFG.OUTPUT_DIR)
        / sensor / model / "stage_agnostic" / "all"
        / f"{run_name}{suffix}"
    )


def _load_method_bands(sensor: str, source_model: str, method: str, strategy: str, k: int) -> dict:
    path = _selection_json(sensor, source_model, method)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as f:
        data = json.load(f)
    selected = data["strategies"][strategy][str(k)]
    return {
        "path": str(path),
        "method": method,
        "band_indices": [int(x) for x in selected["band_indices"]],
        "wavelengths_nm": [float(x) for x in selected["wavelengths_nm"]],
    }


def _combine_bands(method_bands: list[dict], combine: str) -> list[int]:
    sets = [set(x["band_indices"]) for x in method_bands]
    if combine == "union":
        return sorted(set().union(*sets))
    if combine == "intersection":
        return sorted(set.intersection(*sets))
    raise ValueError(combine)


def _wavelengths(sensor: str, band_indices: list[int]) -> list[float]:
    wl = CFG.WAVELENGTHS[sensor]
    return [float(wl[i]) for i in band_indices]


def _save_combined_selection(
    out_dir: Path,
    sensor: str,
    source_model: str,
    methods: list[str],
    strategy: str,
    source_k: int,
    combine: str,
    band_indices: list[int],
    method_bands: list[dict],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sensor": sensor,
        "source_model": source_model,
        "methods": methods,
        "strategy": strategy,
        "source_k_per_method": source_k,
        "combine": combine,
        "n_combined_bands": len(band_indices),
        "band_indices": band_indices,
        "wavelengths_nm": _wavelengths(sensor, band_indices),
        "source_selection_jsons": {x["method"]: x["path"] for x in method_bands},
        "source_method_bands": {
            x["method"]: {
                "band_indices": x["band_indices"],
                "wavelengths_nm": x["wavelengths_nm"],
            }
            for x in method_bands
        },
    }
    json_path = out_dir / (
        f"{sensor}__{source_model}__{'_'.join(methods)}__{combine}"
        f"__{strategy}_k{source_k}_selection.json"
    )
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)

    csv_path = json_path.with_suffix(".csv")
    rows = []
    source_sets = {x["method"]: set(x["band_indices"]) for x in method_bands}
    for idx in band_indices:
        rows.append({
            "sensor": sensor,
            "source_model": source_model,
            "combine": combine,
            "strategy": strategy,
            "source_k_per_method": source_k,
            "combined_k": len(band_indices),
            "band_idx": idx,
            "wavelength_nm": CFG.WAVELENGTHS[sensor][idx],
            **{f"selected_by_{m}": idx in source_sets[m] for m in methods},
        })
    import pandas as pd
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  Combined selection JSON → {json_path}")
    print(f"  Combined selection CSV  → {csv_path}")
    return json_path


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(CFG.SEED)

    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    train_models = [x.strip() for x in args.train_models.split(",") if x.strip()]
    method_bands = [
        _load_method_bands(args.sensor, args.source_model, method, args.strategy, args.k)
        for method in methods
    ]
    combined = _combine_bands(method_bands, args.combine)
    if not combined:
        raise ValueError("Combined band set is empty.")

    print("\nCombined attribution-selected bands")
    print(f"  Sensor       : {args.sensor}")
    print(f"  Source model : {args.source_model}")
    print(f"  Methods      : {methods}")
    print(f"  Strategy     : {args.strategy}")
    print(f"  Combine      : {args.combine}")
    print(f"  Source K     : {args.k} per method")
    print(f"  Combined K   : {len(combined)}")
    print(f"  Wavelengths  : {[round(x, 2) for x in _wavelengths(args.sensor, combined)]}")

    output_root = Path(args.output_root)
    selection_dir = output_root / "combined_selections"
    _save_combined_selection(
        selection_dir,
        args.sensor,
        args.source_model,
        methods,
        args.strategy,
        args.k,
        args.combine,
        combined,
        method_bands,
    )

    manifest_rows = _load_manifest(args.shard_dir)

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

    band_arr = np.array(combined, dtype=np.intp)
    nir_sel = band_arr if args.sensor == "nir" else None
    vnir_sel = band_arr if args.sensor == "vnir" else None
    strategy_name = (
        f"{args.combine}_{'_'.join(methods)}_from_{args.source_model}"
        f"_{args.strategy}_k{args.k}"
    )

    summaries = []
    for train_model in train_models:
        summary = run_one(
            sensor=args.sensor,
            model_name=train_model,
            train_mode="stage_agnostic",
            rip_filter=None,
            boards_nir=boards_nir,
            boards_vnir=boards_vnir,
            nir_sel=nir_sel,
            vnir_sel=vnir_sel,
            k=len(combined),
            strategy=strategy_name,
            output_root=output_root,
            smoke_test=args.smoke_test,
            skip_existing=args.skip_existing,
            manifest_rows=manifest_rows,
        )
        if summary:
            summary["source_model"] = args.source_model
            summary["source_methods"] = methods
            summary["source_k_per_method"] = args.k
            summary["combine"] = args.combine
            summaries.append(summary)

    if summaries:
        out_csv = output_root / "combined_attribution_multispectral_summary.csv"
        fields = sorted({k for row in summaries for k in row.keys() if not isinstance(row.get(k), (list, dict))})
        with out_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summaries)
        print(f"\nSummary CSV → {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multispectral models from combined IG/SHAP bands.")
    parser.add_argument("--shard_dir", required=True)
    parser.add_argument("--sensor", default="vnir", choices=["nir", "vnir"])
    parser.add_argument("--source_model", default="cnn3d_transformer", choices=["cnn3d", "cnn3d_transformer"])
    parser.add_argument("--train_models", default="cnn3d_transformer")
    parser.add_argument("--methods", default="ig,shap")
    parser.add_argument("--strategy", default="averaged", choices=["infested", "averaged"])
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--combine", default="union", choices=["union", "intersection"])
    parser.add_argument("--output_root", type=Path, default=Path(CFG.OUTPUT_DIR) / "combined_attribution_selected")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--demo_run", dest="smoke_test", action="store_true")
    parser.add_argument("--smoke_test", dest="smoke_test", action="store_true",
                        help=argparse.SUPPRESS)
    main(parser.parse_args())
