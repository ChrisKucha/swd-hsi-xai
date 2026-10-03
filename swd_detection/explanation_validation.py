"""
explanation_validation.py - explanation stability and faithfulness checks.

This script adds quantitative explainability validation for the trained
full-spectrum deep-learning models.

Analyses
--------
1. Explanation stability across ripening conditions
   - Uses saved spectral_importance.csv files.
   - Computes Top-K Jaccard overlap and Spearman rank correlation between
     Ripe, Mid-ripe, Unripe, and Combined explanations.

2. Explanation faithfulness by perturbation
   - Loads trained stage-agnostic full-spectrum checkpoints.
   - Evaluates baseline performance.
   - Masks top-K, random-K, and bottom-K wavelengths at inference.
   - A faithful explanation should show a larger performance drop after
     masking top-K important wavelengths than random or bottom-K wavelengths.

3. Cross-method agreement
   - Compares attribution sources, currently IG versus SHAP, for the same
     sensor/model/stage.
   - Reports Top-K Jaccard overlap, overlap counts, and Spearman rank
     correlations.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import config as CFG
from cross_stage_eval import (
    build_full_model,
    load_manifest,
    load_weights,
    make_dataset,
    metrics,
    predict,
)


STAGES = ["Ripe", "Midripe", "Unripe", "all"]
STAGE_DISPLAY = {
    "Ripe": "Ripe",
    "Midripe": "Mid-ripe",
    "Unripe": "Unripe",
    "all": "Combined",
}
MODEL_DISPLAY = {
    "cnn3d": "3DCNN",
    "cnn3d_transformer": "3DCNN-Transformer",
}


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_name(sensor: str, model_name: str, stage: str) -> Tuple[str, str, str]:
    if stage == "all":
        mode = "stage_agnostic"
        rip = "all"
    else:
        mode = "per_stage"
        rip = stage
    return f"{sensor}__{model_name}__{mode}__{rip}", mode, rip


def importance_path(outputs_dir: Path, sensor: str, model_name: str, stage: str, method: str) -> Path:
    name, mode, rip = run_name(sensor, model_name, stage)
    suffix = "_shap_spectral_importance.csv" if method == "shap" else "_spectral_importance.csv"
    return outputs_dir / sensor / model_name / mode / rip / f"{name}{suffix}"


def checkpoint_path(outputs_dir: Path, sensor: str, model_name: str) -> Path:
    name, mode, rip = run_name(sensor, model_name, "all")
    return outputs_dir / sensor / model_name / mode / rip / f"{name}_best.pt"


def load_importance(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "band_idx" not in df.columns:
        return None
    class_cols = [c for c in df.columns if c not in ("band_idx", "wavelength_nm")]
    if not class_cols:
        return None
    df = df.copy()
    df["importance"] = df[class_cols].abs().mean(axis=1)
    df = df.sort_values("band_idx").reset_index(drop=True)
    return df[["band_idx", "wavelength_nm", "importance"]]


def topk(df: pd.DataFrame, k: int, largest: bool = True) -> List[int]:
    return (
        df.sort_values("importance", ascending=not largest)
        .head(k)["band_idx"]
        .astype(int)
        .tolist()
    )


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    return float(len(sa & sb) / len(union)) if union else float("nan")


def explanation_stability(
    outputs_dir: Path,
    out_dir: Path,
    sensors: List[str],
    models: List[str],
    methods: List[str],
    k_values: List[int],
) -> pd.DataFrame:
    records = []
    loaded: Dict[Tuple[str, str, str, str], pd.DataFrame] = {}

    for sensor in sensors:
        for model_name in models:
            for method in methods:
                for stage in STAGES:
                    df = load_importance(importance_path(outputs_dir, sensor, model_name, stage, method))
                    if df is not None:
                        loaded[(sensor, model_name, method, stage)] = df

                available = [
                    s for s in STAGES
                    if (sensor, model_name, method, s) in loaded
                ]
                for i, stage_a in enumerate(available):
                    for stage_b in available[i:]:
                        df_a = loaded[(sensor, model_name, method, stage_a)]
                        df_b = loaded[(sensor, model_name, method, stage_b)]
                        merged = df_a[["band_idx", "importance"]].merge(
                            df_b[["band_idx", "importance"]],
                            on="band_idx",
                            suffixes=("_a", "_b"),
                        )
                        spearman = float(merged["importance_a"].corr(merged["importance_b"], method="spearman"))
                        for k in k_values:
                            records.append({
                                "sensor": sensor,
                                "model": model_name,
                                "method": method,
                                "stage_a": stage_a,
                                "stage_b": stage_b,
                                "k": int(k),
                                "jaccard_topk": jaccard(topk(df_a, k), topk(df_b, k)),
                                "spearman_rank": spearman,
                            })

    df = pd.DataFrame(records)
    ensure_dir(out_dir)
    df.to_csv(out_dir / "explanation_stability_pairwise.csv", index=False)
    save_stability_heatmaps(df, out_dir)
    return df


def _ranked_bands(df: pd.DataFrame, k: int) -> pd.DataFrame:
    return (
        df.sort_values("importance", ascending=False)
        .head(k)
        .loc[:, ["band_idx", "wavelength_nm", "importance"]]
        .reset_index(drop=True)
    )


def cross_method_agreement(
    outputs_dir: Path,
    out_dir: Path,
    sensors: List[str],
    models: List[str],
    method_a: str,
    method_b: str,
    k_values: List[int],
) -> pd.DataFrame:
    records = []
    top_records = []

    for sensor in sensors:
        for model_name in models:
            for stage in STAGES:
                df_a = load_importance(importance_path(outputs_dir, sensor, model_name, stage, method_a))
                df_b = load_importance(importance_path(outputs_dir, sensor, model_name, stage, method_b))
                if df_a is None or df_b is None:
                    missing = []
                    if df_a is None:
                        missing.append(method_a)
                    if df_b is None:
                        missing.append(method_b)
                    print(
                        f"[SKIP] Missing {'/'.join(missing)} importance for "
                        f"{sensor} {model_name} {stage}"
                    )
                    continue

                merged = df_a[["band_idx", "wavelength_nm", "importance"]].merge(
                    df_b[["band_idx", "importance"]],
                    on="band_idx",
                    suffixes=(f"_{method_a}", f"_{method_b}"),
                )
                spearman = float(
                    merged[f"importance_{method_a}"].corr(
                        merged[f"importance_{method_b}"],
                        method="spearman",
                    )
                )
                pearson = float(
                    merged[f"importance_{method_a}"].corr(
                        merged[f"importance_{method_b}"],
                        method="pearson",
                    )
                )

                for k in k_values:
                    top_a = topk(df_a, k)
                    top_b = topk(df_b, k)
                    overlap = sorted(set(top_a) & set(top_b))
                    records.append({
                        "sensor": sensor,
                        "model": model_name,
                        "stage": stage,
                        "method_a": method_a,
                        "method_b": method_b,
                        "k": int(k),
                        "overlap_count": int(len(overlap)),
                        "overlap_fraction": float(len(overlap) / max(k, 1)),
                        "jaccard_topk": jaccard(top_a, top_b),
                        "spearman_rank": spearman,
                        "pearson_importance": pearson,
                        "overlap_band_idx": ";".join(str(x) for x in overlap),
                    })

                    for method, ranked in (
                        (method_a, _ranked_bands(df_a, k)),
                        (method_b, _ranked_bands(df_b, k)),
                    ):
                        for rank, row in ranked.iterrows():
                            top_records.append({
                                "sensor": sensor,
                                "model": model_name,
                                "stage": stage,
                                "method": method,
                                "k": int(k),
                                "rank": int(rank + 1),
                                "band_idx": int(row["band_idx"]),
                                "wavelength_nm": float(row["wavelength_nm"]),
                                "importance": float(row["importance"]),
                                "in_other_method_topk": bool(
                                    int(row["band_idx"]) in (set(top_b) if method == method_a else set(top_a))
                                ),
                            })

    df = pd.DataFrame(records)
    ensure_dir(out_dir)
    df.to_csv(out_dir / "cross_method_agreement.csv", index=False)
    pd.DataFrame(top_records).to_csv(out_dir / "cross_method_top_bands.csv", index=False)
    save_cross_method_plots(df, out_dir)
    return df


def save_cross_method_plots(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    plot_dir = ensure_dir(out_dir / "cross_method_plots")
    for (sensor, model_name, k), sub in df.groupby(["sensor", "model", "k"]):
        stages = [s for s in STAGES if s in set(sub["stage"])]
        vals = []
        labels = []
        for stage in stages:
            row = sub[sub["stage"] == stage]
            if row.empty:
                continue
            vals.append(float(row.iloc[0]["jaccard_topk"]))
            labels.append(STAGE_DISPLAY.get(stage, stage))

        if not vals:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(labels, vals, color="#4c78a8")
        ax.set_ylim(0, 1)
        ax.set_ylabel("IG-SHAP Top-K Jaccard", fontsize=12, fontweight="bold")
        ax.set_title(
            f"{MODEL_DISPLAY.get(model_name, model_name)} {sensor.upper()} IG-SHAP agreement Top-{k}",
            fontsize=13,
            fontweight="bold",
        )
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontsize(11)
            tick.set_fontweight("bold")
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
            )
        fig.tight_layout()
        fig.savefig(plot_dir / f"{sensor}__{model_name}__top{k}_ig_shap_jaccard.png", dpi=180)
        plt.close(fig)

    for (sensor, k), sub in df.groupby(["sensor", "k"]):
        row_labels = []
        values = []
        for model_name in sorted(sub["model"].unique()):
            model_sub = sub[sub["model"] == model_name]
            row_labels.append(MODEL_DISPLAY.get(model_name, model_name))
            values.append([
                float(model_sub[model_sub["stage"] == stage]["jaccard_topk"].iloc[0])
                if not model_sub[model_sub["stage"] == stage].empty else np.nan
                for stage in STAGES
            ])
        mat = np.asarray(values, dtype=float)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Top-K Jaccard", fontsize=11, fontweight="bold")
        ax.set_xticks(np.arange(len(STAGES)), [STAGE_DISPLAY[s] for s in STAGES])
        ax.set_yticks(np.arange(len(row_labels)), row_labels)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(11)
            label.set_fontweight("bold")
        ax.set_title(
            f"{sensor.upper()} IG-SHAP cross-method agreement Top-{k}",
            fontsize=13,
            fontweight="bold",
        )
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                ax.text(j, i, "" if np.isnan(val) else f"{val:.2f}",
                        ha="center", va="center", fontweight="bold",
                        color="white" if (not np.isnan(val) and val < 0.35) else "black")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{sensor}__top{k}_ig_shap_jaccard_heatmap.png", dpi=180)
        plt.close(fig)


def save_stability_heatmaps(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    for (sensor, model_name, method, k), sub in df.groupby(["sensor", "model", "method", "k"]):
        if len(sub) == 0:
            continue
        stages = [s for s in STAGES if s in set(sub["stage_a"]) | set(sub["stage_b"])]
        mat = pd.DataFrame(index=stages, columns=stages, dtype=float)
        for _, row in sub.iterrows():
            mat.loc[row["stage_a"], row["stage_b"]] = row["jaccard_topk"]
            mat.loc[row["stage_b"], row["stage_a"]] = row["jaccard_topk"]
        disp = mat.rename(index=STAGE_DISPLAY, columns=STAGE_DISPLAY)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(disp.to_numpy(float), cmap="RdYlGn", vmin=0, vmax=1)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Jaccard overlap", fontsize=11, fontweight="bold")
        ax.set_xticks(np.arange(len(disp.columns)), disp.columns, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(disp.index)), disp.index)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(11)
            label.set_fontweight("bold")
        ax.set_title(
            f"{MODEL_DISPLAY.get(model_name, model_name)} {sensor.upper()} {method.upper()} Top-{k}\n"
            "ripeness-specific and combined model stability",
            fontsize=13,
            fontweight="bold",
        )
        for i in range(disp.shape[0]):
            for j in range(disp.shape[1]):
                val = disp.iat[i, j]
                ax.text(j, i, "" if pd.isna(val) else f"{val:.2f}",
                        ha="center", va="center", fontweight="bold",
                        color="white" if (not pd.isna(val) and val < 0.35) else "black")
        fig.tight_layout()
        path = ensure_dir(out_dir / "stability_heatmaps") / (
            f"{sensor}__{model_name}__{method}__top{k}_jaccard.png"
        )
        fig.savefig(path, dpi=180)
        plt.close(fig)


class MaskBandsDataset(Dataset):
    def __init__(self, base: Dataset, bands: Optional[Iterable[int]] = None, mode: str = "mask"):
        self.base = base
        self.bands = None if bands is None else np.asarray(list(bands), dtype=np.intp)
        self.mode = mode

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if self.bands is not None and len(self.bands) > 0:
            x = x.clone()
            if self.mode == "mask":
                x[self.bands] = 0.0
            elif self.mode == "keep":
                keep = torch.zeros(x.shape[0], dtype=torch.bool)
                keep[self.bands] = True
                x[~keep] = 0.0
            else:
                raise ValueError(self.mode)
        return x, y


def evaluate_variant(model, dataset, device, batch_size: int, num_workers: int) -> Dict[str, float]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
        prefetch_factor=(2 if num_workers > 0 else None),
    )
    y_true, y_pred, probs = predict(model, loader, device)
    return {"n_test": int(len(y_true)), **metrics(y_true, y_pred, probs)}


def faithfulness_ablation(
    outputs_dir: Path,
    shard_dir: Path,
    out_dir: Path,
    sensors: List[str],
    models: List[str],
    methods: List[str],
    k_values: List[int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    random_repeats: int,
    max_samples: Optional[int],
    seed: int,
) -> pd.DataFrame:
    manifest = load_manifest(shard_dir)
    rng = random.Random(seed)
    records = []

    for sensor in sensors:
        base_ds = make_dataset(manifest, sensor, "all", max_samples=max_samples)
        n_bands = int(getattr(base_ds, "n_bands"))

        for model_name in models:
            ckpt = checkpoint_path(outputs_dir, sensor, model_name)
            if not ckpt.exists():
                print(f"[SKIP] Missing checkpoint: {ckpt}")
                continue
            print(f"\n[Faithfulness] {sensor} {model_name}")
            model = build_full_model(model_name, sensor, manifest, device)
            load_weights(model, ckpt, device)

            baseline = evaluate_variant(model, base_ds, device, batch_size, num_workers)
            records.append({
                "sensor": sensor, "model": model_name, "method": "none",
                "k": 0, "variant": "baseline", "repeat": 0,
                **baseline,
                "accuracy_drop": 0.0,
                "f1_macro_drop": 0.0,
            })
            print(f"  baseline acc={baseline['accuracy']:.4f} f1={baseline['f1_macro']:.4f}")

            for method in methods:
                imp = load_importance(importance_path(outputs_dir, sensor, model_name, "all", method))
                if imp is None:
                    print(f"  [SKIP] missing {method} importance for {sensor} {model_name}")
                    continue
                for k in k_values:
                    variants = {
                        "mask_topk": topk(imp, k, largest=True),
                        "mask_bottomk": topk(imp, k, largest=False),
                        "keep_topk_only": topk(imp, k, largest=True),
                    }
                    for variant, bands in variants.items():
                        mode = "keep" if variant == "keep_topk_only" else "mask"
                        ds = MaskBandsDataset(base_ds, bands, mode=mode)
                        res = evaluate_variant(model, ds, device, batch_size, num_workers)
                        records.append({
                            "sensor": sensor, "model": model_name, "method": method,
                            "k": int(k), "variant": variant, "repeat": 0,
                            **res,
                            "accuracy_drop": baseline["accuracy"] - res["accuracy"],
                            "f1_macro_drop": baseline["f1_macro"] - res["f1_macro"],
                        })
                        print(f"  {method} k={k} {variant}: acc_drop={baseline['accuracy'] - res['accuracy']:.4f}")

                    top = set(variants["mask_topk"])
                    bottom = set(variants["mask_bottomk"])
                    available = [i for i in range(n_bands) if i not in top and i not in bottom]
                    for rep in range(1, random_repeats + 1):
                        bands = rng.sample(available, min(k, len(available)))
                        ds = MaskBandsDataset(base_ds, bands, mode="mask")
                        res = evaluate_variant(model, ds, device, batch_size, num_workers)
                        records.append({
                            "sensor": sensor, "model": model_name, "method": method,
                            "k": int(k), "variant": "mask_randomk", "repeat": rep,
                            **res,
                            "accuracy_drop": baseline["accuracy"] - res["accuracy"],
                            "f1_macro_drop": baseline["f1_macro"] - res["f1_macro"],
                        })

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "faithfulness_ablation.csv", index=False)
    save_faithfulness_plots(df, out_dir)
    return df


def save_faithfulness_plots(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    plot_dir = ensure_dir(out_dir / "faithfulness_plots")
    variants = ["mask_topk", "mask_randomk", "mask_bottomk", "keep_topk_only"]
    for (sensor, model_name, method, k), sub in df[df["variant"] != "baseline"].groupby(["sensor", "model", "method", "k"]):
        vals = []
        labels = []
        for variant in variants:
            v = sub[sub["variant"] == variant]
            if v.empty:
                continue
            vals.append(v["f1_macro_drop"].mean() * 100.0)
            labels.append(variant.replace("_", " "))
        if not vals:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(labels, vals, color=["#b2182b", "#fdae61", "#2166ac", "#4daf4a"][:len(vals)])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Macro-F1 drop (%)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"{MODEL_DISPLAY.get(model_name, model_name)} {sensor.upper()} {method.upper()} faithfulness Top-{k}",
            fontsize=13,
            fontweight="bold",
        )
        ax.tick_params(axis="x", labelrotation=20, labelsize=10)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}",
                    ha="center", va="bottom" if val >= 0 else "top", fontweight="bold")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{sensor}__{model_name}__{method}__top{k}_faithfulness.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Validate explanation stability and faithfulness.")
    parser.add_argument("--shard_dir", type=Path, required=True)
    parser.add_argument("--outputs_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/explanation_validation"))
    parser.add_argument("--sensors", default="nir,vnir")
    parser.add_argument("--models", default="cnn3d,cnn3d_transformer")
    parser.add_argument("--methods", default="ig,shap")
    parser.add_argument("--top_k", default="10,20,30")
    parser.add_argument("--analysis", default="stability,faithfulness",
                        help="Comma list: stability,faithfulness,agreement")
    parser.add_argument("--agreement_methods", default="ig,shap",
                        help="Two methods to compare for cross-method agreement.")
    parser.add_argument("--batch_size", type=int, default=getattr(CFG, "EVAL_BATCH_SIZE", CFG.BATCH_SIZE))
    parser.add_argument("--num_workers", type=int, default=getattr(CFG, "NUM_WORKERS", 4))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--random_repeats", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional debug cap on all-stage test samples.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    sensors = parse_csv_list(args.sensors)
    models = parse_csv_list(args.models)
    methods = parse_csv_list(args.methods)
    k_values = [int(x) for x in parse_csv_list(args.top_k)]
    analyses = set(parse_csv_list(args.analysis))
    device = torch.device(args.device)

    if "stability" in analyses:
        explanation_stability(args.outputs_dir, out_dir, sensors, models, methods, k_values)
    if "agreement" in analyses:
        pair = parse_csv_list(args.agreement_methods)
        if len(pair) != 2:
            raise ValueError("--agreement_methods must contain exactly two methods, e.g. ig,shap")
        cross_method_agreement(args.outputs_dir, out_dir, sensors, models, pair[0], pair[1], k_values)
    if "faithfulness" in analyses:
        faithfulness_ablation(
            args.outputs_dir, args.shard_dir, out_dir, sensors, models, methods,
            k_values, args.batch_size, args.num_workers, device,
            args.random_repeats, args.max_samples, args.seed,
        )

    print(f"\nDone. Explanation validation outputs -> {out_dir}")


if __name__ == "__main__":
    main()
