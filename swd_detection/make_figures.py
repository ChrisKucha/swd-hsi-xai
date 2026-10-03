"""
Prepare manuscript figures from paired 3-seed reruns.

This script is a 3-seed companion to the paired manuscript plotting scripts.
It reads per-seed ``*_summary.json`` and ``*_results.csv`` files from
``outputs_paired_3seed_20260807`` and reports mean ± sample SD across the
available seeds for each plotted condition.

Notes:
  * Seed 42 full-spectrum outputs are read from ``outputs_paired_cnn3d`` and
    ``outputs_paired_cnn3d_transformer``.
  * Seed 42 selected-wavelength outputs are read from the 3-seed rerun root.
  * Confusion matrices are pooled across the available seeds for the condition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd


CLASS_NAMES = ["Infested", "Healthy"]
SCRIPT_DIR = Path(__file__).resolve().parent

STAGES = [
    ("Ripe", "Ripe"),
    ("Midripe", "Mid-ripe"),
    ("Unripe", "Unripe"),
]

MODEL_ORDER = [
    ("nir", "cnn3d", "NIR\n3DCNN"),
    ("nir", "cnn3d_transformer", "NIR\n3DCNN-Transformer"),
    ("vnir", "cnn3d", "VNIR\n3DCNN"),
    ("vnir", "cnn3d_transformer", "VNIR\n3DCNN-Transformer"),
]

CROSS_STAGE_PANELS = [
    ("nir", "cnn3d", "NIR 3DCNN"),
    ("nir", "cnn3d_transformer", "NIR 3DCNN-Transformer"),
    ("vnir", "cnn3d", "VNIR 3DCNN"),
    ("vnir", "cnn3d_transformer", "VNIR 3DCNN-Transformer"),
]

STAGE_DISPLAY = {
    "Ripe": "Ripe",
    "Midripe": "Mid-ripe",
    "Unripe": "Unripe",
}

MODEL_TITLE = {
    "cnn3d": "3DCNN",
    "cnn3d_transformer": "3DCNN-Transformer",
}
ALLOWED_MODELS = set(MODEL_TITLE)

COMBO_COLORS = {
    ("NIR", "3DCNN"): "#4c78a8",
    ("NIR", "3DCNN-Transformer"): "#72b7b2",
    ("VNIR", "3DCNN"): "#f58518",
    ("VNIR", "3DCNN-Transformer"): "#e45756",
}

METHOD_DISPLAY = {"ig": "IG", "shap": "SHAP"}
METHOD_COLORS = {"ig": "#7B3294", "shap": "#008837"}
METHOD_MARKERS = {"ig": "|", "shap": "|"}
STAGE_ORDER_WITH_ALL = ["Ripe", "Midripe", "Unripe", "all"]
STAGE_DISPLAY_WITH_ALL = {**STAGE_DISPLAY, "all": "Stage-agnostic"}
STAGE_COLORS = {
    "Ripe": "#B2182B",
    "Midripe": "#EF8A62",
    "Unripe": "#2166AC",
    "all": "#1B7837",
}

METRIC_KEYS = [
    ("accuracy", "Accuracy"),
    ("f1_macro", "Macro-F1"),
    ("roc_auc", "ROC-AUC"),
]


def color_cmap(name: str, color: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, ["#ffffff", color])


def sample_sd(values: Iterable[float]) -> float:
    vals = np.asarray(list(values), dtype=float)
    if vals.size <= 1:
        return 0.0
    return float(np.std(vals, ddof=1))


def mean_sd(values: Iterable[float]) -> Tuple[float, float]:
    vals = np.asarray(list(values), dtype=float)
    return float(np.mean(vals)), sample_sd(vals)


def read_summary(path: Path) -> Dict[str, object]:
    with open(path) as fh:
        summary = json.load(fh)
    summary["_summary_path"] = str(path)
    return summary


def result_path_for_summary(summary_path: Path) -> Path:
    stem = summary_path.name.removesuffix("_summary.json")
    return summary_path.with_name(f"{stem}_results.csv")


def confusion_from_results(paths: Iterable[Path]) -> np.ndarray:
    cm = np.zeros((2, 2), dtype=int)
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for true, pred in zip(df["true_label"].astype(int), df["pred_label"].astype(int)):
            cm[true, pred] += 1
    return cm


def row_from_summary(path: Path, summary: Dict[str, object], seed: int, selected: bool, method: str | None) -> Dict[str, object] | None:
    model = str(summary["model"])
    if model not in ALLOWED_MODELS:
        return None
    return {
        "seed": seed,
        "sensor": str(summary["sensor"]),
        "model": model,
        "train_mode": str(summary["train_mode"]),
        "stage": str(summary["ripeness_filter"]),
        "method": method,
        "selected": selected,
        "summary_path": path,
        "results_path": result_path_for_summary(path),
        "accuracy": float(summary["accuracy"]),
        "f1_macro": float(summary["f1_macro"]),
        "precision_macro": float(summary.get("precision_macro", np.nan)),
        "recall_macro": float(summary.get("recall_macro", np.nan)),
        "roc_auc": float(summary["roc_auc"]),
        "best_val_acc": float(summary["best_val_acc"]),
        "best_val_loss": float(summary["best_val_loss"]),
        "epochs_trained": int(summary["epochs_trained"]),
        "train_time_min": float(summary["train_time_min"]),
    }


def discover_summaries(root: Path, seed42_full_roots: Iterable[Path] = ()) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for path in root.glob("seed_*//**/*_summary.json"):
        summary = read_summary(path)
        parts = path.parts
        seed = int(summary.get("seed", next(p for p in parts if p.startswith("seed_")).split("_")[1]))
        is_selected = "selected_bands_val" in parts
        if seed == 42 and not is_selected:
            continue
        method = None
        if is_selected:
            idx = parts.index("selected_bands_val")
            method = parts[idx + 1]
        row = row_from_summary(path, summary, seed=seed, selected=is_selected, method=method)
        if row is not None:
            rows.append(row)

    for legacy_root in seed42_full_roots:
        if not legacy_root.exists():
            continue
        for path in legacy_root.glob("**/*_summary.json"):
            summary = read_summary(path)
            row = row_from_summary(path, summary, seed=42, selected=False, method=None)
            if row is not None:
                rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No *_summary.json files found under {root}")
    df = pd.DataFrame(rows)
    return df.drop_duplicates(
        subset=["seed", "selected", "method", "sensor", "model", "train_mode", "stage"],
        keep="first",
    ).reset_index(drop=True)


def aggregate_group(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(sub["seed"].nunique())
        row["seeds"] = ",".join(str(s) for s in sorted(sub["seed"].unique()))
        for metric, _ in METRIC_KEYS:
            mean, sd = mean_sd(sub[metric].astype(float))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
        for metric in ["precision_macro", "recall_macro", "best_val_acc", "best_val_loss", "epochs_trained", "train_time_min"]:
            mean, sd = mean_sd(sub[metric].astype(float))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
        row["results_paths"] = list(sub["results_path"])
        rows.append(row)
    return pd.DataFrame(rows)


def annotate_bars(ax, bars, means: List[float], sds: List[float], ns: List[int], as_percent: bool = True) -> None:
    for bar, mean, _sd, _n in zip(bars, means, sds, ns):
        if not np.isfinite(mean):
            continue
        scale = 100.0 if as_percent else 1.0
        label_y = (mean + _sd) * scale + 1.35
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{mean * scale:.2f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            rotation=90,
        )


def style_perf_axis(ax, ylim=(60, 105)) -> None:
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def annotate_cm(ax, cm: np.ndarray, vmax: int, cmap="YlGnBu", show_y_labels: bool = True) -> None:
    denom = cm.sum(axis=1, keepdims=True)
    row_pct = np.divide(cm, denom, out=np.zeros_like(cm, dtype=float), where=denom != 0) * 100.0
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=vmax)
    threshold = vmax * 0.50
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({row_pct[i, j]:.2f}%)",
                ha="center",
                va="center",
                fontsize=11.5,
                fontweight="bold",
                color=color,
            )
    ax.set_xticks(np.arange(2), CLASS_NAMES)
    if show_y_labels:
        ax.set_yticks(np.arange(2), CLASS_NAMES)
    else:
        ax.set_yticks(np.arange(2), [])
    ax.set_xlabel("Predicted label", fontsize=10.5, fontweight="bold")
    if show_y_labels:
        ax.set_ylabel("True label", fontsize=10.5, fontweight="bold")
    ax.tick_params(length=0)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(9.5)
        tick.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_visible(False)


def metric_row(agg: pd.DataFrame, sensor: str, model: str, **extra) -> pd.Series:
    sub = agg[(agg["sensor"] == sensor) & (agg["model"] == model)]
    for key, value in extra.items():
        sub = sub[sub[key] == value]
    if sub.empty:
        raise KeyError(f"Missing aggregate row for sensor={sensor}, model={model}, extra={extra}")
    return sub.iloc[0]


def plot_full(full: pd.DataFrame, out_dir: Path) -> None:
    fig = plt.figure(figsize=(14.5, 8.5), constrained_layout=False)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.25], hspace=0.34, wspace=0.34)
    ax_bar = fig.add_subplot(gs[0, :])
    x = np.arange(len(MODEL_ORDER))
    labels = [label for _, _, label in MODEL_ORDER]
    width = 0.24
    metric_colors = ["#4c78a8", "#72b7b2", "#e45756"]

    for metric_idx, (metric, metric_label) in enumerate(METRIC_KEYS):
        means, sds, ns = [], [], []
        for sensor, model, _ in MODEL_ORDER:
            row = metric_row(full, sensor, model)
            means.append(float(row[f"{metric}_mean"]))
            sds.append(float(row[f"{metric}_sd"]))
            ns.append(int(row["n_seeds"]))
        bars = ax_bar.bar(
            x + (metric_idx - 1) * width,
            np.asarray(means) * 100.0,
            width,
            yerr=np.asarray(sds) * 100.0,
            capsize=2.5,
            label=metric_label,
            color=metric_colors[metric_idx],
            edgecolor="black",
            linewidth=0.5,
        )
        annotate_bars(ax_bar, bars, means, sds, ns)
    ax_bar.set_title("A. Full spectrum model performance", fontsize=12.5, fontweight="bold", pad=8)
    ax_bar.set_xticks(x, labels)
    style_perf_axis(ax_bar, ylim=(65, 100))
    ax_bar.set_ylabel("Performance (%)", fontsize=11, fontweight="bold")
    ax_bar.legend(loc="upper left", frameon=True, fontsize=8.5)
    for tick in ax_bar.get_xticklabels():
        tick.set_fontsize(7.8)

    cm_rows = [metric_row(full, sensor, model) for sensor, model, _ in MODEL_ORDER]
    cms = [confusion_from_results(row["results_paths"]) for row in cm_rows]
    vmax = max(int(cm.max()) for cm in cms)
    bottom_axes = [fig.add_subplot(gs[1, idx]) for idx in range(4)]
    for idx, (ax, row, cm, (sensor, model, _)) in enumerate(zip(bottom_axes, cm_rows, cms, MODEL_ORDER)):
        annotate_cm(ax, cm, vmax, cmap="YlGnBu", show_y_labels=True)
        ax.set_title(
            f"{chr(66 + idx)}. {sensor.upper()} {MODEL_TITLE[model]}\n"
            f"Acc={row['accuracy_mean']*100:.2f}±{row['accuracy_sd']*100:.2f}; F1={row['f1_macro_mean']*100:.2f}±{row['f1_macro_sd']*100:.2f}",
            fontsize=10.5,
            fontweight="bold",
            pad=8,
        )

    fig.suptitle("Full-spectrum model performance and pooled confusion matrices (mean ± SD)", fontsize=16, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.08, top=0.88)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "paired_full_spectrum_results_confusion_panel_labeled_alt_cmap_3seed.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "paired_full_spectrum_results_confusion_panel_labeled_alt_cmap_3seed.pdf", bbox_inches="tight")
    full.drop(columns=["results_paths"]).to_csv(out_dir / "paired_full_spectrum_results_confusion_3seed.csv", index=False)
    plt.close(fig)


def plot_stage_specific(stage_agg: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(11.9, 7.35), sharey=True, constrained_layout=False)
    metric_colors = ["#4c78a8", "#72b7b2", "#e45756"]

    for row_idx, (stage_key, stage_label) in enumerate(STAGES):
        for col_idx, (sensor, model, combo_label) in enumerate(MODEL_ORDER):
            ax = axes[row_idx, col_idx]
            data = metric_row(stage_agg, sensor, model, stage=stage_key)
            x = np.array([-0.24, 0.0, 0.24])
            means = [float(data[f"{m}_mean"]) for m, _ in METRIC_KEYS]
            sds = [float(data[f"{m}_sd"]) for m, _ in METRIC_KEYS]
            ns = [int(data["n_seeds"])] * 3
            bars = ax.bar(
                x,
                np.asarray(means) * 100.0,
                yerr=np.asarray(sds) * 100.0,
                capsize=2.5,
                color=metric_colors,
                edgecolor="black",
                linewidth=0.5,
                width=0.20,
            )
            ax.set_ylim(60, 105)
            ax.set_xlim(-0.50, 0.50)
            ax.set_xticks([])
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
            for tick in ax.get_yticklabels():
                tick.set_fontsize(10.5)
                tick.set_fontweight("bold")
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            annotate_bars(ax, bars, means, sds, ns)
            if row_idx == 0:
                ax.set_title(combo_label.replace(" ", "\n", 1), fontsize=12.2, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{stage_label}\n(%)", fontsize=12.5, fontweight="bold")

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.5, label=label)
        for (_, label), color in zip(METRIC_KEYS, metric_colors)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        fontsize=11.5,
        bbox_to_anchor=(0.5, 0.915),
    )
    fig.suptitle("Stage-specific model performance (mean ± SD)", fontsize=18, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.075, top=0.78, hspace=0.43, wspace=0.12)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "paired_stage_specific_model_performance_grid_3x4_compact_3seed.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "paired_stage_specific_model_performance_grid_3x4_compact_3seed.pdf", bbox_inches="tight")
    stage_agg.drop(columns=["results_paths"]).to_csv(out_dir / "paired_stage_specific_model_performance_3seed.csv", index=False)
    plt.close(fig)


def plot_selected(selected_agg: pd.DataFrame, out_dir: Path) -> None:
    fig = plt.figure(figsize=(15.5, 8.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.22], hspace=0.16)
    ax_bar = fig.add_subplot(gs[0, :])
    labels = []
    ordered_rows = []
    for method in ["ig", "shap"]:
        for sensor, model, _ in MODEL_ORDER:
            model_label = "3DCNN-\nTransformer" if model == "cnn3d_transformer" else "3DCNN"
            labels.append(f"{sensor.upper()}\n{model_label}\n{method.upper()}")
            ordered_rows.append(metric_row(selected_agg, sensor, model, method=method))

    x = np.arange(len(ordered_rows))
    width = 0.24
    metric_colors = ["#4c78a8", "#72b7b2", "#e45756"]
    for metric_idx, (metric, metric_label) in enumerate(METRIC_KEYS):
        means = [float(row[f"{metric}_mean"]) for row in ordered_rows]
        sds = [float(row[f"{metric}_sd"]) for row in ordered_rows]
        ns = [int(row["n_seeds"]) for row in ordered_rows]
        bars = ax_bar.bar(
            x + (metric_idx - 1) * width,
            np.asarray(means) * 100.0,
            width,
            yerr=np.asarray(sds) * 100.0,
            capsize=2.5,
            label=metric_label,
            color=metric_colors[metric_idx],
            edgecolor="black",
            linewidth=0.5,
        )
        annotate_bars(ax_bar, bars, means, sds, ns)

    ax_bar.set_title("A. Selected-wavelength model performance", fontsize=15, fontweight="bold", pad=10)
    ax_bar.set_ylabel("Performance (%)", fontsize=12, fontweight="bold")
    ax_bar.set_ylim(60, 100)
    ax_bar.set_xticks(x, labels)
    ax_bar.grid(axis="y", alpha=0.25)
    ax_bar.set_axisbelow(True)
    ax_bar.legend(loc="upper left", frameon=True, fontsize=10)
    for tick in ax_bar.get_xticklabels():
        tick.set_fontsize(8.0)
        tick.set_fontweight("bold")
    for tick in ax_bar.get_yticklabels():
        tick.set_fontsize(11)
        tick.set_fontweight("bold")
    for spine in ["top", "right"]:
        ax_bar.spines[spine].set_visible(False)

    selected_for_cm = selected_agg.sort_values("accuracy_mean", ascending=False).head(3)
    cms = [confusion_from_results(row["results_paths"]) for _, row in selected_for_cm.iterrows()]
    vmax = max(int(cm.max()) for cm in cms)
    for idx, ((_, row), cm) in enumerate(zip(selected_for_cm.iterrows(), cms)):
        ax = fig.add_subplot(gs[1, idx])
        annotate_cm(ax, cm, vmax, cmap="YlGnBu", show_y_labels=True)
        ax.set_title(
            f"{chr(66 + idx)}. {str(row['method']).upper()} Top-30\n"
            f"{str(row['sensor']).upper()} {MODEL_TITLE[str(row['model'])]}\n"
            f"Acc={row['accuracy_mean']*100:.2f}±{row['accuracy_sd']*100:.2f}; F1={row['f1_macro_mean']*100:.2f}±{row['f1_macro_sd']*100:.2f}",
            fontsize=11.5,
            fontweight="bold",
            pad=8,
        )

    fig.suptitle("Selected-wavelength model performance and pooled confusion matrices (mean ± SD)", fontsize=17, fontweight="bold")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "paired_selected_wavelength_results_confusion_3seed.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "paired_selected_wavelength_results_confusion_3seed.pdf", bbox_inches="tight")
    selected_agg.drop(columns=["results_paths"]).to_csv(out_dir / "paired_selected_wavelength_results_confusion_3seed.csv", index=False)
    plt.close(fig)


def discover_cross_stage_results(cross_stage_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(cross_stage_root.glob("seed_*/*/cross_stage_results.csv")):
        seed = int(path.parts[-3].split("_")[1])
        df = pd.read_csv(path)
        df["seed"] = seed
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No cross_stage_results.csv files found under {cross_stage_root}")
    return pd.concat(rows, ignore_index=True)


def aggregate_cross_stage(df: pd.DataFrame) -> pd.DataFrame:
    stages = [stage for stage, _ in STAGES]
    df = df[df["train_stage"].isin(stages) & df["test_stage"].isin(stages)].copy()
    rows = []
    metrics = ["accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc"]
    for keys, sub in df.groupby(["sensor", "model", "train_stage", "test_stage"], dropna=False):
        row = dict(zip(["sensor", "model", "train_stage", "test_stage"], keys))
        row["n_seeds"] = int(sub["seed"].nunique())
        row["seeds"] = ",".join(str(s) for s in sorted(sub["seed"].unique()))
        row["n_test"] = int(sub["n_test"].iloc[0]) if "n_test" in sub else np.nan
        for metric in metrics:
            mean, sd = mean_sd(sub[metric].astype(float))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_mean_percent"] = mean * 100.0
            row[f"{metric}_sd_percent"] = sd * 100.0
        rows.append(row)
    return pd.DataFrame(rows)


def cross_stage_matrix(cross_agg: pd.DataFrame, sensor: str, model: str, value_col: str) -> pd.DataFrame:
    stages = [stage for stage, _ in STAGES]
    matrix = pd.DataFrame(index=stages, columns=stages, dtype=float)
    sub = cross_agg[(cross_agg["sensor"] == sensor) & (cross_agg["model"] == model)]
    for _, row in sub.iterrows():
        matrix.loc[row["train_stage"], row["test_stage"]] = float(row[value_col])
    return matrix.rename(index=STAGE_DISPLAY, columns=STAGE_DISPLAY)


def plot_cross_stage_accuracy(cross_agg: pd.DataFrame, out_dir: Path) -> None:
    matrices = [
        cross_stage_matrix(cross_agg, sensor, model, "accuracy_mean_percent")
        for sensor, model, _ in CROSS_STAGE_PANELS
    ]
    vmin = min(float(np.nanmin(m.to_numpy())) for m in matrices)
    vmax = max(float(np.nanmax(m.to_numpy())) for m in matrices)
    vmin = max(0.0, np.floor(vmin / 5.0) * 5.0)
    vmax = min(100.0, np.ceil(vmax / 5.0) * 5.0)

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.6), constrained_layout=True)
    last_im = None
    for idx, (ax, matrix, (_, _, title)) in enumerate(zip(axes.flat, matrices, CROSS_STAGE_PANELS)):
        arr = matrix.to_numpy(float)
        last_im = ax.imshow(arr, cmap="YlGnBu", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(f"{chr(65 + idx)}. {title}", fontsize=14, fontweight="bold", pad=10)
        ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns)
        ax.set_yticks(np.arange(matrix.shape[0]), matrix.index)
        ax.set_xlabel("Test stage", fontsize=12, fontweight="bold")
        ax.set_ylabel("Training stage", fontsize=12, fontweight="bold")
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontsize(11)
            tick.set_fontweight("bold")
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                ax.text(
                    j,
                    i,
                    "" if pd.isna(val) else f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color="white" if (not pd.isna(val) and (val < 45.0 or val > 78.0)) else "black",
                )
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)

    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02)
    cbar.set_label("Accuracy (%)", fontsize=12, fontweight="bold")
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontsize(10)
        tick.set_fontweight("bold")

    fig.suptitle("Full-wavelength cross-stage accuracy (%) (mean ± SD)", fontsize=17, fontweight="bold")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig3_full_wavelength_cross_stage_accuracy_3seed.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig3_full_wavelength_cross_stage_accuracy_3seed.pdf", bbox_inches="tight")
    cross_agg.to_csv(out_dir / "fig3_full_wavelength_cross_stage_accuracy_3seed_mean_sd.csv", index=False)
    plt.close(fig)


def selection_json_path(root: Path, seed: int, sensor: str, model: str, method: str) -> Path:
    run = f"{sensor}__{model}__stage_agnostic__all"
    suffix = "_val_shap_band_selection.json" if method == "shap" else "_val_band_selection.json"
    if seed == 42:
        return root / sensor / model / "stage_agnostic" / "all" / f"{run}{suffix}"
    return root / f"seed_{seed}" / "full" / model / sensor / model / "stage_agnostic" / "all" / f"{run}{suffix}"


def load_selection_records(root: Path, seed42_full_roots: Iterable[Path], k: int = 30) -> pd.DataFrame:
    legacy_roots = {"cnn3d": None, "cnn3d_transformer": None}
    for path in seed42_full_roots:
        if "transformer" in path.name:
            legacy_roots["cnn3d_transformer"] = path
        else:
            legacy_roots["cnn3d"] = path

    rows = []
    for seed in [42, 43, 44]:
        for sensor, model, _ in MODEL_ORDER:
            source_root = legacy_roots[model] if seed == 42 else root
            if source_root is None:
                continue
            for method in ["ig", "shap"]:
                path = selection_json_path(source_root, seed, sensor, model, method)
                if not path.exists():
                    continue
                data = json.loads(path.read_text())
                selected = data["strategies"]["averaged"][str(k)]
                for rank, (band_idx, wl) in enumerate(zip(selected["band_indices"], selected["wavelengths_nm"]), start=1):
                    rows.append({
                        "seed": seed,
                        "sensor": sensor,
                        "model": model,
                        "method": method,
                        "k": k,
                        "rank": rank,
                        "band_idx": int(band_idx),
                        "wavelength_nm": float(wl),
                        "source_path": str(path),
                    })
    return pd.DataFrame(rows)


def plot_selected_wavelengths_3seed(records: pd.DataFrame, out_dir: Path) -> None:
    if records.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    records.to_csv(out_dir / "fig4_validation_selected_wavelengths_top30_3seed_table.csv", index=False)
    overlap = (
        records.groupby(["sensor", "wavelength_nm"])
        .agg(
            n_selected=("wavelength_nm", "size"),
            n_seeds=("seed", "nunique"),
            sources=("source_path", "nunique"),
        )
        .reset_index()
    )
    overlap.to_csv(out_dir / "fig4_validation_selected_wavelengths_top30_3seed_overlap.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(11.4, 6.2), constrained_layout=True)
    row_order = [("ig", "cnn3d"), ("ig", "cnn3d_transformer"), ("shap", "cnn3d"), ("shap", "cnn3d_transformer")]
    row_positions = {pair: i for i, pair in enumerate(reversed(row_order))}
    row_labels = [
        f"{METHOD_DISPLAY[m]}-{'3DT' if model == 'cnn3d_transformer' else '3DCNN'}"
        for m, model in reversed(row_order)
    ]
    for ax, sensor, panel in zip(axes, ["nir", "vnir"], ["A", "B"]):
        sub = records[records["sensor"] == sensor].merge(overlap, on=["sensor", "wavelength_nm"], how="left")
        for method, model in row_order:
            m = sub[(sub["method"] == method) & (sub["model"] == model)].copy()
            if m.empty:
                continue
            y = np.full(len(m), row_positions[(method, model)], dtype=float)
            jitter = (m["seed"].astype(int).to_numpy() - 43) * 0.07
            repeated = m["n_seeds"].astype(int).to_numpy() > 1
            ax.scatter(
                m["wavelength_nm"].to_numpy(float),
                y + jitter,
                marker=METHOD_MARKERS[method],
                s=np.where(repeated, 260, 165),
                linewidths=np.where(repeated, 2.7, 1.5),
                color=METHOD_COLORS[method],
                alpha=0.86,
                zorder=3,
            )
            if repeated.any():
                ax.scatter(
                    m.loc[repeated, "wavelength_nm"].to_numpy(float),
                    y[repeated] + jitter[repeated],
                    marker="o",
                    s=22,
                    facecolor="none",
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=4,
                )
        ax.set_yticks(np.arange(len(row_labels)), row_labels)
        ax.set_title(f"{panel}. {sensor.upper()} selected wavelengths", fontsize=14, fontweight="bold")
        ax.set_xlabel("Wavelength (nm)", fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.24)
        ax.grid(axis="y", alpha=0.12)
        ax.set_ylim(-0.55, len(row_labels) - 0.45)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontsize(10.5)
            tick.set_fontweight("bold")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="|", linestyle="", markersize=13, markeredgewidth=2.0, color=METHOD_COLORS["ig"], label="IG"),
        Line2D([0], [0], marker="|", linestyle="", markersize=13, markeredgewidth=2.0, color=METHOD_COLORS["shap"], label="SHAP"),
        Line2D([0], [0], marker="o", linestyle="", markersize=6, markerfacecolor="none", markeredgecolor="black", label="Repeated across seeds"),
    ]
    axes[0].legend(handles=handles, frameon=True, facecolor="white", edgecolor="black", fontsize=9.5, loc="upper left")
    fig.suptitle("IG/SHAP top-30 selected wavelengths across three seeds", fontsize=16, fontweight="bold")
    fig.savefig(out_dir / "fig4_validation_selected_wavelengths_top30_3seed.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig4_validation_selected_wavelengths_top30_3seed.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_selection_stability_3seed(records: pd.DataFrame, out_dir: Path) -> None:
    if records.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for (sensor, model, method), sub in records.groupby(["sensor", "model", "method"]):
        seed_sets = {
            int(seed): set(seed_df["band_idx"].astype(int))
            for seed, seed_df in sub.groupby("seed")
        }
        seeds = sorted(seed_sets)
        for i, seed_a in enumerate(seeds):
            for seed_b in seeds[i + 1:]:
                rows.append({
                    "sensor": sensor,
                    "model": model,
                    "method": method,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "jaccard_top30": jaccard_sets(seed_sets[seed_a], seed_sets[seed_b]),
                    "overlap_count": len(seed_sets[seed_a] & seed_sets[seed_b]),
                    "overlap_fraction": len(seed_sets[seed_a] & seed_sets[seed_b]) / 30.0,
                })
    pairwise = pd.DataFrame(rows)
    pairwise.to_csv(out_dir / "fig7_validation_selected_wavelength_stability_seed_pairs.csv", index=False)
    agg_rows = []
    for keys, sub in pairwise.groupby(["sensor", "model", "method"], dropna=False):
        row = dict(zip(["sensor", "model", "method"], keys))
        row["n_seed_pairs"] = int(len(sub))
        row["seeds"] = ",".join(str(s) for s in sorted(set(sub["seed_a"].astype(int)) | set(sub["seed_b"].astype(int))))
        for metric in ["jaccard_top30", "overlap_fraction"]:
            row[f"{metric}_mean"], row[f"{metric}_sd"] = mean_sd(sub[metric].astype(float))
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(out_dir / "fig7_validation_selected_wavelength_stability_3seed_mean_sd.csv", index=False)

    combos = [(sensor, model) for sensor, model, _ in MODEL_ORDER]
    labels = [f"{sensor.upper()}\n{MODEL_TITLE[model]}" for sensor, model in combos]
    x = np.arange(len(combos))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), constrained_layout=True)
    panels = [
        ("A. Top-30 Jaccard", "jaccard_top30", (0, 1), "Jaccard"),
        ("B. Overlap fraction", "overlap_fraction", (0, 1), "Overlap fraction"),
    ]
    for ax, (title, metric, ylim, ylabel) in zip(axes, panels):
        for method, offset in [("ig", -width / 2), ("shap", width / 2)]:
            means, sds = [], []
            for sensor, model in combos:
                row = agg[(agg["sensor"] == sensor) & (agg["model"] == model) & (agg["method"] == method)]
                means.append(np.nan if row.empty else float(row.iloc[0][f"{metric}_mean"]))
                sds.append(0.0 if row.empty else float(row.iloc[0][f"{metric}_sd"]))
            bars = ax.bar(x + offset, means, width=width, yerr=sds, capsize=3, color=METHOD_COLORS[method], edgecolor="black", linewidth=0.6, label=METHOD_DISPLAY[method])
            for bar, val, sd in zip(bars, means, sds):
                if not np.isfinite(val):
                    continue
                ax.text(bar.get_x() + bar.get_width() / 2, val + sd + 0.025, f"{val:.2f}", ha="center", va="bottom", fontsize=8.8, fontweight="bold", rotation=90)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_xticks(x, labels)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontsize(9.5)
            tick.set_fontweight("bold")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    axes[0].legend(frameon=True, facecolor="white", edgecolor="black", fontsize=10)
    fig.suptitle("Selected wavelength stability across seed pairs", fontsize=16, fontweight="bold")
    fig.savefig(out_dir / "fig7_validation_selected_wavelength_stability_3seed_1x2.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig7_validation_selected_wavelength_stability_3seed_1x2.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_selection_cross_method_3seed(records: pd.DataFrame, out_dir: Path) -> None:
    if records.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for (seed, sensor, model), sub in records.groupby(["seed", "sensor", "model"]):
        method_sets = {
            method: set(method_df["band_idx"].astype(int))
            for method, method_df in sub.groupby("method")
        }
        if "ig" not in method_sets or "shap" not in method_sets:
            continue
        overlap = method_sets["ig"] & method_sets["shap"]
        rows.append({
            "seed": int(seed),
            "sensor": sensor,
            "model": model,
            "jaccard_top30": jaccard_sets(method_sets["ig"], method_sets["shap"]),
            "overlap_count": len(overlap),
            "overlap_fraction": len(overlap) / 30.0,
        })
    agreement = pd.DataFrame(rows)
    agreement.to_csv(out_dir / "fig8_validation_selected_ig_shap_agreement_seed_level.csv", index=False)
    agg = seed_aggregate(agreement, ["sensor", "model"], ["jaccard_top30", "overlap_fraction"])
    agg.to_csv(out_dir / "fig8_validation_selected_ig_shap_agreement_3seed_mean_sd.csv", index=False)

    combos = [(sensor, model) for sensor, model, _ in MODEL_ORDER]
    labels = [f"{sensor.upper()}\n{MODEL_TITLE[model]}" for sensor, model in combos]
    x = np.arange(len(combos))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), constrained_layout=True)
    panels = [
        ("A. IG-SHAP Top-30 Jaccard", "jaccard_top30", "Jaccard"),
        ("B. IG-SHAP overlap fraction", "overlap_fraction", "Overlap fraction"),
    ]
    for ax, (title, metric, ylabel) in zip(axes, panels):
        means, sds = [], []
        for sensor, model in combos:
            row = agg[(agg["sensor"] == sensor) & (agg["model"] == model)]
            means.append(np.nan if row.empty else float(row.iloc[0][f"{metric}_mean"]))
            sds.append(0.0 if row.empty else float(row.iloc[0][f"{metric}_sd"]))
        bars = ax.bar(x, means, yerr=sds, capsize=3, color="#4c78a8", edgecolor="black", linewidth=0.6)
        for bar, val, sd in zip(bars, means, sds):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val + sd + 0.025, f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold", rotation=90)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontsize(9.5)
            tick.set_fontweight("bold")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    fig.suptitle("IG-SHAP selected wavelength agreement across three seeds", fontsize=16, fontweight="bold")
    fig.savefig(out_dir / "fig8_validation_selected_ig_shap_agreement_3seed_1x2.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig8_validation_selected_ig_shap_agreement_3seed_1x2.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_cross_method_original_style(csv_path: Path, out_dir: Path) -> None:
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "fig8_cross_method_agreement_plot_data.csv", index=False)

    combos = [(sensor, model) for sensor, model, _ in MODEL_ORDER]
    labels = [f"{sensor.upper()}\n{MODEL_TITLE[model]}" for sensor, model in combos]
    x = np.arange(len(combos))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(STAGE_ORDER_WITH_ALL))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True)
    panels = [
        ("A. Top-30 Jaccard overlap", "jaccard_topk", (0.0, 1.0), "Jaccard"),
        ("B. Spearman rank correlation", "spearman_rank", (-1.0, 1.0), "Spearman"),
    ]

    handles = []
    for ax, (title, metric, ylim, ylabel) in zip(axes, panels):
        for stage, offset in zip(STAGE_ORDER_WITH_ALL, offsets):
            vals = []
            for sensor, model in combos:
                row = df[
                    (df["sensor"] == sensor)
                    & (df["model"] == model)
                    & (df["stage"] == stage)
                    & (df["k"] == 30)
                ]
                vals.append(np.nan if row.empty else float(row.iloc[0][metric]))
            bars = ax.bar(
                x + offset,
                vals,
                width=width,
                color=STAGE_COLORS[stage],
                edgecolor="black",
                linewidth=0.45,
                label=STAGE_DISPLAY_WITH_ALL[stage],
            )
            if ax is axes[0]:
                handles.append(bars[0])
            for bar, val in zip(bars, vals):
                if not np.isfinite(val):
                    continue
                pad = 0.025 * (ylim[1] - ylim[0])
                y = val + pad if val >= 0 else val - pad
                va = "bottom" if val >= 0 else "top"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y,
                    f"{val:.2f}",
                    ha="center",
                    va=va,
                    fontsize=8.5,
                    fontweight="bold",
                    rotation=90,
                )

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_ylim(*ylim)
        ax.set_xticks(x, labels)
        ax.tick_params(axis="x", pad=10)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontsize(10)
            tick.set_fontweight("bold")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[0].legend(
        handles=handles,
        labels=[STAGE_DISPLAY_WITH_ALL[s] for s in STAGE_ORDER_WITH_ALL],
        loc="upper left",
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=0.92,
        fontsize=9,
    )
    fig.suptitle("Cross-method agreement between IG and SHAP", fontsize=17, fontweight="bold")
    fig.savefig(out_dir / "fig8_cross_method_ig_shap_jaccard_spearman_1x2.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig8_cross_method_ig_shap_jaccard_spearman_1x2.pdf", bbox_inches="tight")
    plt.close(fig)


def importance_csv_path_test(root: Path, seed: int, sensor: str, model: str, method: str, stage: str) -> Path:
    run_mode = "stage_agnostic" if stage == "all" else "per_stage"
    stage_dir = "all" if stage == "all" else stage
    run_stage = "all" if stage == "all" else stage
    suffix = "_shap_spectral_importance.csv" if method == "shap" else "_spectral_importance.csv"
    run = f"{sensor}__{model}__{run_mode}__{run_stage}"
    if seed == 42:
        return root / sensor / model / run_mode / stage_dir / f"{run}{suffix}"
    return root / f"seed_{seed}" / "full" / model / sensor / model / run_mode / stage_dir / f"{run}{suffix}"


def collect_cross_method_stagewise_3seed(root: Path, seed42_full_roots: Iterable[Path], k: int = 30) -> pd.DataFrame:
    legacy_roots = {"cnn3d": None, "cnn3d_transformer": None}
    for path in seed42_full_roots:
        if "transformer" in path.name:
            legacy_roots["cnn3d_transformer"] = path
        else:
            legacy_roots["cnn3d"] = path

    rows = []
    for seed in [42, 43, 44]:
        for sensor, model, _ in MODEL_ORDER:
            source_root = legacy_roots[model] if seed == 42 else root
            if source_root is None:
                continue
            for stage in STAGE_ORDER_WITH_ALL:
                ig = load_importance_table(importance_csv_path_test(source_root, seed, sensor, model, "ig", stage))
                shap = load_importance_table(importance_csv_path_test(source_root, seed, sensor, model, "shap", stage))
                if ig is None or shap is None:
                    continue
                top_ig = topk_bands(ig, k)
                top_shap = topk_bands(shap, k)
                merged = ig[["band_idx", "importance"]].merge(
                    shap[["band_idx", "importance"]],
                    on="band_idx",
                    suffixes=("_ig", "_shap"),
                )
                overlap = top_ig & top_shap
                rows.append({
                    "seed": seed,
                    "sensor": sensor,
                    "model": model,
                    "stage": stage,
                    "k": k,
                    "overlap_count": int(len(overlap)),
                    "overlap_fraction": float(len(overlap) / max(k, 1)),
                    "jaccard_topk": jaccard_sets(top_ig, top_shap),
                    "spearman_rank": float(merged["importance_ig"].corr(merged["importance_shap"], method="spearman")),
                    "ig_source": str(importance_csv_path_test(source_root, seed, sensor, model, "ig", stage)),
                    "shap_source": str(importance_csv_path_test(source_root, seed, sensor, model, "shap", stage)),
                })
    return pd.DataFrame(rows)


def plot_cross_method_stagewise_3seed(agreement: pd.DataFrame, out_dir: Path) -> None:
    if agreement.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    agreement.to_csv(out_dir / "fig8_cross_method_agreement_seed_level.csv", index=False)
    agg = seed_aggregate(agreement, ["sensor", "model", "stage"], ["jaccard_topk", "spearman_rank", "overlap_fraction"])
    agg.to_csv(out_dir / "fig8_cross_method_agreement_3seed_mean_sd.csv", index=False)

    combos = [(sensor, model) for sensor, model, _ in MODEL_ORDER]
    labels = [f"{sensor.upper()}\n{MODEL_TITLE[model]}" for sensor, model in combos]
    x = np.arange(len(combos))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(STAGE_ORDER_WITH_ALL))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True)
    panels = [
        ("A. Top-30 Jaccard overlap", "jaccard_topk", (0.0, 1.0), "Jaccard"),
        ("B. Spearman rank correlation", "spearman_rank", (-1.0, 1.0), "Spearman"),
    ]
    handles = []
    for ax, (title, metric, ylim, ylabel) in zip(axes, panels):
        for stage, offset in zip(STAGE_ORDER_WITH_ALL, offsets):
            means, sds = [], []
            for sensor, model in combos:
                row = agg[(agg["sensor"] == sensor) & (agg["model"] == model) & (agg["stage"] == stage)]
                means.append(np.nan if row.empty else float(row.iloc[0][f"{metric}_mean"]))
                sds.append(0.0 if row.empty else float(row.iloc[0][f"{metric}_sd"]))
            bars = ax.bar(
                x + offset,
                means,
                width=width,
                yerr=sds,
                capsize=2.0,
                color=STAGE_COLORS[stage],
                edgecolor="black",
                linewidth=0.45,
                label=STAGE_DISPLAY_WITH_ALL[stage],
            )
            if ax is axes[0]:
                handles.append(bars[0])
            for bar, val, sd in zip(bars, means, sds):
                if not np.isfinite(val):
                    continue
                pad = 0.025 * (ylim[1] - ylim[0])
                y = val + sd + pad if val >= 0 else val - sd - pad
                va = "bottom" if val >= 0 else "top"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y,
                    f"{val:.2f}",
                    ha="center",
                    va=va,
                    fontsize=8.2,
                    fontweight="bold",
                    rotation=90,
                )

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_ylim(*ylim)
        ax.set_xticks(x, labels)
        ax.tick_params(axis="x", pad=10)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontsize(10)
            tick.set_fontweight("bold")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[0].legend(
        handles=handles,
        labels=[STAGE_DISPLAY_WITH_ALL[s] for s in STAGE_ORDER_WITH_ALL],
        loc="upper left",
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=0.92,
        fontsize=9,
    )
    fig.suptitle("Cross-method agreement between IG and SHAP (mean ± SD)", fontsize=17, fontweight="bold")
    fig.savefig(out_dir / "fig8_cross_method_ig_shap_jaccard_spearman_3seed_1x2.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig8_cross_method_ig_shap_jaccard_spearman_3seed_1x2.pdf", bbox_inches="tight")
    plt.close(fig)


def importance_csv_path(root: Path, seed: int, sensor: str, model: str, method: str, stage: str) -> Path:
    run_mode = "stage_agnostic" if stage == "all" else "per_stage"
    stage_dir = "all" if stage == "all" else stage
    run_stage = "all" if stage == "all" else stage
    suffix = "_val_shap_spectral_importance.csv" if stage == "all" and method == "shap" else None
    if suffix is None:
        suffix = "_val_spectral_importance.csv" if stage == "all" and method == "ig" else (
            "_shap_spectral_importance.csv" if method == "shap" else "_spectral_importance.csv"
        )
    run = f"{sensor}__{model}__{run_mode}__{run_stage}"
    if seed == 42:
        return root / sensor / model / run_mode / stage_dir / f"{run}{suffix}"
    return root / f"seed_{seed}" / "full" / model / sensor / model / run_mode / stage_dir / f"{run}{suffix}"


def load_importance_table(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    class_cols = [c for c in df.columns if c not in ("band_idx", "wavelength_nm")]
    if "band_idx" not in df.columns or not class_cols:
        return None
    out = df[["band_idx", "wavelength_nm"]].copy()
    out["importance"] = df[class_cols].abs().mean(axis=1)
    return out.sort_values("band_idx").reset_index(drop=True)


def topk_bands(df: pd.DataFrame, k: int) -> set[int]:
    return set(df.sort_values("importance", ascending=False).head(k)["band_idx"].astype(int))


def jaccard_sets(a: set[int], b: set[int]) -> float:
    return float(len(a & b) / len(a | b)) if (a | b) else np.nan


def collect_stability_and_agreement(root: Path, seed42_full_roots: Iterable[Path], k: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    legacy_roots = {"cnn3d": None, "cnn3d_transformer": None}
    for path in seed42_full_roots:
        if "transformer" in path.name:
            legacy_roots["cnn3d_transformer"] = path
        else:
            legacy_roots["cnn3d"] = path

    loaded = {}
    for seed in [42, 43, 44]:
        for sensor, model, _ in MODEL_ORDER:
            source_root = legacy_roots[model] if seed == 42 else root
            if source_root is None:
                continue
            for method in ["ig", "shap"]:
                for stage in STAGE_ORDER_WITH_ALL:
                    imp = load_importance_table(importance_csv_path(source_root, seed, sensor, model, method, stage))
                    if imp is not None:
                        loaded[(seed, sensor, model, method, stage)] = imp

    stability_rows = []
    for seed in [42, 43, 44]:
        for sensor, model, _ in MODEL_ORDER:
            for method in ["ig", "shap"]:
                stages = [s for s in STAGE_ORDER_WITH_ALL if (seed, sensor, model, method, s) in loaded]
                for i, stage_a in enumerate(stages):
                    for stage_b in stages[i + 1:]:
                        df_a = loaded[(seed, sensor, model, method, stage_a)]
                        df_b = loaded[(seed, sensor, model, method, stage_b)]
                        merged = df_a[["band_idx", "importance"]].merge(df_b[["band_idx", "importance"]], on="band_idx", suffixes=("_a", "_b"))
                        stability_rows.append({
                            "seed": seed,
                            "sensor": sensor,
                            "model": model,
                            "method": method,
                            "stage_a": stage_a,
                            "stage_b": stage_b,
                            "k": k,
                            "jaccard_topk": jaccard_sets(topk_bands(df_a, k), topk_bands(df_b, k)),
                            "spearman_rank": float(merged["importance_a"].corr(merged["importance_b"], method="spearman")),
                        })

    agreement_rows = []
    for seed in [42, 43, 44]:
        for sensor, model, _ in MODEL_ORDER:
            for stage in STAGE_ORDER_WITH_ALL:
                key_a = (seed, sensor, model, "ig", stage)
                key_b = (seed, sensor, model, "shap", stage)
                if key_a not in loaded or key_b not in loaded:
                    continue
                df_a = loaded[key_a]
                df_b = loaded[key_b]
                top_a = topk_bands(df_a, k)
                top_b = topk_bands(df_b, k)
                merged = df_a[["band_idx", "importance"]].merge(df_b[["band_idx", "importance"]], on="band_idx", suffixes=("_ig", "_shap"))
                agreement_rows.append({
                    "seed": seed,
                    "sensor": sensor,
                    "model": model,
                    "stage": stage,
                    "k": k,
                    "overlap_count": len(top_a & top_b),
                    "overlap_fraction": len(top_a & top_b) / float(k),
                    "jaccard_topk": jaccard_sets(top_a, top_b),
                    "spearman_rank": float(merged["importance_ig"].corr(merged["importance_shap"], method="spearman")),
                })

    return pd.DataFrame(stability_rows), pd.DataFrame(agreement_rows)


def seed_aggregate(df: pd.DataFrame, group_cols: List[str], metrics: List[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(sub["seed"].nunique())
        row["seeds"] = ",".join(str(s) for s in sorted(sub["seed"].unique()))
        for metric in metrics:
            row[f"{metric}_mean"], row[f"{metric}_sd"] = mean_sd(sub[metric].astype(float))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_stability_3seed(stability: pd.DataFrame, out_dir: Path) -> None:
    if stability.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    stability.to_csv(out_dir / "fig7_attribution_stability_seed_level.csv", index=False)
    cross = stability[stability["stage_a"] != stability["stage_b"]].copy()
    agg = seed_aggregate(cross, ["sensor", "model", "method"], ["jaccard_topk", "spearman_rank"])
    agg.to_csv(out_dir / "fig7_attribution_stability_3seed_mean_sd.csv", index=False)

    combos = [(sensor, model) for sensor, model, _ in MODEL_ORDER]
    labels = [f"{sensor.upper()}\n{MODEL_TITLE[model]}" for sensor, model in combos]
    x = np.arange(len(combos))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), constrained_layout=True)
    panels = [("A. Top-30 Jaccard", "jaccard_topk", (0, 1), "Jaccard"), ("B. Spearman rank", "spearman_rank", (-1, 1), "Spearman")]
    for ax, (title, metric, ylim, ylabel) in zip(axes, panels):
        for method, offset in [("ig", -width / 2), ("shap", width / 2)]:
            means, sds = [], []
            for sensor, model in combos:
                row = agg[(agg["sensor"] == sensor) & (agg["model"] == model) & (agg["method"] == method)]
                means.append(np.nan if row.empty else float(row.iloc[0][f"{metric}_mean"]))
                sds.append(0.0 if row.empty else float(row.iloc[0][f"{metric}_sd"]))
            bars = ax.bar(x + offset, means, width=width, yerr=sds, capsize=3, color=METHOD_COLORS[method], edgecolor="black", linewidth=0.6, label=METHOD_DISPLAY[method])
            for bar, val, sd in zip(bars, means, sds):
                if not np.isfinite(val):
                    continue
                pad = 0.025 * (ylim[1] - ylim[0])
                y = val + sd + pad if val >= 0 else val - sd - pad
                ax.text(bar.get_x() + bar.get_width() / 2, y, f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8.8, fontweight="bold", rotation=90)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_xticks(x, labels)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontsize(9.5)
            tick.set_fontweight("bold")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    axes[0].legend(frameon=True, facecolor="white", edgecolor="black", fontsize=10)
    fig.suptitle("Attribution stability across ripeness-stage explanations (mean ± SD)", fontsize=16, fontweight="bold")
    fig.savefig(out_dir / "fig7_attribution_stability_3seed_1x2.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig7_attribution_stability_3seed_1x2.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_cross_method_3seed(agreement: pd.DataFrame, out_dir: Path) -> None:
    if agreement.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    agreement.to_csv(out_dir / "fig8_cross_method_seed_level.csv", index=False)
    agg = seed_aggregate(agreement, ["sensor", "model", "stage"], ["jaccard_topk", "spearman_rank"])
    agg.to_csv(out_dir / "fig8_cross_method_3seed_mean_sd.csv", index=False)

    combos = [(sensor, model) for sensor, model, _ in MODEL_ORDER]
    labels = [f"{sensor.upper()}\n{MODEL_TITLE[model]}" for sensor, model in combos]
    x = np.arange(len(combos))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(STAGE_ORDER_WITH_ALL))
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), constrained_layout=True)
    panels = [("A. Top-30 Jaccard overlap", "jaccard_topk", (0, 1), "Jaccard"), ("B. Spearman rank correlation", "spearman_rank", (-1, 1), "Spearman")]
    for ax, (title, metric, ylim, ylabel) in zip(axes, panels):
        for stage, offset in zip(STAGE_ORDER_WITH_ALL, offsets):
            means, sds = [], []
            for sensor, model in combos:
                row = agg[(agg["sensor"] == sensor) & (agg["model"] == model) & (agg["stage"] == stage)]
                means.append(np.nan if row.empty else float(row.iloc[0][f"{metric}_mean"]))
                sds.append(0.0 if row.empty else float(row.iloc[0][f"{metric}_sd"]))
            ax.bar(x + offset, means, width=width, yerr=sds, capsize=2, color=STAGE_COLORS[stage], edgecolor="black", linewidth=0.45, label=STAGE_DISPLAY_WITH_ALL[stage])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_xticks(x, labels)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontsize(9.5)
            tick.set_fontweight("bold")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    axes[0].legend(frameon=True, facecolor="white", edgecolor="black", fontsize=9, ncol=2)
    fig.suptitle("IG-SHAP cross-method agreement across three seeds", fontsize=16, fontweight="bold")
    fig.savefig(out_dir / "fig8_cross_method_ig_shap_3seed_1x2.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig8_cross_method_ig_shap_3seed_1x2.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_mean_spectra_with_sd(aggregate_csv: Path, out_dir: Path) -> None:
    if not aggregate_csv.exists():
        return
    df = pd.read_csv(aggregate_csv)
    stage_df = df[(df["group"] == "stage") & (df["stage"].isin([s for s, _ in STAGES]))].copy()
    if stage_df.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_df.to_csv(out_dir / "fig9_mean_spectra_by_stage_with_sd_table.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.8), constrained_layout=True, sharey=False)
    colors = {"Infested": "#B2182B", "Healthy": "#2166AC"}
    for col, (stage_key, stage_label) in enumerate(STAGES):
        for row_idx, sensor in enumerate(["vnir", "nir"]):
            ax = axes[row_idx, col]
            sub = stage_df[(stage_df["sensor"] == sensor) & (stage_df["stage"] == stage_key)]
            for class_name in ["Infested", "Healthy"]:
                c = sub[sub["class_name"] == class_name].sort_values("wavelength_nm")
                if c.empty:
                    continue
                x = c["wavelength_nm"].to_numpy(float)
                y = c["mean_reflectance"].to_numpy(float)
                sd = c["std"].to_numpy(float)
                ax.plot(x, y, color=colors[class_name], linewidth=1.8, label=class_name)
                ax.fill_between(x, y - sd, y + sd, color=colors[class_name], alpha=0.16, linewidth=0)
            ax.set_title(f"{chr(65 + row_idx * 3 + col)}. {sensor.upper()} {stage_label}", fontsize=12.5, fontweight="bold")
            ax.set_xlabel("Wavelength (nm)", fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel("Mean reflectance", fontsize=11, fontweight="bold")
            ax.grid(alpha=0.22)
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontsize(9)
                tick.set_fontweight("bold")
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
    axes[0, 0].legend(frameon=True, facecolor="white", edgecolor="black", fontsize=9)
    fig.suptitle("Mean spectra by class and maturity stage with SD bands", fontsize=16, fontweight="bold")
    fig.savefig(out_dir / "fig9_mean_spectra_by_stage_2x3_3seed.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig9_mean_spectra_by_stage_2x3_3seed.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_faithfulness_if_available(out_dir: Path) -> bool:
    faith_dir = out_dir / "faithfulness"
    csvs = sorted(faith_dir.glob("seed_*/faithfulness_ablation.csv"))
    if not csvs:
        return False
    frames = []
    for csv_path in csvs:
        seed = int(csv_path.parent.name.split("_")[1])
        df = pd.read_csv(csv_path)
        df["seed"] = seed
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    plot_df = (
        df[(df["k"] == 30) & (df["variant"].isin(["mask_topk", "mask_randomk", "mask_bottomk"])) & (df["method"].isin(["ig", "shap"]))]
        .groupby(["seed", "sensor", "model", "method", "variant"], as_index=False)["f1_macro_drop"]
        .mean()
    )
    agg = seed_aggregate(plot_df, ["sensor", "model", "method", "variant"], ["f1_macro_drop"])
    faith_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(faith_dir / "fig6_faithfulness_seed_level_all.csv", index=False)
    agg.to_csv(faith_dir / "fig6_faithfulness_3seed_mean_sd.csv", index=False)

    variants = ["mask_topk", "mask_randomk", "mask_bottomk"]
    variant_labels = {"mask_topk": "Top-30", "mask_randomk": "Random-30", "mask_bottomk": "Bottom-30"}
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.8), constrained_layout=False)
    for ax, label, (sensor, model, _) in zip(axes.flat, ["A", "B", "C", "D"], MODEL_ORDER):
        x = np.arange(len(variants))
        width = 0.34
        for method, offset in [("ig", -width / 2), ("shap", width / 2)]:
            means, sds = [], []
            for variant in variants:
                row = agg[(agg["sensor"] == sensor) & (agg["model"] == model) & (agg["method"] == method) & (agg["variant"] == variant)]
                means.append(np.nan if row.empty else float(row.iloc[0]["f1_macro_drop_mean"]) * 100.0)
                sds.append(0.0 if row.empty else float(row.iloc[0]["f1_macro_drop_sd"]) * 100.0)
            bars = ax.bar(x + offset, means, width=width, yerr=sds, capsize=3, color=METHOD_COLORS[method], edgecolor="black", linewidth=0.7, label=METHOD_DISPLAY[method])
            for bar, val, sd in zip(bars, means, sds):
                if np.isfinite(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, val + sd + 0.8, f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold", rotation=90)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, [variant_labels[v] for v in variants])
        ax.set_ylabel("Macro-F1 drop (%)", fontsize=12, fontweight="bold")
        ax.set_title(f"{label}. {sensor.upper()} {MODEL_TITLE[model]}", fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontsize(10)
            tick.set_fontweight("bold")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    axes[0, 0].legend(frameon=True, facecolor="white", edgecolor="black", fontsize=10, ncol=2)
    fig.suptitle("Faithfulness ablation of selected wavelengths (mean ± SD)", fontsize=17, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.075, top=0.90, hspace=0.28, wspace=0.18)
    fig.savefig(faith_dir / "fig6_faithfulness_3seed_2x2.png", dpi=300, bbox_inches="tight")
    fig.savefig(faith_dir / "fig6_faithfulness_3seed_2x2.pdf", bbox_inches="tight")
    plt.close(fig)
    return True


def write_manifest(df: pd.DataFrame, out_dir: Path, generated_extra: Dict[str, bool] | None = None) -> None:
    generated_extra = generated_extra or {}
    lines = [
        "# Paired 3-seed Manuscript Figure Manifest",
        "",
        f"Source root: `{df.attrs.get('source_root', '')}`",
        f"Seed 42 full-spectrum source: `{df.attrs.get('seed42_full_roots', '')}`",
        "Seed 42 selected-wavelength source: 3-seed rerun root `seed_42/selected_bands_val`",
        "",
        "Available summaries by seed:",
        "",
    ]
    counts = df.groupby(["seed", "selected", "train_mode"]).size().reset_index(name="n")
    for _, row in counts.iterrows():
        lines.append(f"- seed {int(row['seed'])}, selected={bool(row['selected'])}, train_mode={row['train_mode']}: {int(row['n'])}")
    lines.extend([
        "",
        "Generated files:",
        "",
        "- `paired_full_spectrum_results_confusion_panel_labeled_alt_cmap_3seed.*`: full-spectrum mean ± SD bars and pooled confusion matrices.",
        "- `paired_stage_specific_model_performance_grid_3x4_compact_3seed.*`: stage-specific mean ± SD bars.",
        "- `paired_selected_wavelength_results_confusion_3seed.*`: selected-wavelength mean ± SD bars and pooled confusion matrices. These use n=3 where seeds 42, 43, and 44 are all present.",
        "- `fig3_full_wavelength_cross_stage_accuracy_3seed.*`: full-wavelength cross-stage accuracy heatmaps with mean accuracy across seeds.",
        "- `fig4_validation_selected_wavelengths_top30_3seed.*`: validation-selected IG/SHAP wavelength plot across seeds.",
        "- `fig7_validation_selected_wavelength_stability_3seed_1x2.*`: validation-selected wavelength stability mean ± SD across seed pairs.",
        "- `fig8_cross_method_ig_shap_jaccard_spearman_3seed_1x2.*`: stage-wise IG-SHAP cross-method agreement in the same layout as manuscript Fig. 8, with mean ± SD across seeds.",
        "- `fig9_mean_spectra_by_stage_2x3_3seed.*`: mean spectra by class and maturity stage with SD bands.",
        "",
    ])
    if generated_extra.get("faithfulness"):
        lines.append("- `fig6_faithfulness_3seed_2x2.*`: faithfulness ablation mean ± SD across seed-level ablation CSVs.")
    else:
        lines.extend([
            "",
            "Faithfulness note:",
            "",
            "- Per-seed faithfulness ablation CSVs were not found under `faithfulness/seed_*/faithfulness_ablation.csv`, so the script did not create a seed-level faithfulness figure.",
            "- The older paired faithfulness figure remains available under `outputs/manuscript_figures/faithfulness_paired`, but its error bars are repeat-evaluation variability, not seed-to-seed SD.",
        ])
    lines.append("")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "paired_3seed_figure_manifest.md").write_text("\n".join(lines))


def percent_table(df: pd.DataFrame, id_cols: List[str]) -> pd.DataFrame:
    out = df[id_cols + ["n_seeds", "seeds"]].copy()
    metric_labels = [
        ("accuracy", "Accuracy (%)"),
        ("f1_macro", "Macro-F1 (%)"),
        ("roc_auc", "ROC-AUC (%)"),
        ("precision_macro", "Precision (%)"),
        ("recall_macro", "Recall (%)"),
    ]
    for metric, label in metric_labels:
        mean_col = f"{metric}_mean"
        sd_col = f"{metric}_sd"
        out[label] = [
            f"{mean * 100.0:.2f} ± {sd * 100.0:.2f}"
            for mean, sd in zip(df[mean_col].astype(float), df[sd_col].astype(float))
        ]
    return out.rename(columns={"n_seeds": "n seeds"})


def individual_run_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "seed",
        "selected",
        "method",
        "sensor",
        "model",
        "train_mode",
        "stage",
        "accuracy",
        "f1_macro",
        "roc_auc",
        "precision_macro",
        "recall_macro",
        "best_val_acc",
        "best_val_loss",
        "epochs_trained",
        "train_time_min",
    ]
    out = df[cols].copy()
    for col in ["accuracy", "f1_macro", "roc_auc", "precision_macro", "recall_macro", "best_val_acc"]:
        out[col] = out[col].astype(float) * 100.0
    return out.rename(columns={
        "accuracy": "Accuracy (%)",
        "f1_macro": "Macro-F1 (%)",
        "roc_auc": "ROC-AUC (%)",
        "precision_macro": "Precision (%)",
        "recall_macro": "Recall (%)",
        "best_val_acc": "Best val acc (%)",
        "best_val_loss": "Best val loss",
        "epochs_trained": "Epochs",
        "train_time_min": "Train time (min)",
    }).sort_values(["selected", "method", "sensor", "model", "train_mode", "stage", "seed"], na_position="first")


def write_combined_mean_sd_csv(
    out_dir: Path,
    full_agg: pd.DataFrame,
    stage_agg: pd.DataFrame,
    selected_agg: pd.DataFrame,
) -> Path:
    frames = []
    specs = [
        ("full_spectrum", full_agg, {"selected": False, "method": "", "train_mode": "stage_agnostic", "stage": "all"}),
        ("stage_specific", stage_agg, {"selected": False, "method": "", "train_mode": "per_stage"}),
        ("selected_wavelength", selected_agg, {"selected": True, "train_mode": "stage_agnostic", "stage": "all"}),
    ]
    for analysis_scope, agg, defaults in specs:
        if agg.empty:
            continue
        frame = agg.drop(columns=["results_paths"], errors="ignore").copy()
        frame.insert(0, "analysis_scope", analysis_scope)
        for col, value in defaults.items():
            if col not in frame.columns:
                frame[col] = value
        frames.append(frame)

    if not frames:
        raise ValueError("No aggregate result tables are available to export.")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["model_label"] = combined["model"].map(MODEL_TITLE).fillna(combined["model"])
    combined["sensor"] = combined["sensor"].astype(str).str.upper()

    rename_std = {col: col.removesuffix("_sd") + "_std" for col in combined.columns if col.endswith("_sd")}
    combined = combined.rename(columns=rename_std)

    for metric in ["accuracy", "f1_macro", "roc_auc", "precision_macro", "recall_macro", "best_val_acc"]:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col in combined.columns:
            combined[f"{metric}_mean_percent"] = combined[mean_col].astype(float) * 100.0
        if std_col in combined.columns:
            combined[f"{metric}_std_percent"] = combined[std_col].astype(float) * 100.0

    preferred = [
        "analysis_scope",
        "selected",
        "method",
        "train_mode",
        "stage",
        "sensor",
        "model",
        "model_label",
        "n_seeds",
        "seeds",
    ]
    metric_cols = [
        "accuracy_mean_percent", "accuracy_std_percent", "accuracy_mean", "accuracy_std",
        "f1_macro_mean_percent", "f1_macro_std_percent", "f1_macro_mean", "f1_macro_std",
        "roc_auc_mean_percent", "roc_auc_std_percent", "roc_auc_mean", "roc_auc_std",
        "precision_macro_mean_percent", "precision_macro_std_percent", "precision_macro_mean", "precision_macro_std",
        "recall_macro_mean_percent", "recall_macro_std_percent", "recall_macro_mean", "recall_macro_std",
        "best_val_acc_mean_percent", "best_val_acc_std_percent", "best_val_acc_mean", "best_val_acc_std",
        "best_val_loss_mean", "best_val_loss_std",
        "epochs_trained_mean", "epochs_trained_std",
        "train_time_min_mean", "train_time_min_std",
    ]
    ordered = [col for col in preferred + metric_cols if col in combined.columns]
    remaining = [col for col in combined.columns if col not in ordered]
    combined = combined[ordered + remaining].sort_values(
        ["analysis_scope", "method", "stage", "sensor", "model"],
        na_position="first",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "paired_3seed_full_results_mean_std.csv"
    combined.to_csv(path, index=False)
    return path


def add_dataframe_table(doc, df: pd.DataFrame, title: str) -> None:
    from docx.shared import Pt

    doc.add_heading(title, level=2)
    table = doc.add_table(rows=1, cols=len(df.columns))
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    for idx, col in enumerate(df.columns):
        hdr[idx].text = str(col)
    for _, row in df.iterrows():
        cells = table.add_row().cells
        for idx, col in enumerate(df.columns):
            value = row[col]
            if pd.isna(value):
                text = ""
            elif isinstance(value, (float, np.floating)):
                text = f"{float(value):.2f}"
            else:
                text = str(value)
            cells[idx].text = text
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(7)
    doc.add_paragraph()


def build_word_doc(
    out_dir: Path,
    full_agg: pd.DataFrame,
    stage_agg: pd.DataFrame,
    selected_agg: pd.DataFrame,
    discovered: pd.DataFrame,
) -> Path:
    from docx import Document
    from docx.enum.section import WD_ORIENT
    from docx.shared import Inches

    doc = Document()
    section = doc.sections[-1]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width = Inches(11)
    section.page_height = Inches(8.5)
    section.top_margin = Inches(0.45)
    section.bottom_margin = Inches(0.45)
    section.left_margin = Inches(0.45)
    section.right_margin = Inches(0.45)

    doc.add_heading("Paired 3-Seed SWD Detection Results", level=1)
    doc.add_paragraph(
        "Figures show mean ± SD performance with SD error bars. Bar labels show the mean only. "
        "Confusion matrices are pooled across seeds 42, 43, and 44. "
    )

    figure_specs = [
        (
            "Full-Spectrum Model Performance",
            out_dir / "full_spectrum" / "paired_full_spectrum_results_confusion_panel_labeled_alt_cmap_3seed.png",
        ),
        (
            "Stage-Specific Model Performance",
            out_dir / "stage_specific" / "paired_stage_specific_model_performance_grid_3x4_compact_3seed.png",
        ),
        (
            "Selected-Wavelength Model Performance",
            out_dir / "selected_wavelength" / "paired_selected_wavelength_results_confusion_3seed.png",
        ),
        (
            "Full-Wavelength Cross-Stage Accuracy",
            out_dir / "cross_stage" / "fig3_full_wavelength_cross_stage_accuracy_3seed.png",
        ),
        (
            "Validation-Selected Wavelengths",
            out_dir / "selected_wavelengths" / "fig4_validation_selected_wavelengths_top30_3seed.png",
        ),
        (
            "Attribution Stability",
            out_dir / "stability" / "fig7_validation_selected_wavelength_stability_3seed_1x2.png",
        ),
        (
            "IG-SHAP Cross-Method Agreement",
            out_dir / "cross_method" / "fig8_cross_method_ig_shap_jaccard_spearman_3seed_1x2.png",
        ),
        (
            "Mean Spectra by Class and Stage",
            out_dir / "mean_spectra" / "fig9_mean_spectra_by_stage_2x3_3seed.png",
        ),
        (
            "Faithfulness Ablation",
            out_dir / "faithfulness" / "fig6_faithfulness_3seed_2x2.png",
        ),
    ]
    for title, path in figure_specs:
        doc.add_heading(title, level=2)
        if path.exists():
            doc.add_picture(str(path), width=Inches(9.6))
        else:
            doc.add_paragraph(f"Missing figure: {path}")

    add_dataframe_table(
        doc,
        percent_table(full_agg.drop(columns=["results_paths"]), ["sensor", "model"]),
        "Full-Spectrum Aggregate Results",
    )
    add_dataframe_table(
        doc,
        percent_table(stage_agg.drop(columns=["results_paths"]), ["stage", "sensor", "model"]),
        "Stage-Specific Aggregate Results",
    )
    add_dataframe_table(
        doc,
        percent_table(selected_agg.drop(columns=["results_paths"]), ["method", "sensor", "model"]),
        "Selected-Wavelength Aggregate Results",
    )
    add_dataframe_table(
        doc,
        individual_run_table(discovered.drop(columns=["summary_path", "results_path"])),
        "Individual Seed-Level Results",
    )

    path = out_dir / "paired_3seed_full_results.docx"
    doc.save(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare paired 3-seed manuscript figures.")
    parser.add_argument(
        "--root",
        type=Path,
        default=SCRIPT_DIR / "outputs_paired_3seed_20260807",
        help="3-seed rerun output root.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "manuscript_figures_paired_3seed",
        help="Directory for generated manuscript figures.",
    )
    parser.add_argument(
        "--seed42-full-roots",
        type=Path,
        nargs="*",
        default=[SCRIPT_DIR / "outputs_paired_cnn3d", SCRIPT_DIR / "outputs_paired_cnn3d_transformer"],
        help="Corrected full-spectrum roots to treat as seed 42.",
    )
    parser.add_argument(
        "--cross-stage-root",
        type=Path,
        default=SCRIPT_DIR / "outputs_paired_3seed_20260807" / "cross_stage_full",
        help="Root containing per-seed cross_stage_results.csv files.",
    )
    parser.add_argument(
        "--mean-spectra-aggregate",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "manuscript_figures" / "mean_spectra_balanced" / "mean_spectra_aggregates.csv",
        help="Mean spectra aggregate CSV containing mean_reflectance and std columns.",
    )
    parser.add_argument(
        "--cross-method-csv",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "explanation_validation_paired" / "cross_method_agreement.csv",
        help="Cross-method agreement CSV for manuscript-style Fig. 8.",
    )
    args = parser.parse_args()

    df = discover_summaries(args.root, args.seed42_full_roots)
    df.attrs["source_root"] = str(args.root)
    df.attrs["seed42_full_roots"] = ", ".join(str(path) for path in args.seed42_full_roots)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["summary_path", "results_path"]).to_csv(args.out_dir / "paired_3seed_discovered_runs.csv", index=False)

    full = df[(~df["selected"]) & (df["train_mode"] == "stage_agnostic") & (df["stage"] == "all")]
    stage = df[(~df["selected"]) & (df["train_mode"] == "per_stage")]
    selected = df[df["selected"]]

    full_agg = aggregate_group(full, ["sensor", "model"]) if not full.empty else pd.DataFrame()
    stage_agg = aggregate_group(stage, ["stage", "sensor", "model"]) if not stage.empty else pd.DataFrame()
    selected_agg = aggregate_group(selected, ["method", "sensor", "model"]) if not selected.empty else pd.DataFrame()

    if not full_agg.empty:
        plot_full(full_agg, args.out_dir / "full_spectrum")
    if not stage_agg.empty:
        plot_stage_specific(stage_agg, args.out_dir / "stage_specific")
    if not selected_agg.empty:
        plot_selected(selected_agg, args.out_dir / "selected_wavelength")
    if args.cross_stage_root.exists():
        cross_stage = discover_cross_stage_results(args.cross_stage_root)
        cross_stage_agg = aggregate_cross_stage(cross_stage)
        plot_cross_stage_accuracy(cross_stage_agg, args.out_dir / "cross_stage")
    selected_records = load_selection_records(args.root, args.seed42_full_roots, k=30)
    plot_selected_wavelengths_3seed(selected_records, args.out_dir / "selected_wavelengths")
    plot_selection_stability_3seed(selected_records, args.out_dir / "stability")
    cross_method_seed = collect_cross_method_stagewise_3seed(args.root, args.seed42_full_roots, k=30)
    plot_cross_method_stagewise_3seed(cross_method_seed, args.out_dir / "cross_method")
    plot_mean_spectra_with_sd(args.mean_spectra_aggregate, args.out_dir / "mean_spectra")
    faithfulness_generated = plot_faithfulness_if_available(args.out_dir)
    combined_csv_path = write_combined_mean_sd_csv(args.out_dir, full_agg, stage_agg, selected_agg)
    print(f"Saved combined mean/std CSV: {combined_csv_path}")
    if not full_agg.empty and not stage_agg.empty and not selected_agg.empty:
        docx_path = build_word_doc(args.out_dir, full_agg, stage_agg, selected_agg, df)
        print(f"Saved Word results document: {docx_path}")
    write_manifest(df, args.out_dir, {"faithfulness": faithfulness_generated})
    print(f"Saved 3-seed manuscript figures under {args.out_dir}")


if __name__ == "__main__":
    main()
