"""
cross_stage_eval.py - cross-ripeness evaluation for trained full-spectrum models.

This script does not train. It loads existing checkpoints from outputs/ and
evaluates each trained source condition against each target test condition:

    trained on Ripe -> tested on Ripe/Midripe/Unripe/all
    trained on Midripe        -> tested on Ripe/Midripe/Unripe/all
    trained on Unripe         -> tested on Ripe/Midripe/Unripe/all
    trained on all stages     -> tested on Ripe/Midripe/Unripe/all

Outputs:
    cross_stage_results.csv
    per sensor/model metric matrices
    heatmaps for accuracy and macro-F1
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

import config as CFG
from data.dataset import ShardedBlueberryDataset
from models import build_model


STAGES = ["Ripe", "Midripe", "Unripe"]
SOURCE_TAGS = STAGES + ["all"]
TARGET_TAGS = STAGES + ["all"]

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
METRIC_DISPLAY = {
    "accuracy": "accuracy",
    "f1_macro": "macro-F1",
    "precision_macro": "macro-precision",
    "recall_macro": "macro-recall",
    "roc_auc": "ROC-AUC",
}


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_manifest(shard_dir: Path) -> List[dict]:
    manifest = shard_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"manifest.csv not found: {manifest}")
    with open(manifest, newline="") as fh:
        return list(csv.DictReader(fh))


def filter_rows(rows: List[dict], sensor: str, split: str, stage: Optional[str]) -> List[dict]:
    return [
        r for r in rows
        if r["sensor"] == sensor
        and r["split"] == split
        and (stage is None or r["ripeness"] == stage)
    ]


def make_dataset(rows: List[dict], sensor: str, target_stage: str, max_samples: Optional[int] = None):
    stage = None if target_stage == "all" else target_stage
    sensor_rows = filter_rows(rows, sensor, "test", stage)
    if max_samples:
        sensor_rows = sensor_rows[:max_samples]
    return ShardedBlueberryDataset(sensor_rows, label_mode="binary", transform=None)


def infer_dims(rows: List[dict], sensor: str) -> Tuple[int, int, int, int]:
    """Return (nir_bands, vnir_bands, cell_h, cell_w)."""
    nir_row = next((r for r in rows if r["sensor"] == "nir"), None)
    vnir_row = next((r for r in rows if r["sensor"] == "vnir"), None)
    if nir_row is None and sensor == "nir":
        raise RuntimeError("No NIR rows found in manifest.")
    if vnir_row is None and sensor == "vnir":
        raise RuntimeError("No VNIR rows found in manifest.")

    nir_bands = int(nir_row["n_bands"]) if nir_row else 0
    vnir_bands = int(vnir_row["n_bands"]) if vnir_row else 0
    ref = nir_row if sensor == "nir" else vnir_row
    return nir_bands, vnir_bands, int(ref["cell_h"]), int(ref["cell_w"])


def build_full_model(model_name: str, sensor: str, rows: List[dict], device: torch.device):
    nir_bands, vnir_bands, cell_h, cell_w = infer_dims(rows, sensor)
    n_bands = {
        "nir": nir_bands,
        "vnir": vnir_bands,
    }[sensor]

    kwargs = dict(
        n_bands=n_bands,
        cell_h=cell_h,
        cell_w=cell_w,
        num_classes=CFG.NUM_CLASSES,
        dropout=CFG.DROPOUT,
        label_mode="binary",
    )
    if model_name == "cnn3d_transformer":
        kwargs.update(
            n_layers=CFG.TRANSFORMER_N_LAYERS,
            n_heads=CFG.TRANSFORMER_N_HEADS,
        )
    elif model_name != "cnn3d":
        raise ValueError(f"This script currently supports cnn3d and cnn3d_transformer, got {model_name!r}")

    return build_model(model_name, **kwargs).to(device)


def checkpoint_path(outputs_dir: Path, sensor: str, model_name: str, source_stage: str) -> Tuple[Path, str, str, str]:
    if source_stage == "all":
        train_mode = "stage_agnostic"
        rip_tag = "all"
    else:
        train_mode = "per_stage"
        rip_tag = source_stage
    run_name = f"{sensor}__{model_name}__{train_mode}__{rip_tag}"
    path = outputs_dir / sensor / model_name / train_mode / rip_tag / f"{run_name}_best.pt"
    return path, run_name, train_mode, rip_tag


def load_weights(model, path: Path, device: torch.device) -> None:
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)


@torch.no_grad()
def predict(model, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    true, pred, prob = [], [], []
    model.eval()
    use_amp = device.type == "cuda"
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        probs = torch.softmax(logits, dim=1)
        true.append(y.numpy())
        pred.append(logits.argmax(dim=1).cpu().numpy())
        prob.append(probs.cpu().numpy())
    return np.concatenate(true), np.concatenate(pred), np.concatenate(prob)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
    except ValueError:
        out["roc_auc"] = float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out.update({
        "tn_infested": int(cm[0, 0]),
        "fp_infested_as_healthy": int(cm[0, 1]),
        "fn_healthy_as_infested": int(cm[1, 0]),
        "tp_healthy": int(cm[1, 1]),
    })
    return out


def save_heatmap(matrix: pd.DataFrame, path: Path, title: str, vmin: float = 0.0, vmax: float = 100.0) -> None:
    plot_matrix = matrix.astype(float) * 100.0
    plot_matrix = plot_matrix.rename(index=STAGE_DISPLAY, columns=STAGE_DISPLAY)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(plot_matrix.to_numpy(float), cmap="RdYlGn", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Percent (%)", fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)
    ax.set_xticks(np.arange(len(plot_matrix.columns)), plot_matrix.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(plot_matrix.index)), plot_matrix.index)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(11)
        label.set_fontweight("bold")
    ax.set_xlabel("Test stage", fontsize=12, fontweight="bold")
    ax.set_ylabel("Training stage", fontsize=12, fontweight="bold")
    ax.set_title(f"{title} (%)", fontsize=14, fontweight="bold")
    for i in range(plot_matrix.shape[0]):
        for j in range(plot_matrix.shape[1]):
            val = plot_matrix.iat[i, j]
            text = "" if pd.isna(val) else f"{val:.2f}"
            ax.text(
                j, i, text,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white" if (not pd.isna(val) and (val < 35.0 or val > 75.0)) else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_metric_matrices(df: pd.DataFrame, out_dir: Path, source_stages: List[str], target_stages: List[str]) -> None:
    for (sensor, model_name), sub in df.groupby(["sensor", "model"]):
        group_dir = ensure_dir(out_dir / sensor / model_name)
        for metric in ["accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc"]:
            mat = pd.DataFrame(index=source_stages, columns=target_stages, dtype=float)
            for _, row in sub.iterrows():
                mat.loc[row["train_stage"], row["test_stage"]] = row[metric]
            mat.to_csv(group_dir / f"{sensor}__{model_name}__cross_stage_{metric}.csv")
            if metric in ("accuracy", "f1_macro"):
                title = (
                    f"{MODEL_DISPLAY.get(model_name, model_name)} cross-stage "
                    f"{METRIC_DISPLAY.get(metric, metric)}"
                )
                save_heatmap(
                    mat,
                    group_dir / f"{sensor}__{model_name}__cross_stage_{metric}.png",
                    title,
                )


def main():
    parser = argparse.ArgumentParser(description="Cross-stage evaluation of trained full-wavelength models.")
    parser.add_argument("--shard_dir", type=Path, required=True)
    parser.add_argument("--outputs_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/cross_stage_eval"))
    parser.add_argument("--sensors", default="nir,vnir",
                        help="Comma list: nir,vnir")
    parser.add_argument("--models", default="cnn3d,cnn3d_transformer",
                        help="Comma list: cnn3d,cnn3d_transformer")
    parser.add_argument("--source_stages", default="Ripe,Midripe,Unripe,all",
                        help="Comma list of checkpoint training stages to evaluate.")
    parser.add_argument("--target_stages", default="Ripe,Midripe,Unripe,all",
                        help="Comma list of test stages to evaluate.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional debug cap per target test set.")
    parser.add_argument("--batch_size", type=int, default=getattr(CFG, "EVAL_BATCH_SIZE", CFG.BATCH_SIZE))
    parser.add_argument("--num_workers", type=int, default=getattr(CFG, "NUM_WORKERS", 4))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_predictions", action="store_true",
                        help="Do not save per-sample prediction CSVs.")
    parser.add_argument("--plots_only", action="store_true",
                        help="Regenerate matrices/heatmaps from an existing cross_stage_results.csv.")
    args = parser.parse_args()

    sensors = parse_csv_list(args.sensors)
    models = parse_csv_list(args.models)
    source_stages = parse_csv_list(args.source_stages)
    target_stages = parse_csv_list(args.target_stages)
    out_dir = ensure_dir(args.out_dir)
    device = torch.device(args.device)

    if args.plots_only:
        result_path = out_dir / "cross_stage_results.csv"
        if not result_path.exists():
            raise FileNotFoundError(f"Cannot use --plots_only; missing {result_path}")
        result_df = pd.read_csv(result_path)
        save_metric_matrices(result_df, out_dir, source_stages, target_stages)
        print(f"Regenerated percentage heatmaps under: {out_dir}")
        return

    rows = load_manifest(args.shard_dir)

    all_rows = []
    for sensor in sensors:
        for model_name in models:
            for source_stage in source_stages:
                ckpt, source_run_name, train_mode, rip_tag = checkpoint_path(
                    args.outputs_dir, sensor, model_name, source_stage
                )
                if not ckpt.exists():
                    print(f"[SKIP] Missing checkpoint: {ckpt}")
                    continue

                print(f"\n[MODEL] {sensor} {model_name} trained_on={source_stage}")
                model = build_full_model(model_name, sensor, rows, device)
                load_weights(model, ckpt, device)

                for target_stage in target_stages:
                    ds = make_dataset(rows, sensor, target_stage, max_samples=args.max_samples)
                    if len(ds) == 0:
                        print(f"  [SKIP] target={target_stage}: no samples")
                        continue
                    loader = DataLoader(
                        ds,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"),
                        persistent_workers=False,
                        prefetch_factor=(2 if args.num_workers > 0 else None),
                    )
                    y_true, y_pred, probs = predict(model, loader, device)
                    m = metrics(y_true, y_pred, probs)
                    record = {
                        "sensor": sensor,
                        "model": model_name,
                        "checkpoint": str(ckpt),
                        "source_run_name": source_run_name,
                        "train_mode": train_mode,
                        "train_stage": source_stage,
                        "test_stage": target_stage,
                        "n_test": int(len(y_true)),
                        **m,
                    }
                    all_rows.append(record)
                    print(
                        f"  test={target_stage:<15s} "
                        f"acc={m['accuracy']:.4f} f1={m['f1_macro']:.4f} n={len(y_true)}"
                    )

                    if not args.skip_predictions:
                        pred_dir = ensure_dir(out_dir / sensor / model_name / "predictions")
                        pred_df = pd.DataFrame({
                            "true_label": y_true,
                            "pred_label": y_pred,
                            "prob_infested": probs[:, 0],
                            "prob_healthy": probs[:, 1],
                        })
                        pred_df.to_csv(
                            pred_dir / f"{sensor}__{model_name}__train_{source_stage}__test_{target_stage}_predictions.csv",
                            index=False,
                        )

                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    result_df = pd.DataFrame(all_rows)
    result_path = out_dir / "cross_stage_results.csv"
    result_df.to_csv(result_path, index=False)
    with open(out_dir / "cross_stage_results.json", "w") as fh:
        json.dump(all_rows, fh, indent=2)
    print(f"\nSaved: {result_path}")

    if not result_df.empty:
        save_metric_matrices(result_df, out_dir, source_stages, target_stages)
        print(f"Saved matrices and heatmaps under: {out_dir}")


if __name__ == "__main__":
    main()
