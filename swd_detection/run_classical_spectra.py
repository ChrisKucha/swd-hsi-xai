"""
run_classical_spectra.py - classical and 1D-CNN models on berry mean spectra.

This script uses the segmented shard manifest created by
create_segmented_shards.py.  Each per-berry shard is converted to one mean
spectrum by averaging the pixels inside its saved mask, then models are trained
on the same train/val/test split stored in the manifest.

Models:
  - SVM
  - PLS-DA
  - Random Forest
  - Gradient Boosting
  - 1D CNN

Feature sets:
  - full spectrum
  - IG-selected bands from saved *_band_selection.json files
  - SHAP-selected bands from saved *_shap_band_selection.json files
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data.discovery import discover_and_split
from data.dataset import extract_cell, load_cube
from scipy.ndimage import binary_erosion

from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config as CFG


CLASS_NAMES = ["Infested", "Healthy"]
META_COLS = [
    "sample_id", "sensor", "split", "label", "class_name", "ripeness",
    "file_stem", "day", "cell_idx", "mask_area_fraction",
    "nir_sample_id", "vnir_sample_id",
    "nir_mask_area_fraction", "vnir_mask_area_fraction",
]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_manifest(shard_dir: Path) -> List[dict]:
    path = shard_dir / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _board_lookup(sensor: str) -> Dict[Tuple[str, str, str, int], str]:
    boards = discover_and_split(sensor=sensor, paths=CFG.PATHS[sensor], verbose=False)
    return {
        (b["class_name"], b["ripeness"], b["file_stem"], int(b["day"])): b["path"]
        for b in boards
    }


def _correct_reflectance_scale(cube: np.ndarray, board_path: str, sensor: str, mode: str) -> np.ndarray:
    """Correct rare raw cubes that are stored on a different reflectance scale."""
    if mode == "none":
        return cube
    p99 = float(np.nanpercentile(cube, 99))
    if mode in ("auto", "divide100_if_p99_gt_2"):
        if p99 > 2.0:
            print(
                f"  [WARN] {sensor.upper()} cube appears off-scale "
                f"(p99={p99:.3f}); dividing by 100: {board_path}"
            )
            return cube / 100.0
        return cube
    raise ValueError(f"Unknown reflectance scale correction mode: {mode}")


def _mean_spectrum_from_raw_cell(
    row: dict,
    board_paths: Dict[Tuple[str, str, str, int], str],
    cube_cache: Dict[str, np.ndarray],
    erode_mask_pixels: int = 0,
    min_pixel_mean_reflectance: float = 0.0,
    reflectance_scale_correction: str = "auto",
) -> np.ndarray:
    key = (
        row["class_name"],
        row["ripeness"],
        row["file_stem"],
        int(row["day"]),
    )
    board_path = board_paths.get(key)
    if board_path is None:
        raise KeyError(
            "Could not find original board for "
            f"{row['sensor']} {row['class_name']} {row['ripeness']} "
            f"{row['file_stem']} Day{row['day']}"
        )

    if board_path not in cube_cache:
        if len(cube_cache) >= 2:
            cube_cache.clear()
        cube = load_cube(board_path, row["sensor"])
        cube_cache[board_path] = _correct_reflectance_scale(
            cube, board_path, row["sensor"], reflectance_scale_correction
        )

    cube = cube_cache[board_path]
    cell = extract_cell(
        cube,
        int(row["cell_idx"]),
        int(row.get("cell_h") or 0) or None,
        int(row.get("cell_w") or 0) or None,
    ).astype(np.float32)  # (B, H, W), raw reflectance scale
    mask_path = row.get("mask_path", "")
    if mask_path and Path(mask_path).exists():
        mask = np.load(mask_path).astype(bool)
    else:
        mask = np.any(cell != 0, axis=0)

    if mask.shape != cell.shape[1:]:
        raise ValueError(
            f"Mask shape {mask.shape} does not match cell spatial shape "
            f"{cell.shape[1:]} for {row['shard_path']}"
        )
    if erode_mask_pixels > 0 and mask.any():
        structure = np.ones((3, 3), dtype=bool)
        mask = binary_erosion(mask, structure=structure, iterations=erode_mask_pixels)
    if not mask.any():
        return np.full(cell.shape[0], np.nan, dtype=np.float32)
    pixels = cell[:, mask].T  # (N, B)
    if min_pixel_mean_reflectance > 0:
        keep = pixels.mean(axis=1) >= min_pixel_mean_reflectance
        pixels = pixels[keep]
    if pixels.size == 0:
        return np.full(cell.shape[0], np.nan, dtype=np.float32)
    return pixels.mean(axis=0).astype(np.float32)


def _wavelengths(sensor: str) -> np.ndarray:
    if sensor == "nir":
        return np.asarray(CFG.NIR_WAVELENGTHS, dtype=np.float32)
    if sensor == "vnir":
        return np.asarray(CFG.VNIR_WAVELENGTHS, dtype=np.float32)
    raise ValueError(sensor)


def _spectra_csv_path(out_dir: Path, sensor: str) -> Path:
    return out_dir / "features" / f"{sensor}_mean_spectra.csv"


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))


def _save_mean_spectra_tables_and_plots(df: pd.DataFrame, sensor: str, out_dir: Path) -> None:
    """Save aggregate mean spectra tables and simple inspection plots."""
    if sensor not in ("nir", "vnir"):
        return

    wave_cols = [c for c in df.columns if c not in META_COLS]
    if not wave_cols:
        return

    wavelengths = np.asarray([float(c) for c in wave_cols], dtype=np.float32)
    feature_dir = _ensure_dir(out_dir / "features")
    plot_dir = _ensure_dir(out_dir / "plots" / "spectra")

    by_class = (
        df.groupby("class_name", sort=True)[wave_cols]
        .mean()
        .reset_index()
    )
    by_stage_class = (
        df.groupby(["ripeness", "class_name"], sort=True)[wave_cols]
        .mean()
        .reset_index()
    )

    by_class_path = feature_dir / f"{sensor}_mean_spectra_by_class.csv"
    by_stage_path = feature_dir / f"{sensor}_mean_spectra_by_ripeness_class.csv"
    by_class.to_csv(by_class_path, index=False)
    by_stage_class.to_csv(by_stage_path, index=False)

    colors = {"Infested": "#c2410c", "Healthy": "#15803d"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in by_class.iterrows():
        label = row["class_name"]
        ax.plot(
            wavelengths,
            row[wave_cols].to_numpy(dtype=np.float32),
            label=label,
            linewidth=2,
            color=colors.get(label),
        )
    ax.set_title(f"{sensor.upper()} mean spectra by class")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Mean intensity")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / f"{sensor}_mean_spectra_by_class.png", dpi=180)
    plt.close(fig)

    for ripeness in sorted(df["ripeness"].dropna().unique()):
        sub = by_stage_class[by_stage_class["ripeness"] == ripeness]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for _, row in sub.iterrows():
            label = row["class_name"]
            ax.plot(
                wavelengths,
                row[wave_cols].to_numpy(dtype=np.float32),
                label=label,
                linewidth=2,
                color=colors.get(label),
            )
        ax.set_title(f"{sensor.upper()} mean spectra - {ripeness}")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Mean intensity")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{sensor}_mean_spectra_{_safe_name(ripeness)}.png", dpi=180)
        plt.close(fig)

    print(f"  mean spectra tables -> {by_class_path}, {by_stage_path}")
    print(f"  mean spectra plots  -> {plot_dir}")


def _mask_area_ok(row: dict, min_area: float, max_area: float) -> bool:
    try:
        area = float(row.get("mask_area_fraction", np.nan))
    except (TypeError, ValueError):
        return False
    return np.isfinite(area) and min_area <= area <= max_area


def build_sensor_spectra(rows: List[dict], sensor: str, out_dir: Path, args) -> pd.DataFrame:
    out_path = _spectra_csv_path(out_dir, sensor)
    if out_path.exists() and not args.overwrite_features:
        return pd.read_csv(out_path)

    sensor_rows = [r for r in rows if r["sensor"] == sensor]
    if not sensor_rows:
        raise RuntimeError(f"No manifest rows found for sensor {sensor!r}")

    wave_cols = [f"{w:.2f}" for w in _wavelengths(sensor)]
    records = []
    board_paths = _board_lookup(sensor)
    cube_cache: Dict[str, np.ndarray] = {}
    skipped_area = 0
    print(f"\n[Features] Extracting {sensor.upper()} mean spectra: {len(sensor_rows)} berries")
    for i, row in enumerate(sensor_rows, 1):
        if i % 2000 == 0:
            print(f"  {sensor}: {i}/{len(sensor_rows)}")
        if not _mask_area_ok(row, args.min_mask_area_fraction, args.max_mask_area_fraction):
            skipped_area += 1
            continue
        spec = _mean_spectrum_from_raw_cell(
            row,
            board_paths,
            cube_cache,
            erode_mask_pixels=args.erode_mask_pixels,
            min_pixel_mean_reflectance=args.min_pixel_mean_reflectance,
            reflectance_scale_correction=args.reflectance_scale_correction,
        )
        sample_id = (
            f"{row['sensor']}__{row['class_name']}__{row['ripeness']}__"
            f"{row['file_stem']}__Day{row['day']}__c{int(row['cell_idx']):02d}"
        )
        rec = {
            "sample_id": sample_id,
            "sensor": row["sensor"],
            "split": row["split"],
            "label": int(row["label"]),
            "class_name": row["class_name"],
            "ripeness": row["ripeness"],
            "file_stem": row["file_stem"],
            "day": int(row["day"]),
            "cell_idx": int(row["cell_idx"]),
            "mask_area_fraction": float(row.get("mask_area_fraction", np.nan)),
        }
        rec.update({c: float(v) for c, v in zip(wave_cols, spec)})
        records.append(rec)

    df = pd.DataFrame(records)
    df = df.dropna(subset=wave_cols).reset_index(drop=True)
    _ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False)
    print(f"  saved -> {out_path} ({len(df)} rows)")
    if skipped_area:
        print(
            f"  skipped {skipped_area} rows outside mask area range "
            f"[{args.min_mask_area_fraction}, {args.max_mask_area_fraction}]"
        )
    return df


def _read_band_json(path: Path, k: int, strategy: str) -> Optional[List[int]]:
    if not path.exists():
        return None
    with open(path) as fh:
        data = json.load(fh)
    try:
        return [int(x) for x in data["strategies"][strategy][str(k)]["band_indices"]]
    except KeyError:
        return None


def load_band_selections(
    outputs_dir: Path,
    sensors: Iterable[str],
    k: int,
    strategy: str,
    sources: Iterable[str],
) -> Dict[str, Dict[str, List[int]]]:
    selections: Dict[str, Dict[str, List[int]]] = {s: {} for s in sensors}
    for sensor in sensors:
        for source in sources:
            base = outputs_dir / sensor / source / "stage_agnostic" / "all"
            stem = f"{sensor}__{source}__stage_agnostic__all"
            ig = _read_band_json(base / f"{stem}_band_selection.json", k, strategy)
            shap = _read_band_json(base / f"{stem}_shap_band_selection.json", k, strategy)
            if ig:
                selections[sensor][f"ig_{source}_top{k}"] = ig
            if shap:
                selections[sensor][f"shap_{source}_top{k}"] = shap
    return selections


def feature_columns(df: pd.DataFrame, sensor: str, feature_set: str, selections: Dict[str, Dict[str, List[int]]]) -> List[str]:
    all_features = [c for c in df.columns if c not in META_COLS]
    if feature_set == "full":
        return all_features

    if sensor in ("nir", "vnir"):
        idxs = selections.get(sensor, {}).get(feature_set)
        if idxs is None:
            raise KeyError(f"No selection {feature_set!r} for {sensor}")
        return [all_features[i] for i in idxs if i < len(all_features)]

    raise KeyError(f"Unsupported sensor {sensor!r}; use nir or vnir")


def split_xy(
    df: pd.DataFrame,
    cols: List[str],
    max_rows_per_split: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    parts = {}
    rng = np.random.default_rng(seed)
    used_frames = []
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split].copy()
        if max_rows_per_split and len(sub) > max_rows_per_split:
            idx = rng.choice(sub.index.to_numpy(), size=max_rows_per_split, replace=False)
            sub = sub.loc[idx].copy()
        sub = sub.reset_index(drop=True)
        parts[split] = sub
        used_frames.append(sub)

    def xy(split: str):
        sub = parts[split]
        return sub[cols].to_numpy(np.float32), sub["label"].to_numpy(np.int64)

    X_train, y_train = xy("train")
    X_val, y_val = xy("val")
    X_test, y_test = xy("test")
    return X_train, y_train, X_val, y_val, X_test, y_test, pd.concat(used_frames, ignore_index=True)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, prob_pos: Optional[np.ndarray]) -> Dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if prob_pos is not None and len(np.unique(y_true)) == 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, prob_pos))
        except ValueError:
            out["roc_auc"] = float("nan")
    return out


def _save_confusion(y_true: np.ndarray, y_pred: np.ndarray, path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _classification_outputs(
    out_dir: Path,
    run_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prob_pos: Optional[np.ndarray],
    summary: Dict[str, object],
) -> Dict[str, object]:
    _ensure_dir(out_dir)
    pred_df = pd.DataFrame({"true_label": y_true, "pred_label": y_pred})
    if prob_pos is not None:
        pred_df["prob_healthy"] = prob_pos
    pred_df.to_csv(out_dir / f"{run_name}_predictions.csv", index=False)
    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(out_dir / f"{run_name}_classification_report.csv")
    _save_confusion(y_true, y_pred, out_dir / f"{run_name}_confusion_matrix.png", run_name)
    with open(out_dir / f"{run_name}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def _prob_pos(model, X: np.ndarray) -> Optional[np.ndarray]:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))
    return None


class PLSDA:
    def __init__(self, n_components: int = 8):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.model = PLSRegression(n_components=n_components)

    def fit(self, X, y):
        n_comp = max(1, min(self.n_components, X.shape[1], X.shape[0] - 1))
        self.model = PLSRegression(n_components=n_comp)
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y.astype(float))
        return self

    def predict(self, X):
        score = self.predict_proba(X)[:, 1]
        return (score >= 0.5).astype(int)

    def predict_proba(self, X):
        Xs = self.scaler.transform(X)
        score = self.model.predict(Xs).reshape(-1)
        score = np.clip(score, 0.0, 1.0)
        return np.stack([1.0 - score, score], axis=1)


def train_sklearn_model(
    model_name: str,
    X_train, y_train, X_val, y_val,
) -> Tuple[object, Dict[str, object]]:
    candidates = []
    if model_name == "svm":
        for C in [0.1, 1.0, 10.0]:
            for gamma in ["scale", "auto"]:
                candidates.append(Pipeline([
                    ("scale", StandardScaler()),
                    ("model", SVC(C=C, gamma=gamma, kernel="rbf", probability=True, class_weight="balanced")),
                ]))
    elif model_name == "plsda":
        for n in [2, 4, 8, 12, 16, 24]:
            candidates.append(PLSDA(n_components=n))
    elif model_name == "rf":
        for depth in [None, 8, 16]:
            candidates.append(RandomForestClassifier(
                n_estimators=400, max_depth=depth, class_weight="balanced",
                random_state=42, n_jobs=-1,
            ))
    elif model_name == "gb":
        for lr in [0.03, 0.05, 0.1]:
            candidates.append(GradientBoostingClassifier(
                n_estimators=250, learning_rate=lr, max_depth=2, random_state=42,
            ))
    else:
        raise ValueError(model_name)

    best_model = None
    best_info = None
    best_f1 = -1.0
    for cand in candidates:
        model = clone(cand) if not isinstance(cand, PLSDA) else PLSDA(cand.n_components)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        f1 = f1_score(y_val, pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_model = model
            best_info = {"val_f1_macro": float(f1), "selected_model": str(model)}
    return best_model, best_info or {}


class SpectralCNN1D(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 2, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(self.net(x))


def train_cnn1d(
    X_train, y_train, X_val, y_val, out_dir: Path, run_name: str,
    epochs: int, batch_size: int, patience: int, lr: float, weight_decay: float,
) -> Tuple[SpectralCNN1D, Dict[str, object], StandardScaler]:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectralCNN1D(X_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    train_ds = TensorDataset(
        torch.from_numpy(X_train[:, None, :]),
        torch.from_numpy(y_train.astype(np.int64)),
    )
    val_x = torch.from_numpy(X_val[:, None, :]).to(device)
    val_y = torch.from_numpy(y_val.astype(np.int64)).to(device)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    best_state = None
    best_loss = float("inf")
    best_acc = 0.0
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(yb)
            n_seen += len(yb)

        model.eval()
        with torch.no_grad():
            logits = model(val_x)
            val_loss = float(crit(logits, val_y).item())
            pred = logits.argmax(dim=1)
            val_acc = float((pred == val_y).float().mean().item())
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / max(n_seen, 1),
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_loss < best_loss:
            best_loss = val_loss
            best_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    _ensure_dir(out_dir)
    torch.save({"model_state": model.state_dict(), "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_},
               out_dir / f"{run_name}_best.pt")
    pd.DataFrame(history).to_csv(out_dir / f"{run_name}_history.csv", index=False)
    return model, {"best_val_loss": best_loss, "best_val_acc": best_acc, "best_epoch": best_epoch}, scaler


def predict_cnn1d(model: nn.Module, scaler: StandardScaler, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    Xs = scaler.transform(X).astype(np.float32)
    model.eval()
    probs_all = []
    with torch.no_grad():
        for start in range(0, len(Xs), 512):
            xb = torch.from_numpy(Xs[start:start + 512, None, :]).to(device)
            probs_all.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    probs = np.concatenate(probs_all, axis=0)
    return probs.argmax(axis=1), probs[:, 1]


def run_one(
    df: pd.DataFrame,
    sensor: str,
    feature_set: str,
    cols: List[str],
    model_name: str,
    out_root: Path,
    args,
) -> Dict[str, object]:
    X_train, y_train, X_val, y_val, X_test, y_test, used_df = split_xy(
        df, cols, max_rows_per_split=args.max_rows_per_split, seed=args.seed
    )
    run_name = f"{sensor}__{model_name}__{feature_set}"
    out_dir = _ensure_dir(out_root / sensor / model_name / feature_set)

    if model_name == "cnn1d":
        model, train_info, scaler = train_cnn1d(
            X_train, y_train, X_val, y_val, out_dir, run_name,
            epochs=args.cnn_epochs, batch_size=args.cnn_batch_size,
            patience=args.cnn_patience, lr=args.cnn_lr,
            weight_decay=args.cnn_weight_decay,
        )
        y_pred, prob = predict_cnn1d(model, scaler, X_test)
    else:
        model, train_info = train_sklearn_model(model_name, X_train, y_train, X_val, y_val)
        y_pred = model.predict(X_test)
        prob = _prob_pos(model, X_test)

    metrics = _metrics(y_test, y_pred, prob)
    summary = {
        "sensor": sensor,
        "model": model_name,
        "feature_set": feature_set,
        "n_features": len(cols),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "feature_columns": cols,
        **train_info,
        **metrics,
    }
    _classification_outputs(out_dir, run_name, y_test, y_pred, prob, summary)
    print(
        f"  {run_name}: acc={summary['accuracy']:.4f} "
        f"f1_macro={summary['f1_macro']:.4f} n_features={len(cols)}"
    )
    return summary


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Classical and 1D-CNN models on segmented berry mean spectra.")
    parser.add_argument("--shard_dir", type=Path, default=Path("/media/kuchalab/New Volume/swd_detection_segmented_shards"))
    parser.add_argument("--outputs_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--out_dir", type=Path, default=Path("outputs_classical_spectra"))
    parser.add_argument("--sensors", default="nir,vnir",
                        help="Comma list: nir,vnir")
    parser.add_argument("--models", default="svm,plsda,rf,gb,cnn1d",
                        help="Comma list: svm,plsda,rf,gb,cnn1d")
    parser.add_argument("--feature_sets", default="full,ig_cnn3d_top30,shap_cnn3d_top30",
                        help="Comma list. Use full plus names from selection JSONs.")
    parser.add_argument("--selection_sources", default="cnn3d,cnn3d_transformer")
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--strategy", default="averaged")
    parser.add_argument("--overwrite_features", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--erode_mask_pixels", type=int, default=0,
                        help="Erode saved berry masks by this many pixels before averaging spectra.")
    parser.add_argument("--min_mask_area_fraction", type=float, default=0.02,
                        help="Skip masks smaller than this cell-area fraction.")
    parser.add_argument("--max_mask_area_fraction", type=float, default=0.40,
                        help="Skip masks larger than this cell-area fraction.")
    parser.add_argument("--min_pixel_mean_reflectance", type=float, default=0.0,
                        help="Within each mask, discard pixels whose mean spectrum is below this value.")
    parser.add_argument("--reflectance_scale_correction", default="auto",
                        choices=["auto", "divide100_if_p99_gt_2", "none"],
                        help="Correct raw cubes that appear to be stored above the 0-1 reflectance scale.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_rows_per_split", type=int, default=None,
                        help="Debug option: cap rows per split.")
    parser.add_argument("--cnn_epochs", type=int, default=120)
    parser.add_argument("--cnn_batch_size", type=int, default=256)
    parser.add_argument("--cnn_patience", type=int, default=18)
    parser.add_argument("--cnn_lr", type=float, default=1e-3)
    parser.add_argument("--cnn_weight_decay", type=float, default=1e-4)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sensors = parse_csv_list(args.sensors)
    models = parse_csv_list(args.models)
    requested_feature_sets = parse_csv_list(args.feature_sets)
    sources = parse_csv_list(args.selection_sources)
    _ensure_dir(args.out_dir)

    rows = _load_manifest(args.shard_dir)
    dfs: Dict[str, pd.DataFrame] = {}
    if "nir" in sensors:
        dfs["nir"] = build_sensor_spectra(rows, "nir", args.out_dir, args)
    if "vnir" in sensors:
        dfs["vnir"] = build_sensor_spectra(rows, "vnir", args.out_dir, args)
    for sensor in ("nir", "vnir"):
        if sensor in dfs:
            _save_mean_spectra_tables_and_plots(dfs[sensor], sensor, args.out_dir)

    selections = load_band_selections(args.outputs_dir, ["nir", "vnir"], args.top_k, args.strategy, sources)
    with open(args.out_dir / "band_selections_loaded.json", "w") as fh:
        json.dump(selections, fh, indent=2)

    print("\n[Selections]")
    for sensor, vals in selections.items():
        print(f"  {sensor}: {', '.join(vals) if vals else 'none'}")

    summaries = []
    for sensor in sensors:
        df = dfs[sensor]
        for feature_set in requested_feature_sets:
            try:
                cols = feature_columns(df, sensor, feature_set, selections)
            except KeyError as exc:
                print(f"  [SKIP] {sensor} {feature_set}: {exc}")
                continue
            if not cols:
                print(f"  [SKIP] {sensor} {feature_set}: no feature columns")
                continue
            for model_name in models:
                run_name = f"{sensor}__{model_name}__{feature_set}"
                summary_path = args.out_dir / sensor / model_name / feature_set / f"{run_name}_summary.json"
                if args.skip_existing and summary_path.exists():
                    with open(summary_path) as fh:
                        summaries.append(json.load(fh))
                    print(f"  [SKIP existing] {run_name}")
                    continue
                summaries.append(run_one(df, sensor, feature_set, cols, model_name, args.out_dir, args))

    comp = pd.DataFrame(summaries)
    comp_path = args.out_dir / "classical_spectra_comparison.csv"
    comp.to_csv(comp_path, index=False)
    print(f"\nDone. Comparison table -> {comp_path}")


if __name__ == "__main__":
    main()
