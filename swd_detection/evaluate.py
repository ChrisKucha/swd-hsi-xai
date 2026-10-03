"""
evaluate.py — Evaluation and results reporting for the SWD blueberry HSI pipeline.

Public API
----------
    from swd_detection.evaluate import evaluate, compare_runs

    # Per-run evaluation
    metrics = evaluate(
        model         = model,
        test_loader   = test_loader,
        output_dir    = "/path/to/outputs",
        run_name      = "nir_cnn3d_binary",
        label_mode    = "binary",          # or "multi_task"
        ripeness_names = ["Ripe", "Midripe", "Unripe"],
        class_names   = ["Infested", "Healthy"],
        wavelengths   = None,              # optional np.ndarray of band wavelengths
    )

    # Cross-run comparison
    compare_runs(results_dir="/path/to/outputs", output_dir="/path/to/outputs")

evaluate() saves
----------------
    {run_name}_confusion_matrix.png
    {run_name}_roc_curve.png
    {run_name}_results.csv   — one row per test sample
    {run_name}_summary.json  — all scalar metrics

compare_runs() saves
--------------------
    comparison_table.xlsx
    comparison_table.png     — bar chart of F1 scores

Dependencies: torch, numpy, sklearn, matplotlib, pandas. No others.
"""

import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _collect_predictions(
    model,
    test_loader: DataLoader,
    device:      torch.device,
    label_mode:  str,
) -> Dict[str, np.ndarray]:
    """
    Run inference over the full test set and return a dict of arrays:
        true_labels   : (N,) int
        pred_labels   : (N,) int
        pred_probs    : (N, num_classes) float — softmax probabilities
        rip_true      : (N,) int  (only if label_mode == "multi_task")
        rip_pred      : (N,) int  (only if label_mode == "multi_task")
    """
    model.eval()
    all_true_cls, all_pred_cls, all_probs = [], [], []
    all_true_rip, all_pred_rip = [], []

    use_amp = device.type == "cuda"

    with torch.no_grad():
        for batch in test_loader:
            if label_mode == "binary":
                x, cls_lbl = batch
                rip_lbl    = None
            else:
                x, cls_lbl, rip_lbl = batch

            x       = x.to(device, non_blocking=True)
            cls_lbl = cls_lbl.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(x)

            # Handle models that return (cls_logits, rip_logits) or just cls_logits
            if isinstance(out, (tuple, list)):
                cls_logits = out[0]
                rip_logits = out[1] if (label_mode == "multi_task" and len(out) >= 2) else None
            else:
                cls_logits = out
                rip_logits = None

            probs      = torch.softmax(cls_logits, dim=1)
            pred_cls   = cls_logits.argmax(dim=1)

            all_true_cls.append(cls_lbl.cpu().numpy())
            all_pred_cls.append(pred_cls.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

            if label_mode == "multi_task":
                all_true_rip.append(rip_lbl.cpu().numpy() if rip_lbl is not None else
                                    np.full(cls_lbl.shape, -1))
                if rip_logits is not None:
                    all_pred_rip.append(rip_logits.argmax(dim=1).cpu().numpy())
                else:
                    all_pred_rip.append(np.full(cls_lbl.shape, -1))

    result = {
        "true_labels": np.concatenate(all_true_cls),
        "pred_labels": np.concatenate(all_pred_cls),
        "pred_probs":  np.concatenate(all_probs, axis=0),
    }
    if label_mode == "multi_task":
        result["rip_true"] = np.concatenate(all_true_rip)
        result["rip_pred"] = np.concatenate(all_pred_rip)

    return result


def _plot_confusion_matrix(
    cm:          np.ndarray,
    class_names: List[str],
    save_path:   str,
    title:       str = "Confusion Matrix",
) -> None:
    fig, ax = plt.subplots(figsize=(max(5, len(class_names) * 1.5),
                                    max(4, len(class_names) * 1.5)))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=11)

    ax.set_ylabel("True label", fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_roc_curve(
    true_labels: np.ndarray,
    pred_probs:  np.ndarray,
    class_names: List[str],
    save_path:   str,
    run_name:    str,
) -> float:
    """
    Plot ROC curve and return AUC.
    For binary classification uses the probability of class 1 (index 1).
    For multi-class uses macro-OvR.
    """
    n_classes = pred_probs.shape[1]
    fig, ax   = plt.subplots(figsize=(6, 5))

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(true_labels, pred_probs[:, 1])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2,
                label=f"{class_names[1]} (AUC = {roc_auc:.3f})")
    else:
        roc_auc = roc_auc_score(true_labels, pred_probs,
                                multi_class="ovr", average="macro")
        for i, name in enumerate(class_names):
            bin_labels = (true_labels == i).astype(int)
            fpr, tpr, _ = roc_curve(bin_labels, pred_probs[:, i])
            ax.plot(fpr, tpr, lw=1.5, label=f"{name} (AUC={auc(fpr,tpr):.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"ROC Curve — {run_name}", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return float(roc_auc)


def _per_ripeness_breakdown(
    true_cls:      np.ndarray,
    pred_cls:      np.ndarray,
    true_rip:      np.ndarray,
    ripeness_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute accuracy and macro-F1 for disease classification within each
    ripeness stage.  Returns {stage_name: {accuracy, f1}}.
    """
    results = {}
    for idx, name in enumerate(ripeness_names):
        mask = true_rip == idx
        if mask.sum() == 0:
            results[name] = {"accuracy": float("nan"), "f1_macro": float("nan"),
                             "n_samples": 0}
            continue
        acc = accuracy_score(true_cls[mask], pred_cls[mask])
        f1  = f1_score(true_cls[mask], pred_cls[mask],
                       average="macro", zero_division=0)
        results[name] = {
            "accuracy":  float(acc),
            "f1_macro":  float(f1),
            "n_samples": int(mask.sum()),
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Public: evaluate
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model,
    test_loader:    DataLoader,
    output_dir:     str,
    run_name:       str,
    label_mode:     str,
    ripeness_names: List[str],
    class_names:    List[str],
    wavelengths:    Optional[np.ndarray] = None,
) -> Dict:
    """
    Run evaluation on test_loader and save all artifacts.

    Parameters
    ----------
    model          : trained nn.Module
    test_loader    : DataLoader (same label_mode as training)
    output_dir     : directory for saved files
    run_name       : prefix for all output filenames
    label_mode     : "binary" or "multi_task"
    ripeness_names : list of ripeness stage names (e.g. ["Ripe", ...])
    class_names    : list of disease class names  (e.g. ["Infested", "Healthy"])
    wavelengths    : optional 1-D array of band wavelengths (unused in plots
                     currently; reserved for spectral attribution extensions)

    Returns
    -------
    dict of all scalar metrics (same content as *_summary.json)
    """
    os.makedirs(output_dir, exist_ok=True)

    device = next(model.parameters()).device
    # If DataParallel, unwrap for clean attribute access but keep on device
    bare_model = model.module if isinstance(model, nn.DataParallel) else model
    bare_model.eval()

    print(f"\n{'='*60}")
    print(f"  Evaluating  : {run_name}  [{label_mode}]")
    print(f"  Device      : {device}")
    print(f"{'='*60}")

    # ── Collect predictions ───────────────────────────────────────────────────
    preds = _collect_predictions(bare_model, test_loader, device, label_mode)

    true_cls  = preds["true_labels"]
    pred_cls  = preds["pred_labels"]
    pred_prob = preds["pred_probs"]

    # ── Overall metrics ───────────────────────────────────────────────────────
    acc        = accuracy_score(true_cls, pred_cls)
    precision_macro = precision_score(true_cls, pred_cls, average="macro", zero_division=0)
    precision_weighted = precision_score(true_cls, pred_cls, average="weighted", zero_division=0)
    precision_micro = precision_score(true_cls, pred_cls, average="micro", zero_division=0)
    recall_macro = recall_score(true_cls, pred_cls, average="macro", zero_division=0)
    recall_weighted = recall_score(true_cls, pred_cls, average="weighted", zero_division=0)
    recall_micro = recall_score(true_cls, pred_cls, average="micro", zero_division=0)
    f1_macro   = f1_score(true_cls, pred_cls, average="macro",    zero_division=0)
    f1_weighted= f1_score(true_cls, pred_cls, average="weighted", zero_division=0)
    cls_precision, cls_recall, cls_f1, cls_support = precision_recall_fscore_support(
        true_cls,
        pred_cls,
        labels=list(range(len(class_names))),
        zero_division=0,
    )

    # AUC
    n_classes = pred_prob.shape[1]
    try:
        if n_classes == 2:
            roc_auc = roc_auc_score(true_cls, pred_prob[:, 1])
        else:
            roc_auc = roc_auc_score(true_cls, pred_prob,
                                    multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = float("nan")

    print(f"\n  Overall accuracy : {acc:.4f}")
    print(f"  Precision macro  : {precision_macro:.4f}")
    print(f"  Recall macro     : {recall_macro:.4f}")
    print(f"  F1 macro         : {f1_macro:.4f}")
    print(f"  F1 weighted      : {f1_weighted:.4f}")
    print(f"  ROC-AUC          : {roc_auc:.4f}")
    print()
    print(classification_report(true_cls, pred_cls,
                                 target_names=class_names, zero_division=0))

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(true_cls, pred_cls)
    cm_path = os.path.join(output_dir, f"{run_name}_confusion_matrix.png")
    _plot_confusion_matrix(cm, class_names, cm_path,
                           title=f"Confusion Matrix — {run_name}")
    print(f"  Saved: {cm_path}")

    # ── ROC curve ─────────────────────────────────────────────────────────────
    roc_path = os.path.join(output_dir, f"{run_name}_roc_curve.png")
    roc_auc  = _plot_roc_curve(true_cls, pred_prob, class_names, roc_path, run_name)
    print(f"  Saved: {roc_path}")

    # ── Per-ripeness breakdown ────────────────────────────────────────────────
    rip_breakdown: Dict = {}
    if label_mode == "multi_task" and "rip_true" in preds:
        true_rip = preds["rip_true"]
        rip_breakdown = _per_ripeness_breakdown(
            true_cls, pred_cls, true_rip, ripeness_names
        )
        print("\n  Per-ripeness-stage disease classification:")
        for stage, m in rip_breakdown.items():
            print(f"    {stage:<20s}  acc={m['accuracy']:.4f}  "
                  f"f1_macro={m['f1_macro']:.4f}  n={m['n_samples']}")

    # ── Ripeness classification accuracy (multi_task) ─────────────────────────
    rip_acc: Optional[float] = None
    if label_mode == "multi_task" and "rip_true" in preds and "rip_pred" in preds:
        rip_true = preds["rip_true"]
        rip_pred = preds["rip_pred"]
        valid    = rip_pred != -1
        if valid.sum() > 0:
            rip_acc = float(accuracy_score(rip_true[valid], rip_pred[valid]))
            print(f"\n  Ripeness classification accuracy: {rip_acc:.4f}")

    # ── Results CSV (one row per sample) ─────────────────────────────────────
    N = len(true_cls)
    # Build columns
    rows: Dict[str, list] = {
        "true_label": true_cls.tolist(),
        "pred_label": pred_cls.tolist(),
    }
    # Add per-class prob columns
    for i, cname in enumerate(class_names):
        safe = cname.replace(" ", "_")
        rows[f"prob_{safe}"] = pred_prob[:, i].tolist()

    # Scalar pred_prob (probability of positive / class-1)
    if n_classes == 2:
        rows["pred_prob"] = pred_prob[:, 1].tolist()
    else:
        rows["pred_prob"] = pred_prob.max(axis=1).tolist()

    # Ripeness columns
    if label_mode == "multi_task" and "rip_true" in preds:
        rows["ripeness_true"] = [
            (ripeness_names[r] if 0 <= r < len(ripeness_names) else str(r))
            for r in preds["rip_true"].tolist()
        ]
    else:
        rows["ripeness_true"] = ["N/A"] * N

    if label_mode == "multi_task" and "rip_pred" in preds:
        rows["ripeness_pred"] = [
            (ripeness_names[r] if 0 <= r < len(ripeness_names) else str(r))
            for r in preds["rip_pred"].tolist()
        ]
    else:
        rows["ripeness_pred"] = ["N/A"] * N

    df_results = pd.DataFrame(rows)
    csv_path   = os.path.join(output_dir, f"{run_name}_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary: Dict = {
        "run_name":       run_name,
        "label_mode":     label_mode,
        "n_test_samples": N,
        "accuracy":       float(acc),
        "precision_macro": float(precision_macro),
        "precision_weighted": float(precision_weighted),
        "precision_micro": float(precision_micro),
        "recall_macro": float(recall_macro),
        "recall_weighted": float(recall_weighted),
        "recall_micro": float(recall_micro),
        "f1_macro":       float(f1_macro),
        "f1_weighted":    float(f1_weighted),
        "roc_auc":        float(roc_auc) if not np.isnan(roc_auc) else None,
        "per_class": {
            class_names[i]: {
                "n_true": int((true_cls == i).sum()),
                "n_pred": int((pred_cls == i).sum()),
                "precision": float(cls_precision[i]),
                "recall": float(cls_recall[i]),
                "f1": float(cls_f1[i]),
                "support": int(cls_support[i]),
            }
            for i in range(len(class_names))
        },
        "per_ripeness": rip_breakdown,
    }
    if rip_acc is not None:
        summary["ripeness_accuracy"] = rip_acc

    json_path = os.path.join(output_dir, f"{run_name}_summary.json")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Saved: {json_path}")

    print(f"\nEvaluation complete — {run_name}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Public: compare_runs
# ─────────────────────────────────────────────────────────────────────────────

def compare_runs(results_dir: str, output_dir: str) -> pd.DataFrame:
    """
    Read all *_summary.json files in results_dir, build a comparison table,
    and save comparison_table.xlsx and comparison_table.png.

    Parameters
    ----------
    results_dir : directory containing *_summary.json files
    output_dir  : directory where comparison outputs are saved

    Returns
    -------
    pd.DataFrame — one row per run, columns = scalar metrics
    """
    os.makedirs(output_dir, exist_ok=True)

    json_files = sorted(
        p for p in os.listdir(results_dir)
        if p.endswith("_summary.json")
    )
    if not json_files:
        print(f"  compare_runs: no *_summary.json files found in {results_dir}")
        return pd.DataFrame()

    records = []
    for fname in json_files:
        fpath = os.path.join(results_dir, fname)
        with open(fpath) as fh:
            data = json.load(fh)
        row = {
            "run_name":        data.get("run_name",       fname.replace("_summary.json", "")),
            "label_mode":      data.get("label_mode",     ""),
            "n_test_samples":  data.get("n_test_samples", None),
            "accuracy":        data.get("accuracy",       None),
            "precision_macro": data.get("precision_macro", None),
            "precision_weighted": data.get("precision_weighted", None),
            "recall_macro":    data.get("recall_macro",    None),
            "recall_weighted": data.get("recall_weighted", None),
            "f1_macro":        data.get("f1_macro",       None),
            "f1_weighted":     data.get("f1_weighted",    None),
            "roc_auc":         data.get("roc_auc",        None),
        }
        if "ripeness_accuracy" in data:
            row["ripeness_accuracy"] = data["ripeness_accuracy"]
        records.append(row)

    df = pd.DataFrame(records)

    # ── Save XLSX ─────────────────────────────────────────────────────────────
    xlsx_path = os.path.join(output_dir, "comparison_table.xlsx")
    df.to_excel(xlsx_path, index=False)
    print(f"  Saved: {xlsx_path}")

    # ── Bar chart of F1 scores ────────────────────────────────────────────────
    png_path = os.path.join(output_dir, "comparison_table.png")

    # Determine which F1 columns are present and non-null
    f1_cols = [c for c in ["f1_macro", "f1_weighted"] if c in df.columns]
    plot_df = df[["run_name"] + f1_cols].dropna(subset=f1_cols, how="all")

    if plot_df.empty:
        print("  compare_runs: no F1 data to plot.")
    else:
        n_runs = len(plot_df)
        n_cols = len(f1_cols)
        x      = np.arange(n_runs)
        width  = 0.35

        fig, ax = plt.subplots(figsize=(max(8, n_runs * 1.2 + 2), 5))

        colors = ["#2a6fa8", "#e07b2e"]
        for k, col in enumerate(f1_cols):
            vals = plot_df[col].fillna(0).values
            offset = (k - (n_cols - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width, label=col, color=colors[k % len(colors)],
                          edgecolor="white", linewidth=0.6)
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.005,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["run_name"].tolist(),
                           rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("F1 Score", fontsize=11)
        ax.set_title("Model Comparison — F1 Scores", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {png_path}")

    print(f"\nComparison complete — {len(df)} run(s) compared.")
    print(df.to_string(index=False))
    return df
