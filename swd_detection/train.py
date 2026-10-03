"""
train.py — Unified training loop for the SWD blueberry HSI pipeline.

Public API
----------
    from swd_detection.train import train

    result = train(
        model      = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        config_dict  = {
            "num_epochs":              100,
            "learning_rate":           1e-4,
            "weight_decay":            1e-4,
            "dropout":                 0.5,
            "early_stop_patience":     15,
            "lr_patience":             7,
            "use_amp":                 True,
            "label_smoothing":         0.05,
            "label_mode":              "binary",   # or "multi_task"
            "vit_ripeness_loss_weight": 0.3,
        },
        output_dir = "/path/to/outputs",
        run_name   = "nir_cnn3d_binary",
    )

config_dict keys
----------------
num_epochs              : int   — maximum training epochs
learning_rate           : float — AdamW base LR
weight_decay            : float — AdamW weight decay
dropout                 : float — stored for reference; applied inside model
early_stop_patience     : int   — epochs without val-loss improvement → stop
lr_patience             : int   — epochs before ReduceLROnPlateau fires
use_amp                 : bool  — mixed-precision (requires CUDA)
label_smoothing         : float — applied to CrossEntropyLoss (binary mode)
label_mode              : str   — "binary" or "multi_task"
vit_ripeness_loss_weight: float — weight for auxiliary ripeness loss
                                  (multi_task mode only)

Returns
-------
dict with keys:
    best_val_loss   : float
    best_val_acc    : float
    epochs_trained  : int
    history         : list[dict]  — one dict per epoch with
                      epoch, train_loss, val_loss, val_acc, lr
"""

import os
import csv
import copy
import time
import warnings
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    # Graceful fallback: tqdm becomes a no-op pass-through
    def tqdm(iterable, **kwargs):
        return iterable


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_class_weights(loader: DataLoader, device: torch.device) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from a DataLoader whose dataset
    has a .class_weights() method (BlueberryDataset / FusedBlueberryDataset).
    Falls back to uniform weights if the method is absent.
    """
    ds = loader.dataset
    if hasattr(ds, "class_weights"):
        return ds.class_weights().to(device)
    return None


def _get_ripeness_weights(loader: DataLoader, device: torch.device) -> torch.Tensor:
    ds = loader.dataset
    if hasattr(ds, "ripeness_weights"):
        return ds.ripeness_weights().to(device)
    return None


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def _save_history_csv(history: List[Dict], output_dir: str, run_name: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{run_name}_history.csv")
    if not history:
        return
    fieldnames = list(history[0].keys())
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    config_dict:  Dict[str, Any],
    output_dir:   str,
    run_name:     str,
) -> Dict[str, Any]:
    """
    Unified training loop.

    Parameters
    ----------
    model        : nn.Module — must be on CPU; this function moves it to device
    train_loader : DataLoader
    val_loader   : DataLoader
    config_dict  : see module docstring for key descriptions
    output_dir   : directory where checkpoints and CSV are saved
    run_name     : prefix for all saved files

    Returns
    -------
    dict — best_val_loss, best_val_acc, epochs_trained, history
    """
    # ── Unpack config ─────────────────────────────────────────────────────────
    num_epochs        = int(config_dict.get("num_epochs", 100))
    learning_rate     = float(config_dict.get("learning_rate", 1e-4))
    weight_decay      = float(config_dict.get("weight_decay", 1e-4))
    early_stop_pat    = int(config_dict.get("early_stop_patience", 15))
    lr_patience       = int(config_dict.get("lr_patience", 7))
    use_amp           = bool(config_dict.get("use_amp", True))
    label_smoothing   = float(config_dict.get("label_smoothing", 0.0))
    label_mode        = str(config_dict.get("label_mode", "binary"))
    rip_loss_weight   = float(config_dict.get("vit_ripeness_loss_weight", 0.3))
    max_grad_norm     = float(config_dict.get("max_grad_norm", 1.0))
    resume_existing   = bool(config_dict.get("resume_existing", False))

    if label_mode not in ("binary", "multi_task"):
        raise ValueError(f"label_mode must be 'binary' or 'multi_task', got {label_mode!r}")

    os.makedirs(output_dir, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    cuda_available = torch.cuda.is_available()
    device    = torch.device("cuda" if cuda_available else "cpu")
    n_gpus    = torch.cuda.device_count() if cuda_available else 0
    use_amp   = use_amp and device.type == "cuda"

    print(f"\n{'='*60}")
    print(f"  Run   : {run_name}")
    print(f"  Mode  : {label_mode}")
    print(f"  Device: {device}  ({n_gpus} GPU(s) found)")
    print(f"  AMP   : {use_amp}")
    print(f"{'='*60}")

    # ── Multi-GPU ─────────────────────────────────────────────────────────────
    if device.type == "cuda" and n_gpus > 1:
        print(f"  Wrapping model in DataParallel across {n_gpus} GPUs.")
        model = nn.DataParallel(model)
    model = model.to(device)

    # ── Loss functions ────────────────────────────────────────────────────────
    cls_weights = _get_class_weights(train_loader, device)
    rip_weights = _get_ripeness_weights(train_loader, device) if label_mode == "multi_task" else None

    if label_mode == "binary":
        criterion_cls = nn.CrossEntropyLoss(
            weight=cls_weights,
            label_smoothing=label_smoothing,
        )
    else:  # multi_task
        criterion_cls = nn.CrossEntropyLoss(
            weight=cls_weights,
            label_smoothing=label_smoothing,
        )
        criterion_rip = nn.CrossEntropyLoss(weight=rip_weights)

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=lr_patience,
        factor=0.5,
        min_lr=1e-7,
    )

    # ── Mixed precision scaler ────────────────────────────────────────────────
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── State ─────────────────────────────────────────────────────────────────
    best_val_loss   = float("inf")
    best_val_acc    = 0.0
    best_weights    = None
    no_improve_cnt  = 0
    start_epoch     = 1
    history: List[Dict] = []

    checkpoint_path = os.path.join(output_dir, f"{run_name}_best.pt")
    last_path       = os.path.join(output_dir, f"{run_name}_last.pt")

    def _bare_state_dict() -> Dict[str, torch.Tensor]:
        return (
            model.module.state_dict()
            if isinstance(model, nn.DataParallel)
            else model.state_dict()
        )

    def _load_model_state(state_dict: Dict[str, torch.Tensor]) -> None:
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)

    def _save_last_checkpoint(epoch: int) -> None:
        payload = {
            "epoch": epoch,
            "model_state": _bare_state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "no_improve_cnt": no_improve_cnt,
            "history": history,
            "config": dict(config_dict),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }
        tmp_path = f"{last_path}.tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, last_path)

    if resume_existing and os.path.exists(last_path):
        last = torch.load(last_path, map_location=device)
        _load_model_state(last["model_state"])
        optimizer.load_state_dict(last["optimizer_state"])
        scheduler.load_state_dict(last["scheduler_state"])
        if "scaler_state" in last:
            scaler.load_state_dict(last["scaler_state"])
        best_val_loss  = float(last.get("best_val_loss", best_val_loss))
        best_val_acc   = float(last.get("best_val_acc", best_val_acc))
        no_improve_cnt = int(last.get("no_improve_cnt", no_improve_cnt))
        history        = list(last.get("history", []))
        start_epoch    = int(last.get("epoch", 0)) + 1
        if "torch_rng_state" in last:
            try:
                torch.set_rng_state(last["torch_rng_state"].detach().cpu().to(torch.uint8))
            except Exception as exc:
                warnings.warn(
                    f"Could not restore CPU RNG state from {last_path}: {exc}. "
                    "Continuing checkpoint resume without RNG restoration.",
                    RuntimeWarning,
                )
        cuda_rng = last.get("cuda_rng_state_all")
        if cuda_rng is not None and torch.cuda.is_available():
            try:
                cuda_states = [
                    state.detach().cpu().to(torch.uint8)
                    for state in cuda_rng
                    if torch.is_tensor(state)
                ]
                if cuda_states:
                    torch.cuda.set_rng_state_all(cuda_states)
            except Exception as exc:
                warnings.warn(
                    f"Could not restore CUDA RNG state from {last_path}: {exc}. "
                    "Continuing checkpoint resume without CUDA RNG restoration.",
                    RuntimeWarning,
                )
        if os.path.exists(checkpoint_path):
            best_weights = torch.load(checkpoint_path, map_location=device)
        tqdm.write(
            f"  [RESUME] Loaded {last_path}; continuing at epoch "
            f"{start_epoch}/{num_epochs}."
        )

    if start_epoch > num_epochs:
        tqdm.write(
            f"  [RESUME] Last checkpoint already reached epoch {start_epoch - 1}; "
            "skipping training loop."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Epoch loop
    # ─────────────────────────────────────────────────────────────────────────
    epoch_bar = tqdm(range(start_epoch, num_epochs + 1), desc="Epochs", unit="ep",
                     dynamic_ncols=True, position=0, leave=True)

    for epoch in epoch_bar:
        t0 = time.time()

        # ── Training phase ───────────────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0
        train_batches  = 0

        train_bar = tqdm(train_loader, desc=f"  Train {epoch:>3d}", unit="batch",
                         leave=False, dynamic_ncols=True, position=1)
        for batch in train_bar:
            optimizer.zero_grad()

            if label_mode == "binary":
                x, cls_lbl = batch
                x       = x.to(device,       non_blocking=True)
                cls_lbl = cls_lbl.to(device,  non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    # Models may return (cls_logits, rip_logits) or just cls_logits
                    out = model(x)
                    if isinstance(out, (tuple, list)):
                        cls_logits = out[0]
                    else:
                        cls_logits = out
                    loss = criterion_cls(cls_logits, cls_lbl)

            else:  # multi_task
                x, cls_lbl, rip_lbl = batch
                x       = x.to(device,       non_blocking=True)
                cls_lbl = cls_lbl.to(device,  non_blocking=True)
                rip_lbl = rip_lbl.to(device,  non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(x)
                    if isinstance(out, (tuple, list)) and len(out) >= 2:
                        cls_logits, rip_logits = out[0], out[1]
                    else:
                        cls_logits = out[0] if isinstance(out, (tuple, list)) else out
                        rip_logits = None

                    loss_cls = criterion_cls(cls_logits, cls_lbl)
                    if rip_logits is not None:
                        loss_rip = criterion_rip(rip_logits, rip_lbl)
                        loss = (1.0 - rip_loss_weight) * loss_cls + rip_loss_weight * loss_rip
                    else:
                        loss = loss_cls

            scaler.scale(loss).backward()
            # Unscale before clipping so the norm is in fp32 units
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            train_batches  += 1
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss_sum / max(train_batches, 1)

        # ── Validation phase ─────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_correct  = 0
        val_total    = 0

        val_bar = tqdm(val_loader, desc=f"  Val   {epoch:>3d}", unit="batch",
                       leave=False, dynamic_ncols=True, position=1)
        with torch.no_grad():
            for batch in val_bar:
                if label_mode == "binary":
                    x, cls_lbl = batch
                    x       = x.to(device, non_blocking=True)
                    cls_lbl = cls_lbl.to(device, non_blocking=True)

                    with torch.amp.autocast("cuda", enabled=use_amp):
                        out = model(x)
                        cls_logits = out[0] if isinstance(out, (tuple, list)) else out
                        loss = criterion_cls(cls_logits, cls_lbl)

                else:  # multi_task
                    x, cls_lbl, rip_lbl = batch
                    x       = x.to(device, non_blocking=True)
                    cls_lbl = cls_lbl.to(device, non_blocking=True)
                    rip_lbl = rip_lbl.to(device, non_blocking=True)

                    with torch.amp.autocast("cuda", enabled=use_amp):
                        out = model(x)
                        if isinstance(out, (tuple, list)) and len(out) >= 2:
                            cls_logits, rip_logits = out[0], out[1]
                        else:
                            cls_logits = out[0] if isinstance(out, (tuple, list)) else out
                            rip_logits = None

                        loss_cls = criterion_cls(cls_logits, cls_lbl)
                        if rip_logits is not None:
                            loss_rip = criterion_rip(rip_logits, rip_lbl)
                            loss = (1.0 - rip_loss_weight) * loss_cls + rip_loss_weight * loss_rip
                        else:
                            loss = loss_cls

                val_loss_sum += loss.item() * cls_lbl.size(0)
                preds         = cls_logits.argmax(dim=1)
                val_correct  += (preds == cls_lbl).sum().item()
                val_total    += cls_lbl.size(0)

        avg_val_loss = val_loss_sum / max(val_total, 1)
        val_acc      = val_correct  / max(val_total, 1)

        # ── LR scheduler ─────────────────────────────────────────────────────
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Logging ──────────────────────────────────────────────────────────
        elapsed = time.time() - t0
        epoch_bar.set_postfix(
            tr_loss=f"{avg_train_loss:.4f}",
            vl_loss=f"{avg_val_loss:.4f}",
            vl_acc=f"{val_acc:.4f}",
            lr=f"{current_lr:.1e}",
        )
        tqdm.write(
            f"Epoch {epoch:>4d}/{num_epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={avg_val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"lr={current_lr:.2e} | "
            f"{elapsed:.1f}s"
        )

        epoch_record = {
            "epoch":      epoch,
            "train_loss": avg_train_loss,
            "val_loss":   avg_val_loss,
            "val_acc":    val_acc,
            "lr":         current_lr,
        }
        history.append(epoch_record)

        # ── Checkpointing ─────────────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss  = avg_val_loss
            best_val_acc   = val_acc
            # Unwrap DataParallel for saving
            state_dict = _bare_state_dict()
            best_weights = copy.deepcopy(state_dict)
            torch.save(state_dict, checkpoint_path)
            tqdm.write(f"  --> New best val_loss={best_val_loss:.4f}  (saved to {checkpoint_path})")
            no_improve_cnt = 0
        else:
            no_improve_cnt += 1

        _save_history_csv(history, output_dir, run_name)
        _save_last_checkpoint(epoch)

        # ── Early stopping ────────────────────────────────────────────────────
        if no_improve_cnt >= early_stop_pat:
            tqdm.write(f"\nEarly stopping: no improvement for {early_stop_pat} epochs.")
            break

    # ── Save training history CSV ─────────────────────────────────────────────
    _save_history_csv(history, output_dir, run_name)
    print(f"\nTraining complete.  Best val_loss={best_val_loss:.4f}  "
          f"val_acc={best_val_acc:.4f}  epochs={len(history)}")

    # ── Reload best weights into model ────────────────────────────────────────
    if best_weights is not None:
        _load_model_state(best_weights)

    return {
        "best_val_loss":  best_val_loss,
        "best_val_acc":   best_val_acc,
        "epochs_trained": len(history),
        "history":        history,
    }
