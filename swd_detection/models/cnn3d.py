"""
cnn3d.py
--------
3D-CNN baseline model for hyperspectral blueberry SWD (Spotted Wing Drosophila)
disease classification.

Input tensor shape  : (B_batch, n_bands, H_cell, W_cell)
                      A channel dimension is added in forward() → (B_batch, 1, n_bands, H, W)

Supported label modes
    "binary"     – returns cls_logits  shape (B_batch, num_classes)
    "multi_task" – returns (cls_logits, rip_logits)
                   rip_logits shape (B_batch, 3)  [Ripe / Midripe / Unripe]
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helper: conv block
# ---------------------------------------------------------------------------

class ConvBlock3D(nn.Module):
    """Conv3d → BatchNorm3d → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class CNN3D(nn.Module):
    """
    3-block 3D-CNN for hyperspectral SWD classification.

    Parameters
    ----------
    n_bands     : number of spectral bands (depth dimension)
    cell_h      : spatial height of each cell patch
    cell_w      : spatial width  of each cell patch
    num_classes : number of disease classes (default 2: Healthy / Infested)
    fc_dim      : hidden size of the fully-connected head
    dropout     : dropout probability before the final FC layer
    label_mode  : "binary" | "multi_task"
    """

    def __init__(
        self,
        n_bands: int,
        cell_h: int,
        cell_w: int,
        num_classes: int = 2,
        fc_dim: int = 256,
        dropout: float = 0.5,
        label_mode: str = "binary",
    ):
        super().__init__()

        assert label_mode in ("binary", "multi_task"), (
            f"label_mode must be 'binary' or 'multi_task', got {label_mode!r}"
        )
        self.label_mode = label_mode
        self.n_bands = n_bands
        self.cell_h = cell_h
        self.cell_w = cell_w

        # ------------------------------------------------------------------
        # Block 1: spectral kernel (7,3,3), pool only along spectral axis
        # ------------------------------------------------------------------
        self.block1 = nn.Sequential(
            ConvBlock3D(1, 32, kernel=(7, 3, 3), padding=(3, 1, 1)),
            nn.MaxPool3d(kernel_size=(2, 1, 1)),   # halve spectral depth
        )

        # ------------------------------------------------------------------
        # Block 2: mixed kernel, spatial + spectral pooling
        # ------------------------------------------------------------------
        self.block2 = nn.Sequential(
            ConvBlock3D(32, 64, kernel=(5, 3, 3), padding=(2, 1, 1)),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
        )

        # ------------------------------------------------------------------
        # Block 3: cubic kernel, adaptive pool to fixed 4×4×4 output
        # ------------------------------------------------------------------
        self.block3 = nn.Sequential(
            ConvBlock3D(64, 128, kernel=(3, 3, 3), padding=(1, 1, 1)),
            nn.AdaptiveAvgPool3d((4, 4, 4)),       # → (B, 128, 4, 4, 4)
        )

        # ------------------------------------------------------------------
        # Classification head
        # ------------------------------------------------------------------
        flat_dim = 128 * 4 * 4 * 4  # = 8192

        self.fc_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        self.head_cls = nn.Linear(fc_dim, num_classes)

        # Optional ripeness head (3 classes)
        if label_mode == "multi_task":
            self.head_rip = nn.Linear(fc_dim, 3)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B_batch, n_bands, H_cell, W_cell)

        Returns
        -------
        binary     : cls_logits  (B_batch, num_classes)
        multi_task : (cls_logits, rip_logits)  shapes (B, num_classes), (B, 3)
        """
        # Add channel dim → (B, 1, n_bands, H, W)
        x = x.unsqueeze(1)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)     # (B, 128, 4, 4, 4)

        features = self.fc_head(x)   # (B, fc_dim)

        cls_logits = self.head_cls(features)

        if self.label_mode == "multi_task":
            rip_logits = self.head_rip(features)
            return cls_logits, rip_logits

        return cls_logits


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def model_info(n_bands: int = 224, cell_h: int = 33, cell_w: int = 30):
    """Print parameter count and a sample forward-pass output shape."""
    model = CNN3D(n_bands=n_bands, cell_h=cell_h, cell_w=cell_w, label_mode="binary")
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CNN3D  |  total params: {total:,}  |  trainable: {trainable:,}")

    dummy = torch.zeros(2, n_bands, cell_h, cell_w)
    with torch.no_grad():
        out = model(dummy)
    print(f"  input shape : {tuple(dummy.shape)}")
    print(f"  output shape: {tuple(out.shape)}")

    # Also test multi-task
    model_mt = CNN3D(n_bands=n_bands, cell_h=cell_h, cell_w=cell_w, label_mode="multi_task")
    model_mt.eval()
    with torch.no_grad():
        cls_out, rip_out = model_mt(dummy)
    print(f"  multi-task cls  shape: {tuple(cls_out.shape)}")
    print(f"  multi-task rip  shape: {tuple(rip_out.shape)}")


if __name__ == "__main__":
    model_info()
