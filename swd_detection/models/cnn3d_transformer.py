"""
cnn3d_transformer.py
--------------------
3D-CNN + Spectral Transformer hybrid model for hyperspectral SWD classification.

Architecture overview
    1. Three-block 3D-CNN backbone (same design as cnn3d.py) extracts
       volumetric features → (B, 128, D', H', W').
    2. Each spatial-spectral voxel becomes a token of dimension 128.
       A learnable CLS token is prepended.
    3. A standard Transformer encoder (n_layers=4, n_heads=8) processes
       the token sequence.
    4. The CLS token output feeds the classification head(s).

Supported label modes
    "binary"     – returns cls_logits  shape (B_batch, num_classes)
    "multi_task" – returns (cls_logits, rip_logits)
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Re-use the same conv-block helper
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
# Shared CNN backbone (no FC layers)
# ---------------------------------------------------------------------------

class CNN3DBackbone(nn.Module):
    """
    3-block 3D-CNN feature extractor.
    Output shape: (B, 128, D', H', W')
    where D'/H'/W' depend on input size after pooling and AdaptiveAvgPool3d(4,4,4).
    """

    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            ConvBlock3D(1, 32, kernel=(7, 3, 3), padding=(3, 1, 1)),
            nn.MaxPool3d(kernel_size=(2, 1, 1)),
        )
        self.block2 = nn.Sequential(
            ConvBlock3D(32, 64, kernel=(5, 3, 3), padding=(2, 1, 1)),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
        )
        self.block3 = nn.Sequential(
            ConvBlock3D(64, 128, kernel=(3, 3, 3), padding=(1, 1, 1)),
            nn.AdaptiveAvgPool3d((4, 4, 4)),   # fixed 4×4×4 → 64 voxels
        )

    def forward(self, x):
        # x: (B, 1, n_bands, H, W)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)   # (B, 128, 4, 4, 4)
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class CNN3DTransformer(nn.Module):
    """
    3D-CNN + Spectral Transformer for hyperspectral SWD classification.

    Parameters
    ----------
    n_bands     : number of spectral bands
    cell_h      : spatial height of cell patch
    cell_w      : spatial width  of cell patch
    num_classes : number of disease output classes (default 2)
    n_layers    : Transformer encoder depth (default 4)
    n_heads     : number of attention heads (default 8)
    dropout     : dropout for Transformer and classifier head (default 0.5)
    label_mode  : "binary" | "multi_task"
    """

    # d_model is fixed to 128 to match backbone output channels
    D_MODEL = 128

    def __init__(
        self,
        n_bands: int,
        cell_h: int,
        cell_w: int,
        num_classes: int = 2,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.5,
        label_mode: str = "binary",
    ):
        super().__init__()

        assert label_mode in ("binary", "multi_task"), (
            f"label_mode must be 'binary' or 'multi_task', got {label_mode!r}"
        )
        self.label_mode = label_mode
        d_model = self.D_MODEL

        # ---- CNN backbone ------------------------------------------------
        self.backbone = CNN3DBackbone()

        # Number of voxel tokens after AdaptiveAvgPool3d(4,4,4)
        n_voxels = 4 * 4 * 4  # = 64

        # ---- CLS token and positional embedding --------------------------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # +1 for the CLS token
        self.pos_embed = nn.Parameter(torch.zeros(1, n_voxels + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ---- Transformer encoder -----------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=256,
            dropout=0.1,         # internal Transformer dropout
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ---- Classification heads ----------------------------------------
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

        self.head_cls = nn.Linear(d_model, num_classes)

        if label_mode == "multi_task":
            self.head_rip = nn.Linear(d_model, 3)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B_batch, n_bands, H_cell, W_cell)

        Returns
        -------
        binary     : cls_logits  (B, num_classes)
        multi_task : (cls_logits, rip_logits)
        """
        B = x.size(0)

        # --- CNN backbone ---
        x = x.unsqueeze(1)          # (B, 1, n_bands, H, W)
        feat = self.backbone(x)     # (B, 128, 4, 4, 4)

        # --- Tokenise voxels ---
        # (B, 128, 4, 4, 4) → (B, 64, 128)
        tokens = feat.flatten(2).permute(0, 2, 1)  # (B, N_voxels, 128)

        # --- Prepend CLS token ---
        cls = self.cls_token.expand(B, -1, -1)     # (B, 1, 128)
        tokens = torch.cat([cls, tokens], dim=1)   # (B, 65, 128)

        # --- Add positional embedding ---
        tokens = tokens + self.pos_embed           # (B, 65, 128)

        # --- Transformer encoder ---
        encoded = self.transformer(tokens)         # (B, 65, 128)
        encoded = self.norm(encoded)

        # --- Extract CLS token ---
        cls_out = encoded[:, 0, :]                 # (B, 128)
        cls_out = self.dropout(cls_out)

        cls_logits = self.head_cls(cls_out)

        if self.label_mode == "multi_task":
            rip_logits = self.head_rip(cls_out)
            return cls_logits, rip_logits

        return cls_logits


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def model_info(n_bands: int = 224, cell_h: int = 33, cell_w: int = 30):
    """Print parameter count and a sample forward-pass output shape."""
    model = CNN3DTransformer(
        n_bands=n_bands, cell_h=cell_h, cell_w=cell_w, label_mode="binary"
    )
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CNN3DTransformer  |  total params: {total:,}  |  trainable: {trainable:,}")

    dummy = torch.zeros(2, n_bands, cell_h, cell_w)
    with torch.no_grad():
        out = model(dummy)
    print(f"  input shape : {tuple(dummy.shape)}")
    print(f"  output shape: {tuple(out.shape)}")

    model_mt = CNN3DTransformer(
        n_bands=n_bands, cell_h=cell_h, cell_w=cell_w, label_mode="multi_task"
    )
    model_mt.eval()
    with torch.no_grad():
        cls_out, rip_out = model_mt(dummy)
    print(f"  multi-task cls  shape: {tuple(cls_out.shape)}")
    print(f"  multi-task rip  shape: {tuple(rip_out.shape)}")


if __name__ == "__main__":
    model_info()
