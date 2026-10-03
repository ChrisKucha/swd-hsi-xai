"""
transforms.py — Spectral and spatial augmentations for HSI cell tensors.

All transforms operate on a (B, H, W) float32 torch.Tensor where
B = spectral bands, H = cell height, W = cell width.

Usage
-----
    from torchvision import transforms as T
    from data.transforms import SpectralJitter, SpectralDropout, RandomSpatialFlip

    train_transform = T.Compose([
        SpectralJitter(sigma=0.01),
        SpectralDropout(drop_prob=0.05),
        RandomSpatialFlip(),
    ])
"""

import random
import torch
import torch.nn.functional as F


class SpectralJitter:
    """
    Add zero-mean Gaussian noise to each band independently.

    Simulates sensor noise and slight calibration drift between days.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian noise (relative to the data scale).
        A value of 0.01 adds ~1% noise — suitable for z-score normalized cubes.
    """

    def __init__(self, sigma: float = 0.01):
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.sigma
        return x + noise


class SpectralDropout:
    """
    Randomly zero out entire spectral bands.

    Encourages the model to not over-rely on any single wavelength and
    improves robustness to band-specific calibration artifacts.

    Parameters
    ----------
    drop_prob : float
        Probability of zeroing each band independently (default 0.05 = 5%).
    """

    def __init__(self, drop_prob: float = 0.05):
        self.drop_prob = drop_prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        mask = torch.bernoulli(
            torch.full((B, 1, 1), 1.0 - self.drop_prob)
        )
        return x * mask


class RandomSpatialFlip:
    """
    Randomly flip the cell horizontally and/or vertically.

    Blueberries are approximately round so flipping is a valid augmentation.

    Parameters
    ----------
    h_prob : probability of horizontal flip (default 0.5)
    v_prob : probability of vertical flip   (default 0.5)
    """

    def __init__(self, h_prob: float = 0.5, v_prob: float = 0.5):
        self.h_prob = h_prob
        self.v_prob = v_prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.h_prob:
            x = torch.flip(x, dims=[2])   # flip W
        if random.random() < self.v_prob:
            x = torch.flip(x, dims=[1])   # flip H
        return x


class RandomRotation90:
    """
    Randomly rotate the spatial dimensions by 0°, 90°, 180°, or 270°.

    Parameters
    ----------
    prob : probability of applying any rotation (default 0.5)
    """

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.prob:
            k = random.randint(1, 3)
            x = torch.rot90(x, k=k, dims=[1, 2])
        return x


class SpectralSmoothing:
    """
    Apply a mild 1-D Gaussian blur along the spectral axis.

    Simulates slight spectral resolution differences between cameras or
    calibration sessions, improving cross-day generalization.

    Parameters
    ----------
    prob        : probability of applying smoothing (default 0.3)
    kernel_size : spectral kernel width (must be odd, default 5)
    sigma       : Gaussian sigma (default 1.0)
    """

    def __init__(self, prob: float = 0.3, kernel_size: int = 5, sigma: float = 1.0):
        self.prob        = prob
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        # Pre-build Gaussian kernel
        half = kernel_size // 2
        kx   = torch.arange(-half, half + 1, dtype=torch.float32)
        kern = torch.exp(-0.5 * (kx / sigma) ** 2)
        kern = kern / kern.sum()
        self._kernel = kern.view(1, 1, -1)   # (1, 1, K)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.prob:
            return x
        B, H, W = x.shape
        # Apply along spectral axis: treat spatial as batch
        flat  = x.permute(1, 2, 0).reshape(-1, 1, B)   # (H*W, 1, B)
        kern  = self._kernel.to(x.device)
        pad   = self.kernel_size // 2
        flat  = F.conv1d(flat, kern, padding=pad)
        return flat.reshape(H, W, B).permute(2, 0, 1)


class RandomCrop:
    """
    Random spatial crop — pads back to original size.

    Useful when berries do not fill the entire cell and the model should
    be invariant to slight spatial position shifts.

    Parameters
    ----------
    crop_frac : fraction of each spatial dimension to crop away (default 0.1)
    prob      : probability of applying the crop (default 0.3)
    """

    def __init__(self, crop_frac: float = 0.10, prob: float = 0.3):
        self.crop_frac = crop_frac
        self.prob      = prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.prob:
            return x
        B, H, W = x.shape
        dh = max(1, int(H * self.crop_frac))
        dw = max(1, int(W * self.crop_frac))
        top  = random.randint(0, dh)
        left = random.randint(0, dw)
        cropped = x[:, top:H - dh + top, left:W - dw + left]
        # Pad back to original size
        pad_h = H - cropped.shape[1]
        pad_w = W - cropped.shape[2]
        cropped = F.pad(cropped, (0, pad_w, 0, pad_h), mode="reflect")
        return cropped


class Normalize:
    """
    Standardize a tensor to zero mean and unit std.

    Applied per-sample (not per-dataset) so it is safe to use on top of
    per-board normalization for additional stability.

    Parameters
    ----------
    eps : small constant to avoid division by zero
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mu  = x.mean()
        std = x.std()
        return (x - mu) / (std + self.eps)


# ── Preset transform pipelines ────────────────────────────────────────────────

def get_train_transform(spectral_sigma: float = 0.01) -> "torch.nn.Module":
    """Standard training augmentation pipeline.

    Note: RandomRotation90 is intentionally excluded.  When cells are not
    square (e.g. 105×83), 90° / 270° rotations swap H and W and produce
    inconsistent tensor shapes within a batch, crashing the collator.
    RandomSpatialFlip already covers the relevant rotational invariance for
    approximately-round berries.
    """
    from torchvision import transforms as T
    return T.Compose([
        SpectralJitter(sigma=spectral_sigma),
        SpectralDropout(drop_prob=0.05),
        RandomSpatialFlip(h_prob=0.5, v_prob=0.5),
        SpectralSmoothing(prob=0.2, kernel_size=5, sigma=1.0),
    ])


def get_val_transform() -> None:
    """No augmentation for validation / test."""
    return None
