"""Sub-pixel band-to-band registration via phase correlation.

SuperDove acquires each band on a different detector row, so consecutive bands
see the same ground point a few milliseconds apart. After the standard L1B
geometric model is applied there is still a small residual offset — typically
a fraction of a pixel — that we want to either correct or exploit.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import shift as nd_shift
from skimage.registration import phase_cross_correlation


def estimate_offset(
    reference: np.ndarray,
    moving: np.ndarray,
    upsample_factor: int = 20,
) -> tuple[float, float]:
    """Return (dy, dx) sub-pixel offset that aligns ``moving`` to ``reference``.

    Inputs are 2-D float arrays of the same shape. Positive ``dy`` means
    ``moving`` needs to be shifted down to align with ``reference``.
    """
    if reference.shape != moving.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {moving.shape}")
    shift_yx, _, _ = phase_cross_correlation(
        reference, moving, upsample_factor=upsample_factor, normalization=None
    )
    return float(shift_yx[0]), float(shift_yx[1])


def apply_offset(arr: np.ndarray, dy: float, dx: float, order: int = 3) -> np.ndarray:
    """Shift a 2-D array by sub-pixel (dy, dx) using cubic spline interpolation."""
    return nd_shift(arr, shift=(dy, dx), order=order, mode="reflect")


def coarse_align(
    pair: np.ndarray,
    upsample_factor: int = 20,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Align ``pair[1]`` to ``pair[0]`` to the nearest pixel.

    Returns the aligned (2, H, W) array and the residual sub-pixel offset
    (dy, dx) that remains. The residual is bounded to roughly ±0.5 pixels and
    is the bit the network learns to exploit — don't throw it away, log it as
    conditioning if you want a more capable model later.
    """
    dy, dx = estimate_offset(pair[0], pair[1], upsample_factor=upsample_factor)
    int_dy, int_dx = round(dy), round(dx)
    residual_dy, residual_dx = dy - int_dy, dx - int_dx
    if int_dy == 0 and int_dx == 0:
        aligned = pair.copy()
    else:
        shifted = np.roll(pair[1], shift=(int_dy, int_dx), axis=(0, 1))
        aligned = np.stack([pair[0], shifted], axis=0)
    return aligned, (residual_dy, residual_dx)
