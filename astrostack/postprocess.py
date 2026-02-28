"""Post-processing utilities for linear masters (background, stretch, denoise, sharpening)."""

from __future__ import annotations

import itertools
from typing import Optional

import numpy as np
from skimage import exposure, filters, restoration


def to_luminance(image: np.ndarray) -> np.ndarray:
    """Convert mono/RGB image to luminance."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.shape[-1] < 3:
        return np.mean(arr, axis=-1)
    return (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]).astype(np.float32)


def _poly_terms(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    terms = []
    for i, j in itertools.product(range(order + 1), repeat=2):
        if i + j <= order:
            terms.append((x**i) * (y**j))
    return np.stack(terms, axis=1)


def _fit_background_surface(channel: np.ndarray, order: int = 2, sample_step: int = 64) -> np.ndarray:
    h, w = channel.shape
    xs: list[float] = []
    ys: list[float] = []
    vals: list[float] = []

    for y0 in range(0, h, sample_step):
        for x0 in range(0, w, sample_step):
            patch = channel[y0 : min(y0 + sample_step, h), x0 : min(x0 + sample_step, w)]
            pvals = patch[np.isfinite(patch)]
            if pvals.size < 10:
                continue
            # Low percentile sampling is robust to stars.
            vals.append(float(np.percentile(pvals, 20)))
            ys.append((y0 + patch.shape[0] / 2) / max(h - 1, 1))
            xs.append((x0 + patch.shape[1] / 2) / max(w - 1, 1))

    if len(vals) < 6:
        # Fallback to broad Gaussian model.
        smooth = filters.gaussian(channel, sigma=max(h, w) / 16, preserve_range=True)
        return smooth.astype(np.float32, copy=False)

    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    z = np.asarray(vals, dtype=np.float64)

    design = _poly_terms(x, y, order=order)
    coeffs, *_ = np.linalg.lstsq(design, z, rcond=None)

    grid_y, grid_x = np.mgrid[0:h, 0:w]
    grid_x = (grid_x.astype(np.float64) / max(w - 1, 1)).ravel()
    grid_y = (grid_y.astype(np.float64) / max(h - 1, 1)).ravel()
    grid_design = _poly_terms(grid_x, grid_y, order=order)
    model = (grid_design @ coeffs).reshape(h, w)
    return model.astype(np.float32, copy=False)


def remove_background(
    image: np.ndarray,
    order: int = 2,
    sample_step: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract polynomial background model while preserving overall median level."""
    arr = np.asarray(image, dtype=np.float32)

    if arr.ndim == 2:
        model = _fit_background_surface(arr, order=order, sample_step=sample_step)
        corrected = arr - model + float(np.median(model))
        return corrected.astype(np.float32, copy=False), model.astype(np.float32, copy=False)

    models = []
    corrected = arr.copy()
    for c in range(arr.shape[-1]):
        model_c = _fit_background_surface(arr[..., c], order=order, sample_step=sample_step)
        corrected[..., c] = arr[..., c] - model_c + float(np.median(model_c))
        models.append(model_c)
    return corrected.astype(np.float32, copy=False), np.stack(models, axis=-1).astype(np.float32, copy=False)


def color_calibrate_rgb(image: np.ndarray) -> np.ndarray:
    """Simple but robust RGB background neutralization and white balancing."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return arr

    out = arr.copy()

    # Background neutralization per channel.
    for c in range(3):
        bg = float(np.percentile(out[..., c], 10))
        out[..., c] = np.clip(out[..., c] - bg, 0.0, None)

    lum = to_luminance(out)
    threshold = np.percentile(lum, 90)
    star_mask = lum > threshold

    if np.sum(star_mask) < 50:
        channel_means = np.array([np.mean(out[..., c]) for c in range(3)], dtype=np.float32)
    else:
        channel_means = np.array([np.mean(out[..., c][star_mask]) for c in range(3)], dtype=np.float32)

    target = float(np.mean(channel_means))
    scales = target / np.maximum(channel_means, 1e-6)
    for c in range(3):
        out[..., c] *= scales[c]

    return out.astype(np.float32, copy=False)


def auto_stretch_params(image: np.ndarray) -> tuple[float, float]:
    """Estimate black point and white point from robust percentiles."""
    arr = np.asarray(image, dtype=np.float32)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return 0.0, 1.0

    black = float(np.percentile(valid, 0.5))
    white = float(np.percentile(valid, 99.8))
    if white <= black:
        white = black + 1e-6
    return black, white


def _midtone_gamma(midtone: float) -> float:
    mt = np.clip(midtone, 0.05, 0.95)
    return float(np.log(0.5) / np.log(mt))


def _normalize(image: np.ndarray, black: float, white: float) -> np.ndarray:
    span = max(white - black, 1e-6)
    return np.clip((image - black) / span, 0.0, 1.0).astype(np.float32, copy=False)


def stretch_image(
    image: np.ndarray,
    method: str = "asinh",
    black_point: Optional[float] = None,
    midtone: float = 0.25,
    auto: bool = True,
) -> np.ndarray:
    """Stretch linear image data into a display-ready [0,1] range."""
    arr = np.asarray(image, dtype=np.float32)

    if auto or black_point is None:
        black, white = auto_stretch_params(arr)
    else:
        black = float(black_point)
        white = float(np.percentile(arr[np.isfinite(arr)], 99.8))
        if white <= black:
            white = black + 1e-6

    norm = _normalize(arr, black=black, white=white)
    gamma = _midtone_gamma(midtone)
    norm = np.clip(norm**gamma, 0.0, 1.0)

    m = method.lower().strip()
    if m in "asinh", "arcsinh":
        stretch_factor = 8.0 if m == "asinh" else 5.0
        stretched = np.arcsinh(stretch_factor * norm) / np.arcsinh(stretch_factor)
    elif m in "histogram", "histogram stretch":
        if norm.ndim == 2:
            stretched = exposure.equalize_adapthist(norm, clip_limit=0.01)
        else:
            stretched = np.zeros_like(norm)
            for c in range(norm.shape[-1]):
                stretched[..., c] = exposure.equalize_adapthist(norm[..., c], clip_limit=0.01)
    else:
        stretched = norm

    return np.clip(stretched, 0.0, 1.0).astype(np.float32, copy=False)


def denoise_image(image: np.ndarray, strength: float = 0.15) -> np.ndarray:
    """Apply conservative wavelet denoising and blend with original image."""
    arr = np.asarray(image, dtype=np.float32)
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 0:
        return arr

    denoised = restoration.denoise_wavelet(
        arr,
        channel_axis=-1 if arr.ndim == 3 else None,
        convert2ycbcr=False,
        rescale_sigma=True,
    )
    out = (1.0 - s) * arr + s * denoised
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def sharpen_image(image: np.ndarray, amount: float = 0.25, radius: float = 1.5) -> np.ndarray:
    """Apply mild unsharp masking."""
    arr = np.asarray(image, dtype=np.float32)
    amt = float(np.clip(amount, 0.0, 2.0))
    if amt <= 0:
        return arr

    sharp = filters.unsharp_mask(
        arr,
        radius=radius,
        amount=amt,
        preserve_range=True,
        channel_axis=-1 if arr.ndim == 3 else None,
    )
    return np.clip(sharp, 0.0, 1.0).astype(np.float32, copy=False)


def star_reduction(image: np.ndarray, strength: float = 0.2) -> np.ndarray:
    """Conservative star reduction on very bright star cores."""
    arr = np.asarray(image, dtype=np.float32)
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 0:
        return arr

    lum = to_luminance(arr)
    threshold = float(np.percentile(lum, 99.6))
    mask = lum >= threshold

    blur = filters.gaussian(arr, sigma=1.0, preserve_range=True, channel_axis=-1 if arr.ndim == 3 else None)

    out = arr.copy()
    if arr.ndim == 2:
        out[mask] = (1.0 - s) * arr[mask] + s * blur[mask]
    else:
        for c in range(arr.shape[-1]):
            channel = out[..., c]
            channel[mask] = (1.0 - s) * arr[..., c][mask] + s * blur[..., c][mask]
            out[..., c] = channel

    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def to_uint16(image01: np.ndarray) -> np.ndarray:
    """Convert normalized [0,1] image to uint16."""
    arr = np.asarray(image01, dtype=np.float32)
    return np.round(np.clip(arr, 0.0, 1.0) * 65535.0).astype(np.uint16)


def to_uint8(image01: np.ndarray) -> np.ndarray:
    """Convert normalized [0,1] image to uint8."""
    arr = np.asarray(image01, dtype=np.float32)
    return np.round(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)


def normalize_for_preview(image: np.ndarray) -> np.ndarray:
    """Fast automatic stretch for UI previews."""
    return stretch_image(image, method="asinh", auto=True, midtone=0.25)
