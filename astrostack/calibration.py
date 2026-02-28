"""Calibration logic for bias, dark, flat, and cosmetic correction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from astropy.stats import sigma_clip
from scipy.ndimage import median_filter


@dataclass
class MasterDark:
    """Master dark frame with reference exposure metadata."""

    data: np.ndarray
    exposure_s: Optional[float]


def _stack(frames: list[np.ndarray]) -> np.ndarray:
    if not frames:
        raise ValueError("No frames provided for combination.")
    return np.stack([np.asarray(f, dtype=np.float32) for f in frames], axis=0)


def combine_frames(
    frames: list[np.ndarray],
    method: str = "median",
    sigma_clip_enabled: bool = False,
    sigma: float = 3.0,
    maxiters: int = 5,
) -> np.ndarray:
    """Combine a list of frames into one master frame."""
    cube = _stack(frames)
    method_norm = method.lower().strip()

    if sigma_clip_enabled:
        clipped = sigma_clip(cube, sigma=sigma, maxiters=maxiters, axis=0, masked=True)
        if method_norm in "mean", "average", "sigma-clipped mean":
            result = np.ma.mean(clipped, axis=0).filled(np.nan)
        else:
            result = np.ma.median(clipped, axis=0).filled(np.nan)
    else:
        if method_norm in "mean", "average", "sigma-clipped mean":
            result = np.mean(cube, axis=0)
        else:
            result = np.median(cube, axis=0)

    # Replace invalid values with robust central value.
    if np.any(~np.isfinite(result)):
        finite = result[np.isfinite(result)]
        fill = float(np.median(finite)) if finite.size else 0.0
        result = np.nan_to_num(result, nan=fill, posinf=fill, neginf=fill)

    return result.astype(np.float32, copy=False)


def make_master_bias(
    bias_frames: list[np.ndarray],
    sigma_clip_enabled: bool = True,
    sigma: float = 3.0,
) -> np.ndarray:
    """Build a master bias from BIAS frames."""
    return combine_frames(
        bias_frames,
        method="median",
        sigma_clip_enabled=sigma_clip_enabled,
        sigma=sigma,
    )


def make_master_dark(
    dark_frames: list[np.ndarray],
    dark_exposures_s: list[Optional[float]],
    master_bias: Optional[np.ndarray] = None,
    sigma_clip_enabled: bool = True,
    sigma: float = 3.0,
) -> MasterDark:
    """Build a master dark and track reference dark exposure."""
    corrected: list[np.ndarray] = []
    exposure_values: list[float] = []

    for i, dark in enumerate(dark_frames):
        arr = np.asarray(dark, dtype=np.float32).copy()
        if master_bias is not None:
            if master_bias.shape != arr.shape:
                raise ValueError("Bias and dark dimensions do not match.")
            arr -= master_bias
        corrected.append(arr)

        exp = dark_exposures_s[i] if i < len(dark_exposures_s) else None
        if exp is not None and exp > 0:
            exposure_values.append(float(exp))

    master = combine_frames(
        corrected,
        method="median",
        sigma_clip_enabled=sigma_clip_enabled,
        sigma=sigma,
    )
    ref_exposure = float(np.median(exposure_values)) if exposure_values else None
    return MasterDark(data=master, exposure_s=ref_exposure)


def _normalize_flat(flat: np.ndarray) -> np.ndarray:
    if flat.ndim == 2:
        median = float(np.median(flat[np.isfinite(flat)]))
        if median <= 0 or not np.isfinite(median):
            median = 1.0
        return flat / median

    # RGB: normalize per channel.
    out = flat.copy()
    for c in range(flat.shape[-1]):
        channel = flat[..., c]
        median = float(np.median(channel[np.isfinite(channel)]))
        if median <= 0 or not np.isfinite(median):
            median = 1.0
        out[..., c] = channel / median
    return out


def make_master_flat(
    flat_frames: list[np.ndarray],
    flat_exposures_s: list[Optional[float]],
    master_bias: Optional[np.ndarray] = None,
    master_dark: Optional[MasterDark] = None,
    sigma_clip_enabled: bool = True,
    sigma: float = 3.0,
) -> np.ndarray:
    """Build a normalized master flat from FLAT frames."""
    corrected: list[np.ndarray] = []

    for i, flat in enumerate(flat_frames):
        arr = np.asarray(flat, dtype=np.float32).copy()

        if master_bias is not None:
            if master_bias.shape != arr.shape:
                raise ValueError("Bias and flat dimensions do not match.")
            arr -= master_bias

        if master_dark is not None:
            scaled_dark = scale_dark_to_exposure(
                master_dark=master_dark,
                target_exposure_s=flat_exposures_s[i] if i < len(flat_exposures_s) else None,
            )
            if scaled_dark.shape != arr.shape:
                raise ValueError("Dark and flat dimensions do not match.")
            arr -= scaled_dark

        arr = _normalize_flat(arr)
        corrected.append(arr)

    master_flat = combine_frames(
        corrected,
        method="median",
        sigma_clip_enabled=sigma_clip_enabled,
        sigma=sigma,
    )
    return _normalize_flat(master_flat).astype(np.float32, copy=False)


def scale_dark_to_exposure(master_dark: MasterDark, target_exposure_s: Optional[float]) -> np.ndarray:
    """Scale master dark to target exposure if both exposures are known."""
    dark = master_dark.data
    if target_exposure_s is None or master_dark.exposure_s in (None, 0):
        return dark

    scale = float(target_exposure_s) / float(master_dark.exposure_s)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    # Conservative guard against accidental extreme scaling.
    scale = min(max(scale, 0.1), 10.0)
    return (dark * scale).astype(np.float32, copy=False)


def calibrate_light(
    light: np.ndarray,
    light_exposure_s: Optional[float],
    master_bias: Optional[np.ndarray],
    master_dark: Optional[MasterDark],
    master_flat: Optional[np.ndarray],
) -> np.ndarray:
    """Calibrate one light frame with optional bias/dark/flat masters."""
    out = np.asarray(light, dtype=np.float32).copy()

    if master_bias is not None:
        if master_bias.shape != out.shape:
            raise ValueError("Master bias shape does not match light frame shape.")
        out -= master_bias

    if master_dark is not None:
        scaled_dark = scale_dark_to_exposure(master_dark, light_exposure_s)
        if scaled_dark.shape != out.shape:
            raise ValueError("Master dark shape does not match light frame shape.")
        out -= scaled_dark

    if master_flat is not None:
        if master_flat.shape != out.shape:
            raise ValueError("Master flat shape does not match light frame shape.")
        eps = 1e-6
        safe_flat = np.where(np.abs(master_flat) < eps, 1.0, master_flat)
        out = out / safe_flat

    return out.astype(np.float32, copy=False)


def cosmetic_hot_pixel_correction(image: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """Replace strong positive outliers with local median values."""
    arr = np.asarray(image, dtype=np.float32).copy()

    if arr.ndim == 2:
        med = median_filter(arr, size=3)
        diff = arr - med
        mad = np.median(np.abs(diff - np.median(diff))) + 1e-6
        threshold = sigma * 1.4826 * mad
        mask = diff > threshold
        arr[mask] = med[mask]
        return arr

    for c in range(arr.shape[-1]):
        channel = arr[..., c]
        med = median_filter(channel, size=3)
        diff = channel - med
        mad = np.median(np.abs(diff - np.median(diff))) + 1e-6
        threshold = sigma * 1.4826 * mad
        mask = diff > threshold
        channel[mask] = med[mask]
        arr[..., c] = channel

    return arr.astype(np.float32, copy=False)
