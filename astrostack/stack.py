"""Frame integration (stacking) and quality metrics."""

from __future__ import annotations

import numpy as np
from astropy.stats import sigma_clip


def to_luminance(image: np.ndarray) -> np.ndarray:
    """Return a 2D luminance view for mono or RGB arrays."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.shape[-1] < 3:
        return np.mean(arr, axis=-1)
    return (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]).astype(np.float32)


def robust_std(values: np.ndarray) -> float:
    """Median absolute deviation based robust sigma estimate."""
    vals = np.asarray(values, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    return float(1.4826 * mad)


def compute_frame_weights(frames: list[np.ndarray]) -> np.ndarray:
    """Compute per-frame weights based on inverse noise variance."""
    if not frames:
        return np.array([], dtype=np.float32)

    noise: list[float] = []
    for frame in frames:
        lum = to_luminance(frame)
        background = lum[lum <= np.percentile(lum, 60)]
        sigma = robust_std(background)
        sigma = max(sigma, 1e-6)
        noise.append(sigma)

    inv_var = 1.0 / (np.asarray(noise, dtype=np.float32) ** 2)
    inv_var /= np.sum(inv_var)
    return inv_var.astype(np.float32, copy=False)


def integrate_frames(
    frames: list[np.ndarray],
    method: str = "sigma-clipped mean",
    sigma: float = 3.0,
    maxiters: int = 5,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Integrate registered frames into a linear master."""
    if not frames:
        raise ValueError("No frames to integrate.")

    cube = np.stack([np.asarray(f, dtype=np.float32) for f in frames], axis=0)
    method_norm = method.lower().strip()

    stats: dict[str, float] = 
        "n_frames": float(cube.shape[0]),
        "rejected_fraction": 0.0,
    

    if method_norm == "median":
        result = np.median(cube, axis=0)

    elif method_norm == "mean":
        if weights is None or len(weights) != cube.shape[0]:
            result = np.mean(cube, axis=0)
        else:
            w = np.asarray(weights, dtype=np.float32)
            w /= np.sum(w)
            w_shape = (cube.shape[0],) + (1,) * (cube.ndim - 1)
            result = np.sum(cube * w.reshape(w_shape), axis=0)

    elif method_norm in "winsorized", "winsorized mean", "winsorized/sigma clip":
        lo = np.percentile(cube, 5, axis=0)
        hi = np.percentile(cube, 95, axis=0)
        clipped = np.clip(cube, lo, hi)
        result = np.mean(clipped, axis=0)

    else:
        clipped = sigma_clip(cube, sigma=sigma, maxiters=maxiters, axis=0, masked=True)
        if weights is None or len(weights) != cube.shape[0]:
            result = np.ma.mean(clipped, axis=0).filled(np.nan)
        else:
            w = np.asarray(weights, dtype=np.float32)
            w /= np.sum(w)
            w_shape = (cube.shape[0],) + (1,) * (cube.ndim - 1)
            mask = np.ma.getmaskarray(clipped)
            valid_w = np.where(mask, 0.0, w.reshape(w_shape))
            denom = np.sum(valid_w, axis=0)
            denom = np.where(denom <= 1e-9, np.nan, denom)
            numer = np.sum(np.where(mask, 0.0, clipped.data) * valid_w, axis=0)
            result = numer / denom

        rejected = np.ma.getmaskarray(clipped)
        stats["rejected_fraction"] = float(np.mean(rejected))

    if np.any(~np.isfinite(result)):
        finite = result[np.isfinite(result)]
        fill = float(np.median(finite)) if finite.size else 0.0
        result = np.nan_to_num(result, nan=fill, posinf=fill, neginf=fill)

    return result.astype(np.float32, copy=False), stats


def estimate_snr(image: np.ndarray) -> float:
    """Estimate SNR using robust background noise and bright signal percentile."""
    lum = to_luminance(image)
    valid = lum[np.isfinite(lum)]
    if valid.size < 10:
        return 0.0

    bg = np.percentile(valid, 20)
    signal = np.percentile(valid, 99) - bg
    noise_region = valid[valid <= np.percentile(valid, 60)]
    noise = robust_std(noise_region)
    if noise <= 0:
        return 0.0
    return float(signal / noise)


def estimate_snr_improvement(single_frame: np.ndarray, stacked_frame: np.ndarray) -> float:
    """Estimate SNR gain factor from first frame to final stack."""
    single_snr = estimate_snr(single_frame)
    stack_snr = estimate_snr(stacked_frame)
    if single_snr <= 0:
        return 0.0
    return float(stack_snr / single_snr)
