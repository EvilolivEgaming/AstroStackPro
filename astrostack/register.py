"""Star-based image registration for astrophotography stacking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import astroalign as aa
import numpy as np
from skimage.transform import warp


@dataclass
class AlignmentMetric:
    """Per-frame registration quality details."""

    index: int
    success: bool
    rms_error_px: Optional[float]
    matched_stars: int
    error: Optional[str] = None


def to_luminance(image: np.ndarray) -> np.ndarray:
    """Convert mono/RGB image to a mono luminance representation."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.shape[-1] < 3:
        return np.mean(arr, axis=-1)
    return (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]).astype(np.float32)


def _warp_with_transform(image: np.ndarray, transform, output_shape: tuple[int, int]) -> np.ndarray:
    """Apply SimilarityTransform to mono or RGB image."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        warped = warp(
            arr,
            inverse_map=transform.inverse,
            output_shape=output_shape,
            order=3,
            mode="constant",
            cval=0.0,
            preserve_range=True,
        )
        return warped.astype(np.float32, copy=False)

    channels = []
    for c in range(arr.shape[-1]):
        warped_c = warp(
            arr[..., c],
            inverse_map=transform.inverse,
            output_shape=output_shape,
            order=3,
            mode="constant",
            cval=0.0,
            preserve_range=True,
        )
        channels.append(warped_c.astype(np.float32, copy=False))
    return np.stack(channels, axis=-1).astype(np.float32, copy=False)


def register_frames(
    frames: list[np.ndarray],
    max_control_points: int = 100,
    detection_sigma: float = 5.0,
) -> tuple[list[np.ndarray], list[AlignmentMetric]]:
    """Register frames to the first frame using star matching."""
    if not frames:
        raise ValueError("No frames available for registration.")

    ref = np.asarray(frames[0], dtype=np.float32)
    ref_lum = to_luminance(ref)
    output_shape = ref_lum.shape

    aligned = [ref]
    metrics = [AlignmentMetric(index=0, success=True, rms_error_px=0.0, matched_stars=0)]

    for i, frame in enumerate(frames[1:], start=1):
        src = np.asarray(frame, dtype=np.float32)
        src_lum = to_luminance(src)
        try:
            transform, (source_pos, target_pos) = aa.find_transform(
                source=src_lum,
                target=ref_lum,
                max_control_points=max_control_points,
                detection_sigma=detection_sigma,
            )
            warped = _warp_with_transform(src, transform, output_shape=output_shape)

            transformed = transform(source_pos)
            distances = np.linalg.norm(transformed - target_pos, axis=1)
            rms = float(np.sqrt(np.mean(distances**2))) if distances.size else None

            aligned.append(warped)
            metrics.append(
                AlignmentMetric(
                    index=i,
                    success=True,
                    rms_error_px=rms,
                    matched_stars=int(len(distances)),
                )
            )
        except Exception as exc:
            # Preserve frame even on failure so stack can continue with warning.
            aligned.append(src)
            metrics.append(
                AlignmentMetric(
                    index=i,
                    success=False,
                    rms_error_px=None,
                    matched_stars=0,
                    error=str(exc),
                )
            )

    return aligned, metrics


def summarize_alignment(metrics: list[AlignmentMetric]) -> dict[str, float]:
    """Compute summary quality metrics for UI/reporting."""
    if not metrics:
        return "success_ratio": 0.0, "mean_rms_px": float("nan"), "median_rms_px": float("nan")

    success = [m for m in metrics if m.success]
    rms_values = [m.rms_error_px for m in success if m.rms_error_px is not None]

    success_ratio = len(success) / len(metrics)
    if not rms_values:
        return "success_ratio": success_ratio, "mean_rms_px": float("nan"), "median_rms_px": float("nan")

    return 
        "success_ratio": float(success_ratio),
        "mean_rms_px": float(np.mean(rms_values)),
        "median_rms_px": float(np.median(rms_values)),
