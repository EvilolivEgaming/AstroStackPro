"""Debayering utilities for OSC Bayer FITS frames."""

from __future__ import annotations

import numpy as np
import cv2

BAYER_TO_CV2 = 
    "RGGB": cv2.COLOR_BAYER_RG2RGB,
    "BGGR": cv2.COLOR_BAYER_BG2RGB,
    "GRBG": cv2.COLOR_BAYER_GR2RGB,
    "GBRG": cv2.COLOR_BAYER_GB2RGB,



def debayer_osc(image: np.ndarray, bayer_pattern: str = "RGGB") -> np.ndarray:
    """Debayer a single-channel OSC frame to RGB float32."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 2:
        return arr.astype(np.float32, copy=False)

    pattern = str(bayer_pattern or "RGGB").upper().strip()
    if pattern not in BAYER_TO_CV2:
        pattern = "RGGB"

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros((*arr.shape, 3), dtype=np.float32)

    min_val = float(np.min(finite))
    max_val = float(np.max(finite))
    span = max(max_val - min_val, 1e-6)

    # OpenCV demosaic expects integer raw mosaic input.
    scaled = np.clip((arr - min_val) / span, 0.0, 1.0)
    raw_u16 = np.round(scaled * 65535.0).astype(np.uint16)
    rgb_u16 = cv2.cvtColor(raw_u16, BAYER_TO_CV2[pattern])

    rgb = (rgb_u16.astype(np.float32) / 65535.0) * span + min_val
    return rgb.astype(np.float32, copy=False)
