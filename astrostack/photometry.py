"""Aperture photometry and light curve extraction."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from astropy.time import Time
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry


def to_luminance(image: np.ndarray) -> np.ndarray:
    """Convert mono/RGB frame to luminance for photometry."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.shape[-1] < 3:
        return np.mean(arr, axis=-1)
    return (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]).astype(np.float32)


def _parse_date_obs(value: Optional[str]) -> tuple[Optional[str], Optional[float]]:
    if not value:
        return None, None
    token = str(value).strip()
    if token.endswith("Z"):
        token = token[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(token)
        t = Time(dt)
        return dt.isoformat(), float(t.jd)
    except Exception:
        pass
    try:
        t = Time(token, format="isot")
        return str(t.to_datetime().isoformat()), float(t.jd)
    except Exception:
        return None, None


def aperture_light_curve(
    frames: list[np.ndarray],
    date_obs_values: list[Optional[str]],
    x: float,
    y: float,
    aperture_radius: float = 6.0,
    annulus_inner: float = 10.0,
    annulus_outer: float = 15.0,
) -> pd.DataFrame:
    """Measure flux and relative magnitude at a fixed star position."""
    if not frames:
        raise ValueError("No calibrated frames provided for photometry.")

    pos = [(float(x), float(y))]
    aper = CircularAperture(pos, r=float(aperture_radius))
    ann = CircularAnnulus(pos, r_in=float(annulus_inner), r_out=float(annulus_outer))

    rows = []
    for idx, frame in enumerate(frames):
        lum = to_luminance(frame)

        phot = aperture_photometry(lum, [aper, ann])
        star_sum = float(phot["aperture_sum_0"][0])
        ann_sum = float(phot["aperture_sum_1"][0])
        bkg_mean = ann_sum / ann.area
        net_flux = star_sum - (bkg_mean * aper.area)

        date_obs = date_obs_values[idx] if idx < len(date_obs_values) else None
        time_iso, jd = _parse_date_obs(date_obs)

        rows.append(
            
                "frame_index": idx,
                "date_obs": date_obs,
                "time_iso": time_iso,
                "jd": jd,
                "flux": net_flux,
            
        )

    df = pd.DataFrame(rows)
    valid_flux = df["flux"].replace([np.inf, -np.inf], np.nan).dropna()
    if valid_flux.empty:
        df["relative_mag"] = np.nan
        return df

    reference_flux = float(np.median(valid_flux[valid_flux > 0])) if np.any(valid_flux > 0) else np.nan
    if not np.isfinite(reference_flux) or reference_flux <= 0:
        df["relative_mag"] = np.nan
        return df

    safe_flux = np.where(df["flux"] > 0, df["flux"], np.nan)
    df["relative_mag"] = -2.5 * np.log10(safe_flux / reference_flux)

    # Time axis fallback if timestamps are missing.
    if df["jd"].isna().all():
        df["time_index"] = np.arange(len(df), dtype=float)
    return df


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Export DataFrame as UTF-8 CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")
