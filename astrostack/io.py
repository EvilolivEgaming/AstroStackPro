"""FITS input/output and metadata handling for AstroStack."""

from __future__ import annotations

import io as pyio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

FRAME_TYPES = ("LIGHT", "DARK", "FLAT", "BIAS", "UNKNOWN")


@dataclass
class FitsFrame:
    """Container for a FITS frame plus parsed metadata."""

    frame_id: str
    source: str
    filename: str
    data: np.ndarray
    header: fits.Header
    frame_type: str
    user_frame_type: Optional[str]
    exposure: Optional[float]
    gain: Optional[float]
    temperature_c: Optional[float]
    filter_name: Optional[str]
    date_obs: Optional[str]
    telescope: Optional[str]
    camera: Optional[str]
    binning: Optional[tuple[int, int]]
    plate_scale: Optional[float]
    pixel_size_um: Optional[float]
    focal_length_mm: Optional[float]
    bayer_pattern: Optional[str]
    is_rgb: bool
    is_osc_raw: bool
    wcs: Optional[WCS]

    @property
    def effective_frame_type(self) -> str:
        """Return user label if available, otherwise detected frame type."""
        if self.user_frame_type:
            return self.user_frame_type
        return self.frame_type

    @property
    def shape_hw(self) -> tuple[int, int]:
        """Return image shape as height, width regardless of channels."""
        if self.data.ndim == 2:
            return int(self.data.shape[0]), int(self.data.shape[1])
        return int(self.data.shape[0]), int(self.data.shape[1])


def _first_header_value(header: fits.Header, keys: list[str]) -> Optional[Any]:
    for key in keys:
        if key in header and header[key] not in (None, ""):
            return header[key]
    return None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_binning(raw: Any) -> Optional[tuple[int, int]]:
    if raw is None:
        return None
    if isinstance(raw, (tuple, list)) and len(raw) >= 2:
        try:
            return int(raw[0]), int(raw[1])
        except (TypeError, ValueError):
            return None
    if isinstance(raw, str):
        token = raw.lower().replace(" ", "")
        for sep in ("x", "*"):
            if sep in token:
                a, b = token.split(sep, maxsplit=1)
                try:
                    return int(a), int(b)
                except (TypeError, ValueError):
                    return None
        try:
            v = int(token)
            return v, v
        except (TypeError, ValueError):
            return None
    try:
        v = int(raw)
        return v, v
    except (TypeError, ValueError):
        return None


def parse_binning(header: fits.Header) -> Optional[tuple[int, int]]:
    """Parse FITS binning from common keywords."""
    xbin = _to_float(_first_header_value(header, ["XBINNING", "XBIN", "BINX"]))
    ybin = _to_float(_first_header_value(header, ["YBINNING", "YBIN", "BINY"]))
    if xbin and ybin:
        return int(xbin), int(ybin)
    return _normalize_binning(_first_header_value(header, ["BINNING", "CCDBIN1", "CCDSUM"]))


def detect_frame_type(header: fits.Header, filename: str = "") -> str:
    """Detect frame type from common FITS metadata keywords."""
    candidates = [
        _first_header_value(header, ["IMAGETYP", "IMAGETYPE", "FRAME", "OBSTYPE", "EXPTYPE", "TYPE"]),
        _first_header_value(header, ["OBJECT"]),
    ]
    text = " ".join(str(c).upper() for c in candidates if c is not None)
    name_text = filename.upper()
    if "BIAS" in text or "BIAS" in name_text:
        return "BIAS"
    if "DARK" in text or "DARK" in name_text:
        return "DARK"
    if "FLAT" in text or "FLAT" in name_text:
        return "FLAT"
    if "LIGHT" in text or "LIGHT" in name_text:
        return "LIGHT"
    if any(s in text for s in ("SCIENCE", "OBJECT", "TARGET")):
        return "LIGHT"
    return "UNKNOWN"


def detect_bayer_pattern(header: fits.Header) -> Optional[str]:
    """Return normalized Bayer pattern if present."""
    raw = _first_header_value(
        header,
        ["BAYERPAT", "BAYERPATTERN", "BAYERPAT", "COLORTYP", "CFA", "CFAPAT"],
    )
    if raw is None:
        return None
    token = str(raw).upper().strip()
    token = token.replace(" ", "")
    if token in "RGGB", "BGGR", "GRBG", "GBRG":
        return token
    if token.startswith("BAYER_"):
        token = token.replace("BAYER_", "")
    if token in "RGGB", "BGGR", "GRBG", "GBRG":
        return token
    return None


def _normalize_image_array(data: np.ndarray) -> tuple[np.ndarray, bool]:
    """Normalize FITS image orientation to HxW or HxWx3 and flag RGB."""
    arr = np.asarray(data)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr.astype(np.float32, copy=False), False

    if arr.ndim != 3:
        raise ValueError(f"Unsupported FITS dimensionality: arr.shape")

    if arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
    elif arr.shape[-1] in (3, 4):
        pass
    else:
        raise ValueError(
            "3D FITS must be channel-first (3,H,W) or channel-last (H,W,3)."
        )

    if arr.shape[-1] == 4:
        arr = arr[..., :3]

    return arr.astype(np.float32, copy=False), True


def _extract_plate_scale_from_wcs(header: fits.Header) -> Optional[float]:
    # Explicit scalar keys first.
    direct = _to_float(_first_header_value(header, ["PIXSCALE", "SECPIX", "SCALE"]))
    if direct and direct > 0:
        return direct

    # CDELT in degrees/pixel.
    cdelt1 = _to_float(_first_header_value(header, ["CDELT1"]))
    if cdelt1 and cdelt1 != 0:
        return abs(cdelt1) * 3600.0

    # CD matrix in degrees/pixel.
    cd11 = _to_float(_first_header_value(header, ["CD1_1"]))
    cd12 = _to_float(_first_header_value(header, ["CD1_2"]))
    if cd11 is not None:
        if cd12 is None:
            cd12 = 0.0
        return float(np.hypot(cd11, cd12) * 3600.0)

    # Try WCS parsing as a fallback.
    try:
        wcs = WCS(header)
        if wcs.pixel_scale_matrix is not None:
            scale_deg = float(np.hypot(wcs.pixel_scale_matrix[0, 0], wcs.pixel_scale_matrix[0, 1]))
            if scale_deg != 0:
                return abs(scale_deg) * 3600.0
    except Exception:
        return None
    return None


def compute_plate_scale_from_optics(
    pixel_size_um: Optional[float],
    focal_length_mm: Optional[float],
) -> Optional[float]:
    """Compute plate scale in arcsec/pixel from pixel size and focal length."""
    if not pixel_size_um or not focal_length_mm:
        return None
    if pixel_size_um <= 0 or focal_length_mm <= 0:
        return None
    return 206.265 * (pixel_size_um / focal_length_mm)


def parse_date_obs(value: Optional[str]) -> Optional[datetime]:
    """Parse DATE-OBS style values to datetime where possible."""
    if not value:
        return None
    token = str(value).strip()
    if token.endswith("Z"):
        token = token[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(token)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(token, fmt)
        except ValueError:
            continue
    return None


def _extract_primary_hdu(hdul: fits.HDUList) -> tuple[np.ndarray, fits.Header]:
    for hdu in hdul:
        if hdu.data is not None:
            return np.asarray(hdu.data), hdu.header
    raise ValueError("No image data found in FITS file.")


def _build_frame(
    data: np.ndarray,
    header: fits.Header,
    source: str,
    filename: str,
) -> FitsFrame:
    norm_data, is_rgb = _normalize_image_array(data)

    exposure = _to_float(_first_header_value(header, ["EXPTIME", "EXPOSURE", "EXPOSURE"]))
    gain = _to_float(_first_header_value(header, ["GAIN", "EGAIN", "CCDGAIN"]))
    temperature_c = _to_float(_first_header_value(header, ["CCD-TEMP", "SENSOR_TEMP", "TEMPERAT", "SET-TEMP"]))
    filter_name = _first_header_value(header, ["FILTER", "FILT", "FILTNAM", "FILTERID"])
    date_obs = _first_header_value(header, ["DATE-OBS", "DATEOBS", "DATE_OBS", "UTSTART"])
    telescope = _first_header_value(header, ["TELESCOP", "TELESCOPE", "SCOPE"])
    camera = _first_header_value(header, ["INSTRUME", "CAMERA", "DETECTOR"])
    pixel_size_um = _to_float(_first_header_value(header, ["XPIXSZ", "PIXSIZE1", "PIXSIZE", "PIXELSIZE"]))
    focal_length_mm = _to_float(_first_header_value(header, ["FOCALLEN", "FOCAL", "FOCALLENGTH"]))
    binning = parse_binning(header)

    bayer_pattern = detect_bayer_pattern(header)
    is_osc_raw = bool((norm_data.ndim == 2) and bayer_pattern)

    plate_scale = _extract_plate_scale_from_wcs(header)
    if plate_scale is None:
        plate_scale = compute_plate_scale_from_optics(pixel_size_um, focal_length_mm)

    try:
        frame_wcs = WCS(header)
        if not frame_wcs.has_celestial:
            frame_wcs = None
    except Exception:
        frame_wcs = None

    return FitsFrame(
        frame_id=f"filename:hash((source, filename))",
        source=source,
        filename=filename,
        data=norm_data.astype(np.float32, copy=False),
        header=header,
        frame_type=detect_frame_type(header, filename=filename),
        user_frame_type=None,
        exposure=exposure,
        gain=gain,
        temperature_c=temperature_c,
        filter_name=str(filter_name) if filter_name is not None else None,
        date_obs=str(date_obs) if date_obs is not None else None,
        telescope=str(telescope) if telescope is not None else None,
        camera=str(camera) if camera is not None else None,
        binning=binning,
        plate_scale=plate_scale,
        pixel_size_um=pixel_size_um,
        focal_length_mm=focal_length_mm,
        bayer_pattern=bayer_pattern,
        is_rgb=is_rgb,
        is_osc_raw=is_osc_raw,
        wcs=frame_wcs,
    )


def load_fits_from_path(path: str | Path) -> FitsFrame:
    """Load a FITS/FITS.GZ frame from disk."""
    path_obj = Path(path)
    with fits.open(path_obj, memmap=False) as hdul:
        data, header = _extract_primary_hdu(hdul)
    return _build_frame(data=data, header=header, source=str(path_obj), filename=path_obj.name)


def load_fits_from_bytes(filename: str, payload: bytes) -> FitsFrame:
    """Load FITS bytes (e.g., from Streamlit upload)."""
    with fits.open(pyio.BytesIO(payload), memmap=False) as hdul:
        data, header = _extract_primary_hdu(hdul)
    return _build_frame(data=data, header=header, source=f"upload:filename", filename=filename)


def scan_folder_for_fits(folder: str | Path) -> list[Path]:
    """Recursively discover FITS files in a folder."""
    root = Path(folder).expanduser()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Folder does not exist: root")

    patterns = ["*.fits", "*.fit", "*.fts", "*.fits.gz", "*.fit.gz", "*.fts.gz"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    dedup = sorted(f.resolve() for f in files)
    return dedup


def to_table(frames: list[FitsFrame]) -> pd.DataFrame:
    """Convert frame metadata into a DataFrame for UI display/editing."""
    rows = []
    for idx, frame in enumerate(frames):
        h, w = frame.shape_hw
        rows.append(
            
                "idx": idx,
                "filename": frame.filename,
                "detected_type": frame.frame_type,
                "assigned_type": frame.effective_frame_type,
                "width": w,
                "height": h,
                "channels": 1 if frame.data.ndim == 2 else int(frame.data.shape[-1]),
                "exposure_s": frame.exposure,
                "gain": frame.gain,
                "temp_c": frame.temperature_c,
                "filter": frame.filter_name,
                "date_obs": frame.date_obs,
                "binning": f"frame.binning[0]xframe.binning[1]" if frame.binning else None,
                "bayer": frame.bayer_pattern,
                "plate_scale_aspp": frame.plate_scale,
                "camera": frame.camera,
                "telescope": frame.telescope,
            
        )
    return pd.DataFrame(rows)


def apply_user_types(frames: list[FitsFrame], table_df: pd.DataFrame) -> None:
    """Apply edited frame type labels back to frame objects."""
    for _, row in table_df.iterrows():
        idx = int(row["idx"])
        label = str(row.get("assigned_type", "")).upper().strip()
        if label not in FRAME_TYPES:
            label = "UNKNOWN"
        frames[idx].user_frame_type = label


def group_light_frames(frames: list[FitsFrame]) -> dict[str, list[FitsFrame]]:
    """Group LIGHT frames by key metadata compatibility."""
    groups: dict[str, list[FitsFrame]] = 
    for frame in frames:
        if frame.effective_frame_type != "LIGHT":
            continue
        h, w = frame.shape_hw
        exp = None if frame.exposure is None else round(frame.exposure, 3)
        gain = None if frame.gain is None else round(frame.gain, 3)
        temp = None if frame.temperature_c is None else round(frame.temperature_c, 1)
        filt = frame.filter_name or "NA"
        binning = frame.binning or (1, 1)
        key = (
            f"wxh",
            f"binbinning[0]xbinning[1]",
            f"exp:exp",
            f"gain:gain",
            f"temp:temp",
            f"filt:filt",
            "rgb" if frame.data.ndim == 3 else "mono",
        )
        group_name = " | ".join(key)
        groups.setdefault(group_name, []).append(frame)
    return groups


def filter_by_type(frames: list[FitsFrame], frame_type: str) -> list[FitsFrame]:
    """Return all frames matching a normalized type label."""
    normalized = frame_type.upper().strip()
    return [f for f in frames if f.effective_frame_type == normalized]


def compatible_frames_for_reference(
    reference: FitsFrame,
    candidates: list[FitsFrame],
) -> list[FitsFrame]:
    """Keep only candidate frames that match reference dimensions and channels."""
    ref_shape = reference.data.shape
    ref_binning = reference.binning
    kept: list[FitsFrame] = []
    for frame in candidates:
        if frame.data.shape != ref_shape:
            continue
        if ref_binning and frame.binning and frame.binning != ref_binning:
            continue
        kept.append(frame)
    return kept


def compute_fov_arcmin(width_px: int, height_px: int, plate_scale_aspp: Optional[float]) -> tuple[Optional[float], Optional[float]]:
    """Compute FOV in arcminutes for X and Y axes."""
    if plate_scale_aspp is None:
        return None, None
    return width_px * plate_scale_aspp / 60.0, height_px * plate_scale_aspp / 60.0


def json_ready_header_subset(frame: FitsFrame) -> dict[str, Any]:
    """Return a compact metadata dictionary suitable for reports."""
    h, w = frame.shape_hw
    return 
        "filename": frame.filename,
        "source": frame.source,
        "frame_type": frame.effective_frame_type,
        "detected_type": frame.frame_type,
        "dimensions": [h, w] if frame.data.ndim == 2 else [h, w, int(frame.data.shape[-1])],
        "exposure_s": frame.exposure,
        "gain": frame.gain,
        "temperature_c": frame.temperature_c,
        "filter": frame.filter_name,
        "date_obs": frame.date_obs,
        "telescope": frame.telescope,
        "camera": frame.camera,
        "binning": frame.binning,
        "plate_scale_aspp": frame.plate_scale,
        "pixel_size_um": frame.pixel_size_um,
        "focal_length_mm": frame.focal_length_mm,
        "bayer_pattern": frame.bayer_pattern,
    


def report_header_lines(frame: FitsFrame) -> list[str]:
    """Human-readable summary lines for one frame."""
    h, w = frame.shape_hw
    dims = f"wxh" if frame.data.ndim == 2 else f"wxhxframe.data.shape[-1]"
    return [
        f"File: frame.filename",
        f"  Type: frame.effective_frame_type (detected=frame.frame_type)",
        f"  Dimensions: dims",
        f"  Exposure(s): frame.exposure",
        f"  Gain: frame.gain",
        f"  Temp(C): frame.temperature_c",
        f"  Filter: frame.filter_name",
        f"  Date-Obs: frame.date_obs",
        f"  Binning: frame.binning",
        f"  Bayer: frame.bayer_pattern",
        f"  PlateScale(as/px): frame.plate_scale",
        f"  PixelSize(um): frame.pixel_size_um",
        f"  FocalLength(mm): frame.focal_length_mm",
        f"  Telescope: frame.telescope",
        f"  Camera: frame.camera",
    ]


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON with stable formatting."""
    out = Path(path)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
