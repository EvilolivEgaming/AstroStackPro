"""Streamlit UI for local offline astrophotography FITS processing and stacking."""

from __future__ import annotations

import json
import textwrap
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tifffile
from astropy.io import fits
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from astrostack import calibration
from astrostack import debayer
from astrostack import io as fits_io
from astrostack import photometry
from astrostack import postprocess
from astrostack import register
from astrostack import stack


@dataclass
class PipelineSettings:
    """User-configurable processing settings."""

    stack_method: str
    sigma_clip_sigma: float
    registration_enabled: bool
    debayer_mode: str
    calibration_enabled: bool
    use_bias: bool
    use_dark: bool
    use_flat: bool
    bias_folder: str
    dark_folder: str
    flat_folder: str
    cosmetic_hot_pixel: bool
    background_extraction: bool
    color_calibration: bool
    stretch_method: str
    stretch_black_point: float
    stretch_midtone: float
    stretch_auto: bool
    noise_reduction_strength: float
    sharpen_strength: float
    star_reduction_enabled: bool
    star_reduction_strength: float
    plate_scale_override: Optional[float]


st.set_page_config(page_title="AstroStack Pro (Offline)", layout="wide")


VALID_EXTENSIONS = (
    ".fits",
    ".fit",
    ".fts",
    ".fits.gz",
    ".fit.gz",
    ".fts.gz",
)


def is_supported_name(name: str) -> bool:
    token = name.lower()
    return token.endswith(VALID_EXTENSIONS)


def log_message(logs: list[str], message: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    logs.append(f"[ts] message")


def load_frames_from_uploads(uploaded_files) -> tuple[list[fits_io.FitsFrame], list[str]]:
    frames: list[fits_io.FitsFrame] = []
    errors: list[str] = []

    for up in uploaded_files:
        if not is_supported_name(up.name):
            errors.append(f"Skipped unsupported file: up.name")
            continue
        try:
            frame = fits_io.load_fits_from_bytes(up.name, up.getvalue())
            frames.append(frame)
        except Exception as exc:
            errors.append(f"Failed to load up.name: exc")

    return frames, errors


def load_frames_from_folder(folder: str) -> tuple[list[fits_io.FitsFrame], list[str]]:
    frames: list[fits_io.FitsFrame] = []
    errors: list[str] = []

    if not folder.strip():
        return frames, ["Folder path is empty."]

    try:
        files = fits_io.scan_folder_for_fits(folder)
    except Exception as exc:
        return frames, [str(exc)]

    if not files:
        return frames, ["No FITS/FITS.GZ files found in folder."]

    for path in files:
        try:
            frame = fits_io.load_fits_from_path(path)
            frames.append(frame)
        except Exception as exc:
            errors.append(f"Failed to load path.name: exc")

    return frames, errors


def load_calibration_folder(folder: str, forced_type: str, logs: list[str]) -> list[fits_io.FitsFrame]:
    if not folder.strip():
        return []

    frames, errors = load_frames_from_folder(folder)
    for err in errors:
        log_message(logs, f"Calibration folder issue (forced_type): err")

    for frame in frames:
        frame.user_frame_type = forced_type

    if frames:
        log_message(logs, f"Loaded len(frames) forced_type frames from folder: folder")
    return frames


def select_plate_scale(lights: list[fits_io.FitsFrame], user_override: Optional[float]) -> Optional[float]:
    if user_override is not None and user_override > 0:
        return user_override

    for frame in lights:
        if frame.plate_scale and frame.plate_scale > 0:
            return frame.plate_scale

    for frame in lights:
        calc = fits_io.compute_plate_scale_from_optics(frame.pixel_size_um, frame.focal_length_mm)
        if calc and calc > 0:
            return calc

    return None


def _format_value(val) -> str:
    if isinstance(val, np.ndarray):
        return np.array2string(val, precision=4, separator=", ")
    if isinstance(val, (list, tuple)):
        arr = np.asarray(val, dtype=np.float32)
        return np.array2string(arr, precision=4, separator=", ")
    if val is None:
        return "N/A"
    try:
        return f"float(val):.6g"
    except Exception:
        return str(val)


def sample_pixel(image: np.ndarray, x: int, y: int):
    arr = np.asarray(image)
    if y < 0 or y >= arr.shape[0] or x < 0 or x >= arr.shape[1]:
        return None
    if arr.ndim == 2:
        return float(arr[y, x])
    return arr[y, x, :].astype(np.float32)


def build_alignment_overlay(reference: np.ndarray, aligned: np.ndarray) -> np.ndarray:
    ref = postprocess.normalize_for_preview(register.to_luminance(reference))
    ali = postprocess.normalize_for_preview(register.to_luminance(aligned))
    zero = np.zeros_like(ref)
    return np.stack([ref, ali, zero], axis=-1)


def run_pipeline(
    lights: list[fits_io.FitsFrame],
    all_frames: list[fits_io.FitsFrame],
    settings: PipelineSettings,
) -> dict:
    logs: list[str] = []
    errors: list[str] = []

    if not lights:
        raise ValueError("No LIGHT frames selected for processing.")

    reference = lights[0]
    h, w = reference.shape_hw
    log_message(logs, f"Selected len(lights) LIGHT frame(s). Reference=reference.filename")

    extra_cal_frames: list[fits_io.FitsFrame] = []
    if settings.calibration_enabled:
        if settings.use_bias:
            extra_cal_frames.extend(load_calibration_folder(settings.bias_folder, "BIAS", logs))
        if settings.use_dark:
            extra_cal_frames.extend(load_calibration_folder(settings.dark_folder, "DARK", logs))
        if settings.use_flat:
            extra_cal_frames.extend(load_calibration_folder(settings.flat_folder, "FLAT", logs))

    combined_frames = list(all_frames) + extra_cal_frames

    bias_candidates = fits_io.filter_by_type(combined_frames, "BIAS") if settings.calibration_enabled and settings.use_bias else []
    dark_candidates = fits_io.filter_by_type(combined_frames, "DARK") if settings.calibration_enabled and settings.use_dark else []
    flat_candidates = fits_io.filter_by_type(combined_frames, "FLAT") if settings.calibration_enabled and settings.use_flat else []

    bias_candidates = fits_io.compatible_frames_for_reference(reference, bias_candidates)
    dark_candidates = fits_io.compatible_frames_for_reference(reference, dark_candidates)
    flat_candidates = fits_io.compatible_frames_for_reference(reference, flat_candidates)

    master_bias = None
    master_dark = None
    master_flat = None

    if settings.calibration_enabled:
        if settings.use_bias and bias_candidates:
            log_message(logs, f"Building master bias from len(bias_candidates) frame(s).")
            master_bias = calibration.make_master_bias([f.data for f in bias_candidates], sigma_clip_enabled=True)
        elif settings.use_bias:
            errors.append("Bias calibration enabled but no compatible BIAS frames found.")

        if settings.use_dark and dark_candidates:
            log_message(logs, f"Building master dark from len(dark_candidates) frame(s).")
            master_dark = calibration.make_master_dark(
                [f.data for f in dark_candidates],
                [f.exposure for f in dark_candidates],
                master_bias=master_bias,
                sigma_clip_enabled=True,
            )
        elif settings.use_dark:
            errors.append("Dark calibration enabled but no compatible DARK frames found.")

        if settings.use_flat and flat_candidates:
            log_message(logs, f"Building master flat from len(flat_candidates) frame(s).")
            master_flat = calibration.make_master_flat(
                [f.data for f in flat_candidates],
                [f.exposure for f in flat_candidates],
                master_bias=master_bias,
                master_dark=master_dark,
                sigma_clip_enabled=True,
            )
        elif settings.use_flat:
            errors.append("Flat calibration enabled but no compatible FLAT frames found.")

    calibrated_pre_register: list[np.ndarray] = []
    calibrated_for_inspector: list[np.ndarray] = []

    for idx, frame in enumerate(lights):
        work = frame.data.astype(np.float32, copy=True)
        try:
            if settings.calibration_enabled:
                work = calibration.calibrate_light(
                    work,
                    light_exposure_s=frame.exposure,
                    master_bias=master_bias,
                    master_dark=master_dark,
                    master_flat=master_flat,
                )

            if settings.cosmetic_hot_pixel:
                work = calibration.cosmetic_hot_pixel_correction(work, sigma=5.0)

            calibrated_for_inspector.append(work.copy())

            if frame.is_osc_raw:
                if settings.debayer_mode == "Auto":
                    pattern = frame.bayer_pattern or "RGGB"
                else:
                    pattern = settings.debayer_mode
                work = debayer.debayer_osc(work, pattern)
            elif frame.data.ndim == 3:
                # Already RGB/3-plane FITS.
                work = np.asarray(work, dtype=np.float32)

            calibrated_pre_register.append(work.astype(np.float32, copy=False))
        except Exception as exc:
            errors.append(f"Frame calibration failed for frame.filename: exc")
            log_message(logs, f"Calibration failed: frame.filename -> exc")

        if idx % 10 == 0:
            log_message(logs, f"Calibrated frame idx + 1/len(lights)")

    if not calibrated_pre_register:
        raise RuntimeError("No calibrated light frames available after preprocessing.")

    # Registration
    if settings.registration_enabled:
        log_message(logs, "Starting star-based registration.")
        registered, align_metrics = register.register_frames(
            calibrated_pre_register,
            max_control_points=120,
            detection_sigma=4.5,
        )
        align_summary = register.summarize_alignment(align_metrics)
    else:
        registered = calibrated_pre_register
        align_metrics = [register.AlignmentMetric(index=i, success=True, rms_error_px=0.0, matched_stars=0) for i in range(len(registered))]
        align_summary = register.summarize_alignment(align_metrics)

    log_message(logs, "Integrating registered frames.")
    weights = stack.compute_frame_weights(registered)
    linear_master_raw, integrate_stats = stack.integrate_frames(
        registered,
        method=settings.stack_method,
        sigma=settings.sigma_clip_sigma,
        maxiters=5,
        weights=weights,
    )

    # Post-processing pipeline on linear master.
    linear_master = linear_master_raw.copy()
    background_model = None

    if settings.background_extraction:
        linear_master, background_model = postprocess.remove_background(linear_master, order=2, sample_step=64)
        log_message(logs, "Applied background extraction.")

    if settings.color_calibration and linear_master.ndim == 3:
        linear_master = postprocess.color_calibrate_rgb(linear_master)
        log_message(logs, "Applied color calibration.")

    stretched = postprocess.stretch_image(
        linear_master,
        method=settings.stretch_method,
        black_point=settings.stretch_black_point,
        midtone=settings.stretch_midtone,
        auto=settings.stretch_auto,
    )

    if settings.noise_reduction_strength > 0:
        stretched = postprocess.denoise_image(stretched, strength=settings.noise_reduction_strength)
        log_message(logs, "Applied wavelet noise reduction.")

    if settings.sharpen_strength > 0:
        stretched = postprocess.sharpen_image(stretched, amount=settings.sharpen_strength, radius=1.6)
        log_message(logs, "Applied gentle sharpening.")

    if settings.star_reduction_enabled and settings.star_reduction_strength > 0:
        stretched = postprocess.star_reduction(stretched, strength=settings.star_reduction_strength)
        log_message(logs, "Applied conservative star reduction.")

    final_processed = np.clip(stretched, 0.0, 1.0).astype(np.float32, copy=False)

    # Metrics
    snr_gain = stack.estimate_snr_improvement(registered[0], linear_master_raw)
    plate_scale_aspp = select_plate_scale(lights, settings.plate_scale_override)
    fov_x_arcmin, fov_y_arcmin = fits_io.compute_fov_arcmin(w, h, plate_scale_aspp)

    # Save outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path.cwd() / "astrostack_outputs" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    linear_master_fits_path = output_dir / "linear_master.fits"
    stretched_tiff_path = output_dir / "stretched_master_16bit.tiff"
    stretched_png_path = output_dir / "stretched_master_preview.png"
    final_tiff_path = output_dir / "final_processed_16bit.tiff"
    final_png_path = output_dir / "final_processed_preview.png"
    report_json_path = output_dir / "processing_report.json"
    report_text_path = output_dir / "processing_report.txt"
    log_path = output_dir / "processing.log"

    fits_data = linear_master.astype(np.float32)
    if fits_data.ndim == 3:
        fits_data = np.moveaxis(fits_data, -1, 0)

    out_header = reference.header.copy()
    out_header["HISTORY"] = "AstroStack Pro linear master output"
    fits.writeto(linear_master_fits_path, fits_data, header=out_header, overwrite=True)

    stretched_u16 = postprocess.to_uint16(stretched)
    final_u16 = postprocess.to_uint16(final_processed)
    tifffile.imwrite(
        stretched_tiff_path,
        stretched_u16,
        photometric="rgb" if stretched_u16.ndim == 3 else "minisblack",
    )
    tifffile.imwrite(
        final_tiff_path,
        final_u16,
        photometric="rgb" if final_u16.ndim == 3 else "minisblack",
    )

    iio.imwrite(stretched_png_path, postprocess.to_uint8(stretched))
    iio.imwrite(final_png_path, postprocess.to_uint8(final_processed))

    report = 
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "pipeline": "AstroStack Pro",
        "settings": asdict(settings),
        "inputs": [fits_io.json_ready_header_subset(f) for f in lights],
        "counts": 
            "lights": len(lights),
            "bias": len(bias_candidates),
            "dark": len(dark_candidates),
            "flat": len(flat_candidates),
        ,
        "alignment": 
            "summary": align_summary,
            "per_frame": [asdict(m) for m in align_metrics],
        ,
        "integration": 
            "method": settings.stack_method,
            "weights": weights.tolist(),
            "stats": integrate_stats,
        ,
        "metrics": 
            "snr_improvement_factor": snr_gain,
            "plate_scale_arcsec_per_px": plate_scale_aspp,
            "fov_arcmin": "x": fov_x_arcmin, "y": fov_y_arcmin,
        ,
        "errors": errors,
        "logs": logs,
        "outputs": 
            "linear_master_fits": str(linear_master_fits_path),
            "stretched_tiff_16bit": str(stretched_tiff_path),
            "stretched_png_preview": str(stretched_png_path),
            "final_tiff_16bit": str(final_tiff_path),
            "final_png_preview": str(final_png_path),
        ,
    

    fits_io.save_json(report_json_path, report)

    text_lines = [
        "AstroStack Processing Report",
        "=" * 32,
        f"Created (UTC): report['created_utc']",
        "",
        "Input Summary",
        "-" * 20,
        f"LIGHT frames: len(lights)",
        f"BIAS frames: len(bias_candidates)",
        f"DARK frames: len(dark_candidates)",
        f"FLAT frames: len(flat_candidates)",
        "",
        "Metrics",
        "-" * 20,
        f"Alignment success ratio: align_summary.get('success_ratio', float('nan')):.3f",
        f"Alignment mean RMS (px): align_summary.get('mean_rms_px', float('nan')):.3f",
        f"SNR improvement factor: snr_gain:.3f",
        f"Plate scale (arcsec/px): plate_scale_aspp",
        f"FOV (arcmin): X=fov_x_arcmin, Y=fov_y_arcmin",
        "",
        "Errors",
        "-" * 20,
    ]
    text_lines.extend(errors if errors else ["None"])
    text_lines.extend(["", "Logs", "-" * 20])
    text_lines.extend(logs)

    report_text_path.write_text("".join(text_lines), encoding="utf-8")
    log_path.write_text("".join(logs), encoding="utf-8")

    log_message(logs, f"Saved outputs to: output_dir")

    return 
        "output_dir": output_dir,
        "linear_master": linear_master,
        "final_processed": final_processed,
        "stretched": stretched,
        "before_preview": postprocess.normalize_for_preview(registered[0]),
        "after_preview": final_processed,
        "alignment_overlay": build_alignment_overlay(registered[0], registered[-1]),
        "alignment_summary": align_summary,
        "snr_improvement": snr_gain,
        "plate_scale_aspp": plate_scale_aspp,
        "fov_x_arcmin": fov_x_arcmin,
        "fov_y_arcmin": fov_y_arcmin,
        "raw_reference": reference.data,
        "calibrated_reference": calibrated_for_inspector[0] if calibrated_for_inspector else reference.data,
        "width": w,
        "height": h,
        "logs": logs,
        "errors": errors,
        "report": report,
        "photometry_frames": registered,
        "photometry_dates": [f.date_obs for f in lights],
        "output_files": 
            "linear_master.fits": linear_master_fits_path,
            "stretched_master_16bit.tiff": stretched_tiff_path,
            "stretched_master_preview.png": stretched_png_path,
            "final_processed_16bit.tiff": final_tiff_path,
            "final_processed_preview.png": final_png_path,
            "processing_report.json": report_json_path,
            "processing_report.txt": report_text_path,
            "processing.log": log_path,
        ,
    


def main() -> None:
    st.title("AstroStack Pro - Offline FITS Processor")
    st.caption(
        "Local FITS/FITS.GZ calibration, registration, sigma-clipped integration, post-processing, and photometry."
    )

    if "loaded_frames" not in st.session_state:
        st.session_state.loaded_frames = []
    if "pipeline_result" not in st.session_state:
        st.session_state.pipeline_result = None
    if "photometry_df" not in st.session_state:
        st.session_state.photometry_df = None

    st.sidebar.header("Pipeline Controls")

    stack_method = st.sidebar.selectbox(
        "Stacking method",
        ["sigma-clipped mean", "median", "mean", "winsorized"],
        index=0,
    )
    sigma_clip_sigma = st.sidebar.slider("Sigma clip threshold", min_value=1.5, max_value=5.0, value=3.0, step=0.1)
    registration_enabled = st.sidebar.toggle("Registration / alignment", value=True)

    debayer_mode = st.sidebar.selectbox("Debayer pattern", ["Auto", "RGGB", "BGGR", "GRBG", "GBRG"], index=0)

    calibration_enabled = st.sidebar.toggle("Enable calibration", value=True)
    use_bias = st.sidebar.checkbox("Use bias", value=True)
    use_dark = st.sidebar.checkbox("Use dark", value=True)
    use_flat = st.sidebar.checkbox("Use flat", value=True)

    st.sidebar.caption("Optional extra calibration folders (local paths):")
    bias_folder = st.sidebar.text_input("Bias folder", value="")
    dark_folder = st.sidebar.text_input("Dark folder", value="")
    flat_folder = st.sidebar.text_input("Flat folder", value="")

    cosmetic_hot_pixel = st.sidebar.toggle("Hot pixel correction", value=True)
    background_extraction = st.sidebar.toggle("Background extraction", value=True)
    color_calibration = st.sidebar.toggle("Color calibration", value=True)

    stretch_method = st.sidebar.selectbox("Stretch method", ["asinh", "histogram", "arcsinh"], index=0)
    stretch_auto = st.sidebar.toggle("Auto stretch", value=True)
    stretch_black_point = st.sidebar.number_input("Black point (linear units)", value=0.0, step=10.0)
    stretch_midtone = st.sidebar.slider("Midtone", min_value=0.05, max_value=0.95, value=0.25, step=0.01)

    noise_reduction_strength = st.sidebar.slider("Noise reduction", min_value=0.0, max_value=1.0, value=0.12, step=0.01)
    sharpen_strength = st.sidebar.slider("Sharpening", min_value=0.0, max_value=1.5, value=0.22, step=0.01)
    star_reduction_enabled = st.sidebar.toggle("Star reduction", value=False)
    star_reduction_strength = st.sidebar.slider("Star reduction strength", min_value=0.0, max_value=1.0, value=0.2, step=0.01)

    plate_scale_override = st.sidebar.number_input(
        "Plate scale override (arcsec/pixel, 0=auto)",
        min_value=0.0,
        value=0.0,
        step=0.01,
    )
    plate_scale_override = plate_scale_override if plate_scale_override > 0 else None

    settings = PipelineSettings(
        stack_method=stack_method,
        sigma_clip_sigma=sigma_clip_sigma,
        registration_enabled=registration_enabled,
        debayer_mode=debayer_mode,
        calibration_enabled=calibration_enabled,
        use_bias=use_bias,
        use_dark=use_dark,
        use_flat=use_flat,
        bias_folder=bias_folder,
        dark_folder=dark_folder,
        flat_folder=flat_folder,
        cosmetic_hot_pixel=cosmetic_hot_pixel,
        background_extraction=background_extraction,
        color_calibration=color_calibration,
        stretch_method=stretch_method,
        stretch_black_point=float(stretch_black_point),
        stretch_midtone=stretch_midtone,
        stretch_auto=stretch_auto,
        noise_reduction_strength=noise_reduction_strength,
        sharpen_strength=sharpen_strength,
        star_reduction_enabled=star_reduction_enabled,
        star_reduction_strength=star_reduction_strength,
        plate_scale_override=plate_scale_override,
    )

    st.subheader("1) Input Frames")
    input_mode = st.radio("Input mode", ["Upload FITS/FITS.GZ files", "Scan local folder"], horizontal=True)

    uploaded_files = None
    folder_path = ""

    if input_mode == "Upload FITS/FITS.GZ files":
        uploaded_files = st.file_uploader(
            "Upload one FITS/FITS.GZ or many files",
            type=None,
            accept_multiple_files=True,
            help="Select one file or a full night of data.",
        )
    else:
        folder_path = st.text_input("Folder path", value="")

    load_clicked = st.button("Load input frames", type="primary")

    if load_clicked:
        with st.spinner("Loading FITS files..."):
            if input_mode == "Upload FITS/FITS.GZ files":
                frames, errors = load_frames_from_uploads(uploaded_files or [])
            else:
                frames, errors = load_frames_from_folder(folder_path)

        st.session_state.loaded_frames = frames
        st.session_state.pipeline_result = None
        st.session_state.photometry_df = None

        if errors:
            for err in errors:
                st.warning(err)
        if frames:
            st.success(f"Loaded len(frames) frame(s).")

    frames: list[fits_io.FitsFrame] = st.session_state.loaded_frames

    if not frames:
        st.info("Load FITS files to continue.")
        return

    frame_table = fits_io.to_table(frames)
    st.write("Detected metadata and frame types (editable):")

    edited = st.data_editor(
        frame_table,
        hide_index=True,
        num_rows="fixed",
        column_config=
            "assigned_type": st.column_config.SelectboxColumn(
                "assigned_type",
                options=list(fits_io.FRAME_TYPES),
                required=True,
            )
        ,
        disabled=[
            "idx",
            "filename",
            "detected_type",
            "width",
            "height",
            "channels",
            "exposure_s",
            "gain",
            "temp_c",
            "filter",
            "date_obs",
            "binning",
            "bayer",
            "plate_scale_aspp",
            "camera",
            "telescope",
        ],
        use_container_width=True,
    )

    fits_io.apply_user_types(frames, edited)

    light_groups = fits_io.group_light_frames(frames)
    if not light_groups:
        st.error("No LIGHT frames found. Label at least one frame as LIGHT.")
        return

    st.subheader("2) Light Group Selection")
    group_name = st.selectbox("Choose LIGHT group", list(light_groups.keys()))
    selected_lights = light_groups[group_name]
    st.caption(f"Group has len(selected_lights) LIGHT frame(s).")

    if len(selected_lights) < 2:
        st.warning("Stacking works best with multiple LIGHT frames. You currently have fewer than 2.")

    if st.button("Run full processing pipeline", type="primary"):
        with st.spinner("Running calibration, registration, integration, and post-processing..."):
            try:
                result = run_pipeline(selected_lights, frames, settings)
                st.session_state.pipeline_result = result
                st.session_state.photometry_df = None
                st.success("Processing complete.")
            except Exception as exc:
                st.session_state.pipeline_result = None
                st.error(f"Pipeline failed: exc")

    result = st.session_state.pipeline_result
    if not result:
        return

    st.subheader("3) Output Summary")
    metrics_cols = st.columns(4)
    metrics_cols[0].metric("Frames stacked", len(selected_lights))
    mean_rms = result["alignment_summary"].get("mean_rms_px", float("nan"))
    metrics_cols[1].metric("Alignment RMS (px)", f"mean_rms:.3f" if np.isfinite(mean_rms) else "N/A")
    metrics_cols[2].metric("SNR improvement", f"xresult['snr_improvement']:.2f")
    plate_scale_label = f"result['plate_scale_aspp']:.3f" if result["plate_scale_aspp"] else "N/A"
    metrics_cols[3].metric("Plate scale ("/px)", plate_scale_label)

    if result["fov_x_arcmin"] and result["fov_y_arcmin"]:
        st.caption(f"FOV estimate: result['fov_x_arcmin']:.2f' x result['fov_y_arcmin']:.2f'")
    else:
        st.caption("FOV estimate unavailable (plate scale not found). Use plate scale override if needed.")

    preview_cols = st.columns(3)
    preview_cols[0].image(result["before_preview"], caption="Before (first calibrated frame)", clamp=True)
    preview_cols[1].image(result["after_preview"], caption="Final processed", clamp=True)
    preview_cols[2].image(result["alignment_overlay"], caption="Alignment overlay (R=ref, G=last)", clamp=True)

    st.subheader("4) Pixel Inspector")
    click_img = Image.fromarray(postprocess.to_uint8(result["after_preview"]))
    click_coords = streamlit_image_coordinates(click_img, key="pixel_click")
    if click_coords:
        x = int(click_coords["x"])
        y = int(click_coords["y"])
        raw_val = sample_pixel(result["raw_reference"], x, y)
        cal_val = sample_pixel(result["calibrated_reference"], x, y)

        st.write(f"Pixel: x=x, y=y")
        st.write(f"Raw value(s): _format_value(raw_val)")
        st.write(f"Calibrated value(s): _format_value(cal_val)")

    st.subheader("5) Photometry / Light Curve")
    st.caption("Click star position below, then run aperture photometry across calibrated frames.")

    star_img = Image.fromarray(postprocess.to_uint8(result["before_preview"]))
    star_coords = streamlit_image_coordinates(star_img, key="star_click")

    col_p = st.columns(4)
    ap_r = col_p[0].number_input("Aperture r", value=6.0, min_value=1.0, step=0.5)
    ann_in = col_p[1].number_input("Annulus in", value=10.0, min_value=2.0, step=0.5)
    ann_out = col_p[2].number_input("Annulus out", value=15.0, min_value=3.0, step=0.5)
    run_phot = col_p[3].button("Run photometry")

    if run_phot:
        if not star_coords:
            st.warning("Click a star position first.")
        else:
            try:
                x = float(star_coords["x"])
                y = float(star_coords["y"])
                df = photometry.aperture_light_curve(
                    result["photometry_frames"],
                    result["photometry_dates"],
                    x=x,
                    y=y,
                    aperture_radius=float(ap_r),
                    annulus_inner=float(ann_in),
                    annulus_outer=float(ann_out),
                )
                st.session_state.photometry_df = df
            except Exception as exc:
                st.error(f"Photometry failed: exc")

    phot_df = st.session_state.photometry_df
    if isinstance(phot_df, pd.DataFrame):
        st.dataframe(phot_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 3.5))
        if "jd" in phot_df.columns and phot_df["jd"].notna().any():
            xvals = phot_df["jd"]
            xlabel = "Julian Date"
        else:
            xvals = phot_df["frame_index"]
            xlabel = "Frame Index"

        yvals = phot_df["relative_mag"]
        ax.plot(xvals, yvals, marker="o", linewidth=1.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Relative Magnitude")
        ax.invert_yaxis()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        st.download_button(
            "Download light curve CSV",
            data=photometry.dataframe_to_csv_bytes(phot_df),
            file_name="light_curve.csv",
            mime="text/csv",
        )

    st.subheader("6) Downloads")
    for label, path in result["output_files"].items():
        data = Path(path).read_bytes()
        mime = "application/octet-stream"
        if str(path).endswith(".png"):
            mime = "image/png"
        elif str(path).endswith(".tiff") or str(path).endswith(".tif"):
            mime = "image/tiff"
        elif str(path).endswith(".fits"):
            mime = "application/fits"
        elif str(path).endswith(".json"):
            mime = "application/json"
        elif str(path).endswith(".txt") or str(path).endswith(".log"):
            mime = "text/plain"

        st.download_button(
            f"Download label",
            data=data,
            file_name=Path(path).name,
            mime=mime,
        )

    st.caption(f"Output directory: result['output_dir']")

    if result["errors"]:
        st.subheader("Warnings / Errors")
        for err in result["errors"]:
            st.warning(err)

    st.subheader("Processing Log")
    st.code("".join(result["logs"]), language="text")

    with st.expander("View JSON report"):
        st.json(result["report"])

    st.markdown(
        textwrap.dedent(
            """
            **Notes**
            - Linear master is saved as FITS before stretch.
            - Stretched and final outputs are saved as 16-bit TIFF + PNG previews.
            - Use plate scale override in sidebar when FITS metadata lacks WCS/pixel scale and optics metadata.
            """
        )
    )


if __name__ == "__main__":
    main()
