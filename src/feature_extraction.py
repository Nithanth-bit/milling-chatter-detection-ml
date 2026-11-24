"""Validated signal processing, HDF5 parsing, and physics stability boundary labeling.

Implements the exact corrected DSP workflow established in JOURNAL.ipynb (Cell 339):
  1. Read 17-column HDF5 simulation array (column 0=omega, 1=depth, 2=time, 11=xtacc).
  2. Compute true sampling step dt = mean(diff(time)) and verify dt > 0.
  3. Detrend xtacc by removing DC mean offset.
  4. Compute time-domain features: RMS, Peak-to-Peak (plus Max/Min for diagnostics).
  5. Apply Hanning window, compute single-sided rFFT normalized by 2 / sum(window),
     suppress DC bin (index 0), and derive true frequencies in Hz via rfftfreq(N, d=dt).
  6. Extract Max FFT and Dominant Frequency (Hz).
  7. Interpolate critical depth b_lim(omega) from the dataset's analytical stability
     boundary HDF5 file to assign ground-truth physics labels (Chatter = 1 if depth > b_lim).
"""

from __future__ import annotations

import io
import os
import re
from typing import Any, BinaryIO, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import h5py
except ImportError:
    h5py = None

RAW_HDF5_COLUMNS = [
    "omega", "b", "time",
    "xtpos", "ytpos", "xwpos", "ywpos",
    "xtvel", "ytvel", "xwvel", "ywvel",
    "xtacc", "ytacc", "xwacc", "ywacc",
    "Forcex", "Forcey",
]

# Core 4 leakage-free vibration features used by the validated ML pipeline (Cell 364)
MODEL_FEATURE_COLUMNS = [
    "RMS",
    "Peak-to-Peak",
    "Max FFT",
    "Dominant Frequency",
]

# Strictly separated experiment metadata / label columns (NEVER fed as ML inputs)
METADATA_COLUMNS = [
    "Dataset",
    "Experiment",
    "File",
    "Omega",
    "Depth",
    "Critical Depth",
    "Chatter",
]


def _require_h5py() -> None:
    if h5py is None:
        raise ImportError(
            "h5py is required to read HDF5 (.h5) files. Install it with `pip install h5py`."
        )


def parse_filename_ids(filename: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract (dataset_number, experiment_number) from filenames like 'time15_42.h5'."""
    base = os.path.basename(filename)
    match = re.search(r"time(\d+)_(\d+)\.h5$", base, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    match_ds = re.search(r"time(\d+)_", base, re.IGNORECASE)
    if match_ds:
        return int(match_ds.group(1)), None
    return None, None


def compute_signal_features_and_spectrum(
    time: np.ndarray,
    acceleration: np.ndarray,
) -> Dict[str, Any]:
    """Compute validated time-domain and frequency-domain vibration features."""
    time_arr = np.asarray(time, dtype=float).ravel()
    acc_arr = np.asarray(acceleration, dtype=float).ravel()

    if len(time_arr) < 16 or len(acc_arr) < 16:
        raise ValueError("Signal is too short (minimum 16 samples required for FFT).")
    if len(time_arr) != len(acc_arr):
        raise ValueError(
            f"Time length ({len(time_arr)}) does not match acceleration length ({len(acc_arr)})."
        )
    if not np.all(np.isfinite(time_arr)) or not np.all(np.isfinite(acc_arr)):
        raise ValueError("Signal contains NaN or Inf values.")

    dt_values = np.diff(time_arr)
    dt = float(np.mean(dt_values))
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError(f"Invalid sampling step dt={dt}. Time vector must be strictly increasing.")

    fs = 1.0 / dt
    n_samples = len(acc_arr)

    # 1. Remove DC offset (detrend by mean subtraction)
    detrended = acc_arr - float(np.mean(acc_arr))

    # 2. Time-domain features
    rms = float(np.sqrt(np.mean(detrended ** 2)))
    max_acc = float(np.max(detrended))
    min_acc = float(np.min(detrended))
    peak_to_peak = float(max_acc - min_acc)

    # 3. Hanning-windowed single-sided Real FFT normalized by 2 / sum(window)
    window = np.hanning(n_samples)
    window_sum = float(np.sum(window))
    windowed = detrended * window
    fft_raw = np.abs(np.fft.rfft(windowed))
    fft_amplitude = (2.0 * fft_raw) / window_sum
    frequencies = np.fft.rfftfreq(n_samples, d=dt)

    # Suppress DC component
    fft_amplitude[0] = 0.0

    dom_idx = int(np.argmax(fft_amplitude))
    max_fft = float(fft_amplitude[dom_idx])
    dominant_freq = float(frequencies[dom_idx])

    return {
        "RMS": rms,
        "Peak-to-Peak": peak_to_peak,
        "Max FFT": max_fft,
        "Dominant Frequency": dominant_freq,
        "Max Acc": max_acc,
        "Min Acc": min_acc,
        "dt": dt,
        "fs": fs,
        "N": n_samples,
        "time": time_arr,
        "raw_signal": acc_arr,
        "detrended_signal": detrended,
        "frequencies": frequencies,
        "fft_amplitude": fft_amplitude,
    }


def read_experiment_hdf5(
    source: Union[str, os.PathLike, bytes, BinaryIO],
    filename: str = "uploaded.h5",
) -> Dict[str, Any]:
    """Read a 17-column Digital Machining Database HDF5 file from path or uploaded buffer."""
    _require_h5py()

    if isinstance(source, (str, os.PathLike)):
        filename = os.path.basename(str(source))
        h5_handle = h5py.File(source, "r")
    elif isinstance(source, (bytes, bytearray)):
        h5_handle = h5py.File(io.BytesIO(source), "r")
    else:
        raw_bytes = source.read()
        if hasattr(source, "seek"):
            source.seek(0)
        h5_handle = h5py.File(io.BytesIO(raw_bytes), "r")

    with h5_handle as f:
        keys = list(f.keys())
        if not keys:
            raise ValueError(f"HDF5 file '{filename}' contains no datasets.")
        dataset_key = keys[0]
        data = np.asarray(f[dataset_key][:], dtype=float)

    if data.ndim != 2 or data.shape[1] < 12:
        raise ValueError(
            f"Unsupported HDF5 shape {data.shape} in '{filename}'. "
            "Expected a 2D time-series simulation matrix with 17 columns "
            "(stability_boundary*.h5 files have only 2 columns and are not time-series experiments)."
        )

    omega = float(data[0, 0])
    depth = float(data[0, 1])
    time_vec = data[:, 2]
    xtacc = data[:, 11]

    dsp = compute_signal_features_and_spectrum(time_vec, xtacc)

    channel_rms: Dict[str, float] = {}
    if data.shape[1] >= 15:
        for ch_name, col_idx in [("xtacc", 11), ("ytacc", 12), ("xwacc", 13), ("ywacc", 14)]:
            sig = data[:, col_idx] - float(np.mean(data[:, col_idx]))
            channel_rms[ch_name] = float(np.sqrt(np.mean(sig ** 2)))

    ds_num, exp_num = parse_filename_ids(filename)

    return {
        "File": filename,
        "Dataset": ds_num,
        "Experiment": exp_num,
        "dataset_key": dataset_key,
        "shape": data.shape,
        "Omega": omega,
        "Depth": depth,
        "channel_rms": channel_rms,
        **dsp,
    }


def load_stability_boundary(
    boundary_path: Union[str, os.PathLike],
) -> Tuple[np.ndarray, np.ndarray]:
    """Load analytical stability boundary (omega_rpm, b_critical_m) from HDF5."""
    _require_h5py()

    with h5py.File(boundary_path, "r") as f:
        keys = list(f.keys())
        if not keys:
            raise ValueError(f"Stability boundary file '{boundary_path}' is empty.")
        data = np.asarray(f[keys[0]][:], dtype=float)

    if data.ndim != 2:
        raise ValueError(f"Expected 2D array in stability boundary, got shape {data.shape}.")
    if data.shape[0] == 2 and data.shape[1] > 2:
        data = data.T
    if data.shape[1] < 2:
        raise ValueError(f"Expected 2 columns [omega, b_lim], got shape {data.shape}.")

    order = np.argsort(data[:, 0])
    omega_vec = data[order, 0]
    b_lim_vec = data[order, 1]
    return omega_vec, b_lim_vec


def evaluate_physics_stability(
    omega: float,
    depth: float,
    omega_boundary: np.ndarray,
    b_critical_boundary: np.ndarray,
) -> Dict[str, Any]:
    """Interpolate critical depth of cut b_lim(omega) and compare against actual depth."""
    omega_min = float(np.min(omega_boundary))
    omega_max = float(np.max(omega_boundary))
    in_range = omega_min <= omega <= omega_max

    critical_depth = float(np.interp(omega, omega_boundary, b_critical_boundary))
    is_chatter = int(depth > critical_depth)
    margin_m = critical_depth - depth
    margin_pct = (margin_m / critical_depth * 100.0) if critical_depth > 0 else 0.0

    return {
        "omega": float(omega),
        "depth": float(depth),
        "critical_depth": critical_depth,
        "physics_label": is_chatter,
        "physics_state": "CHATTER" if is_chatter == 1 else "STABLE",
        "margin_m": float(margin_m),
        "margin_pct": float(margin_pct),
        "in_boundary_range": in_range,
        "omega_range": (omega_min, omega_max),
    }


def list_available_boundaries(boundary_dir: Union[str, os.PathLike]) -> Dict[int, str]:
    """Return mapping {dataset_id: filepath} for all stability_boundary*.h5 files in a directory."""
    found: Dict[int, str] = {}
    if not os.path.isdir(boundary_dir):
        return found
    for fn in sorted(os.listdir(boundary_dir)):
        m = re.fullmatch(r"stability_boundary(\d+)\.h5", fn, re.IGNORECASE)
        if m:
            found[int(m.group(1))] = os.path.join(boundary_dir, fn)
    return found
