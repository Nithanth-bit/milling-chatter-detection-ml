"""Page 4 — Interactive Vibration Signal Analysis & DSP Lab."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from src.feature_extraction import (
    compute_signal_features_and_spectrum,
    read_experiment_hdf5,
)
from src.prediction import (
    DEFAULT_SAMPLES_DIR,
    artifacts_exist,
    load_model_bundle,
)
from src.theme import apply_industrial_theme, render_hero, render_kpi_card, render_sidebar_footer
from src.visualizations import (
    plot_fft_spectrum,
    plot_multichannel_rms,
    plot_time_signal,
)

st.set_page_config(page_title="Vibration Signal Analysis | Chatter Detection AI", page_icon="📡", layout="wide")
apply_industrial_theme()

model_ready = artifacts_exist()
metadata = load_model_bundle()[1] if model_ready else {}
render_sidebar_footer(model_ready, metadata.get("experiment_name", ""))

render_hero(
    kicker="Digital Signal Processing · Time & Frequency Diagnostics",
    title="Vibration Signal Analyser",
    subtitle=(
        "Inspect raw vs. detrended tool acceleration waveforms (xtacc), interactively filter frequency bands "
        "on the Hanning-windowed single-sided FFT spectrum, and verify extracted model features."
    ),
)

source_mode = st.radio(
    "Signal Input Source",
    options=["Upload HDF5 Experiment (.h5)", "Upload Time-Series CSV (time, xtacc)", "Bundled Sample Cuts"],
    horizontal=True,
)

signal_data = None
source_title = ""
channel_rms = {}
cut_meta = {}

if source_mode == "Upload HDF5 Experiment (.h5)":
    up = st.file_uploader("Upload 17-column Simulation HDF5 File", type=["h5", "hdf5"], key="sig_h5")
    if up is not None:
        with st.spinner(f"Reading {up.name} and computing FFT..."):
            try:
                signal_data = read_experiment_hdf5(up, filename=up.name)
                source_title = signal_data["File"]
                channel_rms = signal_data.get("channel_rms", {})
                cut_meta = {
                    "Omega (rpm)": signal_data["Omega"],
                    "Depth of Cut (mm)": round(signal_data["Depth"] * 1000.0, 4),
                    "Sampling Freq (Hz)": round(signal_data["fs"], 1),
                    "Samples (N)": signal_data["N"],
                }
            except Exception as exc:
                st.error(f"Unable to load HDF5 signal: {exc}")

elif source_mode == "Upload Time-Series CSV (time, xtacc)":
    st.caption("CSV must contain a time column (`time` or `Time`) in seconds and an acceleration column (`xtacc` or `acceleration`).")
    up_csv = st.file_uploader("Upload Vibration Waveform CSV", type=["csv"], key="sig_csv")
    if up_csv is not None:
        with st.spinner(f"Processing waveform from {up_csv.name}..."):
            try:
                wdf = pd.read_csv(up_csv)
                t_col = next((c for c in ["time", "Time", "t"] if c in wdf.columns), None)
                a_col = next((c for c in ["xtacc", "acceleration", "acc", "signal"] if c in wdf.columns), None)
                if t_col is None or a_col is None:
                    st.error(
                        f"Missing required waveform columns (`time` and `xtacc`). Found: `{wdf.columns.tolist()}`."
                    )
                else:
                    signal_data = compute_signal_features_and_spectrum(
                        wdf[t_col].to_numpy(dtype=float),
                        wdf[a_col].to_numpy(dtype=float),
                    )
                    source_title = up_csv.name
                    cut_meta = {
                        "Sampling Freq (Hz)": round(signal_data["fs"], 1),
                        "Time Step dt (µs)": round(signal_data["dt"] * 1e6, 2),
                        "Samples (N)": signal_data["N"],
                    }
            except Exception as exc:
                st.error(f"Signal processing failed: {exc}")

else:
    available_samples = []
    if (DEFAULT_SAMPLES_DIR / "sample_stable_cut.csv").is_file():
        available_samples.append(("Stable Cut Sample", "sample_stable_cut.csv", "sample_stable_meta.json"))
    if (DEFAULT_SAMPLES_DIR / "sample_chatter_cut.csv").is_file():
        available_samples.append(("Chatter Cut Sample", "sample_chatter_cut.csv", "sample_chatter_meta.json"))

    if not available_samples:
        st.info(
            "Bundled simulation samples are exported automatically into `data/samples/` when you run "
            "`python train_pipeline.py`. Switch to **Upload HDF5 Experiment (.h5)** or **Upload Time-Series CSV** "
            "above to analyse any file right now."
        )
    else:
        pick = st.selectbox("Choose Bundled Simulation Waveform", options=available_samples, format_func=lambda x: x[0])
        sdf = pd.read_csv(DEFAULT_SAMPLES_DIR / pick[1])
        meta_path = DEFAULT_SAMPLES_DIR / pick[2]
        smeta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        signal_data = compute_signal_features_and_spectrum(sdf["time"].to_numpy(), sdf["xtacc"].to_numpy())
        source_title = smeta.get("File", pick[1])
        cut_meta = {
            "True State": smeta.get("True Label", "N/A"),
            "Omega (rpm)": smeta.get("Omega", "N/A"),
            "Depth of Cut (mm)": round(smeta.get("Depth", 0.0) * 1000.0, 4) if "Depth" in smeta else "N/A",
            "Sampling Freq (Hz)": round(signal_data["fs"], 1),
        }

if signal_data is None:
    st.stop()

# ------------------------------------------------------------------------------
# EXTRACTED FEATURE KPIS
# ------------------------------------------------------------------------------
st.markdown(f"### Signal Diagnostics — `{source_title}`")

if cut_meta:
    meta_str = " · ".join([f"**{k}:** `{v}`" for k, v in cut_meta.items()])
    st.markdown(meta_str)

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    render_kpi_card("RMS", f"{signal_data['RMS']:.3f}", "m/s^2 (detrended)")
with k2:
    render_kpi_card("Peak-to-Peak", f"{signal_data['Peak-to-Peak']:.3f}", "m/s^2 (max - min)", accent="teal")
with k3:
    render_kpi_card("Max FFT", f"{signal_data['Max FFT']:.3f}", "Normalized amplitude", accent="green")
with k4:
    render_kpi_card("Dominant Freq", f"{signal_data['Dominant Frequency']:.1f} Hz", f"fs = {signal_data['fs']:.0f} Hz", accent="amber")
with k5:
    dc_off = float(np.mean(signal_data["raw_signal"]))
    render_kpi_card("DC Offset Removed", f"{dc_off:.4f}", f"N = {signal_data['N']} pts")

st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TIME-DOMAIN & FREQUENCY-DOMAIN INTERACTIVE CONTROLS
# ------------------------------------------------------------------------------
t_arr = signal_data["time"]
t_min, t_max = float(t_arr.min()), float(t_arr.max())
f_arr = signal_data["frequencies"]
f_max_nyq = float(f_arr.max())

c_ctrl1, c_ctrl2 = st.columns(2)
with c_ctrl1:
    t_window = st.slider(
        "Time-Domain Zoom Window (seconds)",
        min_value=t_min,
        max_value=t_max,
        value=(t_min, t_max),
        format="%.4f",
    )
with c_ctrl2:
    default_f_hi = min(f_max_nyq, max(2500.0, signal_data["Dominant Frequency"] * 2.0))
    f_window = st.slider(
        "Frequency Spectrum Band Filter (Hz)",
        min_value=0.0,
        max_value=f_max_nyq,
        value=(0.0, float(default_f_hi)),
        step=10.0,
    )

t_mask = (t_arr >= t_window[0]) & (t_arr <= t_window[1])

tab_raw, tab_det, tab_fft, tab_ch = st.tabs(
    ["Raw vs. Detrended Waveform", "Detrended Signal (RMS Bounds)", "FFT Frequency Spectrum", "4-Channel Comparison"]
)

with tab_raw:
    r1, r2 = st.columns(2)
    with r1:
        st.plotly_chart(
            plot_time_signal(
                t_arr[t_mask],
                signal_data["raw_signal"][t_mask],
                "Raw Acceleration Signal (xtacc)",
                color="#94A3B8",
            ),
            use_container_width=True,
        )
    with r2:
        st.plotly_chart(
            plot_time_signal(
                t_arr[t_mask],
                signal_data["detrended_signal"][t_mask],
                "Detrended Acceleration Signal (DC Offset Removed)",
                rms=signal_data["RMS"],
                color="#38BDF8",
            ),
            use_container_width=True,
        )

with tab_det:
    st.plotly_chart(
        plot_time_signal(
            t_arr[t_mask],
            signal_data["detrended_signal"][t_mask],
            f"Detrended Tool X-Acceleration with ±RMS Envelope ({source_title})",
            rms=signal_data["RMS"],
            color="#38BDF8",
        ),
        use_container_width=True,
    )

with tab_fft:
    st.plotly_chart(
        plot_fft_spectrum(
            f_arr,
            signal_data["fft_amplitude"],
            signal_data["Dominant Frequency"],
            signal_data["Max FFT"],
            freq_range=f_window,
        ),
        use_container_width=True,
    )
    st.caption(
        "Spectrum computed via `np.fft.rfft(detrended * np.hanning(N))`, normalized by `2 / sum(window)`, "
        "with DC bin suppressed (`index 0 = 0`) and frequency axis mapped to true Hz via `np.fft.rfftfreq(N, d=dt)`."
    )

with tab_ch:
    if channel_rms:
        st.plotly_chart(plot_multichannel_rms(channel_rms), use_container_width=True)
        st.caption(
            "Consistent with Dataset 1 EDA in `JOURNAL.ipynb`, tool accelerations (`xtacc`, `ytacc`) "
            "exhibit orders-of-magnitude stronger dynamic response than workpiece accelerations (`xwacc`, `ywacc`)."
        )
    else:
        st.info("4-channel comparison is available when uploading a full 17-column HDF5 simulation file.")
