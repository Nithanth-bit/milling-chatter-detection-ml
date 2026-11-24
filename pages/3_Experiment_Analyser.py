"""Page 3 — Interactive Chatter Prediction & Experiment Analyser."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from src.feature_extraction import (
    MODEL_FEATURE_COLUMNS,
    compute_signal_features_and_spectrum,
    evaluate_physics_stability,
    list_available_boundaries,
    load_stability_boundary,
    read_experiment_hdf5,
)
from src.prediction import (
    DEFAULT_BOUNDARIES_DIR,
    DEFAULT_SAMPLES_DIR,
    artifacts_exist,
    generate_engineering_explanation,
    load_model_bundle,
    predict_batch,
    predict_single,
)
from src.theme import apply_industrial_theme, render_hero, render_kpi_card, render_sidebar_footer
from src.visualizations import (
    plot_fft_spectrum,
    plot_probability_gauge,
    plot_stability_lobes,
    plot_time_signal,
)

st.set_page_config(page_title="Experiment Analyser | Chatter Detection AI", page_icon="🎯", layout="wide")
apply_industrial_theme()

model_ready = artifacts_exist()
model, metadata = (load_model_bundle() if model_ready else (None, {}))
render_sidebar_footer(model_ready, metadata.get("experiment_name", ""))

render_hero(
    kicker="Inference Workbench · Dual ML & Physics Verification",
    title="Chatter Prediction & Experiment Analyser",
    subtitle=(
        "Upload a simulated HDF5 milling experiment, run batch CSV predictions, or enter vibration features manually. "
        "ML vibration predictions are strictly separated from experiment metadata (Ω, b) and analytical SLD checks."
    ),
)

if not model_ready:
    st.warning(
        "**Saved ML model (`models/final_model.joblib`) not found yet.** "
        "Run `python train_pipeline.py` locally to enable ML inference. You can still upload `.h5` files "
        "or enter `(Omega, Depth)` below to extract validated features and run the analytical physics boundary check."
    )

st.info(
    "**Important Engineering Separation:** The ML classifier takes **only vibration-derived inputs** "
    "(`RMS`, `Peak-to-Peak`, `Max FFT`, `Dominant Frequency`). Spindle speed (`Omega`) and depth of cut (`Depth`) "
    "are **experiment metadata** used strictly for the separate analytical Stability Lobe Diagram check—never "
    "as inputs to the ML model."
)

boundaries_map = list_available_boundaries(DEFAULT_BOUNDARIES_DIR)


def _render_single_result(
    feature_dict: Dict[str, float],
    omega: Optional[float] = None,
    depth: Optional[float] = None,
    boundary_ds: Optional[int] = None,
    signal_bundle: Optional[Dict[str, Any]] = None,
    source_name: str = "Manual Input",
) -> None:
    st.markdown("---")
    st.markdown("### Diagnostic Output Panel")

    physics_res: Optional[Dict[str, Any]] = None
    loaded_boundary = None
    if (
        omega is not None
        and depth is not None
        and boundary_ds is not None
        and boundary_ds in boundaries_map
    ):
        ob, bc = load_stability_boundary(boundaries_map[boundary_ds])
        loaded_boundary = {boundary_ds: (ob, bc)}
        physics_res = evaluate_physics_stability(omega, depth, ob, bc)

    pred_res: Optional[Dict[str, Any]] = None
    if model_ready and model is not None:
        pred_res = predict_single(model, feature_dict, metadata.get("feature_columns", MODEL_FEATURE_COLUMNS))

    left_col, right_col = st.columns([1.35, 1.0])

    with left_col:
        st.markdown("#### 1. Machine Learning Prediction (Vibration Inputs Only)")
        if pred_res is not None:
            is_chatter = pred_res["predicted_class"] == 1
            banner_cls = "status-banner-chatter" if is_chatter else "status-banner-stable"
            conf_str = (
                f"Confidence: <b>{pred_res['confidence']*100:.1f}%</b> "
                f"(Stable: {pred_res['prob_stable']*100:.1f}% | Chatter: {pred_res['prob_chatter']*100:.1f}%)"
                if pred_res["confidence"] is not None
                else "Probability output unavailable for this estimator."
            )
            st.markdown(
                f"""
                <div class="{banner_cls}">
                    <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.1em;color:#CBD5E1;">
                        ML Model Verdict ({metadata.get('selected_model', 'Saved Model')})
                    </div>
                    <div style="font-size:1.85rem;font-weight:700;color:#F8FAFC;margin:4px 0;">
                        {pred_res['predicted_label']}
                    </div>
                    <div style="font-size:0.88rem;color:#E2E8F0;">{conf_str}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("ML prediction unavailable until `train_pipeline.py` generates `models/final_model.joblib`.")

        st.markdown("#### 2. Analytical Stability Boundary Calculation (Metadata: Ω & b)")
        if physics_res is not None:
            p_cls = "status-banner-chatter" if physics_res["physics_label"] == 1 else "status-banner-stable"
            range_warn = (
                ""
                if physics_res["in_boundary_range"]
                else f"<br/><span style='color:#F59E0B;'>Note: Ω={physics_res['omega']:.0f} rpm is outside Dataset {boundary_ds} boundary range {physics_res['omega_range']}.</span>"
            )
            st.markdown(
                f"""
                <div class="{p_cls}">
                    <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.1em;color:#CBD5E1;">
                        Physics Reference (Dataset {boundary_ds} SLD)
                    </div>
                    <div style="font-size:1.45rem;font-weight:700;color:#F8FAFC;margin:4px 0;">
                        {physics_res['physics_state']}
                    </div>
                    <div style="font-size:0.88rem;color:#E2E8F0;">
                        Actual Depth: <b>{physics_res['depth']*1000:.3f} mm</b> |
                        Critical Depth b<sub>lim</sub>: <b>{physics_res['critical_depth']*1000:.3f} mm</b> |
                        Safety Margin: <b>{physics_res['margin_pct']:+.1f}%</b>
                        {range_warn}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.caption("Select a reference stability boundary and provide `(Omega, Depth)` metadata to compare against analytical SLDs.")

    with right_col:
        if pred_res is not None and pred_res["prob_chatter"] is not None:
            st.plotly_chart(plot_probability_gauge(pred_res["prob_chatter"]), use_container_width=True)
        else:
            st.info("Probability gauge activates once the model artifact is loaded.")

    # Extracted feature cards
    st.markdown("#### Extracted Vibration Features (Model Inputs)")
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        render_kpi_card("RMS", f"{feature_dict['RMS']:.3f}", "m/s^2 (detrended xtacc)")
    with fc2:
        render_kpi_card("Peak-to-Peak", f"{feature_dict['Peak-to-Peak']:.3f}", "m/s^2 (max - min)", accent="teal")
    with fc3:
        render_kpi_card("Max FFT", f"{feature_dict['Max FFT']:.3f}", "Normalized rFFT peak", accent="green")
    with fc4:
        render_kpi_card("Dominant Freq", f"{feature_dict['Dominant Frequency']:.2f} Hz", "True frequency axis", accent="amber")

    # Engineering explanation
    if pred_res is not None:
        st.markdown("#### Engineering Interpretation")
        notes = generate_engineering_explanation(pred_res, metadata, physics_res)
        for line in notes:
            st.markdown(f"- {line}")

    # Optional plots (signal + SLD location)
    if signal_bundle is not None:
        p1, p2 = st.columns(2)
        with p1:
            st.plotly_chart(
                plot_time_signal(
                    signal_bundle["time"],
                    signal_bundle["detrended_signal"],
                    f"Detrended xtacc Waveform ({source_name})",
                    rms=signal_bundle["RMS"],
                ),
                use_container_width=True,
            )
        with p2:
            st.plotly_chart(
                plot_fft_spectrum(
                    signal_bundle["frequencies"],
                    signal_bundle["fft_amplitude"],
                    signal_bundle["Dominant Frequency"],
                    signal_bundle["Max FFT"],
                ),
                use_container_width=True,
            )

    if loaded_boundary is not None and omega is not None and depth is not None:
        st.plotly_chart(
            plot_stability_lobes(
                loaded_boundary,
                highlight_point=(omega, depth, f"Current Cut ({source_name})"),
            ),
            use_container_width=True,
        )

    # Downloadable JSON report
    report = {
        "source": source_name,
        "vibration_features_ml_inputs": feature_dict,
        "experiment_metadata": {"Omega_rpm": omega, "Depth_m": depth, "Boundary_Dataset": boundary_ds},
        "ml_prediction": pred_res,
        "physics_stability_check": physics_res,
    }
    st.download_button(
        "Download Experiment Diagnostic Report (JSON)",
        data=json.dumps(report, indent=2),
        file_name=f"chatter_report_{os.path.splitext(source_name)[0]}.json",
        mime="application/json",
    )


# ------------------------------------------------------------------------------
# INPUT MODE TABS
# ------------------------------------------------------------------------------
tab_h5, tab_csv, tab_manual = st.tabs(
    [
        "1. Upload HDF5 Experiment (.h5)",
        "2. Batch Feature CSV (.csv)",
        "3. Manual Features & Physics Check",
    ]
)

with tab_h5:
    st.markdown("Upload any 17-column Digital Machining Database simulation file (e.g., `time1_12.h5`, `time15_40.h5`, `time19_85.h5`).")
    col_u1, col_u2 = st.columns([2, 1])
    with col_u1:
        up_h5 = st.file_uploader("Upload Milling Experiment HDF5 File", type=["h5", "hdf5"], key="analyser_h5")
    with col_u2:
        b_options = [None] + sorted(boundaries_map.keys())
        sel_b_h5 = st.selectbox(
            "Reference Stability Boundary (Optional)",
            options=b_options,
            format_func=lambda x: "Auto-match from filename" if x is None else f"Dataset {x} Boundary",
            key="h5_boundary",
        )

    # Built-in sample waveform shortcut if generated by train_pipeline.py
    sample_stable = DEFAULT_SAMPLES_DIR / "sample_stable_cut.csv"
    sample_chatter = DEFAULT_SAMPLES_DIR / "sample_chatter_cut.csv"
    if sample_stable.is_file() or sample_chatter.is_file():
        st.caption("Or test immediately with a bundled real simulation sample:")
        sb1, sb2 = st.columns(2)
        demo_choice = None
        if sample_stable.is_file() and sb1.button("Load Bundled Stable Sample"):
            demo_choice = ("sample_stable_cut.csv", "sample_stable_meta.json")
        if sample_chatter.is_file() and sb2.button("Load Bundled Chatter Sample"):
            demo_choice = ("sample_chatter_cut.csv", "sample_chatter_meta.json")
        if demo_choice is not None:
            sdf = pd.read_csv(DEFAULT_SAMPLES_DIR / demo_choice[0])
            meta_p = DEFAULT_SAMPLES_DIR / demo_choice[1]
            smeta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.is_file() else {}
            dsp = compute_signal_features_and_spectrum(sdf["time"].to_numpy(), sdf["xtacc"].to_numpy())
            feats = {k: dsp[k] for k in MODEL_FEATURE_COLUMNS}
            _render_single_result(
                feats,
                omega=smeta.get("Omega"),
                depth=smeta.get("Depth"),
                boundary_ds=smeta.get("Dataset"),
                signal_bundle=dsp,
                source_name=smeta.get("File", demo_choice[0]),
            )

    if up_h5 is not None:
        with st.spinner(f"Parsing HDF5 & computing Hanning-windowed FFT for {up_h5.name}..."):
            try:
                parsed = read_experiment_hdf5(up_h5, filename=up_h5.name)
            except Exception as exc:
                st.error(f"Could not parse `{up_h5.name}`: {exc}")
                parsed = None

        if parsed is not None:
            auto_ds = parsed.get("Dataset")
            use_ds = sel_b_h5 if sel_b_h5 is not None else (auto_ds if auto_ds in boundaries_map else None)
            feats = {k: parsed[k] for k in MODEL_FEATURE_COLUMNS}
            _render_single_result(
                feats,
                omega=parsed["Omega"],
                depth=parsed["Depth"],
                boundary_ds=use_ds,
                signal_bundle=parsed,
                source_name=parsed["File"],
            )

with tab_csv:
    st.markdown(
        "Upload a CSV file containing the 4 required vibration feature columns: "
        "`RMS`, `Peak-to-Peak`, `Max FFT`, `Dominant Frequency` (if only `Max Acc` and `Min Acc` exist, "
        "`Peak-to-Peak` is computed automatically)."
    )
    up_csv = st.file_uploader("Upload Vibration Features CSV", type=["csv"], key="analyser_csv")

    if up_csv is not None:
        if not model_ready:
            st.error("Run `python train_pipeline.py` first so `models/final_model.joblib` is available for batch inference.")
        else:
            with st.spinner("Validating columns and running batch predictions..."):
                try:
                    raw_df = pd.read_csv(up_csv)
                    pred_df, missing_cols = predict_batch(
                        model, raw_df, metadata.get("feature_columns", MODEL_FEATURE_COLUMNS)
                    )
                    if missing_cols:
                        st.error(
                            f"Uploaded CSV is missing required vibration feature columns: **{missing_cols}**. "
                            f"Found columns: `{raw_df.columns.tolist()}`."
                        )
                    else:
                        n_c = int((pred_df["Predicted_Class"] == 1).sum())
                        n_s = int((pred_df["Predicted_Class"] == 0).sum())
                        st.success(f"Predicted **{len(pred_df)}** experiments: **{n_s} STABLE**, **{n_c} CHATTER**.")
                        st.dataframe(pred_df, use_container_width=True, hide_index=True)
                        st.download_button(
                            "Download Batch Predictions (CSV)",
                            data=pred_df.to_csv(index=False).encode("utf-8"),
                            file_name="chatter_batch_predictions.csv",
                            mime="text/csv",
                            type="primary",
                        )
                except Exception as exc:
                    st.error(f"Batch prediction error: {exc}")

with tab_manual:
    st.markdown("Enter vibration features directly for ML inference, and optionally enter `(Omega, Depth)` metadata for the separate physics SLD check.")
    m_left, m_right = st.columns(2)
    ranges = metadata.get("feature_ranges", {})

    with m_left:
        st.markdown("##### ML Model Inputs (Vibration Signals Only)")
        in_rms = st.number_input("RMS Acceleration (m/s^2)", min_value=0.0, value=float(round(ranges.get("RMS", {}).get("median", 150.0), 3)), format="%.3f")
        in_p2p = st.number_input("Peak-to-Peak Acceleration (m/s^2)", min_value=0.0, value=float(round(ranges.get("Peak-to-Peak", {}).get("median", 450.0), 3)), format="%.3f")
        in_fft = st.number_input("Max FFT Amplitude (m/s^2, normalized)", min_value=0.0, value=float(round(ranges.get("Max FFT", {}).get("median", 120.0), 3)), format="%.3f")
        in_freq = st.number_input("Dominant Frequency (Hz)", min_value=0.0, value=float(round(ranges.get("Dominant Frequency", {}).get("median", 500.0), 2)), format="%.2f")

    with m_right:
        st.markdown("##### Experiment Metadata (Separate Physics Check Only)")
        in_omega = st.number_input("Spindle Speed, Omega (rpm)", min_value=500.0, max_value=40000.0, value=5000.0, step=100.0)
        in_depth_mm = st.number_input("Axial Depth of Cut, b (mm)", min_value=0.05, max_value=25.0, value=2.5, step=0.1, format="%.3f")
        b_opts_m = [None] + sorted(boundaries_map.keys())
        sel_b_manual = st.selectbox(
            "Stability Boundary Reference",
            options=b_opts_m,
            index=1 if len(b_opts_m) > 1 else 0,
            format_func=lambda x: "None (ML only)" if x is None else f"Dataset {x} Analytical SLD",
            key="manual_boundary",
        )

    if st.button("Run Dual Analysis (ML + Physics)", type="primary"):
        feat_map = {
            "RMS": in_rms,
            "Peak-to-Peak": in_p2p,
            "Max FFT": in_fft,
            "Dominant Frequency": in_freq,
        }
        _render_single_result(
            feat_map,
            omega=in_omega,
            depth=in_depth_mm / 1000.0,
            boundary_ds=sel_b_manual,
            source_name="Manual_Entry",
        )
