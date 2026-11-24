"""Page 6 — Honest Research Findings, Limitations & Future Engineering Directions."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.prediction import (
    artifacts_exist,
    load_evaluation_results,
    load_journal_benchmarks,
    load_model_bundle,
)
from src.theme import apply_industrial_theme, render_hero, render_sidebar_footer

st.set_page_config(page_title="Research Conclusion | Chatter Detection AI", page_icon="📝", layout="wide")
apply_industrial_theme()

model_ready = artifacts_exist()
metadata = load_model_bundle()[1] if model_ready else {}
eval_data = load_evaluation_results()
journal_data = load_journal_benchmarks()

render_sidebar_footer(model_ready, metadata.get("experiment_name", ""))

render_hero(
    kicker="Synthesis · Findings, Limitations & Roadmap",
    title="Research Conclusion",
    subtitle=(
        "Transparent summary of what was investigated, implemented, and validated on simulated milling data, "
        "along with remaining scientific limitations and engineering next steps."
    ),
)

c1, c2 = st.columns(2)
with c1:
    st.markdown(
        """
        <div class="info-panel">
            <h3 style="margin-top:0;color:#F8FAFC;font-size:1.12rem;">What Was Investigated &amp; Implemented</h3>
            <ul style="color:#CBD5E1;font-size:0.91rem;line-height:1.65;margin-bottom:0;">
                <li><b>Problem Formulation:</b> Transitioned from an unsupported tool-wear regression goal to physics-grounded binary <b>chatter detection</b> in simulated milling operations.</li>
                <li><b>HDF5 Signal Pipeline:</b> Built a reproducible 17-channel HDF5 parser, verified tool-tip acceleration dominance over workpiece channels, and implemented detrended, Hanning-windowed single-sided FFT extraction in true Hz.</li>
                <li><b>Ground-Truth Labeling:</b> Replaced flawed RMS-median heuristic labels with analytical Stability Lobe Boundary interpolation (<code>stability_boundaryX.h5</code>).</li>
                <li><b>Leakage-Free ML Benchmarking:</b> Compared <b>Logistic Regression</b>, <b>Decision Tree</b>, and <b>Random Forest</b> using strictly vibration-derived features under both <b>Leave-One-Dataset-Out (LODO)</b> and <b>5-Fold StratifiedGroupKFold</b> cross-validation.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="info-panel">
            <h3 style="margin-top:0;color:#F8FAFC;font-size:1.12rem;">What the Validated Results Indicate</h3>
            <ul style="color:#CBD5E1;font-size:0.91rem;line-height:1.65;margin-bottom:0;">
                <li><b>Within-Configuration Separability:</b> Under identical 5-fold grouped CV (Cell 423), tree models achieve ~80.6–83.6% accuracy across both <code>D1+D2+D3</code> (DT 80.58%, RF 80.72%) and <code>D1+D15+D19</code> (DT 83.63%, RF 82.57%).</li>
                <li><b>Inflation Diagnosis:</b> Earlier ~96–99% accuracies were proven to result from RMS target-proxy leakage, duplicate noisy datasets (D61/D91), and uncleaned splits—not true cross-machine generalization.</li>
                <li><b>Cross-Machine Domain Shift:</b> Under strict LODO evaluation on <code>D1+D15+D19</code> (Cells 400 &amp; 404), Random Forest generalized reasonably on 500 Hz systems (D1 ~80%, D15 ~76%) but dropped to ~49% on unseen <b>Dataset 19</b> (1500 Hz / 10 MN/m stiffness, up to 30,000 rpm), pulling mean LODO accuracy to ~68.4%.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

if eval_data is not None:
    st.markdown("### Live Validated Pipeline Summary (Current Deployed Artifact)")
    st.success(eval_data.get("selection_rationale", ""))
    st.dataframe(
        pd.DataFrame(eval_data["lodo"]["summary"])[
            ["Model", "Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1", "MCC"]
        ].round(4),
        use_container_width=True,
        hide_index=True,
    )

st.markdown(
    """
    <div class="status-banner-warn">
        <strong>Explicit Scope &amp; Simulation Disclaimer:</strong><br/>
        All experiments, stability boundaries, and trained models in this project are derived exclusively from
        <b>simulated milling operations</b> in Dr. Tony Schmitz &amp; K. S. Smith's <i>Digital Machining Database
        for Machine Learning</i> (Springer, 2019). This project does <b>not</b> claim real-world CNC shop-floor
        deployment, experimental accelerometer validation, or real-time edge chatter suppression.
    </div>
    """,
    unsafe_allow_html=True,
)

l1, l2 = st.columns(2)
with l1:
    st.markdown("### Remaining Scientific Limitations")
    st.markdown(
        """
        1. **Domain Shift Under Structural Changes:** Raw amplitude scales (`RMS`, `Peak-to-Peak`, `Max FFT`) and absolute `Dominant Frequency` are tightly coupled to a machine's natural frequency ($f_n$) and stiffness ($k$), limiting zero-shot transfer to drastically stiffer or higher-frequency spindles (e.g., D19).
        2. **No Onset Localization:** Whole-cut FFT summarizes the entire time series and cannot pinpoint the exact millisecond when a stable cut transitions into chatter.
        3. **Single-Axis Sensing:** Current model features use only tool X-acceleration (`xtacc`), ignoring orthogonal tool Y-acceleration (`ytacc`) and cutting force channels (`Forcex`, `Forcey`).
        4. **Noise-Free Idealized Simulations:** Datasets 1, 15, and 19 have `0%` sensor noise, whereas real CNC spindles experience runout, thermal drift, bearing harmonics, and background shop-floor vibration.
        """
    )

with l2:
    st.markdown("### Concrete Future Research Directions")
    st.markdown(
        """
        1. **Domain-Invariant Feature Engineering:** Introduce dimensionless spectral ratios (e.g., non-harmonic energy ratio relative to tooth-passing frequency $f_{\\text{TPE}} = m \\cdot \\Omega / 60$, spectral entropy, kurtosis, crest factor) that remain invariant across 500 Hz vs. 1500 Hz spindles.
        2. **Time-Frequency Onset Detection:** Replace global FFT with Short-Time Fourier Transform (STFT) or Wavelet Packet Decomposition (WPD) over sliding windows to detect early chatter onset before amplitude saturates.
        3. **Multi-Sensor Fusion:** Combine `xtacc`, `ytacc`, and force signals (`Forcex`, `Forcey`) to capture anisotropic directional dynamics (e.g., Datasets 33–36, 50).
        4. **Noise Robustness & Physical Validation:** Benchmark trained classifiers against noisy simulation variants (1%–5% sensor noise) and validate via physical dynamometer/accelerometer cutting trials on a 3-axis CNC vertical milling center.
        """
    )
