"""Home — Research Overview for Machine Learning-Based Chatter Detection in Milling Operations."""

from __future__ import annotations

import streamlit as st

from src.feature_extraction import list_available_boundaries, load_stability_boundary
from src.prediction import (
    DEFAULT_BOUNDARIES_DIR,
    artifacts_exist,
    load_dataset_configurations,
    load_journal_benchmarks,
    load_model_bundle,
    load_processed_dataset,
)
from src.theme import apply_industrial_theme, render_hero, render_kpi_card, render_sidebar_footer
from src.visualizations import plot_class_balance_by_dataset, plot_stability_lobes

st.set_page_config(
    page_title="Milling Chatter Detection AI | Research Platform",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_industrial_theme()

model_ready = artifacts_exist()
metadata = {}
df_processed = load_processed_dataset()
journal_data = load_journal_benchmarks()
cfg_df = load_dataset_configurations()

if model_ready:
    _, metadata = load_model_bundle()

render_sidebar_footer(model_ready, metadata.get("experiment_name", "D1 + D15 + D19 (Journal)"))

render_hero(
    kicker="Industrial AI · Production Engineering Research Platform",
    title="Machine Learning-Based Chatter Detection in Milling Operations",
    subtitle=(
        "Physics-grounded vibration classification using analytical Stability Lobe Diagrams (SLDs), "
        "Hanning-windowed spectral features, and Leave-One-Dataset-Out (LODO) cross-validation on "
        "simulated milling data from the Digital Machining Database (Schmitz & Smith, Springer, 2019)."
    ),
)

if not model_ready:
    st.markdown(
        """
        <div class="status-banner-warn">
            <strong>Action Required to Activate Live ML Predictions &amp; Full Metrics:</strong><br/>
            This repository ships with <strong>zero fabricated results</strong>. Run
            <code>python train_pipeline.py --experiment d1_d15_d19</code> (or <code>--experiment d1_d2_d3</code>)
            once on your machine where the raw HDF5 folders live to extract real features, run LODO validation,
            and generate <code>models/final_model.joblib</code> and <code>data/processed_features.csv</code>.
            Meanwhile, you can explore the verified journal benchmarks, analytical Stability Lobe Diagrams,
            and interactive Signal Analysis lab below.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------------------
# KPI ROW
# ------------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)

if model_ready:
    n_exp = metadata.get("total_experiments", 0)
    dist = metadata.get("class_distribution", {})
    n_stable = dist.get("stable_0", 0)
    n_chatter = dist.get("chatter_1", 0)
    chatter_pct = (n_chatter / n_exp * 100.0) if n_exp else 0.0
    lodo_m = metadata.get("selected_model_lodo_metrics", {})

    with c1:
        render_kpi_card(
            "Experiments Used",
            f"{n_exp}",
            f"Datasets: {metadata.get('datasets_used', [])}",
        )
    with c2:
        render_kpi_card(
            "Class Balance",
            f"{n_stable} / {n_chatter}",
            f"Stable vs. Chatter ({chatter_pct:.1f}% chatter)",
            accent="teal",
        )
    with c3:
        render_kpi_card(
            "Selected Model",
            metadata.get("selected_model", "N/A"),
            "4 vibration features (leakage-free)",
            accent="green",
        )
    with c4:
        render_kpi_card(
            "LODO F1 / Recall",
            f"{lodo_m.get('F1', 0):.3f} / {lodo_m.get('Recall', 0):.3f}",
            f"Balanced Acc: {lodo_m.get('Balanced Accuracy', 0):.3f}",
            accent="teal",
        )
    with c5:
        render_kpi_card(
            "LODO MCC / Acc",
            f"{lodo_m.get('MCC', 0):.3f} / {lodo_m.get('Accuracy', 0)*100:.1f}%",
            "Unseen configuration test",
            accent="amber",
        )
else:
    lodo_ref = journal_data.get("journal_documented_d1_d15_d19_lodo", {})
    cv_ref = journal_data.get("journal_documented_apples_to_apples_5fold", {})
    with c1:
        render_kpi_card(
            "Active Study Scope",
            "D1 + D15 + D19",
            "Also benchmarked D1+D2+D3 (394 cuts)",
        )
    with c2:
        render_kpi_card(
            "D1+D2+D3 Split",
            "234 / 160",
            "59.4% Stable vs. 40.6% Chatter (Cell 186)",
            accent="teal",
        )
    with c3:
        render_kpi_card(
            "Model Status",
            "Pending Run",
            "Run train_pipeline.py to serialize",
            accent="amber",
        )
    with c4:
        render_kpi_card(
            "Journal LODO (RF)",
            f"{lodo_ref.get('rf_mean_accuracy_pct', 68.4):.1f}%",
            "D1~80%, D15~76%, D19~49% (Cell 404)",
            accent="red",
        )
    with c5:
        d1519_rf = cv_ref.get("D1_D15_D19", {}).get("Random Forest Accuracy (%)", 82.57)
        render_kpi_card(
            "5-Fold Group CV",
            f"{d1519_rf:.2f}%",
            "RF on D1+D15+D19 (Cell 423)",
            accent="green",
        )

st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# RESEARCH OBJECTIVE & SYSTEM WORKFLOW
# ------------------------------------------------------------------------------
left, right = st.columns([1.25, 1.0])

with left:
    st.markdown(
        """
        <div class="info-panel">
            <h3 style="margin-top:0;color:#F8FAFC;font-size:1.15rem;">Research Objective &amp; Engineering Motivation</h3>
            <p style="color:#CBD5E1;font-size:0.93rem;line-height:1.6;">
                Regenerative chatter is a self-excited dynamic instability in milling caused by feedback between
                cutting forces, tool-tip deflection, and surface waviness left by prior flute passes. Uncontrolled
                chatter ruins surface finish, accelerates tool wear, and risks spindle damage.
            </p>
            <p style="color:#CBD5E1;font-size:0.93rem;line-height:1.6;margin-bottom:0;">
                <b>Core Research Question:</b> <i>Can time-domain and frequency-domain vibration features extracted
                purely from tool X-acceleration (<code>xtacc</code>) accurately predict analytical stability states
                across unseen machining configurations—without leaking spindle speed (Ω) or depth of cut (b)?</i>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        """
        <div class="info-panel">
            <h3 style="margin-top:0;color:#F8FAFC;font-size:1.15rem;">Research Status &amp; Honest Scope</h3>
            <ul style="color:#CBD5E1;font-size:0.9rem;line-height:1.65;padding-left:1.2rem;margin-bottom:0;">
                <li><b>Data Source:</b> 100% <b>simulated milling data</b> (17-channel time-domain numerical simulations + analytical frequency-domain SLDs).</li>
                <li><b>Validated Correction:</b> Replaced early flawed RMS-median labels and D1/D61/D91 duplicates with physics-based SLD labels and leakage-free LODO validation.</li>
                <li><b>Key Finding:</b> Strong in-distribution separation (~80–84% grouped CV), but clear domain shift when predicting unseen 1500 Hz / 10 MN/m dynamics (Dataset 19).</li>
                <li><b>Deployment Boundary:</b> Research prototype only—not validated on live shop-floor CNC hardware or real-time edge streaming.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Visual 6-step system workflow
st.markdown("### How the System Works")
steps = [
    ("01", "HDF5 Ingestion", "Read 17-channel simulated milling time-series (dt ≈ 80 µs, fs ≈ 12.5 kHz)."),
    ("02", "Signal Conditioning", "Extract tool X-acc (xtacc), remove DC offset, apply Hanning window."),
    ("03", "Feature Engineering", "Compute RMS, Peak-to-Peak, normalized Max FFT, and Dominant Freq (Hz)."),
    ("04", "Physics Labeling", "Interpolate critical depth b_lim(Ω) from analytical stability_boundary*.h5."),
    ("05", "Leakage-Free ML", "Fit StandardScaler + classifier strictly inside LODO & StratifiedGroupKFold."),
    ("06", "Dual Verification", "Report ML vibration prediction separately from analytical SLD stability margin."),
]
cols = st.columns(6)
for col, (num, title, desc) in zip(cols, steps):
    with col:
        st.markdown(
            f"""
            <div class="pipeline-step">
                <div class="pipeline-step-num">STEP {num}</div>
                <div class="pipeline-step-title">{title}</div>
                <div class="pipeline-step-desc">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# DATASET OVERVIEW CHARTS
# ------------------------------------------------------------------------------
col_a, col_b = st.columns([1.05, 1.35])

with col_a:
    if df_processed is not None and not df_processed.empty:
        st.plotly_chart(plot_class_balance_by_dataset(df_processed), use_container_width=True)
    else:
        st.markdown("### Simulated Configuration Summary")
        if not cfg_df.empty:
            st.dataframe(
                cfg_df[["Dataset", "Research Role", "fntx (Hz)", "ktx (N/m)", "ktc (N/m2)", "beta (deg)"]],
                use_container_width=True,
                hide_index=True,
            )

with col_b:
    b_paths = list_available_boundaries(DEFAULT_BOUNDARIES_DIR)
    if b_paths:
        loaded_b = {ds: load_stability_boundary(p) for ds, p in b_paths.items()}
        st.plotly_chart(
            plot_stability_lobes(loaded_b, experiments_df=df_processed),
            use_container_width=True,
        )
    else:
        st.info("Place `stability_boundary*.h5` files in `data/stability_boundaries/` to plot analytical SLDs.")

if model_ready and metadata.get("selection_rationale"):
    st.markdown("### Final Model Selection Rationale")
    st.info(metadata["selection_rationale"])
