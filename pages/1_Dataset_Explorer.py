"""Page 1 — Interactive Dataset Explorer for Processed Milling Features & Physics Configs."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.feature_extraction import MODEL_FEATURE_COLUMNS
from src.prediction import (
    artifacts_exist,
    load_dataset_configurations,
    load_model_bundle,
    load_processed_dataset,
)
from src.theme import apply_industrial_theme, render_hero, render_kpi_card, render_sidebar_footer
from src.visualizations import (
    plot_class_balance_by_dataset,
    plot_correlation_heatmap,
    plot_feature_distribution,
    plot_feature_scatter,
)

st.set_page_config(page_title="Dataset Explorer | Chatter Detection AI", page_icon="📊", layout="wide")
apply_industrial_theme()

model_ready = artifacts_exist()
metadata = load_model_bundle()[1] if model_ready else {}
render_sidebar_footer(model_ready, metadata.get("experiment_name", ""))

render_hero(
    kicker="Data Intelligence · Digital Machining Database",
    title="Dataset Explorer",
    subtitle=(
        "Filter simulated milling experiments by dataset configuration, stability state, spindle speed, and "
        "depth of cut. Inspect feature distributions, multicollinearity, and cross-configuration shifts."
    ),
)

df = load_processed_dataset()
cfg_df = load_dataset_configurations()

if df is None or df.empty:
    st.warning(
        "**`data/processed_features.csv` has not been generated yet.** "
        "Run `python train_pipeline.py --experiment d1_d15_d19` (or `--experiment d1_d2_d3`) locally on your "
        "HDF5 folders to populate the interactive explorer with your actual extracted dataset."
    )
    st.markdown("### Verified Machining Configuration Parameters (`Dataset descriptions .xlsx`)")
    if not cfg_df.empty:
        st.dataframe(cfg_df, use_container_width=True, hide_index=True)
    st.stop()

# ------------------------------------------------------------------------------
# INTERACTIVE FILTERS
# ------------------------------------------------------------------------------
st.markdown("### Experiment Filters")
f1, f2, f3, f4 = st.columns([1.2, 1.0, 1.2, 1.2])

all_datasets = sorted(df["Dataset"].unique().tolist())
with f1:
    sel_datasets = st.multiselect(
        "Dataset Configuration",
        options=all_datasets,
        default=all_datasets,
        format_func=lambda x: f"Dataset {x}",
    )

with f2:
    state_filter = st.selectbox("Stability State", ["All Cuts", "Stable (0) Only", "Chatter (1) Only"])

omega_min, omega_max = float(df["Omega"].min()), float(df["Omega"].max())
with f3:
    omega_range = st.slider(
        "Spindle Speed, Omega (rpm)",
        min_value=omega_min,
        max_value=omega_max,
        value=(omega_min, omega_max),
    )

depth_min_mm, depth_max_mm = float(df["Depth"].min() * 1000), float(df["Depth"].max() * 1000)
with f4:
    depth_range_mm = st.slider(
        "Depth of Cut, b (mm)",
        min_value=round(depth_min_mm, 3),
        max_value=round(depth_max_mm, 3),
        value=(round(depth_min_mm, 3), round(depth_max_mm, 3)),
    )

filtered = df[
    df["Dataset"].isin(sel_datasets)
    & (df["Omega"] >= omega_range[0])
    & (df["Omega"] <= omega_range[1])
    & (df["Depth"] * 1000.0 >= depth_range_mm[0] - 1e-6)
    & (df["Depth"] * 1000.0 <= depth_range_mm[1] + 1e-6)
].copy()

if state_filter == "Stable (0) Only":
    filtered = filtered[filtered["Chatter"] == 0]
elif state_filter == "Chatter (1) Only":
    filtered = filtered[filtered["Chatter"] == 1]

if filtered.empty:
    st.warning("No experiments match the selected filters. Widen your filter ranges.")
    st.stop()

# KPIs for filtered slice
k1, k2, k3, k4 = st.columns(4)
n_tot = len(filtered)
n_st = int((filtered["Chatter"] == 0).sum())
n_ch = int((filtered["Chatter"] == 1).sum())
with k1:
    render_kpi_card("Filtered Cuts", f"{n_tot}", f"of {len(df)} total experiments")
with k2:
    render_kpi_card("Stable / Chatter", f"{n_st} / {n_ch}", f"{n_ch / n_tot * 100:.1f}% chatter", accent="teal")
with k3:
    render_kpi_card("Median RMS", f"{filtered['RMS'].median():.2f}", "m/s^2 (detrended xtacc)", accent="green")
with k4:
    render_kpi_card("Median Dom. Freq", f"{filtered['Dominant Frequency'].median():.1f} Hz", "Hanning-windowed rFFT", accent="amber")

st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# VISUAL EXPLORATION TABS
# ------------------------------------------------------------------------------
tab_dist, tab_corr, tab_scatter, tab_ds, tab_table = st.tabs(
    [
        "Feature Distributions",
        "Correlation Heatmap",
        "Feature Comparisons",
        "Dataset-Wise Physics",
        "Data Table & Export",
    ]
)

with tab_dist:
    c_left, c_right = st.columns([2, 1])
    with c_left:
        feat_choice = st.selectbox("Select Vibration Feature", MODEL_FEATURE_COLUMNS)
    with c_right:
        use_log = st.checkbox("Logarithmic Y-Axis (recommended for RMS / Max FFT)", value=(feat_choice != "Dominant Frequency"))
    st.plotly_chart(plot_feature_distribution(filtered, feat_choice, log_y=use_log), use_container_width=True)

    st.markdown("#### Summary Statistics by Stability State")
    stats_tbl = (
        filtered.assign(State=filtered["Chatter"].map({0: "Stable (0)", 1: "Chatter (1)"}))
        .groupby("State")[MODEL_FEATURE_COLUMNS]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(3)
    )
    st.dataframe(stats_tbl, use_container_width=True)

with tab_corr:
    corr_cols = MODEL_FEATURE_COLUMNS + [c for c in ["Max Acc", "Min Acc", "Omega", "Depth", "Chatter"] if c in filtered.columns]
    st.plotly_chart(plot_correlation_heatmap(filtered, corr_cols), use_container_width=True)
    st.caption(
        "Note: `Omega` and `Depth` are shown here for physical insight only—they are strictly excluded "
        "from ML model inputs. `Max Acc` and `Min Acc` exhibit near-perfect negative collinearity (~ -0.99), "
        "which motivated consolidating them into `Peak-to-Peak`."
    )

with tab_scatter:
    sc1, sc2, sc3, sc4 = st.columns([1.2, 1.2, 0.8, 0.8])
    with sc1:
        x_feat = st.selectbox("X-Axis Feature", MODEL_FEATURE_COLUMNS, index=0)
    with sc2:
        y_feat = st.selectbox("Y-Axis Feature", MODEL_FEATURE_COLUMNS, index=2)
    with sc3:
        log_x = st.checkbox("Log X", value=(x_feat != "Dominant Frequency"))
    with sc4:
        log_y = st.checkbox("Log Y", value=(y_feat != "Dominant Frequency"))
    st.plotly_chart(plot_feature_scatter(filtered, x_feat, y_feat, log_x=log_x, log_y=log_y), use_container_width=True)

with tab_ds:
    st.plotly_chart(plot_class_balance_by_dataset(filtered), use_container_width=True)
    st.markdown("#### Dataset-Wise Feature Medians")
    ds_med = filtered.groupby("Dataset")[MODEL_FEATURE_COLUMNS + ["Omega", "Depth"]].median().round(4)
    st.dataframe(ds_med, use_container_width=True)

    if not cfg_df.empty:
        st.markdown("#### Physical Simulation Parameters (`Dataset descriptions .xlsx`)")
        st.dataframe(cfg_df[cfg_df["Dataset"].isin(sel_datasets)], use_container_width=True, hide_index=True)

with tab_table:
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download Filtered Dataset ({len(filtered)} rows, CSV)",
        data=csv_bytes,
        file_name="chatter_filtered_features.csv",
        mime="text/csv",
        type="primary",
    )
