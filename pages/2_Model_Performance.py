"""Page 2 — Validated Model Performance, LODO Generalization & Journal Benchmark Audit."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.prediction import (
    artifacts_exist,
    load_evaluation_results,
    load_journal_benchmarks,
    load_model_bundle,
)
from src.theme import apply_industrial_theme, render_hero, render_kpi_card, render_sidebar_footer
from src.visualizations import (
    plot_confusion_matrix,
    plot_feature_importances,
    plot_model_comparison_bars,
    plot_per_dataset_lodo,
)

st.set_page_config(page_title="Model Performance | Chatter Detection AI", page_icon="📈", layout="wide")
apply_industrial_theme()

model_ready = artifacts_exist()
metadata = load_model_bundle()[1] if model_ready else {}
eval_data = load_evaluation_results()
journal_data = load_journal_benchmarks()

render_sidebar_footer(model_ready, metadata.get("experiment_name", ""))

render_hero(
    kicker="Scientific Evaluation · Leakage-Free Validation",
    title="Model Performance & Cross-Configuration Generalization",
    subtitle=(
        "Honest comparison of Logistic Regression, Decision Tree, and Random Forest using "
        "Leave-One-Dataset-Out (LODO) and 5-Fold StratifiedGroupKFold cross-validation."
    ),
)

if eval_data is not None:
    st.success(eval_data.get("selection_rationale", ""))

    scheme = st.radio(
        "Validation Protocol",
        options=["Leave-One-Dataset-Out (LODO) — Primary", "5-Fold StratifiedGroupKFold — In-Distribution"],
        horizontal=True,
    )

    if scheme.startswith("Leave-One-Dataset-Out"):
        block = eval_data["lodo"]
        summary_df = pd.DataFrame(block["summary"])
        st.plotly_chart(
            plot_model_comparison_bars(summary_df, "LODO Mean Performance Across Unseen Configurations"),
            use_container_width=True,
        )

        st.markdown("### Summary Metrics Table (Mean Across LODO Folds)")
        disp_cols = ["Model", "Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1", "MCC"]
        st.dataframe(
            summary_df[disp_cols].style.format({c: "{:.4f}" for c in disp_cols if c != "Model"}),
            use_container_width=True,
            hide_index=True,
        )

        # Per-dataset LODO breakdown
        st.markdown("### Per-Dataset Generalization Breakdown")
        fold_df = pd.DataFrame(block["folds"])
        metric_pick = st.selectbox(
            "Inspect Per-Dataset Metric",
            ["F1", "MCC", "Balanced Accuracy", "Recall", "Precision", "Accuracy"],
            index=0,
        )
        st.plotly_chart(plot_per_dataset_lodo(fold_df, metric=metric_pick), use_container_width=True)

        fold_disp = ["Model", "Fold", "Train Datasets", "Test Dataset", "Test Samples", "Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1", "MCC"]
        st.dataframe(
            fold_df[fold_disp].style.format(
                {c: "{:.4f}" for c in ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1", "MCC"]}
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        block = eval_data["grouped_kfold"]
        summary_df = pd.DataFrame(block["summary"])
        st.plotly_chart(
            plot_model_comparison_bars(summary_df, f"{block['n_splits']}-Fold StratifiedGroupKFold (Grouped by Operating Condition)"),
            use_container_width=True,
        )
        disp_cols = ["Model", "Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1", "MCC"]
        st.dataframe(
            summary_df[disp_cols].style.format({c: "{:.4f}" for c in disp_cols if c != "Model"}),
            use_container_width=True,
            hide_index=True,
        )

    # Confusion matrices & Classification Reports
    st.markdown("### Out-of-Fold Confusion Matrices")
    cms = block["aggregate_confusion_matrices"]
    cm_cols = st.columns(len(cms))
    for col, (m_name, cm_mat) in zip(cm_cols, cms.items()):
        with col:
            st.plotly_chart(plot_confusion_matrix(cm_mat, f"{m_name}"), use_container_width=True)

    with st.expander("View Detailed Classification Reports"):
        reports = block["aggregate_classification_reports"]
        rep_model = st.selectbox("Select Model Report", list(reports.keys()))
        rep_df = pd.DataFrame(reports[rep_model]).T.round(4)
        st.dataframe(rep_df, use_container_width=True)

    # Feature Importances
    importances = eval_data.get("feature_importances", {})
    if importances:
        st.markdown("### Feature Importance Analysis")
        imp_cols = st.columns(min(len(importances), 3))
        for col, (m_name, imp_dict) in zip(imp_cols, importances.items()):
            with col:
                lbl = f"{m_name} (Gini Importance)" if m_name != "Logistic Regression" else f"{m_name} (Normalized |Coef|)"
                st.plotly_chart(plot_feature_importances(imp_dict, lbl), use_container_width=True)

    # Download evaluation report JSON
    st.download_button(
        "Download Full Evaluation Metrics (JSON)",
        data=pd.Series(eval_data).to_json(indent=2),
        file_name="chatter_evaluation_results.json",
        mime="application/json",
    )
else:
    st.warning(
        "**Live trained evaluation metrics (`models/evaluation_results.json`) are not generated yet.** "
        "Run `python train_pipeline.py` on your HDF5 folders to compute exact confusion matrices, MCC, "
        "Balanced Accuracy, Recall, F1, and feature importances."
    )

# ------------------------------------------------------------------------------
# VERIFIED JOURNAL BENCHMARKS & REJECTED ITERATION AUDIT (ALWAYS VISIBLE)
# ------------------------------------------------------------------------------
st.markdown("---")
st.markdown("### Documented Research Journal Findings (`JOURNAL.ipynb`)")

lodo_j = journal_data.get("journal_documented_d1_d15_d19_lodo", {})
cv_j = journal_data.get("journal_documented_apples_to_apples_5fold", {})

if lodo_j and cv_j:
    j1, j2 = st.columns(2)
    with j1:
        st.markdown("#### LODO Domain-Shift Finding (Cells 400 & 404)")
        per_ds = lodo_j.get("per_dataset_approx_accuracy_pct", {})
        df_lodo_j = pd.DataFrame(
            [{"Unseen Test Dataset": k, "Random Forest Accuracy (%)": v} for k, v in per_ds.items()]
        )
        st.dataframe(df_lodo_j, use_container_width=True, hide_index=True)
        st.caption(f"Mean LODO Accuracy: **{lodo_j.get('rf_mean_accuracy_pct')}%**. {lodo_j.get('key_insight')}")

    with j2:
        st.markdown("#### Apples-to-Apples 5-Fold Grouped CV (Cell 423)")
        rows_cv = []
        for ds_combo in ["D1_D2_D3", "D1_D15_D19"]:
            if ds_combo in cv_j:
                rows_cv.append({"Dataset Combination": ds_combo.replace("_", " + "), **cv_j[ds_combo]})
        st.dataframe(pd.DataFrame(rows_cv), use_container_width=True, hide_index=True)
        st.caption(cv_j.get("key_insight", ""))

early = journal_data.get("early_invalid_experiments", [])
if early:
    st.markdown("#### Audit of Early Invalid Experiments (Explicitly Excluded from Deployment)")
    st.dataframe(pd.DataFrame(early), use_container_width=True, hide_index=True)

st.markdown(
    """
    <div class="info-panel">
        <h4 style="margin-top:0;color:#F8FAFC;">Key Model Limitations</h4>
        <ul style="color:#CBD5E1;font-size:0.9rem;line-height:1.6;margin-bottom:0;">
            <li><b>Structural Domain Shift (Dataset 19):</b> Absolute amplitude features (<code>RMS</code>, <code>Peak-to-Peak</code>,
            <code>Max FFT</code>) and raw <code>Dominant Frequency</code> shift dramatically when moving from 500 Hz / 5 MN/m
            configurations (D1, D15) to 1500 Hz / 10 MN/m dynamics (D19), causing LODO generalization to drop.</li>
            <li><b>Class Imbalance Sensitivity:</b> Accuracy alone can hide missed chatter cuts; therefore, MCC, Balanced Accuracy,
            and Chatter Recall are prioritized over raw accuracy.</li>
            <li><b>Whole-Cut Features:</b> Global RMS and FFT summarize the full cut and do not localize the exact millisecond of chatter onset.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)
