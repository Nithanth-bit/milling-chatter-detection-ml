"""Page 5 — Complete Research Methodology, Physics & Leakage Correction Audit."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.feature_extraction import list_available_boundaries, load_stability_boundary
from src.prediction import (
    DEFAULT_BOUNDARIES_DIR,
    artifacts_exist,
    load_dataset_configurations,
    load_model_bundle,
    load_processed_dataset,
)
from src.theme import apply_industrial_theme, render_hero, render_sidebar_footer
from src.visualizations import plot_stability_lobes

st.set_page_config(page_title="Research Methodology | Chatter Detection AI", page_icon="🔬", layout="wide")
apply_industrial_theme()

model_ready = artifacts_exist()
metadata = load_model_bundle()[1] if model_ready else {}
render_sidebar_footer(model_ready, metadata.get("experiment_name", ""))

render_hero(
    kicker="Engineering Foundations · Signal Processing & ML Rigor",
    title="Research Methodology & Pipeline Architecture",
    subtitle=(
        "Complete technical documentation of the Digital Machining Database workflow: regenerative chatter physics, "
        "Hanning-windowed spectral engineering, analytical SLD ground-truth labeling, and leakage-free LODO validation."
    ),
)

# End-to-end Pipeline Diagram
st.markdown("### 1. End-to-End Research Pipeline")
st.graphviz_chart(
    """
    digraph G {
        rankdir=LR;
        bgcolor="transparent";
        node [shape=box, style="rounded,filled", fillcolor="#111827", color="#38BDF8", fontcolor="#F1F5F9", fontname="Inter", fontsize=11, margin="0.18,0.12"];
        edge [color="#64748B", penwidth=1.6];

        A [label="HDF5 Simulation\\n(17 Channels, dt≈80µs)"];
        B [label="Signal Conditioning\\nxtacc Detrend + Hanning"];
        C [label="Vibration Features (X)\\nRMS, P2P, Max FFT, Dom Freq"];
        D [label="Analytical SLD\\nstability_boundaryX.h5"];
        E [label="Physics Label (y)\\nChatter = (b > b_lim)"];
        F [label="Leakage-Free ML\\nScaler inside LODO CV"];
        G [label="Deployment Inference\\nML Verdict + Physics Check"];

        A -> B -> C -> F -> G;
        A -> D -> E -> F;
    }
    """,
    use_container_width=True,
)

# Regenerative Chatter Physics
col1, col2 = st.columns(2)
with col1:
    st.markdown("### 2. Physics of Regenerative Chatter")
    st.markdown(
        """
        <div class="info-panel">
            <p style="color:#CBD5E1;font-size:0.92rem;line-height:1.6;">
                In milling, chatter is a <b>self-excited regenerative vibration</b> arising from the closed-loop interaction
                between cutting forces, structural compliance, and chip-thickness modulation:
            </p>
            <ol style="color:#CBD5E1;font-size:0.9rem;line-height:1.65;margin-bottom:0;">
                <li>A flute enters the workpiece and generates a dynamic cutting force.</li>
                <li>The compliant tool-holder structure deflects, leaving a wavy cut surface.</li>
                <li>The following tooth cuts across that wavy surface, modulating instantaneous chip thickness.</li>
                <li>Variable chip thickness alters cutting forces, feeding energy back into tool vibration.</li>
                <li>If depth of cut exceeds the analytical limit <i>b</i><sub>lim</sub>(Ω), damping cannot dissipate the energy and vibration amplifies exponentially.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown("### 3. Time-Domain vs. Frequency-Domain Features")
    st.markdown(
        """
        <div class="info-panel">
            <p style="color:#CBD5E1;font-size:0.92rem;line-height:1.6;">
                Features are extracted strictly from detrended tool X-acceleration (<code>xtacc</code>, Column 11):
            </p>
            <ul style="color:#CBD5E1;font-size:0.9rem;line-height:1.65;margin-bottom:0;">
                <li><b>RMS (Time Domain):</b> Measures overall vibration energy; grows sharply once unstable regenerative feedback develops.</li>
                <li><b>Peak-to-Peak (Time Domain):</b> Captures total oscillation swing (<code>max - min</code>), replacing redundant <code>Max Acc</code> and <code>Min Acc</code> (which correlated at <code>r = -0.995</code>).</li>
                <li><b>Max FFT (Frequency Domain):</b> Normalized peak spectral amplitude using a Hanning window (<code>2 * |rfft(x_w)| / sum(w)</code>) to suppress spectral leakage and length dependence.</li>
                <li><b>Dominant Frequency in Hz (Frequency Domain):</b> True frequency of peak energy via <code>rfftfreq(N, d=dt)</code> with DC bin zeroed—eliminating earlier negative-frequency FFT artifacts.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Interactive Stability Lobe Diagram
st.markdown("### 4. Analytical Stability Lobe Diagrams (Ground-Truth Labeling)")
st.markdown(
    "Each simulation dataset has a paired `stability_boundaryX.h5` file containing `[Omega (rpm), b_critical (m)]`. "
    "Ground-truth labels are computed via linear interpolation: $b_{\\text{critical}}(\\Omega) = \\text{interp}(\\Omega, \\Omega_{\\text{vec}}, b_{\\text{lim}})$, "
    "assigning **Chatter = 1** if $b_{\\text{actual}} > b_{\\text{critical}}$, else **Stable = 0**."
)

b_paths = list_available_boundaries(DEFAULT_BOUNDARIES_DIR)
if b_paths:
    sel_b = st.multiselect(
        "Select Analytical Stability Boundaries to Overlay",
        options=sorted(b_paths.keys()),
        default=sorted(b_paths.keys()),
        format_func=lambda x: f"Dataset {x} Stability Boundary",
    )
    if sel_b:
        loaded_b = {ds: load_stability_boundary(b_paths[ds]) for ds in sel_b}
        df_proc = load_processed_dataset()
        sub_exp = df_proc[df_proc["Dataset"].isin(sel_b)] if df_proc is not None else None
        st.plotly_chart(plot_stability_lobes(loaded_b, experiments_df=sub_exp), use_container_width=True)
        st.caption(
            "Note: Dataset 1 and Dataset 2 boundaries overlap exactly (`np.allclose == True`) because they share "
            "500 Hz / 5 MN/m modal parameters and differ only in milling direction (down vs. up milling)."
        )

# Dataset Physics Comparison
cfg_df = load_dataset_configurations()
if not cfg_df.empty:
    st.markdown("### 5. Simulated Machining Configurations (`Dataset descriptions .xlsx`)")
    st.dataframe(cfg_df, use_container_width=True, hide_index=True)

# Methodology Corrections & Leakage Audit
st.markdown("### 6. Methodological Corrections & Leakage Prevention Audit")
audit_df = pd.DataFrame(
    [
        {
            "Issue Discovered in Research": "RMS-Median Threshold Labeling",
            "Why It Was Invalid": "Defined Chatter = (RMS > median RMS), forcing an artificial 50/50 split and leaking target proxy into amplitude features.",
            "Correction Implemented": "Replaced with physics-based analytical Stability Lobe Boundary interpolation (`depth > b_critical`).",
        },
        {
            "Issue Discovered in Research": "Duplicate Noisy Datasets (D1, D61, D91)",
            "Why It Was Invalid": "D61 (1% noise) and D91 (5% noise) shared 100% identical stability labels with D1, causing memorization across random splits.",
            "Correction Implemented": "Excluded D61/D91; progressed to distinct physical configurations (`D1 + D15 + D19` and `D1 + D2 + D3`).",
        },
        {
            "Issue Discovered in Research": "Unwindowed Two-Sided FFT Without dt",
            "Why It Was Invalid": "Produced negative frequencies (`-0.04`), unnormalized length-dependent FFT spikes, and non-Hz bin indices.",
            "Correction Implemented": "Detrend -> Hanning window -> single-sided `rfft` normalized by `2 / sum(window)` -> true Hz via `rfftfreq(N, d=dt)`.",
        },
        {
            "Issue Discovered in Research": "Metadata Leakage (`Omega`, `Depth` in X)",
            "Why It Was Invalid": "Allowed trees to memorize one machine's SLD coordinates instead of detecting vibration instability.",
            "Correction Implemented": "Restricted X strictly to 4 vibration features (`RMS`, `Peak-to-Peak`, `Max FFT`, `Dominant Frequency`).",
        },
        {
            "Issue Discovered in Research": "Random Train/Test Split & Pre-Split Scaling",
            "Why It Was Invalid": "Tested on cuts from the same machine configuration seen during training, inflating accuracy to ~99%.",
            "Correction Implemented": "Adopted Leave-One-Dataset-Out (LODO) CV + 5-Fold StratifiedGroupKFold with `StandardScaler` inside scikit-learn `Pipeline`.",
        },
    ]
)
st.dataframe(audit_df, use_container_width=True, hide_index=True)
