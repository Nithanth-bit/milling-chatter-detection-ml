"""Dark-themed interactive Plotly chart builders for Industrial AI analytics."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.theme import COLORS

LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0F172A",
    font=dict(family="Inter, sans-serif", color="#E2E8F0", size=12),
    margin=dict(l=55, r=25, t=50, b=50),
    hoverlabel=dict(bgcolor="#1E293B", bordercolor="#38BDF8", font_color="#F8FAFC"),
    xaxis=dict(gridcolor="#1E293B", zerolinecolor="#334155"),
    yaxis=dict(gridcolor="#1E293B", zerolinecolor="#334155"),
)


def _apply_base(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, font=dict(size=15, color="#F8FAFC"), x=0.01),
        height=height,
        legend=dict(
            bgcolor="rgba(17,24,39,0.75)",
            bordercolor="#1E293B",
            borderwidth=1,
        ),
    )
    return fig


def plot_class_balance_by_dataset(df: pd.DataFrame) -> go.Figure:
    counts = (
        df.assign(State=df["Chatter"].map({0: "Stable (0)", 1: "Chatter (1)"}))
        .groupby(["Dataset", "State"])
        .size()
        .reset_index(name="Count")
    )
    counts["Dataset"] = "Dataset " + counts["Dataset"].astype(str)
    fig = px.bar(
        counts,
        x="Dataset",
        y="Count",
        color="State",
        barmode="group",
        text_auto=True,
        color_discrete_map={"Stable (0)": COLORS["stable"], "Chatter (1)": COLORS["chatter"]},
    )
    fig.update_traces(marker_line_width=0)
    return _apply_base(fig, "Stable vs. Chatter Experiment Distribution by Dataset", height=360)


def plot_feature_distribution(
    df: pd.DataFrame,
    feature: str,
    log_y: bool = False,
) -> go.Figure:
    temp = df.assign(State=df["Chatter"].map({0: "Stable (0)", 1: "Chatter (1)"}))
    fig = px.box(
        temp,
        x="State",
        y=feature,
        color="State",
        points="all",
        hover_data=[c for c in ["Dataset", "File", "Omega", "Depth"] if c in temp.columns],
        color_discrete_map={"Stable (0)": COLORS["stable"], "Chatter (1)": COLORS["chatter"]},
    )
    if log_y:
        fig.update_yaxes(type="log")
    return _apply_base(fig, f"{feature} Distribution by Stability State", height=400)


def plot_correlation_heatmap(df: pd.DataFrame, cols: List[str]) -> go.Figure:
    valid_cols = [c for c in cols if c in df.columns]
    corr = df[valid_cols].corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=11),
            hovertemplate="%{y} vs %{x}<br>r = %{z:.3f}<extra></extra>",
        )
    )
    return _apply_base(fig, "Feature Correlation Matrix (Pearson r)", height=430)


def plot_feature_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    log_x: bool = False,
    log_y: bool = False,
) -> go.Figure:
    temp = df.assign(
        State=df["Chatter"].map({0: "Stable (0)", 1: "Chatter (1)"}),
        Dataset_Str="D" + df["Dataset"].astype(str),
    )
    fig = px.scatter(
        temp,
        x=x_col,
        y=y_col,
        color="State",
        symbol="Dataset_Str",
        hover_data=[c for c in ["File", "Omega", "Depth", "Critical Depth"] if c in temp.columns],
        color_discrete_map={"Stable (0)": COLORS["stable"], "Chatter (1)": COLORS["chatter"]},
    )
    fig.update_traces(marker=dict(size=8, opacity=0.85))
    if log_x:
        fig.update_xaxes(type="log")
    if log_y:
        fig.update_yaxes(type="log")
    return _apply_base(fig, f"{y_col} vs. {x_col} Across Datasets", height=430)


def plot_stability_lobes(
    boundaries: Dict[int, Tuple[np.ndarray, np.ndarray]],
    experiments_df: Optional[pd.DataFrame] = None,
    highlight_point: Optional[Tuple[float, float, str]] = None,
) -> go.Figure:
    fig = go.Figure()
    palette = ["#38BDF8", "#2DD4BF", "#F59E0B", "#A78BFA", "#F472B6"]

    for idx, (ds, (omega_b, b_crit)) in enumerate(sorted(boundaries.items())):
        step = max(1, len(omega_b) // 800)
        fig.add_trace(
            go.Scatter(
                x=omega_b[::step],
                y=b_crit[::step] * 1000.0,
                mode="lines",
                name=f"Dataset {ds} Stability Boundary",
                line=dict(color=palette[idx % len(palette)], width=2.2),
                hovertemplate=f"Dataset {ds}<br>Speed: %{{x:.0f}} rpm<br>Critical DOC: %{{y:.3f}} mm<extra></extra>",
            )
        )

    if experiments_df is not None and not experiments_df.empty:
        stable = experiments_df[experiments_df["Chatter"] == 0]
        chatter = experiments_df[experiments_df["Chatter"] == 1]
        if not stable.empty:
            fig.add_trace(
                go.Scatter(
                    x=stable["Omega"],
                    y=stable["Depth"] * 1000.0,
                    mode="markers",
                    name="Stable Cuts (0)",
                    marker=dict(color=COLORS["stable"], size=6, opacity=0.78, symbol="circle"),
                    hovertemplate="Stable<br>Omega: %{x:.0f} rpm<br>DOC: %{y:.3f} mm<extra></extra>",
                )
            )
        if not chatter.empty:
            fig.add_trace(
                go.Scatter(
                    x=chatter["Omega"],
                    y=chatter["Depth"] * 1000.0,
                    mode="markers",
                    name="Chatter Cuts (1)",
                    marker=dict(color=COLORS["chatter"], size=7, opacity=0.85, symbol="x"),
                    hovertemplate="Chatter<br>Omega: %{x:.0f} rpm<br>DOC: %{y:.3f} mm<extra></extra>",
                )
            )

    if highlight_point is not None:
        hx, hy_m, hlabel = highlight_point
        fig.add_trace(
            go.Scatter(
                x=[hx],
                y=[hy_m * 1000.0],
                mode="markers",
                name=hlabel,
                marker=dict(color="#FACC15", size=14, symbol="star", line=dict(color="#0B1120", width=1.5)),
                hovertemplate=f"{hlabel}<br>Omega: %{{x:.0f}} rpm<br>DOC: %{{y:.3f}} mm<extra></extra>",
            )
        )

    fig.update_xaxes(title_text="Spindle Speed, Omega (rpm)")
    fig.update_yaxes(title_text="Axial Depth of Cut, b (mm)")
    return _apply_base(fig, "Analytical Stability Lobe Diagram (SLD)", height=460)


def plot_model_comparison_bars(summary_df: pd.DataFrame, title: str) -> go.Figure:
    metrics = ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1", "MCC"]
    present = [m for m in metrics if m in summary_df.columns]
    melted = summary_df.melt(id_vars="Model", value_vars=present, var_name="Metric", value_name="Score")
    fig = px.bar(
        melted,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text=melted["Score"].map(lambda v: f"{v:.3f}"),
        color_discrete_sequence=["#38BDF8", "#2DD4BF", "#818CF8"],
    )
    ymin = -0.2 if float(melted["Score"].min()) < 0 else 0.0
    fig.update_yaxes(range=[ymin, 1.08])
    return _apply_base(fig, title, height=420)


def plot_per_dataset_lodo(fold_df: pd.DataFrame, metric: str = "F1") -> go.Figure:
    temp = fold_df.copy()
    temp["Unseen Test Dataset"] = "Dataset " + temp["Test Dataset"].astype(str)
    fig = px.bar(
        temp,
        x="Unseen Test Dataset",
        y=metric,
        color="Model",
        barmode="group",
        text=temp[metric].map(lambda v: f"{v:.3f}"),
        color_discrete_sequence=["#38BDF8", "#2DD4BF", "#818CF8"],
    )
    ymin = -0.2 if float(temp[metric].min()) < 0 else 0.0
    fig.update_yaxes(range=[ymin, 1.08])
    return _apply_base(fig, f"Per-Dataset Generalization (LODO {metric})", height=400)


def plot_confusion_matrix(cm: List[List[int]], title: str) -> go.Figure:
    z = np.asarray(cm, dtype=int)
    labels = ["Stable (0)", "Chatter (1)"]
    total = int(z.sum()) or 1
    text = [[f"{z[r, c]}<br>({z[r, c] / total * 100:.1f}%)" for c in range(2)] for r in range(2)]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=14, color="#F8FAFC"),
            colorscale=[[0.0, "#0F172A"], [0.5, "#1D4ED8"], [1.0, "#38BDF8"]],
            showscale=False,
        )
    )
    fig.update_xaxes(title_text="Predicted Class")
    fig.update_yaxes(title_text="True Physics Class", autorange="reversed")
    return _apply_base(fig, title, height=340)


def plot_feature_importances(importances: Dict[str, float], title: str) -> go.Figure:
    s = pd.Series(importances).sort_values(ascending=True)
    fig = go.Figure(
        go.Bar(
            x=s.values,
            y=s.index,
            orientation="h",
            marker_color=COLORS["cyan"],
            text=[f"{v:.3f}" for v in s.values],
            textposition="outside",
        )
    )
    fig.update_xaxes(title_text="Relative Importance", range=[0, max(float(s.values.max()) * 1.18, 0.1)])
    return _apply_base(fig, title, height=340)


def plot_time_signal(
    time: np.ndarray,
    signal: np.ndarray,
    title: str,
    rms: Optional[float] = None,
    color: str = "#38BDF8",
    max_points: int = 8000,
) -> go.Figure:
    step = max(1, len(time) // max_points)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time[::step],
            y=signal[::step],
            mode="lines",
            name="Acceleration (xtacc)",
            line=dict(color=color, width=1.1),
            hovertemplate="Time: %{x:.5f} s<br>Acc: %{y:.2f} m/s^2<extra></extra>",
        )
    )
    if rms is not None:
        fig.add_hline(
            y=rms,
            line_dash="dot",
            line_color="#F59E0B",
            annotation_text=f"+RMS ({rms:.2f})",
            annotation_font_color="#F59E0B",
        )
        fig.add_hline(y=-rms, line_dash="dot", line_color="#F59E0B")
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Tool X-Acceleration (m/s^2)")
    return _apply_base(fig, title, height=360)


def plot_fft_spectrum(
    freqs: np.ndarray,
    amp: np.ndarray,
    dom_freq: float,
    max_fft: float,
    freq_range: Optional[Tuple[float, float]] = None,
) -> go.Figure:
    mask = np.ones(len(freqs), dtype=bool)
    if freq_range is not None:
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])

    f_plot = freqs[mask]
    a_plot = amp[mask]
    step = max(1, len(f_plot) // 8000)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=f_plot[::step],
            y=a_plot[::step],
            mode="lines",
            name="Windowed rFFT Magnitude",
            line=dict(color=COLORS["teal"], width=1.4),
            fill="tozeroy",
            fillcolor="rgba(45, 212, 191, 0.12)",
            hovertemplate="Freq: %{x:.1f} Hz<br>Amp: %{y:.3f} m/s^2<extra></extra>",
        )
    )
    if freq_range is None or (freq_range[0] <= dom_freq <= freq_range[1]):
        fig.add_trace(
            go.Scatter(
                x=[dom_freq],
                y=[max_fft],
                mode="markers+text",
                name="Dominant Peak",
                text=[f"Peak: {dom_freq:.1f} Hz ({max_fft:.2f})"],
                textposition="top right",
                marker=dict(color=COLORS["chatter"], size=10, symbol="diamond"),
                textfont=dict(color="#FCA5A5"),
            )
        )
    fig.update_xaxes(title_text="Frequency (Hz)")
    fig.update_yaxes(title_text="Normalized Amplitude (m/s^2)")
    return _apply_base(fig, "Single-Sided Hanning-Windowed FFT Spectrum (DC Removed)", height=380)


def plot_multichannel_rms(channel_rms: Dict[str, float]) -> go.Figure:
    names = list(channel_rms.keys())
    vals = [channel_rms[k] for k in names]
    colors = ["#38BDF8", "#2DD4BF", "#64748B", "#475569"]
    fig = go.Figure(
        go.Bar(
            x=names,
            y=vals,
            marker_color=colors[: len(names)],
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
        )
    )
    fig.update_xaxes(title_text="Accelerometer Channel (Tool vs. Workpiece)")
    fig.update_yaxes(title_text="RMS Acceleration (m/s^2)")
    return _apply_base(fig, "4-Channel Acceleration RMS Comparison (xtacc/ytacc vs. xwacc/ywacc)", height=340)


def plot_probability_gauge(prob_chatter: float) -> go.Figure:
    pct = prob_chatter * 100.0
    bar_color = COLORS["chatter"] if prob_chatter >= 0.5 else COLORS["stable"]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"color": "#F8FAFC", "size": 32}},
            title={"text": "Chatter Probability", "font": {"color": "#94A3B8", "size": 14}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748B"},
                "bar": {"color": bar_color},
                "bgcolor": "#0F172A",
                "bordercolor": "#1E293B",
                "steps": [
                    {"range": [0, 50], "color": "rgba(34, 197, 94, 0.16)"},
                    {"range": [50, 100], "color": "rgba(239, 68, 68, 0.16)"},
                ],
                "threshold": {
                    "line": {"color": "#F8FAFC", "width": 2},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F1F5F9"),
        height=260,
        margin=dict(l=25, r=25, t=40, b=15),
    )
    return fig
