"""Industrial AI dark-theme styling helpers and reusable UI components."""

import streamlit as st

COLORS = {
    "bg": "#0B1120",
    "surface": "#111827",
    "surface_alt": "#162032",
    "border": "#1E293B",
    "text": "#F1F5F9",
    "muted": "#94A3B8",
    "cyan": "#38BDF8",
    "teal": "#2DD4BF",
    "blue": "#3B82F6",
    "stable": "#22C55E",
    "chatter": "#EF4444",
    "warning": "#F59E0B",
}


def apply_industrial_theme() -> None:
    """Inject clean dark industrial analytics CSS across all Streamlit pages."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        code, pre, .mono {
            font-family: 'JetBrains Mono', monospace;
        }
        .stApp {
            background-color: #0B1120;
            color: #F1F5F9;
        }
        section[data-testid="stSidebar"] {
            background-color: #0F172A;
            border-right: 1px solid #1E293B;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1320px;
        }
        .hero-card {
            background: linear-gradient(135deg, #111827 0%, #0F1E36 100%);
            border: 1px solid #1E3A5F;
            border-radius: 12px;
            padding: 1.75rem 2rem;
            margin-bottom: 1.5rem;
        }
        .hero-kicker {
            color: #38BDF8;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }
        .hero-title {
            font-size: 1.95rem;
            font-weight: 700;
            color: #F8FAFC;
            line-height: 1.2;
            margin: 0 0 0.6rem 0;
        }
        .hero-subtitle {
            color: #94A3B8;
            font-size: 0.96rem;
            line-height: 1.55;
            margin: 0;
        }
        .kpi-card {
            background-color: #111827;
            border: 1px solid #1E293B;
            border-top: 3px solid #38BDF8;
            border-radius: 10px;
            padding: 1.1rem 1.25rem;
            height: 100%;
        }
        .kpi-card.teal { border-top-color: #2DD4BF; }
        .kpi-card.green { border-top-color: #22C55E; }
        .kpi-card.red { border-top-color: #EF4444; }
        .kpi-card.amber { border-top-color: #F59E0B; }
        .kpi-label {
            color: #94A3B8;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }
        .kpi-value {
            color: #F8FAFC;
            font-size: 1.55rem;
            font-weight: 700;
            line-height: 1.18;
            font-family: 'JetBrains Mono', monospace;
        }
        .kpi-sub {
            color: #64748B;
            font-size: 0.79rem;
            margin-top: 0.35rem;
        }
        .info-panel {
            background-color: #111827;
            border: 1px solid #1E293B;
            border-radius: 10px;
            padding: 1.25rem 1.4rem;
            margin-bottom: 1rem;
        }
        .status-banner-stable {
            background: rgba(34, 197, 94, 0.12);
            border: 1px solid #22C55E;
            border-left: 5px solid #22C55E;
            border-radius: 10px;
            padding: 1.2rem 1.45rem;
            margin-bottom: 1rem;
        }
        .status-banner-chatter {
            background: rgba(239, 68, 68, 0.12);
            border: 1px solid #EF4444;
            border-left: 5px solid #EF4444;
            border-radius: 10px;
            padding: 1.2rem 1.45rem;
            margin-bottom: 1rem;
        }
        .status-banner-warn {
            background: rgba(245, 158, 11, 0.10);
            border: 1px solid #F59E0B;
            border-left: 5px solid #F59E0B;
            border-radius: 10px;
            padding: 1rem 1.25rem;
            margin-bottom: 1rem;
        }
        .pipeline-step {
            background-color: #111827;
            border: 1px solid #1E293B;
            border-radius: 8px;
            padding: 0.9rem;
            text-align: center;
            height: 100%;
        }
        .pipeline-step-num {
            color: #38BDF8;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.08em;
        }
        .pipeline-step-title {
            color: #F8FAFC;
            font-size: 0.92rem;
            font-weight: 600;
            margin: 0.3rem 0;
        }
        .pipeline-step-desc {
            color: #94A3B8;
            font-size: 0.78rem;
            line-height: 1.4;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #1E293B;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-kicker">{kicker}</div>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value: str, sub: str = "", accent: str = "") -> None:
    cls = f"kpi-card {accent}".strip()
    st.markdown(
        f"""
        <div class="{cls}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_footer(model_ready: bool, active_experiment: str = "") -> None:
    with st.sidebar:
        st.markdown("---")
        status_color = "#22C55E" if model_ready else "#F59E0B"
        status_text = "Model Artifacts Loaded" if model_ready else "Awaiting Local Training"
        exp_line = (
            f"<div style='color:#94A3B8;font-size:0.78rem;margin-top:4px;'>Experiment: <b>{active_experiment}</b></div>"
            if active_experiment
            else ""
        )
        st.markdown(
            f"""
            <div style="background:#111827;border:1px solid #1E293B;border-radius:8px;padding:0.85rem;">
                <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;color:#64748B;">System Status</div>
                <div style="margin-top:4px;font-size:0.85rem;font-weight:600;color:{status_color};">● {status_text}</div>
                {exp_line}
                <div style="color:#64748B;font-size:0.74rem;margin-top:8px;">
                    Simulated Milling Research Prototype<br/>
                    Digital Machining Database (Schmitz &amp; Smith)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
