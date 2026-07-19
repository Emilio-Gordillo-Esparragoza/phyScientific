"""
Streamlit dashboard:
  Sidebar — dataset picker (active_matter / gray_scott / acoustic_scattering)
  Tab 1 — Interactive ANOVA teaching sandbox
  Tab 2 — Real-data ANOVA on selected dataset features
  Tab 3 — Physics validation & anomalies

Visual language: lab notebook / chart paper — ink, graphite, muted olive & ochre.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_catalog import (  # noqa: E402
    DATASET_IDS,
    DatasetSpec,
    feature_path_for,
    get_dataset,
)

# Lab-board palette (ink / paper / graphite / olive / ochre)
PALETTE = {
    "paper": "#E6E2D6",
    "paper_deep": "#D8D2C2",
    "ink": "#1A1814",
    "graphite": "#5C574E",
    "grid": "#C4BDAE",
    "olive": "#5F6B45",
    "olive_soft": "#8A9470",
    "ochre": "#9A7340",
    "ochre_soft": "#C4A06A",
    "pass": "#4A5D3A",
    "fail": "#7A3E32",
    "muted": "#7A756A",
}

VERDICT_COLORS = {
    "strong_evidence": PALETTE["olive"],
    "moderate_evidence": PALETTE["ochre"],
    "weak_evidence": PALETTE["ochre_soft"],
    "no_visible_difference": PALETTE["muted"],
}

PLOTLY_COLORS = [
    PALETTE["olive"],
    PALETTE["ochre"],
    "#3D5A56",
    "#7A3E32",
    "#5C574E",
    "#8A9470",
    "#9A7340",
    "#4A5D3A",
]


def inject_lab_css() -> None:
    # Use st.html (not st.markdown): markdown sanitization strips <style>,
    # which leaks raw CSS rules as visible page text.
    css = f"""
<style>
@import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Source+Sans+3:ital,wght@0,400;0,500;0,600;1,400&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600;8..60,700&display=swap");

:root {{
    --paper: {PALETTE["paper"]};
    --paper-deep: {PALETTE["paper_deep"]};
    --ink: {PALETTE["ink"]};
    --graphite: {PALETTE["graphite"]};
    --grid: {PALETTE["grid"]};
    --olive: {PALETTE["olive"]};
    --ochre: {PALETTE["ochre"]};
    --muted: {PALETTE["muted"]};
}}

html, body, [data-testid="stAppViewContainer"] {{
    background:
        linear-gradient(90deg, transparent 49px, var(--grid) 49px, var(--grid) 50px, transparent 50px),
        linear-gradient(var(--grid) 1px, transparent 1px),
        linear-gradient(90deg, var(--grid) 1px, transparent 1px),
        var(--paper) !important;
    background-size: 50px 50px, 10px 10px, 10px 10px, auto !important;
    background-position: -1px -1px, 0 0, 0 0, 0 0 !important;
    color: var(--ink);
    font-family: "Source Sans 3", "Segoe UI", sans-serif;
}}

[data-testid="stHeader"] {{
    background: transparent !important;
}}

.block-container {{
    padding-top: 1.4rem !important;
    padding-bottom: 3rem !important;
    max-width: 1180px !important;
}}

h1, h2, h3, .lab-brand {{
    font-family: "Source Serif 4", Georgia, serif !important;
    color: var(--ink) !important;
    letter-spacing: -0.01em;
    font-weight: 600 !important;
}}

h1 {{
    font-size: 2.05rem !important;
    margin-bottom: 0.15rem !important;
    border-bottom: 1.5px solid var(--ink);
    padding-bottom: 0.45rem;
}}

.lab-kicker {{
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--olive);
    margin: 0 0 0.35rem 0;
}}

.lab-kicker a {{
    color: var(--ochre);
    text-decoration: underline;
    text-underline-offset: 0.15em;
}}

.lab-lede {{
    font-family: "Source Serif 4", Georgia, serif;
    font-size: 1.05rem;
    color: var(--graphite);
    max-width: 42rem;
    line-height: 1.45;
    margin: 0.55rem 0 1.1rem 0;
}}

.lab-meta {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem 1.1rem;
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.75rem;
    color: var(--graphite);
    border-top: 1px dashed var(--grid);
    border-bottom: 1px dashed var(--grid);
    padding: 0.55rem 0;
    margin-bottom: 1.15rem;
}}

.lab-meta b {{
    color: var(--ink);
    font-weight: 600;
}}

.lab-meta a {{
    color: var(--olive);
    font-weight: 600;
    text-decoration: underline;
    text-underline-offset: 0.12em;
}}

.lab-dataset {{
    background: var(--paper-deep);
    border: 1px solid var(--grid);
    border-left: 3px solid var(--graphite);
    padding: 0.7rem 1rem;
    margin: 0 0 1.05rem 0;
    max-width: 52rem;
}}
.lab-dataset-label {{
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--graphite);
    margin-bottom: 0.35rem;
}}
.lab-dataset p {{
    font-family: "Source Sans 3", "Segoe UI", sans-serif;
    font-size: 0.92rem;
    color: var(--ink);
    line-height: 1.5;
    margin: 0;
}}
.lab-dataset code {{
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.85em;
    color: var(--olive);
}}

.lab-findings {{
    background: var(--paper-deep);
    border: 1px solid var(--ink);
    border-left: 4px solid var(--olive);
    padding: 0.75rem 1rem;
    margin: 0 0 1.2rem 0;
}}
.lab-findings-label {{
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--olive);
    margin-bottom: 0.35rem;
}}
.lab-findings p {{
    font-family: "Source Serif 4", Georgia, serif;
    font-size: 0.98rem;
    color: var(--ink);
    line-height: 1.45;
    margin: 0;
}}

.lab-footer {{
    margin-top: 2.4rem;
    padding-top: 0.85rem;
    border-top: 1px solid var(--ink);
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.72rem;
    color: var(--graphite);
    letter-spacing: 0.02em;
    line-height: 1.55;
}}
.lab-footer b {{
    color: var(--ink);
    font-weight: 600;
}}
.lab-footer a {{
    color: var(--olive);
    text-decoration: underline;
    text-underline-offset: 0.12em;
}}

/* Section switcher (replaces tabs so only one panel runs) */
.stRadio [role="radiogroup"] {{
    gap: 0.15rem;
    border-bottom: 1.5px solid var(--ink);
    padding-bottom: 0.15rem;
    margin-bottom: 0.85rem;
}}
.stRadio [data-testid="stMarkdownContainer"] p {{
    font-family: "IBM Plex Mono", ui-monospace, monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.04em;
}}
.stTabs [data-baseweb="tab-list"] {{
    gap: 0.15rem;
    border-bottom: 1.5px solid var(--ink);
    background: transparent;
}}
.stTabs [data-baseweb="tab"] {{
    font-family: "IBM Plex Mono", ui-monospace, monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.04em;
    color: var(--graphite) !important;
    background: transparent !important;
    border: none !important;
    padding: 0.55rem 0.9rem !important;
}}
.stTabs [aria-selected="true"] {{
    color: var(--ink) !important;
    background: var(--paper-deep) !important;
    border-bottom: 3px solid var(--ochre) !important;
}}

/* Instrument readouts */
div[data-testid="stMetric"] {{
    background: var(--paper-deep);
    border: 1px solid var(--ink);
    padding: 0.65rem 0.75rem 0.55rem;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.35);
}}
div[data-testid="stMetric"] label {{
    font-family: "IBM Plex Mono", ui-monospace, monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--graphite) !important;
}}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    font-family: "IBM Plex Mono", ui-monospace, monospace !important;
    font-size: 1.35rem !important;
    font-weight: 600 !important;
    color: var(--ink) !important;
}}

/* Controls */
.stSelectbox label, .stSlider label, .stNumberInput label {{
    font-family: "IBM Plex Mono", ui-monospace, monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--graphite) !important;
}}

/* Align slider tracks when labels differ in length */
div[data-testid="column"] .stSlider {{
    margin-top: 0;
}}
.ctrl-label {{
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.72rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--graphite);
    min-height: 2.4rem;
    display: flex;
    align-items: flex-end;
    margin: 0 0 0.35rem 0;
    line-height: 1.25;
}}
.lab-note {{
    font-family: "Source Serif 4", Georgia, serif;
    font-size: 0.95rem;
    color: var(--graphite);
    line-height: 1.45;
    max-width: 46rem;
    margin: 0.35rem 0 1.1rem 0;
}}
.lab-controls {{
    margin: 0.85rem 0 1.1rem 0;
    padding: 0;
    background: transparent;
    border: none;
}}

/* Prevent empty HTML “chrome” boxes from covering sliders / radios */
div.lab-controls:empty,
div.stElementContainer:has(> div.lab-controls:empty) {{
    display: none !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
}}
.stRadio, .stRadio > div, .stRadio [role="radiogroup"],
div[data-testid="stSlider"],
div[data-baseweb="slider"],
div[data-testid="stNumberInput"],
div[data-testid="stSelectbox"],
div[data-testid="stWidgetLabel"],
div[data-testid="column"] {{
    background: transparent !important;
}}
div[data-testid="stHorizontalBlock"] {{
    background: transparent !important;
}}
/* Kill opaque Streamlit widget chrome that reads as a beige slab */
[data-baseweb="select"] > div,
[data-baseweb="base-input"],
[data-baseweb="input"],
[data-baseweb="textarea"],
div[data-testid="stNumberInputContainer"],
div[data-testid="stSelectbox"] > div,
div[data-testid="stSlider"] input {{
    background-color: transparent !important;
    background-image: none !important;
    box-shadow: none !important;
}}
div[data-testid="stSlider"] input {{
    border: 1px solid var(--grid) !important;
    color: var(--ink) !important;
}}
/* Keep metric / verdict cards as intentional paper-deep panels only */

/* Expanders / dataframes */
.streamlit-expanderHeader {{
    font-family: "Source Serif 4", Georgia, serif !important;
    color: var(--ink) !important;
}}
[data-testid="stDataFrame"] {{
    border: 1px solid var(--grid);
}}

/* Info / alert restyle */
[data-testid="stAlert"] {{
    background: var(--paper-deep) !important;
    border: 1px solid var(--olive) !important;
    color: var(--ink) !important;
}}

/* Hamburger (collapsed sidebar control) — three-line drawer */
[data-testid="collapsedControl"] {{
    background: var(--paper-deep) !important;
    border: 1.5px solid var(--ink) !important;
    border-radius: 2px !important;
    color: var(--ink) !important;
    top: 0.65rem !important;
    left: 0.65rem !important;
    width: 2.35rem !important;
    height: 2.35rem !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 1px 1px 0 var(--grid);
    z-index: 1000 !important;
}}
[data-testid="collapsedControl"] svg {{
    fill: var(--ink) !important;
    stroke: var(--ink) !important;
}}
[data-testid="collapsedControl"]:hover {{
    background: var(--paper) !important;
    border-color: var(--olive) !important;
}}

/* Sidebar panel — same lab board language */
section[data-testid="stSidebar"] {{
    background:
        linear-gradient(90deg, transparent 49px, var(--grid) 49px, var(--grid) 50px, transparent 50px),
        linear-gradient(var(--grid) 1px, transparent 1px),
        linear-gradient(90deg, var(--grid) 1px, transparent 1px),
        var(--paper) !important;
    background-size: 50px 50px, 10px 10px, 10px 10px, auto !important;
    border-right: 1.5px solid var(--ink) !important;
}}
section[data-testid="stSidebar"] > div {{
    background: transparent !important;
}}
.lab-sidebar-kicker {{
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--olive);
    margin: 0.35rem 0 0.75rem 0;
}}
.lab-sidebar-hint {{
    font-family: "Source Sans 3", "Segoe UI", sans-serif;
    font-size: 0.85rem;
    color: var(--graphite);
    line-height: 1.4;
    margin: 0.5rem 0 1rem 0;
}}

/* Mobile */
@media (max-width: 768px) {{
    h1 {{ font-size: 1.55rem !important; }}
    .lab-lede {{ font-size: 0.98rem; }}
    .block-container {{ padding-left: 1rem !important; padding-right: 1rem !important; }}
}}

@media (prefers-reduced-motion: reduce) {{
    * {{ animation: none !important; transition: none !important; }}
}}
</style>
"""
    st.html(css)


def apply_plotly_theme(fig: go.Figure, height: int = 400) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=48, r=24, t=56, b=44),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(230,226,214,0.55)",
        font=dict(family="Source Sans 3, sans-serif", color=PALETTE["ink"], size=13),
        title=dict(
            font=dict(family="Source Serif 4, Georgia, serif", size=16, color=PALETTE["ink"]),
            x=0.0,
            xanchor="left",
        ),
        legend=dict(
            bgcolor="rgba(230,226,214,0.85)",
            bordercolor=PALETTE["grid"],
            borderwidth=1,
            font=dict(family="IBM Plex Mono, monospace", size=11),
        ),
        colorway=PLOTLY_COLORS,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=PALETTE["grid"],
        zeroline=False,
        linecolor=PALETTE["ink"],
        tickfont=dict(family="IBM Plex Mono, monospace", size=11),
        title_font=dict(family="Source Sans 3, sans-serif", size=12),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=PALETTE["grid"],
        zeroline=False,
        linecolor=PALETTE["ink"],
        tickfont=dict(family="IBM Plex Mono, monospace", size=11),
        title_font=dict(family="Source Sans 3, sans-serif", size=12),
    )
    return fig


@st.cache_data(show_spinner=False)
def load_features(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_parquet(path)


def load_features_or_stop(spec: DatasetSpec) -> pd.DataFrame:
    path = feature_path_for(spec.id)
    try:
        return load_features(str(path))
    except FileNotFoundError:
        st.error(
            f"Missing `{path}`. Run:\n\n"
            f"`python -m src.extract_features --dataset {spec.id} --splits train`\n\n"
            "or, as a demo fallback only:\n\n"
            f"`python -m src.extract_features --dataset {spec.id} --synthetic`"
        )
        st.stop()


def data_provenance(df: pd.DataFrame, spec: DatasetSpec) -> dict:
    is_synth = bool(df.get("synthetic", pd.Series([False] * len(df))).fillna(False).any())
    fa, fb = spec.factor_a, spec.factor_b
    n_a = int(df[fa].nunique()) if fa in df.columns else 0
    n_b = int(df[fb].nunique()) if fb in df.columns else 0
    n_cells = (
        int(df.groupby([fa, fb]).ngroups)
        if {fa, fb} <= set(df.columns)
        else 0
    )
    return {
        "synthetic": is_synth,
        "n_rows": len(df),
        "n_factor_a": n_a,
        "n_factor_b": n_b,
        "n_cells": n_cells,
    }


AUTHOR = "Emilio Gordillo Esparragoza"


def render_dataset_sidebar() -> DatasetSpec:
    """Collapsed sidebar drawer: pick which Well feature table to analyze."""
    if "dataset_id" not in st.session_state:
        st.session_state["dataset_id"] = "active_matter"

    with st.sidebar:
        st.markdown('<p class="lab-sidebar-kicker">Dataset</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="lab-sidebar-hint">Choose a The Well ensemble. '
            "active_matter keeps ANOVA; gray_scott and acoustic use phase / "
            "interaction views instead.</p>",
            unsafe_allow_html=True,
        )
        labels = {
            "active_matter": "active_matter",
            "gray_scott": "gray_scott",
            "acoustic_scattering": "acoustic_scattering",
        }
        choice = st.radio(
            "Ensemble",
            list(DATASET_IDS),
            format_func=lambda i: labels.get(i, i),
            key="dataset_id",
            label_visibility="collapsed",
        )
        spec = get_dataset(choice)
        st.caption(f"HF · `{spec.hf_name}`")
        st.markdown(f"[Dataset card]({spec.hf_url})")
        st.markdown(
            f"<p class='lab-sidebar-hint'>Factors: "
            f"<code>{spec.factor_a}</code> × <code>{spec.factor_b}</code></p>",
            unsafe_allow_html=True,
        )
    return get_dataset(st.session_state["dataset_id"])


def lab_header(df: pd.DataFrame, spec: DatasetSpec) -> None:
    meta = data_provenance(df, spec)
    findings_html = ""
    if not meta["synthetic"] and spec.findings_real:
        findings_html = f"""
        <div class="lab-findings">
          <div class="lab-findings-label">Results at a glance</div>
          <p>
            {spec.findings_real.format(**meta)}
          </p>
        </div>
        """
    st.html(
        f"""
        <div class="lab-kicker">
          Statistical laboratory
        </div>
        <h1 class="lab-brand">{spec.title}</h1>
        <p class="lab-lede">
          {spec.lede}
        </p>
        <div class="lab-meta">
          <span>trajectories <b>{meta["n_rows"]}</b></span>
          <span>{spec.factor_a_label} levels <b>{meta["n_factor_a"]}</b></span>
          <span>{spec.factor_b_label} levels <b>{meta["n_factor_b"]}</b></span>
          <span>factorial cells <b>{meta["n_cells"]}</b></span>
        </div>
        <div class="lab-dataset">
          <div class="lab-dataset-label">What is the {spec.id} dataset?</div>
          <p>
            {spec.blurb}
          </p>
        </div>
        {findings_html}
        """
    )


def lab_footer(spec: DatasetSpec) -> None:
    st.html(
        f"""
        <div class="lab-footer">
          Author · <b>{AUTHOR}</b><br/>
          Dataset · {spec.footer_label}
        </div>
        """
    )


def verdict_banner(text: str, level: str) -> None:
    color = VERDICT_COLORS.get(level, PALETTE["ink"])
    st.markdown(
        f"""
        <div style="padding:0.85rem 1rem;margin:0.4rem 0 1rem 0;
                    background:{PALETTE["paper_deep"]};
                    border:1px solid {PALETTE["ink"]};
                    border-left:5px solid {color};
                    font-family:'Source Serif 4',Georgia,serif;
                    font-size:1.12rem;font-weight:600;color:{PALETTE["ink"]};">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
                         letter-spacing:0.12em;text-transform:uppercase;
                         color:{color};display:block;margin-bottom:0.25rem;">
                Verdict
            </span>
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def variance_decomposition_fig(ss_between: float, ss_within: float) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(
                x=["Between groups (SSB)", "Within groups (SSW)"],
                y=[ss_between, ss_within],
                marker_color=[PALETTE["olive"], PALETTE["ochre"]],
                marker_line=dict(color=PALETTE["ink"], width=1),
                text=[f"{ss_between:.3f}", f"{ss_within:.3f}"],
                textposition="auto",
                textfont=dict(family="IBM Plex Mono, monospace"),
            )
        ]
    )
    fig.update_layout(
        title="Variance decomposition",
        yaxis_title="Sum of squares",
        showlegend=False,
    )
    return apply_plotly_theme(fig, height=360)


def fig_note(text: str) -> None:
    """Short caption under a plot or table (supports **bold** via HTML)."""
    import html
    import re

    escaped = html.escape(text)
    rich = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    st.markdown(f'<p class="lab-note">{rich}</p>', unsafe_allow_html=True)


def ctrl_label(text: str) -> None:
    st.markdown(f'<div class="ctrl-label">{text}</div>', unsafe_allow_html=True)


def glossary_block() -> None:
    with st.expander("Glossary — variables, plots, and statistics", expanded=False):
        st.markdown(
            r"""
**Control / physics factors**
- **α (alpha)** — dipole strength of the active rods (more negative → stronger activity).
- **ζ (zeta)** — alignment strength; larger ζ favors ordered (nematic-like) states.
- **L** — domain linear size (fixed at 10 in this extract).

**Response features (per trajectory)**
- **nematic_order_S** — orientation-order magnitude from the orientation tensor (higher ⇒ more ordered).
- **nematic_order_S_final** — order near the end of the trajectory.
- **kinetic_energy** — mean \(\tfrac12|u|^2\) of the velocity field.
- **enstrophy** — mean squared vorticity (spin / shear intensity).
- **mean_concentration / std_concentration** — average particle density and its spatial spread.
- **div_u_rms** — RMS of discrete \(\nabla\cdot u\) (incompressibility residual; coarse grids inflate it).
- **spectral_slope** — log–log slope of the kinetic-energy spectrum.
- **time_to_steady** — fraction of the run until order plateaus.

**ANOVA / inference**
- **F** — between-group variance ÷ within-group variance (\(\mathrm{MSB}/\mathrm{MSW}\)). Large F ⇒ group means differ relative to noise.
- **p-value** — probability of seeing an F this large (or larger) if all group means were equal.
- **SSB / SSW (SSE)** — sum of squares between groups / within groups (error).
- **η² (eta squared)** — fraction of total variance explained by the factor (effect size).
- **ω² (omega squared)** — less biased effect-size cousin of η².
- **Tukey HSD** — which pairs of levels differ after ANOVA.
- **Levene / Shapiro** — checks equal variances and roughly normal residuals.
- **MAD / z-score / IQR anomalies** — trajectories unusual *within* their (α, ζ) cell.
            """
        )


# ---------------------------------------------------------------------------
# Tab 1 — teaching sandbox
# ---------------------------------------------------------------------------
def tab_teaching() -> None:
    from src.stats import simulate_anova_groups

    st.markdown("### Interactive ANOVA sandbox")
    st.markdown(
        "Change how far group means sit apart and how noisy each group is, "
        "then watch **F**, **p**, **SSE (SSW)**, and **η²** update. "
        r"Recall $F = \mathrm{MSB}/\mathrm{MSW}$ "
        "(variation between groups / variation within groups)."
    )

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        ctrl_label("Number of groups")
        n_groups = st.slider(
            "Number of groups",
            2,
            6,
            3,
            label_visibility="collapsed",
            help="How many populations to compare.",
        )
    with c2:
        ctrl_label("n per group")
        n_per = st.slider(
            "n per group",
            5,
            80,
            20,
            label_visibility="collapsed",
            help="Sample size within each group.",
        )
    with c3:
        ctrl_label("Mean difference")
        mean_diff = st.slider(
            "Difference between means",
            0.0,
            5.0,
            1.0,
            0.05,
            label_visibility="collapsed",
            help="Spacing between adjacent group means (effect size knob).",
        )
    with c4:
        ctrl_label("Within-group SD")
        within_sd = st.slider(
            "Dispersion within groups (SD)",
            0.1,
            5.0,
            1.0,
            0.05,
            label_visibility="collapsed",
            help="Noise inside each group. Larger SD shrinks F and raises p.",
        )

    s1, _, _ = st.columns([1, 1, 2], gap="medium")
    with s1:
        ctrl_label("Random seed")
        seed = st.number_input(
            "Random seed",
            min_value=0,
            value=0,
            step=1,
            label_visibility="collapsed",
            help="Fixes the random draws so the same knobs give the same sample.",
        )

    result = simulate_anova_groups(
        n_groups=n_groups,
        n_per_group=n_per,
        mean_diff=mean_diff,
        within_sd=within_sd,
        seed=int(seed),
    )

    verdict_banner(result.verdict, result.verdict_level)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("F", f"{result.f:.3f}")
    m2.metric("p-value", f"{result.p:.4g}")
    m3.metric("SSE (SSW)", f"{result.ss_within:.3f}")
    m4.metric("η² (eta squared)", f"{result.eta_sq:.3f}")
    m5.metric("ω² (omega squared)", f"{result.omega_sq:.3f}")
    fig_note(
        "Readouts: **F** and **p** say whether group means differ relative to noise; "
        "**SSE (SSW)** is leftover within-group scatter; **η² / ω²** measure how large "
        "that difference is (effect size), not only whether it is “significant.”"
    )

    rows = []
    for label, g in zip(result.labels, result.groups):
        for v in g:
            rows.append({"group": label, "value": float(v)})
    plot_df = pd.DataFrame(rows)
    fig = px.box(
        plot_df,
        x="group",
        y="value",
        points="all",
        color="group",
        title="Simulated observations by group",
        color_discrete_sequence=PLOTLY_COLORS,
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(apply_plotly_theme(fig, height=400), use_container_width=True)
    fig_note(
        "Box plot of the synthetic sample: boxes show the middle 50% of each group; "
        "dots are individual draws. Wider overlap between boxes usually means weaker "
        "evidence for a mean difference (lower F, higher p)."
    )

    st.plotly_chart(
        variance_decomposition_fig(result.ss_between, result.ss_within),
        use_container_width=True,
    )
    fig_note(
        "Variance decomposition: **SSB** is how much group means differ from the grand mean; "
        "**SSW** is scatter around each group mean. ANOVA compares these two piles of variance."
    )

    with st.expander("ANOVA table details"):
        st.table(
            pd.DataFrame(
                [
                    {
                        "Source": "Between",
                        "SS": result.ss_between,
                        "df": result.df_between,
                        "MS": result.ms_between,
                        "F": result.f,
                        "p": result.p,
                    },
                    {
                        "Source": "Within (error)",
                        "SS": result.ss_within,
                        "df": result.df_within,
                        "MS": result.ms_within,
                        "F": np.nan,
                        "p": np.nan,
                    },
                    {
                        "Source": "Total",
                        "SS": result.ss_total,
                        "df": result.df_between + result.df_within,
                        "MS": np.nan,
                        "F": np.nan,
                        "p": np.nan,
                    },
                ]
            )
        )
        fig_note(
            "Classic ANOVA table: **SS** = sum of squares, **df** = degrees of freedom, "
            "**MS = SS/df**, and **F = MS_between / MS_within**."
        )

    glossary_block()

# ---------------------------------------------------------------------------
# Tab 2 — real-data ANOVA
# ---------------------------------------------------------------------------
def tab_realdata(df: pd.DataFrame, spec: DatasetSpec) -> None:
    from src.stats import (
        assumption_checks,
        one_way_anova,
        pairwise_ttests,
        two_way_anova,
    )

    fa, fb = spec.factor_a, spec.factor_b
    responses = [c for c in spec.response_options if c in df.columns]
    if not responses:
        st.error(f"No response columns found for {spec.id}.")
        return
    default_idx = (
        responses.index(spec.default_response)
        if spec.default_response in responses
        else 0
    )

    st.markdown(f"### ANOVA on `{spec.id}` features")
    st.markdown(
        f"Apply the same ANOVA ideas to **{spec.id}** trajectories. "
        f"Pick a response feature and a factor (`{fa}` or `{fb}`), or run the full "
        "two-way model with interaction."
    )
    if df.get("synthetic", pd.Series([False])).fillna(False).any():
        st.info(
            f"Currently using a **synthetic / demo** feature table for `{spec.id}`. "
            "Replace it with real data:\n\n"
            f"`python -m src.extract_features --dataset {spec.id} --splits train "
            "--time-stride 8 --space-stride 16`"
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        response = st.selectbox("Response feature", responses, index=default_idx)
    with c2:
        analysis = st.selectbox(
            "Analysis",
            [
                "One-way ANOVA",
                f"Two-way ANOVA ({fa} × {fb})",
                "Pairwise t-tests",
            ],
        )
    with c3:
        factor = st.selectbox(
            "Factor (one-way / t-tests)",
            [fa, fb],
            index=0,
        )

    two_way_label = f"Two-way ANOVA ({fa} × {fb})"

    if analysis == "One-way ANOVA":
        res = one_way_anova(df, response, factor)
        verdict_banner(res["verdict"], res["verdict_level"])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("F", f"{res['f']:.3f}")
        m2.metric("p-value", f"{res['p']:.4g}")
        m3.metric("SSE (SSW)", f"{res['ss_within']:.4g}")
        m4.metric("η²", f"{res['eta_sq']:.3f}")
        fig_note(
            f"One-way ANOVA of **{response}** across levels of **{factor}**. "
            "Large η² means this factor explains much of the feature’s variance."
        )

        fig = px.box(
            df,
            x=df[factor].astype(str),
            y=response,
            color=df[factor].astype(str),
            points="all",
            title=f"{response} by {factor}",
            labels={factor: factor, "x": factor},
            color_discrete_sequence=PLOTLY_COLORS,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(apply_plotly_theme(fig, height=420), use_container_width=True)
        fig_note(
            f"Distribution of **{response}** at each **{factor}** level. "
            "Separated boxes support a factor effect; heavy overlap does not."
        )
        st.plotly_chart(
            variance_decomposition_fig(res["ss_between"], res["ss_within"]),
            use_container_width=True,
        )
        fig_note(
            "Same SSB vs SSW split as in the sandbox, now computed on the feature table."
        )

        st.markdown("#### Assumption checks")
        assumptions = assumption_checks(df, response, factor)
        a1, a2, a3 = st.columns(3)
        a1.metric("Levene p (equal variance)", f"{assumptions['levene_p']:.4g}")
        a2.metric("Shapiro p (normal residuals)", f"{assumptions['shapiro_p']:.4g}")
        a3.write(
            "Recommend nonparametric: **"
            + ("yes" if assumptions["recommend_nonparametric"] else "no")
            + "**"
        )
        fig_note(
            "Levene tests similar within-group variances; Shapiro checks residual normality. "
            "If either fails badly, prefer Kruskal–Wallis / robust methods (see notebook)."
        )

        st.markdown("#### Tukey HSD post-hoc")
        st.dataframe(res["tukey"], use_container_width=True)
        fig_note(
            "Tukey HSD lists pairwise level comparisons. "
            "**reject = True** means that pair’s means differ after multiple-comparison control."
        )

    elif analysis == two_way_label:
        res = two_way_anova(df, response, fa, fb)
        verdict_banner(res["verdict"], res["verdict_level"])
        st.dataframe(res["table"], use_container_width=True)
        fig_note(
            f"Two-way ANOVA table: main effects of **{fa}** and **{fb}**, plus their **interaction**. "
            "Partial η² shows which term explains the most unique variance in the response."
        )
        fig = px.box(
            df,
            x=df[fb].astype(str),
            y=response,
            color=df[fa].astype(str),
            title=f"{response} — {fb} × {fa}",
            labels={"x": fb, "color": fa},
            color_discrete_sequence=PLOTLY_COLORS,
        )
        st.plotly_chart(apply_plotly_theme(fig, height=420), use_container_width=True)
        fig_note(
            f"**{response}** vs **{fb}**, colored by **{fa}**. "
            f"If boxes for different {fa} diverge as {fb} changes, that visualizes an interaction."
        )
        with st.expander("Model summary"):
            st.text(res["model_summary"])

    else:
        res = one_way_anova(df, response, factor)
        tt = pairwise_ttests(df, response, factor)
        verdict_banner(res["verdict"], res["verdict_level"])
        st.markdown("#### Pairwise Welch t-tests (Holm-adjusted)")
        st.dataframe(tt, use_container_width=True)
        fig_note(
            "Each row compares two levels of the factor (Welch t-test). "
            "**p_corr** is Holm-adjusted; use it instead of raw p for many pairs."
        )
        fig = px.strip(
            df,
            x=df[factor].astype(str),
            y=response,
            color=df[factor].astype(str),
            title=f"{response} by {factor}",
            color_discrete_sequence=PLOTLY_COLORS,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(apply_plotly_theme(fig, height=420), use_container_width=True)
        fig_note(
            "Strip plot of every trajectory. Useful for spotting outliers that a box plot might hide."
        )

    glossary_block()


# ---------------------------------------------------------------------------
# Tab 3 — physics + anomalies
# ---------------------------------------------------------------------------
def tab_physics(df: pd.DataFrame, spec: DatasetSpec) -> None:
    from src.stats import detect_anomalies, physics_validation

    fa, fb = spec.factor_a, spec.factor_b
    responses = [c for c in spec.response_options if c in df.columns]
    default_idx = (
        responses.index(spec.default_response)
        if spec.default_response in responses
        else 0
    )

    st.markdown("### Physics validation & anomalies")
    st.markdown(
        f"Sanity-check the `{spec.id}` feature table against expected physics, "
        f"then flag trajectories that look atypical inside their ({fa}, {fb}) cell."
    )

    phys = physics_validation(df, checks=spec.physics_checks)
    for key, block in phys.items():
        color = PALETTE["pass"] if block["pass"] else PALETTE["fail"]
        title = block.get("title", key.replace("_", " ").title())
        st.markdown(
            f"""
            <div style="padding:0.7rem 1rem;margin-bottom:0.55rem;
                        background:{PALETTE["paper_deep"]};
                        border:1px solid {PALETTE["ink"]};
                        border-left:5px solid {color};">
                <span style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
                             letter-spacing:0.1em;text-transform:uppercase;color:{color};">
                    {"Pass" if block["pass"] else "Fail"}
                </span>
                <div style="font-family:'Source Serif 4',Georgia,serif;font-weight:600;
                            color:{PALETTE["ink"]};margin-top:0.15rem;">
                    {title}
                </div>
                <div style="color:{PALETTE["graphite"]};font-size:0.95rem;margin-top:0.2rem;">
                    {block["message"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    fig_note(
        "Pass/fail cards summarize dataset-specific conservation, bounds, and association checks."
    )

    sx, sy, sc = spec.scatter_x, spec.scatter_y, spec.scatter_color
    if sx in df.columns and sy in df.columns and sc in df.columns:
        st.markdown(f"#### Response vs circumstance: {sy} vs {sx}")
        fig = px.scatter(
            df,
            x=sx,
            y=sy,
            color=df[sc].astype(str),
            labels={"color": sc},
            title=spec.scatter_title or f"{sy} vs {sx}",
            opacity=0.7,
            color_discrete_sequence=PLOTLY_COLORS,
        )
        means = (
            df.groupby([sc, sx], as_index=False)[sy]
            .mean()
            .sort_values([sc, sx])
        )
        for a, sub in means.groupby(sc):
            fig.add_trace(
                go.Scatter(
                    x=sub[sx],
                    y=sub[sy],
                    mode="lines",
                    name=f"mean {sc}={a}",
                    line=dict(width=2),
                )
            )
        st.plotly_chart(apply_plotly_theme(fig, height=440), use_container_width=True)
        fig_note(
            f"Each point is one trajectory. Lines are cell means of **{sy}** vs **{sx}**, "
            f"split by **{sc}**."
        )

    st.markdown("#### Within-cell anomalies")
    c1, c2, c3 = st.columns(3)
    with c1:
        anom_feat = st.selectbox(
            "Anomaly feature",
            responses or [sy],
            index=default_idx if responses else 0,
            key="anom_feat",
        )
    with c2:
        method = st.selectbox("Method", ["mad", "zscore", "iqr"], index=0)
    with c3:
        threshold = st.number_input("Threshold", value=3.5, min_value=1.0, step=0.5)

    flagged = detect_anomalies(
        df, anom_feat, group_cols=[fa, fb], method=method, threshold=float(threshold)
    )
    n_flag = int(flagged["is_anomaly"].sum())
    st.metric("Flagged trajectories", f"{n_flag} / {len(flagged)}")
    fig_note(
        f"Anomalies are scored **within each ({fa}, {fb}) cell**, so unusual runs are "
        "relative to peers with the same circumstance factors — not global outliers."
    )

    plot_df = flagged.copy()
    plot_df["status"] = np.where(plot_df["is_anomaly"], "anomaly", "ok")
    fig = px.scatter(
        plot_df,
        x=df[fb].astype(str) if fb in df.columns else list(range(len(df))),
        y=anom_feat,
        color="status",
        color_discrete_map={"anomaly": PALETTE["fail"], "ok": PALETTE["olive"]},
        title=f"Anomalies in {anom_feat}",
        opacity=0.75,
    )
    st.plotly_chart(apply_plotly_theme(fig, height=400), use_container_width=True)

    show_cols = [
        c
        for c in [
            fa,
            fb,
            anom_feat,
            "anomaly_score",
            "is_anomaly",
            "file",
            "traj_idx",
            "synthetic",
        ]
        if c in flagged.columns
    ]
    st.dataframe(
        flagged.loc[flagged["is_anomaly"], show_cols],
        use_container_width=True,
    )
    fig_note("Table lists only flagged rows. Empty table ⇒ no within-cell anomalies at this threshold.")

    g1, g2 = st.columns(2)
    with g1:
        hist_col = None
        for cand in ("mean_concentration", "mean_A", "wall_fraction", "mean_abs_pressure"):
            if cand in df.columns:
                hist_col = cand
                break
        if hist_col:
            fig_c = px.histogram(
                df,
                x=hist_col,
                nbins=30,
                title=hist_col,
                color_discrete_sequence=[PALETTE["olive"]],
            )
            if hist_col == "mean_concentration":
                fig_c.add_vline(x=1.0, line_dash="dash", line_color=PALETTE["fail"])
            st.plotly_chart(apply_plotly_theme(fig_c, height=320), use_container_width=True)
    with g2:
        hist2 = None
        for cand in ("div_u_rms", "pattern_contrast", "pressure_energy", "kinetic_energy"):
            if cand in df.columns:
                hist2 = cand
                break
        if hist2:
            fig_d = px.histogram(
                df,
                x=hist2,
                nbins=30,
                title=hist2,
                color_discrete_sequence=[PALETTE["ochre"]],
            )
            st.plotly_chart(apply_plotly_theme(fig_d, height=320), use_container_width=True)

    glossary_block()


# ---------------------------------------------------------------------------
# Gray–Scott — F×k phase diagram (not ANOVA)
# ---------------------------------------------------------------------------
def tab_phase_diagram(df: pd.DataFrame, spec: DatasetSpec) -> None:
    st.markdown("### (f, k) phase diagram")
    st.markdown(
        "Each Gray–Scott regime is a point on the **feed–kill plane**. "
        "Cell color shows the mean of a pattern metric — a sparse factorial / "
        "phase diagram rather than an ANOVA table."
    )
    responses = [c for c in spec.response_options if c in df.columns]
    if not responses:
        st.error("No response columns available.")
        return
    default_idx = (
        responses.index(spec.default_response)
        if spec.default_response in responses
        else 0
    )
    metric = st.selectbox("Phase-diagram metric", responses, index=default_idx)

    cell = (
        df.groupby(["f", "k", "pattern"], as_index=False)[metric]
        .mean()
        .sort_values(["k", "f"])
    )
    fig = px.scatter(
        cell,
        x="f",
        y="k",
        color=metric,
        size=metric,
        text="pattern",
        title=f"Gray–Scott phase diagram — mean {metric}",
        color_continuous_scale=["#D8D2C2", PALETTE["ochre"], PALETTE["olive"]],
        labels={"f": "feed f", "k": "kill k"},
    )
    fig.update_traces(textposition="top center", marker=dict(line=dict(width=1, color=PALETTE["ink"])))
    st.plotly_chart(apply_plotly_theme(fig, height=460), use_container_width=True)
    fig_note(
        "Named regimes sit at discrete **(f, k)** pairs. Color/size encode the "
        f"mean **{metric}** within each regime — the classic reaction–diffusion phase map."
    )

    pivot = cell.pivot_table(index="k", columns="f", values=metric)
    fig_h = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[f"{c:g}" for c in pivot.columns],
            y=[f"{r:g}" for r in pivot.index],
            colorscale=[[0, "#E6E2D6"], [0.5, PALETTE["ochre_soft"]], [1, PALETTE["olive"]]],
            colorbar=dict(title=metric),
        )
    )
    fig_h.update_layout(title=f"Sparse (f, k) heatmap of {metric}", xaxis_title="f", yaxis_title="k")
    st.plotly_chart(apply_plotly_theme(fig_h, height=380), use_container_width=True)
    fig_note(
        "Only six cells are occupied (sparse design). Empty grid cells are not "
        "simulated in The Well release — this is a **factorial sample**, not a dense sweep."
    )
    st.dataframe(cell, use_container_width=True)


def tab_pattern_metrics(df: pd.DataFrame, spec: DatasetSpec) -> None:
    st.markdown("### Pattern metrics vs parameters")
    st.markdown(
        "How concentration statistics and contrast change with **f**, **k**, and named regime. "
        "Correlation and scatter views — exploratory data analysis, not hypothesis tests."
    )
    responses = [c for c in spec.response_options if c in df.columns]
    default_idx = (
        responses.index(spec.default_response)
        if spec.default_response in responses
        else 0
    )
    c1, c2 = st.columns(2)
    with c1:
        metric = st.selectbox("Metric", responses, index=default_idx, key="gs_metric")
    with c2:
        x_axis = st.selectbox("X parameter", ["f", "k", "pattern"], index=0)

    color_col = "pattern" if "pattern" in df.columns else spec.factor_b
    fig = px.box(
        df,
        x=df[x_axis].astype(str) if x_axis != "pattern" else df["pattern"],
        y=metric,
        color=df[color_col].astype(str),
        points="all",
        title=f"{metric} vs {x_axis}",
        color_discrete_sequence=PLOTLY_COLORS,
    )
    st.plotly_chart(apply_plotly_theme(fig, height=420), use_container_width=True)
    fig_note(
        f"**{metric}** stratified by **{x_axis}**. Separation across regimes is the "
        "phase-diagram story in distributional form."
    )

    fig_s = px.scatter(
        df,
        x="f",
        y="k",
        color=metric,
        symbol="pattern" if "pattern" in df.columns else None,
        title=f"Trajectories in (f, k) colored by {metric}",
        color_continuous_scale=["#D8D2C2", PALETTE["ochre"], PALETTE["olive"]],
        opacity=0.75,
    )
    st.plotly_chart(apply_plotly_theme(fig_s, height=420), use_container_width=True)

    num_cols = [c for c in ["f", "k", *responses] if c in df.columns]
    corr = df[num_cols].corr(method="spearman")
    fig_c = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale=["#7A3E32", "#E6E2D6", PALETTE["olive"]],
        title="Spearman correlation — params & pattern metrics",
        aspect="auto",
    )
    st.plotly_chart(apply_plotly_theme(fig_c, height=420), use_container_width=True)
    fig_note(
        "Spearman rank correlations among parameters and metrics. Strong |ρ| between "
        "a parameter and a metric suggests a clear phase-map direction."
    )


# ---------------------------------------------------------------------------
# Acoustic — multiparameter / interaction plots (not ANOVA)
# ---------------------------------------------------------------------------
def tab_geometry_sources(df: pd.DataFrame, spec: DatasetSpec) -> None:
    st.markdown("### Geometry × sources response surface")
    st.markdown(
        "Maze **path width** and **source count** jointly set the acoustic field. "
        "Heatmaps and mean response surfaces replace ANOVA for this ensemble."
    )
    responses = [c for c in spec.response_options if c in df.columns]
    default_idx = (
        responses.index(spec.default_response)
        if spec.default_response in responses
        else 0
    )
    metric = st.selectbox("Response metric", responses, index=default_idx, key="ac_surf")

    cell = (
        df.groupby(["maze_width", "n_sources"], as_index=False)[metric]
        .mean()
        .sort_values(["maze_width", "n_sources"])
    )
    pivot = cell.pivot(index="maze_width", columns="n_sources", values=metric)
    fig_h = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=[str(r) for r in pivot.index],
            colorscale=[[0, "#E6E2D6"], [0.5, PALETTE["ochre_soft"]], [1, PALETTE["olive"]]],
            colorbar=dict(title=metric),
        )
    )
    fig_h.update_layout(
        title=f"Mean {metric} — maze_width × n_sources",
        xaxis_title="n_sources",
        yaxis_title="maze_width",
    )
    st.plotly_chart(apply_plotly_theme(fig_h, height=420), use_container_width=True)
    fig_note(
        "Each cell is the mean response at that geometry × source count. "
        "Look for **rows** (geometry main effect), **columns** (source main effect), "
        "or **tilted gradients** (interaction)."
    )

    fig = px.line(
        cell,
        x="n_sources",
        y=metric,
        color=cell["maze_width"].astype(str),
        markers=True,
        title=f"Interaction plot: {metric} vs n_sources by maze_width",
        color_discrete_sequence=PLOTLY_COLORS,
        labels={"color": "maze_width"},
    )
    st.plotly_chart(apply_plotly_theme(fig, height=420), use_container_width=True)
    fig_note(
        "Classic **interaction plot**: non-parallel curves mean geometry changes how "
        "source count maps to the response."
    )


def tab_response_interactions(df: pd.DataFrame, spec: DatasetSpec) -> None:
    st.markdown("### Multiparameter effects & frequency-content proxy")
    st.markdown(
        "Pressure energy and **spectral slope** (frequency-content proxy of the scattered "
        "field) versus geometry and sources. Correlation and faceted scatters — EDA / "
        "response analysis, not ANOVA."
    )
    responses = [c for c in spec.response_options if c in df.columns]
    default_idx = (
        responses.index(spec.default_response)
        if spec.default_response in responses
        else 0
    )
    c1, c2 = st.columns(2)
    with c1:
        y_metric = st.selectbox("Y response", responses, index=default_idx, key="ac_y")
    with c2:
        x_param = st.selectbox("X factor", ["n_sources", "maze_width", "wall_fraction"], index=0)

    color = "maze_width" if x_param != "maze_width" else "n_sources"
    fig = px.scatter(
        df,
        x=x_param,
        y=y_metric,
        color=df[color].astype(str),
        title=f"{y_metric} vs {x_param}",
        color_discrete_sequence=PLOTLY_COLORS,
        opacity=0.7,
        labels={"color": color},
    )
    st.plotly_chart(apply_plotly_theme(fig, height=420), use_container_width=True)
    fig_note(
        f"Scatter of **{y_metric}** vs **{x_param}**, colored by **{color}** — "
        "multiparameter structure without assuming an ANOVA model."
    )

    if "spectral_slope" in df.columns:
        fig_f = px.box(
            df,
            x=df["maze_width"].astype(str),
            y="spectral_slope",
            color=df["n_sources"].astype(str),
            title="Spectral slope (frequency-content proxy) by geometry × sources",
            color_discrete_sequence=PLOTLY_COLORS,
            labels={"x": "maze_width", "color": "n_sources"},
        )
        st.plotly_chart(apply_plotly_theme(fig_f, height=400), use_container_width=True)
        fig_note(
            "**Spectral slope** summarizes how energy is distributed across spatial "
            "frequencies after scattering — a frequency-response style readout for this maze ensemble."
        )

    num_cols = [
        c
        for c in [
            "maze_width",
            "n_sources",
            "wall_fraction",
            *responses,
        ]
        if c in df.columns
    ]
    corr = df[num_cols].corr(method="spearman")
    fig_c = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale=["#7A3E32", "#E6E2D6", PALETTE["olive"]],
        title="Spearman correlation — geometry, sources, responses",
        aspect="auto",
    )
    st.plotly_chart(apply_plotly_theme(fig_c, height=440), use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="statistical laboratory · The Well",
        page_icon="∫",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_lab_css()
    spec = render_dataset_sidebar()
    df = load_features_or_stop(spec)
    lab_header(df, spec)

    # Dataset-specific panels: ANOVA only for active_matter
    labels = list(spec.panel_labels)
    # Reset panel key when dataset changes so we don't keep a stale ANOVA label
    panel_key = f"lab_panel_{spec.id}"
    panel = st.radio(
        "Section",
        labels,
        horizontal=True,
        label_visibility="collapsed",
        key=panel_key,
    )

    if spec.analysis_mode == "anova":
        if panel == labels[0]:
            tab_teaching()
        elif panel == labels[1]:
            tab_realdata(df, spec)
        else:
            tab_physics(df, spec)
    elif spec.analysis_mode == "phase":
        if panel == labels[0]:
            tab_phase_diagram(df, spec)
        elif panel == labels[1]:
            tab_pattern_metrics(df, spec)
        else:
            tab_physics(df, spec)
    else:  # interaction
        if panel == labels[0]:
            tab_geometry_sources(df, spec)
        elif panel == labels[1]:
            tab_response_interactions(df, spec)
        else:
            tab_physics(df, spec)

    lab_footer(spec)


if __name__ == "__main__":
    main()
