"""
Streamlit dashboard:
  Tab 1 — Interactive ANOVA teaching sandbox
  Tab 2 — Real-data ANOVA on active_matter features
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

FEATURES_PATH = ROOT / "data" / "features.parquet"
RESPONSE_OPTIONS = [
    "nematic_order_S",
    "nematic_order_S_final",
    "kinetic_energy",
    "enstrophy",
    "std_concentration",
    "mean_concentration",
    "div_u_rms",
    "spectral_slope",
    "time_to_steady",
]

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
    padding: 0.85rem 1rem 0.55rem;
    background: var(--paper-deep);
    border: 1px solid var(--grid);
}}

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
def load_features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(str(FEATURES_PATH))
    return pd.read_parquet(FEATURES_PATH)


def load_features_or_stop() -> pd.DataFrame:
    try:
        return load_features()
    except FileNotFoundError:
        st.error(
            f"Missing `{FEATURES_PATH}`. Run:\n\n"
            "`python -m src.extract_features --splits train`\n\n"
            "or, as a demo fallback only:\n\n"
            "`python -m src.extract_features --synthetic`"
        )
        st.stop()


def data_provenance(df: pd.DataFrame) -> dict:
    is_synth = bool(df.get("synthetic", pd.Series([False] * len(df))).fillna(False).any())
    n_alpha = int(df["alpha"].nunique()) if "alpha" in df.columns else 0
    n_zeta = int(df["zeta"].nunique()) if "zeta" in df.columns else 0
    n_cells = int(df.groupby(["alpha", "zeta"]).ngroups) if {"alpha", "zeta"} <= set(df.columns) else 0
    return {
        "synthetic": is_synth,
        "n_rows": len(df),
        "n_alpha": n_alpha,
        "n_zeta": n_zeta,
        "n_cells": n_cells,
    }


AUTHOR = "Emilio Gordillo Esparragoza"


def lab_header(df: pd.DataFrame) -> None:
    meta = data_provenance(df)
    findings_html = ""
    if not meta["synthetic"]:
        findings_html = f"""
        <div class="lab-findings">
          <div class="lab-findings-label">Results at a glance</div>
          <p>
            On <b>{meta["n_rows"]}</b> real trajectories covering all
            <b>{meta["n_cells"]}</b> α×ζ cells, two-way ANOVA shows
            <b>alignment (ζ)</b> dominates nematic order
            (partial η² ≈ 0.97), while <b>dipole (α)</b> has little main effect
            and acts mainly through an α×ζ interaction. Concentration stays at 1;
            order rises with ζ (Spearman ρ ≈ 0.86). See the README Findings section
            for full numbers and caveats.
          </p>
        </div>
        """
    st.html(
        f"""
        <div class="lab-kicker">
          Statistical laboratory
        </div>
        <h1 class="lab-brand">active_matter · circumstance &amp; response</h1>
        <p class="lab-lede">
          Quantify how initial control factors (α, ζ) shape nematic order and flow,
          with ANOVA evidence, physics-law checks, and within-cell anomaly flags.
        </p>
        <div class="lab-meta">
          <span>trajectories <b>{meta["n_rows"]}</b></span>
          <span>α levels <b>{meta["n_alpha"]}</b></span>
          <span>ζ levels <b>{meta["n_zeta"]}</b></span>
          <span>factorial cells <b>{meta["n_cells"]}</b></span>
        </div>
        <div class="lab-dataset">
          <div class="lab-dataset-label">What is the active_matter dataset?</div>
          <p>
            <code>active_matter</code> (from PolymathicAI <b>The Well</b>) is a continuum
            simulation ensemble of <b>rod-like active particles</b> in a <b>Stokes fluid</b>.
            Each run is controlled by two initial factors: dipole strength
            <code>α</code> (alpha) and alignment strength <code>ζ</code> (zeta).
            From each trajectory we extract scalar responses — nematic order
            <code>S</code>, kinetic energy, enstrophy, concentration, divergence residual —
            then ask how α and ζ shape those outcomes via ANOVA, physics checks, and
            within-cell anomaly screens.
          </p>
        </div>
        {findings_html}
        """
    )


def lab_footer() -> None:
    st.html(
        f"""
        <div class="lab-footer">
          Author · <b>{AUTHOR}</b><br/>
          Dataset · PolymathicAI · The Well · active_matter
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

    st.markdown('<div class="lab-controls">', unsafe_allow_html=True)
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
    st.markdown("</div>", unsafe_allow_html=True)

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
def tab_realdata(df: pd.DataFrame) -> None:
    from src.stats import (
        assumption_checks,
        one_way_anova,
        pairwise_ttests,
        two_way_anova,
    )

    st.markdown("### ANOVA on active_matter features")
    st.markdown(
        "Apply the same ANOVA ideas to **real** The Well trajectories. "
        "Pick a response feature and a factor (`zeta` or `alpha`), or run the full "
        "two-way model with interaction."
    )
    if df.get("synthetic", pd.Series([False])).fillna(False).any():
        st.info(
            "Currently using a **synthetic / demo** feature table that mirrors the "
            "active_matter factorial design (α × ζ). Replace it with real data:\n\n"
            "`python -m src.extract_features --splits train --time-stride 8 --space-stride 16`"
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        response = st.selectbox("Response feature", RESPONSE_OPTIONS, index=0)
    with c2:
        analysis = st.selectbox(
            "Analysis",
            ["One-way ANOVA", "Two-way ANOVA (alpha × zeta)", "Pairwise t-tests"],
        )
    with c3:
        factor = st.selectbox("Factor (one-way / t-tests)", ["zeta", "alpha"], index=0)

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

    elif analysis == "Two-way ANOVA (alpha × zeta)":
        res = two_way_anova(df, response, "alpha", "zeta")
        verdict_banner(res["verdict"], res["verdict_level"])
        st.dataframe(res["table"], use_container_width=True)
        fig_note(
            "Two-way ANOVA table: main effects of **α** and **ζ**, plus their **interaction**. "
            "Partial η² shows which term explains the most unique variance in the response."
        )
        fig = px.box(
            df,
            x=df["zeta"].astype(str),
            y=response,
            color=df["alpha"].astype(str),
            title=f"{response} — zeta × alpha",
            labels={"x": "zeta", "color": "alpha"},
            color_discrete_sequence=PLOTLY_COLORS,
        )
        st.plotly_chart(apply_plotly_theme(fig, height=420), use_container_width=True)
        fig_note(
            f"**{response}** vs **ζ**, colored by **α**. "
            "If curves/boxes for different α diverge as ζ changes, that visualizes an interaction."
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
def tab_physics(df: pd.DataFrame) -> None:
    from src.stats import detect_anomalies, physics_validation

    st.markdown("### Physics validation & anomalies")
    st.markdown(
        "Sanity-check the feature table against expected physics "
        "(concentration near 1, order rising with alignment), "
        "then flag trajectories that look atypical inside their (α, ζ) cell."
    )

    phys = physics_validation(df)
    for key in ("concentration", "incompressibility", "phase_transition"):
        block = phys[key]
        color = PALETTE["pass"] if block["pass"] else PALETTE["fail"]
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
                    {key.replace("_", " ").title()}
                </div>
                <div style="color:{PALETTE["graphite"]};font-size:0.95rem;margin-top:0.2rem;">
                    {block["message"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    fig_note(
        "Pass/fail cards summarize conservation, discrete divergence, and the "
        "Spearman correlation between nematic order and ζ."
    )

    st.markdown("#### Phase transition: nematic order vs alignment (ζ)")
    fig = px.scatter(
        df,
        x="zeta",
        y="nematic_order_S",
        color=df["alpha"].astype(str),
        labels={"color": "alpha"},
        title="Nematic order S vs zeta (colored by alpha)",
        opacity=0.7,
        color_discrete_sequence=PLOTLY_COLORS,
    )
    means = (
        df.groupby(["alpha", "zeta"], as_index=False)["nematic_order_S"]
        .mean()
        .sort_values(["alpha", "zeta"])
    )
    for a, sub in means.groupby("alpha"):
        fig.add_trace(
            go.Scatter(
                x=sub["zeta"],
                y=sub["nematic_order_S"],
                mode="lines",
                name=f"mean α={a}",
                line=dict(width=2),
            )
        )
    st.plotly_chart(apply_plotly_theme(fig, height=440), use_container_width=True)
    fig_note(
        "Each point is a trajectory. Lines are mean **S** vs **ζ** at fixed **α**. "
        "A rising trend is the isotropic→nematic-like signature in this ensemble."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        anom_feat = st.selectbox("Anomaly feature", RESPONSE_OPTIONS, index=0, key="anom_feat")
    with c2:
        method = st.selectbox("Method", ["mad", "zscore", "iqr"], index=0)
    with c3:
        thresh = st.slider("Threshold", 2.0, 6.0, 3.5, 0.1)

    flagged = detect_anomalies(
        df, anom_feat, group_cols=["alpha", "zeta"], method=method, threshold=thresh
    )
    n_flag = int(flagged["is_anomaly"].sum())
    st.metric("Flagged anomalies", f"{n_flag} / {len(flagged)}")
    fig_note(
        "Anomalies are scored **within** each (α, ζ) cell, so a high-ζ ordered run is not "
        "flagged merely for having large S overall."
    )

    fig_a = px.scatter(
        flagged,
        x="zeta",
        y=anom_feat,
        color=flagged["is_anomaly"].map({True: "anomaly", False: "ok"}),
        symbol=df["alpha"].astype(str),
        title=f"Anomalies in {anom_feat} within (alpha, zeta) cells",
        color_discrete_map={"anomaly": PALETTE["fail"], "ok": PALETTE["olive"]},
    )
    st.plotly_chart(apply_plotly_theme(fig_a, height=420), use_container_width=True)
    fig_note(
        "Red points exceed the robust threshold for the chosen method (MAD / z-score / IQR)."
    )

    st.markdown("#### Anomaly table")
    show_cols = [
        "file",
        "traj_idx",
        "alpha",
        "zeta",
        anom_feat,
        "anomaly_score",
    ]
    if "synthetic" in flagged.columns:
        show_cols.append("synthetic")
    if "injected_anomaly" in flagged.columns:
        show_cols.append("injected_anomaly")
    show = flagged.loc[flagged["is_anomaly"], show_cols]
    st.dataframe(show, use_container_width=True)
    fig_note(
        "Tabular list of flagged trajectories with their anomaly score for auditing or exclusion."
    )

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("#### Concentration conservation")
        fig_c = px.histogram(
            df,
            x="mean_concentration",
            nbins=30,
            title="mean concentration",
            color_discrete_sequence=[PALETTE["olive"]],
        )
        fig_c.add_vline(x=1.0, line_dash="dash", line_color=PALETTE["fail"])
        st.plotly_chart(apply_plotly_theme(fig_c, height=360), use_container_width=True)
        fig_note(
            "Histogram of mean concentration. The dashed line marks the physical target **c = 1**."
        )
    with g2:
        st.markdown("#### Incompressibility residual")
        fig_d = px.histogram(
            df,
            x="div_u_rms",
            nbins=30,
            title="div_u_rms",
            color_discrete_sequence=[PALETTE["ochre"]],
        )
        st.plotly_chart(apply_plotly_theme(fig_d, height=360), use_container_width=True)
        fig_note(
            "Distribution of discrete divergence RMS. Values can look large on a coarse "
            "analysis grid even when the underlying Stokes flow is nearly incompressible."
        )

    glossary_block()


def main() -> None:
    st.set_page_config(
        page_title="active_matter · statistical laboratory",
        page_icon="∫",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_lab_css()
    df = load_features_or_stop()
    lab_header(df)

    # Only render the selected panel (st.tabs runs every tab body every rerun).
    panel = st.radio(
        "Section",
        ["ANOVA sandbox", "Real-data ANOVA", "Physics & anomalies"],
        horizontal=True,
        label_visibility="collapsed",
        key="lab_panel",
    )
    if panel == "ANOVA sandbox":
        tab_teaching()
    elif panel == "Real-data ANOVA":
        tab_realdata(df)
    else:
        tab_physics(df)

    lab_footer()


if __name__ == "__main__":
    main()
