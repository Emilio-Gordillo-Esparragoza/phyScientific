"""
Dataset registry for the multi-dataset statistical laboratory.

Each entry describes branding copy, feature-table path, ANOVA factors,
response columns, and declarative physics checks consumed by stats.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PhysicsCheckSpec:
    """Declarative physics / sanity check for a feature table."""

    key: str
    title: str
    kind: str  # mean_near_target | range | spearman | finite
    # mean_near_target
    column: str | None = None
    target: float | None = None
    tol: float | None = None
    # range
    lo: float | None = None
    hi: float | None = None
    # spearman
    x: str | None = None
    y: str | None = None
    min_rho: float | None = None
    max_p: float | None = None
    # finite
    columns: tuple[str, ...] = ()
    # optional human hint
    pass_message: str | None = None
    fail_message: str | None = None


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    hf_name: str
    title: str
    lede: str
    blurb: str
    footer_label: str
    feature_path: Path
    legacy_feature_paths: tuple[Path, ...] = ()
    # "anova" | "phase" | "interaction" — drives which Streamlit panels appear
    analysis_mode: str = "anova"
    factor_a: str = "alpha"
    factor_b: str = "zeta"
    factor_a_label: str = "α"
    factor_b_label: str = "ζ"
    response_options: tuple[str, ...] = ()
    default_response: str = ""
    physics_checks: tuple[PhysicsCheckSpec, ...] = ()
    findings_real: str = ""
    scatter_x: str = ""
    scatter_y: str = ""
    scatter_color: str = ""
    scatter_title: str = ""
    hf_url: str = ""
    panel_labels: tuple[str, str, str] = (
        "ANOVA sandbox",
        "Real-data ANOVA",
        "Physics & anomalies",
    )


def _resolve_feature_path(spec: DatasetSpec) -> Path:
    """Prefer the canonical path; fall back to legacy locations if present."""
    if spec.feature_path.exists():
        return spec.feature_path
    for alt in spec.legacy_feature_paths:
        if alt.exists():
            return alt
    return spec.feature_path


ACTIVE_MATTER = DatasetSpec(
    id="active_matter",
    hf_name="active_matter",
    title="active_matter · circumstance & response",
    lede=(
        "Quantify how initial control factors (α, ζ) shape nematic order and flow, "
        "with ANOVA evidence, physics-law checks, and within-cell anomaly flags."
    ),
    blurb=(
        "<code>active_matter</code> (from PolymathicAI <b>The Well</b>) is a continuum "
        "simulation ensemble of <b>rod-like active particles</b> in a <b>Stokes fluid</b>. "
        "Each run is controlled by two initial factors: dipole strength "
        "<code>α</code> (alpha) and alignment strength <code>ζ</code> (zeta). "
        "From each trajectory we extract scalar responses — nematic order "
        "<code>S</code>, kinetic energy, enstrophy, concentration, divergence residual — "
        "then ask how α and ζ shape those outcomes via ANOVA, physics checks, and "
        "within-cell anomaly screens."
    ),
    footer_label="PolymathicAI · The Well · active_matter",
    feature_path=ROOT / "data" / "features_active_matter.parquet",
    legacy_feature_paths=(ROOT / "data" / "features.parquet",),
    analysis_mode="anova",
    factor_a="alpha",
    factor_b="zeta",
    factor_a_label="α",
    factor_b_label="ζ",
    response_options=(
        "nematic_order_S",
        "nematic_order_S_final",
        "kinetic_energy",
        "enstrophy",
        "std_concentration",
        "mean_concentration",
        "div_u_rms",
        "spectral_slope",
        "time_to_steady",
    ),
    default_response="nematic_order_S",
    physics_checks=(
        PhysicsCheckSpec(
            key="concentration",
            title="Concentration conservation",
            kind="mean_near_target",
            column="mean_concentration",
            target=1.0,
            tol=0.05,
        ),
        PhysicsCheckSpec(
            key="incompressibility",
            title="Near-incompressibility",
            kind="range",
            column="div_u_rms",
            lo=0.0,
            hi=0.5,
            pass_message="Mean |div u| RMS is acceptable on the analysis grid",
            fail_message="Elevated discrete divergence (check downsampling)",
        ),
        PhysicsCheckSpec(
            key="phase_transition",
            title="Isotropic→nematic signature",
            kind="spearman",
            x="zeta",
            y="nematic_order_S",
            min_rho=0.3,
            max_p=0.05,
            pass_message="Nematic order rises with alignment ζ",
            fail_message="Order does not clearly rise with ζ",
        ),
    ),
    findings_real=(
        "On <b>{n_rows}</b> real trajectories covering all "
        "<b>{n_cells}</b> α×ζ cells, two-way ANOVA shows "
        "<b>alignment (ζ)</b> dominates nematic order "
        "(partial η² ≈ 0.97), while <b>dipole (α)</b> has little main effect "
        "and acts mainly through an α×ζ interaction. Concentration stays at 1; "
        "order rises with ζ (Spearman ρ ≈ 0.86). See the README Findings section "
        "for full numbers and caveats."
    ),
    scatter_x="zeta",
    scatter_y="nematic_order_S",
    scatter_color="alpha",
    scatter_title="Nematic order S vs zeta (colored by alpha)",
    hf_url="https://huggingface.co/datasets/polymathic-ai/active_matter",
    panel_labels=("ANOVA sandbox", "Real-data ANOVA", "Physics & anomalies"),
)

GRAY_SCOTT = DatasetSpec(
    id="gray_scott",
    hf_name="gray_scott_reaction_diffusion",
    title="gray_scott · (f, k) phase diagram",
    lede=(
        "Map reaction–diffusion regimes on the feed–kill plane: phase diagrams of "
        "pattern metrics, and how concentration / contrast respond to (f, k)."
    ),
    blurb=(
        "<code>gray_scott_reaction_diffusion</code> (The Well) is a reaction–diffusion "
        "ensemble of species <b>A</b> and <b>B</b>. Six discrete <code>(f, k)</code> pairs "
        "produce named patterns (Gliders, Bubbles, Maze, Worms, Spirals, Spots). "
        "This lab treats the <b>(f, k) grid as a sparse factorial / phase diagram</b> — "
        "not an ANOVA sandbox — and plots pattern metrics against parameters."
    ),
    footer_label="PolymathicAI · The Well · gray_scott_reaction_diffusion",
    feature_path=ROOT / "data" / "features_gray_scott.parquet",
    analysis_mode="phase",
    factor_a="f",
    factor_b="k",
    factor_a_label="f",
    factor_b_label="k",
    response_options=(
        "pattern_contrast",
        "mean_A",
        "mean_B",
        "std_A",
        "std_B",
        "spectral_slope",
        "time_to_steady",
    ),
    default_response="pattern_contrast",
    physics_checks=(
        PhysicsCheckSpec(
            key="A_bounds",
            title="Species A in [0, 1]",
            kind="range",
            column="mean_A",
            lo=0.0,
            hi=1.0,
        ),
        PhysicsCheckSpec(
            key="B_bounds",
            title="Species B in [0, 1]",
            kind="range",
            column="mean_B",
            lo=0.0,
            hi=1.0,
        ),
        PhysicsCheckSpec(
            key="pattern_diversity",
            title="Contrast vs feed rate",
            kind="spearman",
            x="f",
            y="pattern_contrast",
            min_rho=0.15,
            max_p=0.05,
            pass_message="Pattern contrast covaries with feed rate f",
            fail_message="Weak association between contrast and f",
        ),
    ),
    findings_real=(
        "On <b>{n_rows}</b> trajectories across <b>{n_cells}</b> (f, k) regime cells, "
        "the six named patterns separate cleanly in contrast and concentration space. "
        "Explore the <b>phase diagram</b> and metric-vs-parameter panels (no ANOVA here)."
    ),
    scatter_x="f",
    scatter_y="pattern_contrast",
    scatter_color="pattern",
    scatter_title="Pattern contrast vs feed f (colored by regime)",
    hf_url="https://huggingface.co/datasets/polymathic-ai/gray_scott_reaction_diffusion",
    panel_labels=(
        "F×k phase diagram",
        "Pattern metrics vs params",
        "Physics & anomalies",
    ),
)

ACOUSTIC = DatasetSpec(
    id="acoustic_scattering",
    hf_name="acoustic_scattering_maze",
    title="acoustic_scattering · geometry & response",
    lede=(
        "Explore how maze geometry and source count reshape pressure energy: "
        "multiparameter response surfaces and interaction plots (not ANOVA)."
    ),
    blurb=(
        "<code>acoustic_scattering_maze</code> (The Well) models pressure-wave "
        "propagation through maze-like density fields. Circumstance factors are "
        "<code>maze_width</code> (path geometry) and <code>n_sources</code>. "
        "Spectral slope acts as a <b>frequency-content proxy</b> of the scattered field. "
        "This lab emphasizes <b>response surfaces and interaction plots</b> across "
        "geometry × sources — not factorial ANOVA."
    ),
    footer_label="PolymathicAI · The Well · acoustic_scattering_maze",
    feature_path=ROOT / "data" / "features_acoustic_scattering.parquet",
    analysis_mode="interaction",
    factor_a="maze_width",
    factor_b="n_sources",
    factor_a_label="width",
    factor_b_label="sources",
    response_options=(
        "mean_abs_pressure",
        "pressure_energy",
        "kinetic_energy",
        "wall_fraction",
        "spectral_slope",
        "time_to_steady",
    ),
    default_response="pressure_energy",
    physics_checks=(
        PhysicsCheckSpec(
            key="finite_energy",
            title="Finite pressure energy",
            kind="finite",
            columns=("pressure_energy", "mean_abs_pressure"),
        ),
        PhysicsCheckSpec(
            key="wall_fraction",
            title="Wall fraction in (0, 1)",
            kind="range",
            column="wall_fraction",
            lo=0.05,
            hi=0.95,
        ),
        PhysicsCheckSpec(
            key="energy_vs_sources",
            title="Energy vs source count",
            kind="spearman",
            x="n_sources",
            y="pressure_energy",
            min_rho=0.2,
            max_p=0.05,
            pass_message="Pressure energy rises with n_sources",
            fail_message="Weak association between energy and source count",
        ),
    ),
    findings_real=(
        "On <b>{n_rows}</b> trajectories across <b>{n_cells}</b> width×source cells, "
        "geometry and source count jointly shape pressure energy and spectral slope. "
        "Use the interaction / response panels to see multiparameter effects."
    ),
    scatter_x="n_sources",
    scatter_y="pressure_energy",
    scatter_color="maze_width",
    scatter_title="Pressure energy vs n_sources (colored by maze_width)",
    hf_url="https://huggingface.co/datasets/polymathic-ai/acoustic_scattering_maze",
    panel_labels=(
        "Geometry × sources",
        "Response & interactions",
        "Physics & anomalies",
    ),
)

DATASETS: dict[str, DatasetSpec] = {
    ACTIVE_MATTER.id: ACTIVE_MATTER,
    GRAY_SCOTT.id: GRAY_SCOTT,
    ACOUSTIC.id: ACOUSTIC,
}

DATASET_IDS: tuple[str, ...] = tuple(DATASETS.keys())


def get_dataset(dataset_id: str) -> DatasetSpec:
    if dataset_id not in DATASETS:
        raise KeyError(f"Unknown dataset_id={dataset_id!r}; choose from {list(DATASETS)}")
    return DATASETS[dataset_id]


def feature_path_for(dataset_id: str) -> Path:
    return _resolve_feature_path(get_dataset(dataset_id))


def default_out_path(dataset_id: str) -> Path:
    return get_dataset(dataset_id).feature_path
