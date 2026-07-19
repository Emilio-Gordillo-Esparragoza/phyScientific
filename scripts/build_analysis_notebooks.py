"""Build and execute enriched gray_scott / acoustic analysis notebooks."""
from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def write_and_execute(path: Path, cells: list[dict]) -> None:
    nb = nbformat.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    for c in cells:
        if c["cell_type"] == "markdown":
            nb.cells.append(nbformat.v4.new_markdown_cell("".join(c["source"])))
        else:
            nb.cells.append(nbformat.v4.new_code_cell("".join(c["source"])))

    client = NotebookClient(
        nb,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(NB_DIR)}},
    )
    client.execute()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"wrote+executed {path}")


SETUP = '''\
from pathlib import Path
import sys
ROOT = Path("..").resolve()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = ["#5F6B45", "#9A7340", "#3D5A56", "#7A3E32", "#5C574E", "#8A9470"]
sns.set_palette(PALETTE)
%matplotlib inline
'''

GS_CELLS = [
    md(
        "# Gray–Scott · (f, k) phase diagram\n\n"
        "Exploratory analysis of The Well `gray_scott_reaction_diffusion` feature table.\n\n"
        "**Not ANOVA** — focus on the feed–kill phase map, regime summary tables, "
        "and pattern metrics vs parameters.\n"
    ),
    code(
        SETUP
        + '''
from src.dataset_catalog import get_dataset

spec = get_dataset("gray_scott")
df = pd.read_parquet(spec.feature_path)
print("rows:", len(df), "| synthetic:", bool(df["synthetic"].any()) if "synthetic" in df.columns else "n/a")
print("regimes:", sorted(df["pattern"].unique()))
df.head()
'''
    ),
    md("## Coverage table\n\nHow many trajectories sit in each named (f, k) regime."),
    code(
        '''
coverage = (
    df.groupby(["pattern", "f", "k"], as_index=False)
    .size()
    .rename(columns={"size": "n_traj"})
    .sort_values(["f", "k"])
)
display(coverage)
print(f"Total trajectories: {coverage['n_traj'].sum()} | regimes: {len(coverage)}")
'''
    ),
    md("## Feature dictionary (this table)"),
    code(
        '''
desc = {
    "f": "feed rate",
    "k": "kill rate",
    "pattern": "named regime label",
    "mean_A / mean_B": "time–space mean concentrations",
    "std_A / std_B": "spatial/temporal variability",
    "pattern_contrast": "std_A + std_B (pattern strength proxy)",
    "spectral_slope": "log–log spectral slope of A",
    "time_to_steady": "fraction of run until A plateaus",
}
pd.DataFrame({"column / group": list(desc), "meaning": list(desc.values())})
'''
    ),
    md("## Summary statistics by regime"),
    code(
        '''
metrics = [c for c in [
    "pattern_contrast", "mean_A", "mean_B", "std_A", "std_B",
    "spectral_slope", "time_to_steady",
] if c in df.columns]

summary = (
    df.groupby("pattern")[metrics]
    .agg(["mean", "std", "min", "max"])
    .round(4)
)
display(summary)

# Compact mean-only table for readability
means = df.groupby("pattern")[metrics].mean().round(4).sort_values("pattern_contrast")
display(means)
'''
    ),
    md("## Sparse (f, k) phase diagram"),
    code(
        '''
cell = (
    df.groupby(["f", "k", "pattern"], as_index=False)[metrics]
    .mean()
    .sort_values(["k", "f"])
)
display(cell[["pattern", "f", "k", "pattern_contrast", "mean_A", "mean_B", "spectral_slope"]])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, metric, title in zip(
    axes,
    ["pattern_contrast", "mean_A"],
    ["mean pattern_contrast", "mean mean_A"],
):
    sc = ax.scatter(
        cell["f"], cell["k"],
        c=cell[metric],
        s=180 + 500 * (cell[metric] - cell[metric].min()) / ((cell[metric].max() - cell[metric].min()) + 1e-9),
        cmap="YlGn", edgecolors="k", linewidths=0.8,
    )
    for _, r in cell.iterrows():
        ax.annotate(r["pattern"], (r["f"], r["k"]), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)
    ax.set_xlabel("feed f"); ax.set_ylabel("kill k"); ax.set_title(title)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
plt.suptitle("Gray–Scott phase diagram (sparse factorial)", y=1.02)
plt.tight_layout(); plt.show()
'''
    ),
    md("## Heatmaps on the (f, k) plane"),
    code(
        '''
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, metric in zip(axes, ["pattern_contrast", "mean_A", "mean_B"]):
    piv = cell.pivot(index="k", columns="f", values=metric)
    sns.heatmap(piv, annot=True, fmt=".3f", cmap="YlGn", ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title(metric); ax.set_xlabel("f"); ax.set_ylabel("k")
plt.suptitle("Regime-mean heatmaps (only occupied cells)", y=1.03)
plt.tight_layout(); plt.show()
'''
    ),
    md("## Pattern metrics vs parameters"),
    code(
        '''
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
sns.boxplot(data=df, x="pattern", y="pattern_contrast", ax=axes[0, 0], color="#8A9470")
axes[0, 0].tick_params(axis="x", rotation=30); axes[0, 0].set_title("Contrast by regime")

sns.boxplot(data=df, x="pattern", y="mean_B", ax=axes[0, 1], color="#C4A06A")
axes[0, 1].tick_params(axis="x", rotation=30); axes[0, 1].set_title("mean_B by regime")

sns.scatterplot(data=df, x="f", y="pattern_contrast", hue="pattern", s=70, ax=axes[1, 0])
axes[1, 0].set_title("pattern_contrast vs feed f")

sns.scatterplot(data=df, x="k", y="mean_A", hue="pattern", s=70, ax=axes[1, 1])
axes[1, 1].set_title("mean_A vs kill k")
plt.tight_layout(); plt.show()
'''
    ),
    code(
        '''
# Pairplot of key metrics colored by regime
pair_cols = [c for c in ["mean_A", "mean_B", "pattern_contrast", "spectral_slope"] if c in df.columns]
g = sns.pairplot(df, vars=pair_cols, hue="pattern", corner=True, height=2.2,
                 plot_kws={"s": 40, "alpha": 0.85})
g.fig.suptitle("Metric pairs by Gray–Scott regime", y=1.02)
plt.show()
'''
    ),
    md("## Spearman correlations"),
    code(
        '''
num = list(dict.fromkeys(c for c in ["f", "k", *metrics] if c in df.columns))
corr = df[num].corr(method="spearman")
display(corr.round(3))

plt.figure(figsize=(8, 6.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0, square=True)
plt.title("Spearman correlations — params & pattern metrics")
plt.tight_layout(); plt.show()
'''
    ),
    md("## Ranking regimes by contrast"),
    code(
        '''
rank = (
    df.groupby("pattern")
    .agg(
        n=("pattern_contrast", "size"),
        contrast_mean=("pattern_contrast", "mean"),
        contrast_std=("pattern_contrast", "std"),
        mean_A=("mean_A", "mean"),
        mean_B=("mean_B", "mean"),
        spectral_slope=("spectral_slope", "mean"),
        f=("f", "first"),
        k=("k", "first"),
    )
    .sort_values("contrast_mean", ascending=False)
    .round(4)
)
display(rank)

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=rank.reset_index(), x="pattern", y="contrast_mean",
            color="#5F6B45", ax=ax)
ax.set_title("Regimes ranked by mean pattern_contrast")
ax.tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.show()
'''
    ),
    md(
        "## Takeaway\n\n"
        "Treat `(f, k)` as a **phase diagram / sparse factorial**, not a balanced ANOVA layout. "
        "Named regimes separate in contrast and concentration space; heatmaps and rankings "
        "make those separations readable at a glance.\n"
    ),
]

AC_CELLS = [
    md(
        "# Acoustic scattering · geometry × sources\n\n"
        "Exploratory multiparameter analysis of The Well `acoustic_scattering_maze` feature table.\n\n"
        "**Not ANOVA** — response surfaces, interaction plots, tables, and spectral slope "
        "as a frequency-content proxy.\n"
    ),
    code(
        SETUP
        + '''
from src.dataset_catalog import get_dataset

spec = get_dataset("acoustic_scattering")
df = pd.read_parquet(spec.feature_path)
print("rows:", len(df), "| synthetic:", bool(df["synthetic"].any()) if "synthetic" in df.columns else "n/a")
print("maze_width levels:", sorted(df["maze_width"].unique()))
print("n_sources levels:", sorted(df["n_sources"].unique()))
df.head()
'''
    ),
    md("## Coverage table"),
    code(
        '''
coverage = (
    df.groupby(["maze_width", "n_sources"], as_index=False)
    .size()
    .rename(columns={"size": "n_traj"})
    .sort_values(["maze_width", "n_sources"])
)
display(coverage)
print(f"Cells occupied: {len(coverage)} | total traj: {coverage['n_traj'].sum()}")
'''
    ),
    md("## Feature dictionary"),
    code(
        '''
desc = {
    "maze_width": "path width (px) — geometry factor",
    "n_sources": "initial high-pressure rings",
    "mean_abs_pressure": "mean |p| over space/time",
    "pressure_energy": "mean p² (energy proxy)",
    "kinetic_energy": "mean ½|u|²",
    "wall_fraction": "fraction of dense-wall cells",
    "spectral_slope": "frequency-content proxy of pressure",
    "time_to_steady": "fraction of run until |p| plateaus",
}
pd.DataFrame({"column": list(desc), "meaning": list(desc.values())})
'''
    ),
    md("## Summary statistics"),
    code(
        '''
metrics = [c for c in [
    "mean_abs_pressure", "pressure_energy", "kinetic_energy",
    "wall_fraction", "spectral_slope", "time_to_steady",
] if c in df.columns]

display(df[metrics].describe().T.round(4))

by_width = df.groupby("maze_width")[metrics].agg(["mean", "std"]).round(4)
display(by_width)

by_src = df.groupby("n_sources")[metrics].agg(["mean", "std"]).round(4)
display(by_src)
'''
    ),
    md("## Cell-mean response table (geometry × sources)"),
    code(
        '''
cell = (
    df.groupby(["maze_width", "n_sources"], as_index=False)[metrics]
    .mean()
    .sort_values(["maze_width", "n_sources"])
)
display(cell.round(4))
'''
    ),
    md("## Response-surface heatmaps"),
    code(
        '''
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, metric in zip(axes, ["pressure_energy", "mean_abs_pressure", "spectral_slope"]):
    piv = cell.pivot(index="maze_width", columns="n_sources", values=metric)
    sns.heatmap(piv, annot=True, fmt=".3f", cmap="YlGn", ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title(f"mean {metric}")
    ax.set_xlabel("n_sources"); ax.set_ylabel("maze_width")
plt.suptitle("Geometry × sources response surfaces", y=1.03)
plt.tight_layout(); plt.show()
'''
    ),
    md("## Interaction plots"),
    code(
        '''
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.lineplot(
    data=cell, x="n_sources", y="pressure_energy",
    hue="maze_width", marker="o", ax=axes[0],
)
axes[0].set_title("pressure_energy vs n_sources by maze_width")

sns.lineplot(
    data=cell, x="maze_width", y="pressure_energy",
    hue="n_sources", marker="o", ax=axes[1],
)
axes[1].set_title("pressure_energy vs maze_width by n_sources")
plt.tight_layout(); plt.show()
'''
    ),
    md("## Distributions & scatter relationships"),
    code(
        '''
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
sns.boxplot(data=df, x="maze_width", y="pressure_energy", ax=axes[0, 0], color="#8A9470")
axes[0, 0].set_title("pressure_energy by maze_width")

sns.boxplot(data=df, x="n_sources", y="pressure_energy", ax=axes[0, 1], color="#C4A06A")
axes[0, 1].set_title("pressure_energy by n_sources")

sns.scatterplot(
    data=df, x="n_sources", y="pressure_energy",
    hue="maze_width", s=80, ax=axes[1, 0],
)
axes[1, 0].set_title("Energy vs sources (color = width)")

sns.scatterplot(
    data=df, x="wall_fraction", y="pressure_energy",
    hue="n_sources", style="maze_width", s=80, ax=axes[1, 1],
)
axes[1, 1].set_title("Energy vs wall_fraction")
plt.tight_layout(); plt.show()
'''
    ),
    md("## Frequency-content proxy (spectral slope)"),
    code(
        '''
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.boxplot(data=df, x="maze_width", y="spectral_slope", hue="n_sources", ax=axes[0])
axes[0].set_title("Spectral slope by geometry × sources")

sns.scatterplot(
    data=df, x="spectral_slope", y="pressure_energy",
    hue="maze_width", size="n_sources", sizes=(40, 160), ax=axes[1],
)
axes[1].set_title("Energy vs spectral slope")
plt.tight_layout(); plt.show()

slope_tab = (
    df.groupby(["maze_width", "n_sources"])["spectral_slope"]
    .agg(["count", "mean", "std"])
    .round(4)
)
display(slope_tab)
'''
    ),
    md("## Spearman correlations"),
    code(
        '''
num = list(dict.fromkeys(
    c for c in ["maze_width", "n_sources", "wall_fraction", *metrics] if c in df.columns
))
corr = df[num].corr(method="spearman")
display(corr.round(3))

plt.figure(figsize=(8, 6.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0, square=True)
plt.title("Spearman correlations — geometry, sources, responses")
plt.tight_layout(); plt.show()
'''
    ),
    md("## Pairplot of key responses"),
    code(
        '''
pair_cols = [c for c in [
    "pressure_energy", "mean_abs_pressure", "kinetic_energy", "spectral_slope"
] if c in df.columns]
g = sns.pairplot(
    df, vars=pair_cols, hue="maze_width", corner=True, height=2.2,
    plot_kws={"s": 45, "alpha": 0.85},
)
g.fig.suptitle("Acoustic response pairs (hue = maze_width)", y=1.02)
plt.show()
'''
    ),
    md(
        "## Takeaway\n\n"
        "Geometry and source count **jointly** shape pressure energy; non-parallel interaction "
        "curves signal multiparameter structure. Spectral slope tracks how scattering "
        "redistributes spatial-frequency content across the maze.\n"
    ),
]


def main() -> None:
    # Prefer project venv kernel; fall back to python3
    write_and_execute(NB_DIR / "gray_scott_analysis.ipynb", GS_CELLS)
    write_and_execute(NB_DIR / "acoustic_scattering_analysis.ipynb", AC_CELLS)


if __name__ == "__main__":
    main()
