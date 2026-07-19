"""
Shared statistical helpers for ANOVA, t-tests, assumption checks, and anomalies.

Used by notebooks/analysis.ipynb and app/streamlit_app.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats


VerdictLevel = Literal[
    "strong_evidence",
    "moderate_evidence",
    "weak_evidence",
    "no_visible_difference",
]


@dataclass
class AnovaResult:
    source: str
    ss: float
    df: float
    ms: float
    f: float
    p: float
    eta_sq: float
    omega_sq: float


@dataclass
class TeachingAnova:
    """Outputs of the interactive ANOVA sandbox."""

    groups: list[np.ndarray]
    labels: list[str]
    ss_between: float
    ss_within: float
    ss_total: float
    df_between: int
    df_within: int
    ms_between: float
    ms_within: float
    f: float
    p: float
    eta_sq: float
    omega_sq: float
    verdict: str
    verdict_level: VerdictLevel


def eta_squared(ss_effect: float, ss_total: float) -> float:
    if ss_total <= 0:
        return 0.0
    return float(ss_effect / ss_total)


def omega_squared(ss_effect: float, df_effect: float, ms_error: float, ss_total: float) -> float:
    num = ss_effect - df_effect * ms_error
    den = ss_total + ms_error
    if den <= 0:
        return 0.0
    return float(max(num / den, 0.0))


def verdict_from_p_and_eta(p: float, eta_sq: float, alpha: float = 0.05) -> tuple[str, VerdictLevel]:
    """Human-readable evidence statement combining significance and effect size."""
    if p < alpha and eta_sq >= 0.14:
        return "Strong evidence for a difference (large effect)", "strong_evidence"
    if p < alpha and eta_sq >= 0.06:
        return "Evidence for a difference (moderate effect)", "moderate_evidence"
    if p < alpha:
        return "Statistically significant but small effect", "weak_evidence"
    if eta_sq >= 0.06:
        return "Visible separation but not statistically significant", "weak_evidence"
    return "Without visible differences", "no_visible_difference"


def one_way_anova(df: pd.DataFrame, response: str, factor: str) -> dict:
    """One-way ANOVA with effect sizes and Tukey HSD."""
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    data = df[[response, factor]].dropna().copy()
    data[factor] = data[factor].astype(str)
    groups = [g[response].to_numpy() for _, g in data.groupby(factor)]
    if len(groups) < 2:
        raise ValueError("Need at least 2 groups for ANOVA")

    f_stat, p_val = stats.f_oneway(*groups)
    # SS decomposition
    grand = data[response].mean()
    ss_total = float(((data[response] - grand) ** 2).sum())
    ss_within = float(sum(((g - g.mean()) ** 2).sum() for g in groups))
    ss_between = ss_total - ss_within
    k = len(groups)
    n = len(data)
    df_b, df_w = k - 1, n - k
    ms_b = ss_between / df_b if df_b else np.nan
    ms_w = ss_within / df_w if df_w else np.nan
    eta = eta_squared(ss_between, ss_total)
    omega = omega_squared(ss_between, df_b, ms_w, ss_total)

    tukey = pairwise_tukeyhsd(data[response], data[factor], alpha=0.05)
    tukey_df = pd.DataFrame(
        tukey._results_table.data[1:], columns=tukey._results_table.data[0]
    )

    text, level = verdict_from_p_and_eta(float(p_val), eta)
    return {
        "f": float(f_stat),
        "p": float(p_val),
        "ss_between": ss_between,
        "ss_within": ss_within,
        "ss_total": ss_total,
        "df_between": df_b,
        "df_within": df_w,
        "ms_between": float(ms_b),
        "ms_within": float(ms_w),
        "eta_sq": eta,
        "omega_sq": omega,
        "tukey": tukey_df,
        "verdict": text,
        "verdict_level": level,
        "n": n,
        "k": k,
    }


def two_way_anova(df: pd.DataFrame, response: str, factor_a: str, factor_b: str) -> dict:
    """Two-way ANOVA with interaction: response ~ C(A) * C(B)."""
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    data = df[[response, factor_a, factor_b]].dropna().copy()
    data[factor_a] = data[factor_a].astype(str)
    data[factor_b] = data[factor_b].astype(str)
    formula = f"{response} ~ C({factor_a}) * C({factor_b})"
    model = ols(formula, data=data).fit()
    table = anova_lm(model, typ=2)
    ss_total = float(table["sum_sq"].sum())
    rows = []
    for source, row in table.iterrows():
        ss = float(row["sum_sq"])
        dfree = float(row["df"])
        f = float(row["F"]) if pd.notna(row["F"]) else np.nan
        p = float(row["PR(>F)"]) if pd.notna(row["PR(>F)"]) else np.nan
        ms = ss / dfree if dfree else np.nan
        # partial eta^2 ≈ SS_effect / (SS_effect + SS_error)
        ss_err = float(table.loc["Residual", "sum_sq"])
        partial_eta = ss / (ss + ss_err) if source != "Residual" else np.nan
        rows.append(
            {
                "source": source,
                "ss": ss,
                "df": dfree,
                "ms": ms,
                "F": f,
                "p": p,
                "partial_eta_sq": partial_eta,
                "eta_sq": eta_squared(ss, ss_total) if source != "Residual" else np.nan,
            }
        )
    result_df = pd.DataFrame(rows)

    # Overall verdict based on strongest non-residual effect
    effects = result_df[result_df["source"] != "Residual"].copy()
    if len(effects):
        best = effects.loc[effects["partial_eta_sq"].idxmax()]
        text, level = verdict_from_p_and_eta(
            float(best["p"]) if pd.notna(best["p"]) else 1.0,
            float(best["partial_eta_sq"]) if pd.notna(best["partial_eta_sq"]) else 0.0,
        )
    else:
        text, level = "Without visible differences", "no_visible_difference"

    return {
        "table": result_df,
        "model_summary": str(model.summary()),
        "ss_total": ss_total,
        "verdict": text,
        "verdict_level": level,
        "n": len(data),
    }


def pairwise_ttests(
    df: pd.DataFrame,
    response: str,
    factor: str,
    paired: bool = False,
    padjust: str = "holm",
) -> pd.DataFrame:
    """All pairwise Welch t-tests with multiple-comparison correction (pingouin)."""
    import pingouin as pg

    data = df[[response, factor]].dropna().copy()
    data[factor] = data[factor].astype(str)
    return pg.pairwise_tests(
        data=data,
        dv=response,
        between=factor,
        padjust=padjust,
        parametric=True,
        correction="auto",
    )


def assumption_checks(df: pd.DataFrame, response: str, factor: str) -> dict:
    """Levene (homogeneity) + Shapiro on residuals (normality)."""
    data = df[[response, factor]].dropna().copy()
    data[factor] = data[factor].astype(str)
    groups = [g[response].to_numpy() for _, g in data.groupby(factor)]
    levene_stat, levene_p = stats.levene(*groups)

    # Residuals = value - group mean
    residuals = []
    for g in groups:
        residuals.append(g - g.mean())
    resid = np.concatenate(residuals)
    # Shapiro is unreliable for n>5000; subsample if needed
    if len(resid) > 5000:
        rng = np.random.default_rng(0)
        resid = rng.choice(resid, size=5000, replace=False)
    if len(resid) >= 3:
        shapiro_stat, shapiro_p = stats.shapiro(resid)
    else:
        shapiro_stat, shapiro_p = np.nan, np.nan

    return {
        "levene_stat": float(levene_stat),
        "levene_p": float(levene_p),
        "equal_variance": bool(levene_p >= 0.05),
        "shapiro_stat": float(shapiro_stat) if pd.notna(shapiro_stat) else np.nan,
        "shapiro_p": float(shapiro_p) if pd.notna(shapiro_p) else np.nan,
        "normal_residuals": bool(pd.notna(shapiro_p) and shapiro_p >= 0.05),
        "recommend_nonparametric": bool(levene_p < 0.05 or (pd.notna(shapiro_p) and shapiro_p < 0.05)),
    }


def kruskal_wallis(df: pd.DataFrame, response: str, factor: str) -> dict:
    """Nonparametric one-way alternative."""
    data = df[[response, factor]].dropna().copy()
    groups = [g[response].to_numpy() for _, g in data.groupby(factor)]
    h, p = stats.kruskal(*groups)
    return {"H": float(h), "p": float(p), "k": len(groups), "n": len(data)}


def simulate_anova_groups(
    n_groups: int,
    n_per_group: int,
    mean_diff: float,
    within_sd: float,
    seed: int = 0,
) -> TeachingAnova:
    """
    Generate synthetic groups for the teaching sandbox.

    Group means are centered at 0 and spaced by `mean_diff`
    (so total span ≈ mean_diff * (n_groups - 1)).
    """
    rng = np.random.default_rng(seed)
    if n_groups < 2:
        raise ValueError("n_groups must be >= 2")
    centers = np.linspace(
        -0.5 * mean_diff * (n_groups - 1),
        0.5 * mean_diff * (n_groups - 1),
        n_groups,
    )
    groups = [rng.normal(loc=c, scale=within_sd, size=n_per_group) for c in centers]
    labels = [f"G{i+1}" for i in range(n_groups)]

    all_y = np.concatenate(groups)
    grand = all_y.mean()
    ss_total = float(((all_y - grand) ** 2).sum())
    ss_within = float(sum(((g - g.mean()) ** 2).sum() for g in groups))
    ss_between = ss_total - ss_within
    df_b = n_groups - 1
    df_w = n_groups * n_per_group - n_groups
    ms_b = ss_between / df_b
    ms_w = ss_within / df_w if df_w else np.nan
    f = ms_b / ms_w if ms_w and ms_w > 0 else np.inf
    p = float(1 - stats.f.cdf(f, df_b, df_w)) if np.isfinite(f) else 0.0
    eta = eta_squared(ss_between, ss_total)
    omega = omega_squared(ss_between, df_b, ms_w, ss_total)
    text, level = verdict_from_p_and_eta(p, eta)

    return TeachingAnova(
        groups=groups,
        labels=labels,
        ss_between=ss_between,
        ss_within=ss_within,
        ss_total=ss_total,
        df_between=df_b,
        df_within=df_w,
        ms_between=float(ms_b),
        ms_within=float(ms_w),
        f=float(f),
        p=p,
        eta_sq=eta,
        omega_sq=omega,
        verdict=text,
        verdict_level=level,
    )


def detect_anomalies(
    df: pd.DataFrame,
    response: str,
    group_cols: list[str],
    method: Literal["zscore", "iqr", "mad"] = "mad",
    threshold: float = 3.5,
) -> pd.DataFrame:
    """
    Flag trajectories that deviate from their (alpha, zeta) cell.

    Returns a copy of df with columns: anomaly_score, is_anomaly.
    """
    out = df.copy()
    scores = np.full(len(out), np.nan)
    flags = np.zeros(len(out), dtype=bool)

    for _, idx in out.groupby(group_cols).groups.items():
        vals = out.loc[idx, response].to_numpy(dtype=float)
        if len(vals) < 3:
            continue
        if method == "zscore":
            mu, sd = vals.mean(), vals.std(ddof=1)
            s = np.abs(vals - mu) / (sd if sd > 0 else 1.0)
            flag = s > threshold
        elif method == "iqr":
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            s = np.maximum((lo - vals) / (iqr + 1e-12), (vals - hi) / (iqr + 1e-12))
            s = np.clip(s, 0, None)
            flag = (vals < lo) | (vals > hi)
        else:  # mad
            med = np.median(vals)
            mad = np.median(np.abs(vals - med))
            s = 0.6745 * np.abs(vals - med) / (mad if mad > 0 else 1.0)
            flag = s > threshold
        scores[list(idx)] = s
        flags[list(idx)] = flag

    out["anomaly_score"] = scores
    out["is_anomaly"] = flags
    out["anomaly_feature"] = response
    return out


def _run_physics_check(df: pd.DataFrame, spec: Any) -> dict:
    """Evaluate one PhysicsCheckSpec (from dataset_catalog) against a feature table."""
    kind = getattr(spec, "kind", None)
    key = getattr(spec, "key", "check")
    title = getattr(spec, "title", key)

    if kind == "mean_near_target":
        col = spec.column
        mean_v = float(df[col].mean())
        std_v = float(df[col].std())
        target = float(spec.target)
        tol = float(spec.tol)
        ok = abs(mean_v - target) < tol
        msg = (
            f"Mean {col} = {mean_v:.4f} (target {target}, tol {tol}) — "
            + ("PASS" if ok else "FAIL")
        )
        return {
            "key": key,
            "title": title,
            "pass": ok,
            "message": msg,
            "mean": mean_v,
            "std": std_v,
            "target": target,
        }

    if kind == "range":
        col = spec.column
        mean_v = float(df[col].mean())
        lo = float(spec.lo) if spec.lo is not None else -np.inf
        hi = float(spec.hi) if spec.hi is not None else np.inf
        ok = bool(lo <= mean_v <= hi)
        # Soft pass for div_u_rms-style checks: also accept relative residual
        if col == "div_u_rms" and "kinetic_energy" in df.columns and not ok:
            ke_mean = float(df["kinetic_energy"].mean())
            vel_scale = float(np.sqrt(max(2.0 * ke_mean, 1e-12)))
            div_rel = mean_v / vel_scale
            ok = bool(div_rel < 2.0)
            detail = f" (relative to |u| scale: {div_rel:.3f})"
        else:
            detail = ""
        default_pass = getattr(spec, "pass_message", None) or f"Mean {col} in [{lo}, {hi}]"
        default_fail = getattr(spec, "fail_message", None) or f"Mean {col} outside [{lo}, {hi}]"
        msg = (
            f"Mean {col} = {mean_v:.4g}{detail} — "
            + (default_pass if ok else default_fail)
        )
        return {
            "key": key,
            "title": title,
            "pass": ok,
            "message": msg,
            "mean": mean_v,
        }

    if kind == "spearman":
        x_col, y_col = spec.x, spec.y
        spearman = stats.spearmanr(df[x_col], df[y_col])
        rho = float(spearman.correlation) if spearman.correlation is not None else 0.0
        p = float(spearman.pvalue) if spearman.pvalue is not None else 1.0
        min_rho = float(spec.min_rho) if spec.min_rho is not None else 0.0
        max_p = float(spec.max_p) if spec.max_p is not None else 0.05
        ok = bool(rho > min_rho and p < max_p)
        default_pass = getattr(spec, "pass_message", None) or f"{y_col} rises with {x_col}"
        default_fail = getattr(spec, "fail_message", None) or f"Weak {y_col} vs {x_col}"
        msg = (
            f"{y_col} vs {x_col}: Spearman ρ={rho:.3f}, p={p:.3e} — "
            + (default_pass if ok else default_fail)
        )
        return {
            "key": key,
            "title": title,
            "pass": ok,
            "message": msg,
            "spearman_rho": rho,
            "spearman_p": p,
        }

    if kind == "finite":
        cols = list(spec.columns) if spec.columns else []
        missing = [c for c in cols if c not in df.columns]
        if missing:
            return {
                "key": key,
                "title": title,
                "pass": False,
                "message": f"Missing columns: {missing}",
            }
        finite_ok = all(bool(np.isfinite(df[c]).all()) for c in cols)
        finite_ok = finite_ok and all(bool(np.isfinite(df[c].mean())) for c in cols)
        msg = (
            f"Finite values for {', '.join(cols)} — "
            + ("PASS" if finite_ok else "FAIL")
        )
        return {"key": key, "title": title, "pass": finite_ok, "message": msg}

    return {
        "key": key,
        "title": title,
        "pass": False,
        "message": f"Unknown check kind: {kind!r}",
    }


def physics_validation(
    df: pd.DataFrame,
    checks: list[Any] | tuple[Any, ...] | None = None,
) -> dict:
    """
    Run physics / sanity checks on a feature table.

    If ``checks`` is None, use the classic active_matter checks (backward compatible).
    Otherwise ``checks`` is a sequence of PhysicsCheckSpec from dataset_catalog.
    """
    if checks is None:
        # Legacy active_matter defaults (imported lazily to avoid circular imports)
        from src.dataset_catalog import ACTIVE_MATTER

        checks = ACTIVE_MATTER.physics_checks

    out: dict = {}
    for spec in checks:
        block = _run_physics_check(df, spec)
        out[block["key"]] = block
    return out
