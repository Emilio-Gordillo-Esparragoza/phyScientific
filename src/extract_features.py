"""
Extract scalar physics features from The Well trajectories.

Supports datasets: active_matter, gray_scott, acoustic_scattering.
Streams HDF5 from Hugging Face (or a local mirror), downsamples in
space/time for speed, and writes a tidy features.parquet for ANOVA / t-tests.

Windows note: the_well's WellDataset joins HF URIs with os.path.join (backslashes).
This script reads files directly via fsspec + h5py to avoid that bug.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Iterable

import fsspec
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.dataset_catalog import DATASET_IDS, default_out_path, get_dataset

CHANNEL_LAYOUT = {
    "concentration": 0,
    "velocity_x": 1,
    "velocity_y": 2,
    "D_xx": 3,
    "D_xy": 4,
    "D_yx": 5,
    "D_yy": 6,
    "E_xx": 7,
    "E_xy": 8,
    "E_yx": 9,
    "E_yy": 10,
}

DEFAULT_HF_SPLIT = {
    "train": "hf://datasets/polymathic-ai/active_matter/data/train",
    "valid": "hf://datasets/polymathic-ai/active_matter/data/valid",
    "test": "hf://datasets/polymathic-ai/active_matter/data/test",
}

GRAY_SCOTT_HF_SPLIT = {
    "train": "hf://datasets/polymathic-ai/gray_scott_reaction_diffusion/data/train",
    "valid": "hf://datasets/polymathic-ai/gray_scott_reaction_diffusion/data/valid",
    "test": "hf://datasets/polymathic-ai/gray_scott_reaction_diffusion/data/test",
}

ACOUSTIC_HF_SPLIT = {
    "train": "hf://datasets/polymathic-ai/acoustic_scattering_maze/data/train",
    "valid": "hf://datasets/polymathic-ai/acoustic_scattering_maze/data/valid",
    "test": "hf://datasets/polymathic-ai/acoustic_scattering_maze/data/test",
}

# Named Gray–Scott regimes from The Well dataset card
GRAY_SCOTT_REGIMES: list[tuple[str, float, float]] = [
    ("Gliders", 0.014, 0.054),
    ("Bubbles", 0.098, 0.057),
    ("Maze", 0.029, 0.057),
    ("Worms", 0.058, 0.065),
    ("Spirals", 0.018, 0.051),
    ("Spots", 0.030, 0.062),
]

_ALPHA_ZETA_RE = re.compile(
    r"zeta_(?P<zeta>-?\d+(?:\.\d+)?)_alpha_(?P<alpha>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

_FK_RE = re.compile(
    r"(?:f|F)[_=]?(?P<f>\d+(?:\.\d+)?)[_/].*?(?:k|K)[_=]?(?P<k>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def list_hdf5_files(base: str) -> list[str]:
    """List .hdf5 / .h5 files under an fsspec-compatible path."""
    fs, path = fsspec.core.url_to_fs(base)
    files = fs.glob(path.rstrip("/") + "/*.hdf5") + fs.glob(path.rstrip("/") + "/*.h5")
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No HDF5 files found under {base}")
    # Prefer full URLs when base is an HF URI
    if base.startswith("hf://"):
        return [f if f.startswith("hf://") else f"hf://{f}" for f in files]
    return files


def parse_alpha_zeta(url: str) -> tuple[float | None, float | None]:
    """Parse alpha/zeta from active_matter filenames when possible."""
    name = Path(url.split("?")[0]).name
    m = _ALPHA_ZETA_RE.search(name)
    if not m:
        return None, None
    return float(m.group("alpha")), float(m.group("zeta"))


def select_stratified_files(
    files: list[str],
    alphas: list[float] | None = None,
    zetas: list[float] | None = None,
) -> list[str]:
    """
    Keep files whose (alpha, zeta) match the requested factor levels.

    Default (both None) keeps the full list — the train split is already a
    complete 5×9 factorial (one HDF5 per cell).
    """
    if alphas is None and zetas is None:
        return files
    alpha_set = None if alphas is None else {float(a) for a in alphas}
    zeta_set = None if zetas is None else {float(z) for z in zetas}
    selected: list[str] = []
    for url in files:
        a, z = parse_alpha_zeta(url)
        if a is None or z is None:
            continue
        if alpha_set is not None and a not in alpha_set:
            continue
        if zeta_set is not None and z not in zeta_set:
            continue
        selected.append(url)
    if not selected:
        raise FileNotFoundError("No files matched the requested alpha/zeta filters.")
    return selected


def open_h5(url: str):
    """Open a local or remote HDF5 file (context-manager friendly via nested with)."""
    return fsspec.open(url, "rb")


def nematic_order_from_D(D: np.ndarray) -> np.ndarray:
    """
    Scalar nematic order parameter S from orientation tensor D.

    D shape: (..., 2, 2). For a traceless nematic tensor in 2D,
    S = 2 * |largest eigenvalue| (approx); we use S = sqrt(2) * ||D||_F
    which is proportional and stable for ANOVA.
    """
    # Frobenius norm of the 2x2 tensor field, then spatial mean
    fro = np.sqrt(np.sum(D**2, axis=(-2, -1)))  # (...,)
    return fro


def divergence_rms(vx: np.ndarray, vy: np.ndarray, dx: float = 1.0) -> float:
    """RMS of discrete divergence (periodic central differences along last two axes)."""
    # vx, vy: (T, H, W)
    dvx_dx = (np.roll(vx, -1, axis=-1) - np.roll(vx, 1, axis=-1)) / (2.0 * dx)
    dvy_dy = (np.roll(vy, -1, axis=-2) - np.roll(vy, 1, axis=-2)) / (2.0 * dx)
    div = dvx_dx + dvy_dy
    return float(np.sqrt(np.mean(div**2)))


def vorticity_enstrophy(vx: np.ndarray, vy: np.ndarray, dx: float = 1.0) -> float:
    """Mean enstrophy 0.5 * <omega^2> with omega = dvy/dx - dvx/dy."""
    dvy_dx = (np.roll(vy, -1, axis=-1) - np.roll(vy, 1, axis=-1)) / (2.0 * dx)
    dvx_dy = (np.roll(vx, -1, axis=-2) - np.roll(vx, 1, axis=-2)) / (2.0 * dx)
    omega = dvy_dx - dvx_dy
    return float(0.5 * np.mean(omega**2))


def spectral_slope_ke(vx: np.ndarray, vy: np.ndarray) -> float:
    """
    Log-log slope of the kinetic-energy spectrum E(k) ~ k^slope
    using a radially averaged 2D FFT of |u|^2 (time-averaged).
    """
    # Average over time: (H, W)
    ke = 0.5 * (vx**2 + vy**2).mean(axis=0)
    F = np.fft.fftshift(np.fft.fft2(ke))
    power = np.abs(F) ** 2
    H, W = power.shape
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    r_max = min(cy, cx)
    if r_max < 4:
        return float("nan")
    spectrum = np.bincount(r.ravel(), power.ravel())[: r_max]
    counts = np.bincount(r.ravel())[: r_max]
    counts = np.maximum(counts, 1)
    Ek = spectrum / counts
    k = np.arange(len(Ek))
    # Fit mid-k band (avoid DC and Nyquist)
    mask = (k >= 2) & (k <= r_max // 2) & (Ek > 0)
    if mask.sum() < 3:
        return float("nan")
    slope, _ = np.polyfit(np.log(k[mask]), np.log(Ek[mask]), 1)
    return float(slope)


def time_to_steady(order_t: np.ndarray, frac: float = 0.05) -> float:
    """
    Fraction of trajectory until |S(t) - S_final| stays within frac * range.
    Returns value in [0, 1].
    """
    if order_t.size < 2:
        return 0.0
    s_final = order_t[-1]
    span = max(np.ptp(order_t), 1e-8)
    tol = frac * span
    ok = np.abs(order_t - s_final) <= tol
    # last index from the end that is False, then +1
    if ok.all():
        return 0.0
    # find first index from which all remaining are True
    for i in range(len(ok)):
        if ok[i:].all():
            return float(i / (len(ok) - 1))
    return 1.0


def features_for_trajectory(
    concentration: np.ndarray,
    velocity: np.ndarray,
    D: np.ndarray,
    alpha: float,
    zeta: float,
    L: float,
    split: str,
    file_stem: str,
    traj_idx: int,
    dx: float,
) -> dict:
    """
    concentration: (T, H, W)
    velocity: (T, H, W, 2)
    D: (T, H, W, 2, 2)
    """
    vx = velocity[..., 0]
    vy = velocity[..., 1]

    # Nematic order time series (spatial mean of Frobenius norm)
    S_t = nematic_order_from_D(D).mean(axis=(-2, -1))  # (T,)
    S_mean = float(S_t.mean())
    S_final = float(S_t[-1])

    mean_c = float(concentration.mean())
    std_c = float(concentration.std())
    ke = float(0.5 * np.mean(vx**2 + vy**2))
    enstrophy = vorticity_enstrophy(vx, vy, dx=dx)
    div_rms = divergence_rms(vx, vy, dx=dx)
    slope = spectral_slope_ke(vx, vy)
    t_steady = time_to_steady(S_t)

    return {
        "split": split,
        "file": file_stem,
        "traj_idx": int(traj_idx),
        "replicate": int(traj_idx),
        "alpha": float(alpha),
        "zeta": float(zeta),
        "L": float(L),
        "mean_concentration": mean_c,
        "std_concentration": std_c,
        "nematic_order_S": S_mean,
        "nematic_order_S_final": S_final,
        "kinetic_energy": ke,
        "enstrophy": enstrophy,
        "div_u_rms": div_rms,
        "spectral_slope": slope,
        "time_to_steady": t_steady,
        "n_timesteps_used": int(concentration.shape[0]),
        "spatial_resolution_used": int(concentration.shape[-1]),
    }


def extract_from_file(
    url: str,
    split: str,
    time_stride: int = 4,
    space_stride: int = 8,
    max_traj: int | None = None,
) -> list[dict]:
    """Extract features for all trajectories in one HDF5 file."""
    file_stem = Path(url.split("?")[0]).name
    rows: list[dict] = []

    with fsspec.open(url, "rb") as fo:
        with h5py.File(fo, "r") as f:
            alpha = float(f["scalars/alpha"][()])
            zeta = float(f["scalars/zeta"][()])
            L = float(f["scalars/L"][()])
            x = np.asarray(f["dimensions/x"][()])
            dx = float(x[1] - x[0]) * space_stride if len(x) > 1 else 1.0

            conc = f["t0_fields/concentration"]  # (N, T, H, W)
            vel = f["t1_fields/velocity"]  # (N, T, H, W, 2)
            D = f["t2_fields/D"]  # (N, T, H, W, 2, 2)

            n_traj = conc.shape[0]
            n_use = n_traj if max_traj is None else min(n_traj, max_traj)

            for i in range(n_use):
                c_i = np.asarray(conc[i, ::time_stride, ::space_stride, ::space_stride])
                v_i = np.asarray(
                    vel[i, ::time_stride, ::space_stride, ::space_stride, :]
                )
                D_i = np.asarray(
                    D[i, ::time_stride, ::space_stride, ::space_stride, :, :]
                )
                rows.append(
                    features_for_trajectory(
                        c_i, v_i, D_i, alpha, zeta, L, split, file_stem, i, dx
                    )
                )
    return rows


def generate_synthetic_features(
    n_replicates: int = 5,
    seed: int = 42,
    splits: Iterable[str] = ("train",),
) -> pd.DataFrame:
    """
    Physics-plausible synthetic feature table matching the active_matter design.

    alpha in {-1,-2,-3,-4,-5}, zeta in {1,3,...,17}, beta fixed.
    Encodes: stronger |alpha| and larger zeta -> higher nematic order / energy,
    concentration conserved near 1, small div_u_rms, occasional anomalies.
    """
    rng = np.random.default_rng(seed)
    alphas = np.array([-1.0, -2.0, -3.0, -4.0, -5.0])
    zetas = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0])
    rows = []

    for split in splits:
        for alpha in alphas:
            for zeta in zetas:
                # Phase-transition-like sigmoid in zeta, modulated by |alpha|
                base_S = 0.05 + 0.55 / (1.0 + np.exp(-(zeta - 9.0) / 2.0))
                base_S *= 0.7 + 0.3 * (abs(alpha) / 5.0)
                base_ke = 0.02 + 0.15 * base_S**2 * (abs(alpha) / 3.0)
                base_ens = 0.01 + 0.4 * base_S * (abs(alpha) / 5.0)

                for rep in range(n_replicates):
                    noise = rng.normal(0, 1)
                    S = max(0.0, base_S + 0.04 * noise)
                    ke = max(0.0, base_ke + 0.01 * rng.normal())
                    ens = max(0.0, base_ens + 0.02 * rng.normal())
                    mean_c = 1.0 + 0.002 * rng.normal()
                    std_c = abs(0.05 + 0.08 * S + 0.01 * rng.normal())
                    div_rms = abs(1e-3 + 5e-4 * rng.normal())
                    slope = -2.0 - 0.3 * S + 0.15 * rng.normal()
                    t_steady = float(np.clip(0.35 - 0.1 * S + 0.05 * rng.normal(), 0, 1))

                    # Inject rare anomalies (~3%)
                    is_anom = rng.random() < 0.03
                    if is_anom:
                        S *= 1.0 + rng.choice([-0.8, 1.5])
                        ke *= 1.0 + rng.choice([-0.7, 2.0])
                        div_rms *= 20.0

                    rows.append(
                        {
                            "split": split,
                            "file": f"synthetic_zeta_{zeta}_alpha_{alpha}.hdf5",
                            "traj_idx": rep,
                            "replicate": rep,
                            "alpha": float(alpha),
                            "zeta": float(zeta),
                            "L": 10.0,
                            "mean_concentration": float(mean_c),
                            "std_concentration": float(std_c),
                            "nematic_order_S": float(max(S, 0.0)),
                            "nematic_order_S_final": float(max(S + 0.02 * rng.normal(), 0.0)),
                            "kinetic_energy": float(ke),
                            "enstrophy": float(ens),
                            "div_u_rms": float(div_rms),
                            "spectral_slope": float(slope),
                            "time_to_steady": t_steady,
                            "n_timesteps_used": 21,
                            "spatial_resolution_used": 32,
                            "synthetic": True,
                            "injected_anomaly": bool(is_anom),
                        }
                    )
    return pd.DataFrame(rows)


def generate_synthetic_gray_scott(
    n_replicates: int = 8,
    seed: int = 42,
    splits: Iterable[str] = ("train",),
) -> pd.DataFrame:
    """Synthetic table for the six Gray–Scott (f, k) regimes."""
    rng = np.random.default_rng(seed)
    # Distinct contrast / mean levels per named pattern
    contrast_base = {
        "Gliders": 0.22,
        "Bubbles": 0.35,
        "Maze": 0.48,
        "Worms": 0.40,
        "Spirals": 0.55,
        "Spots": 0.30,
    }
    rows: list[dict] = []
    for split in splits:
        for pattern, f, k in GRAY_SCOTT_REGIMES:
            base_c = contrast_base[pattern]
            base_A = 0.55 - 0.15 * (f / 0.1) + 0.05 * (k / 0.07)
            base_B = 0.20 + 0.25 * base_c
            for rep in range(n_replicates):
                noise = rng.normal(0, 1)
                contrast = max(0.02, base_c + 0.04 * noise)
                mean_A = float(np.clip(base_A + 0.03 * rng.normal(), 0.05, 0.95))
                mean_B = float(np.clip(base_B + 0.03 * rng.normal(), 0.02, 0.85))
                std_A = abs(0.08 + 0.5 * contrast + 0.02 * rng.normal())
                std_B = abs(0.06 + 0.45 * contrast + 0.02 * rng.normal())
                slope = -1.5 - 0.4 * contrast + 0.2 * rng.normal()
                t_steady = float(np.clip(0.45 - 0.2 * contrast + 0.08 * rng.normal(), 0, 1))
                is_anom = rng.random() < 0.04
                if is_anom:
                    contrast *= 1.0 + rng.choice([-0.7, 1.8])
                rows.append(
                    {
                        "split": split,
                        "file": f"synthetic_f_{f}_k_{k}.hdf5",
                        "traj_idx": rep,
                        "replicate": rep,
                        "f": float(f),
                        "k": float(k),
                        "pattern": pattern,
                        "mean_A": mean_A,
                        "mean_B": mean_B,
                        "std_A": float(std_A),
                        "std_B": float(std_B),
                        "pattern_contrast": float(max(contrast, 0.0)),
                        "spectral_slope": float(slope),
                        "time_to_steady": t_steady,
                        "n_timesteps_used": 64,
                        "spatial_resolution_used": 32,
                        "synthetic": True,
                        "injected_anomaly": bool(is_anom),
                    }
                )
    return pd.DataFrame(rows)


def generate_synthetic_acoustic(
    n_replicates: int = 4,
    seed: int = 42,
    splits: Iterable[str] = ("train",),
) -> pd.DataFrame:
    """Synthetic table for acoustic_scattering_maze (width × n_sources)."""
    rng = np.random.default_rng(seed)
    widths = [6, 8, 10, 12, 14, 16]
    sources = [1, 2, 3, 4, 5, 6]
    rows: list[dict] = []
    for split in splits:
        for w in widths:
            for ns in sources:
                # Narrower paths → more trapping / slightly lower transmitted energy
                base_e = 0.8 * ns * (0.6 + 0.4 * (w / 16.0))
                wall_frac = 0.55 - 0.015 * w
                for rep in range(n_replicates):
                    pe = max(0.05, base_e + 0.15 * rng.normal())
                    mean_p = max(0.02, 0.35 * np.sqrt(pe) + 0.05 * rng.normal())
                    ke = max(0.01, 0.4 * pe + 0.08 * rng.normal())
                    wf = float(np.clip(wall_frac + 0.03 * rng.normal(), 0.1, 0.9))
                    slope = -2.2 + 0.05 * ns + 0.15 * rng.normal()
                    t_steady = float(np.clip(0.3 + 0.05 * ns + 0.05 * rng.normal(), 0, 1))
                    is_anom = rng.random() < 0.04
                    if is_anom:
                        pe *= 1.0 + rng.choice([-0.75, 2.2])
                    rows.append(
                        {
                            "split": split,
                            "file": f"synthetic_width_{w}_sources_{ns}.hdf5",
                            "traj_idx": rep,
                            "replicate": rep,
                            "maze_width": int(w),
                            "n_sources": int(ns),
                            "mean_abs_pressure": float(mean_p),
                            "pressure_energy": float(pe),
                            "kinetic_energy": float(ke),
                            "wall_fraction": wf,
                            "spectral_slope": float(slope),
                            "time_to_steady": t_steady,
                            "n_timesteps_used": 50,
                            "spatial_resolution_used": 32,
                            "synthetic": True,
                            "injected_anomaly": bool(is_anom),
                        }
                    )
    return pd.DataFrame(rows)


def generate_synthetic_for_dataset(
    dataset_id: str,
    n_replicates: int = 5,
    seed: int = 42,
    splits: Iterable[str] = ("train",),
) -> pd.DataFrame:
    if dataset_id == "active_matter":
        return generate_synthetic_features(n_replicates, seed, splits)
    if dataset_id == "gray_scott":
        return generate_synthetic_gray_scott(n_replicates, seed, splits)
    if dataset_id == "acoustic_scattering":
        return generate_synthetic_acoustic(n_replicates, seed, splits)
    raise ValueError(f"Unknown dataset_id={dataset_id!r}")


def pattern_name_for_fk(f: float, k: float, tol: float = 1e-4) -> str:
    for name, ff, kk in GRAY_SCOTT_REGIMES:
        if abs(f - ff) < tol and abs(k - kk) < tol:
            return name
    return f"f={f:g}_k={k:g}"


def spectral_slope_field(field: np.ndarray) -> float:
    """Log-log spectral slope of a scalar field averaged over time (T, H, W)."""
    if field.ndim == 3:
        img = field.mean(axis=0)
    else:
        img = field
    F = np.fft.fftshift(np.fft.fft2(img))
    power = np.abs(F) ** 2
    H, W = power.shape
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    r_max = min(cy, cx)
    if r_max < 4:
        return float("nan")
    spectrum = np.bincount(r.ravel(), power.ravel())[:r_max]
    counts = np.maximum(np.bincount(r.ravel())[:r_max], 1)
    Ek = spectrum / counts
    k = np.arange(len(Ek))
    mask = (k >= 2) & (k <= r_max // 2) & (Ek > 0)
    if mask.sum() < 3:
        return float("nan")
    slope, _ = np.polyfit(np.log(k[mask]), np.log(Ek[mask]), 1)
    return float(slope)


def _read_scalar(f: h5py.File, names: list[str]) -> float | None:
    if "scalars" not in f:
        return None
    grp = f["scalars"]
    for name in names:
        if name in grp:
            return float(np.asarray(grp[name][()]).reshape(-1)[0])
    return None


def features_gray_scott_traj(
    A: np.ndarray,
    B: np.ndarray,
    f: float,
    k: float,
    split: str,
    file_stem: str,
    traj_idx: int,
) -> dict:
    mean_A = float(A.mean())
    mean_B = float(B.mean())
    std_A = float(A.std())
    std_B = float(B.std())
    contrast = float(std_A + std_B)
    A_t = A.mean(axis=(-2, -1))
    return {
        "split": split,
        "file": file_stem,
        "traj_idx": int(traj_idx),
        "replicate": int(traj_idx),
        "f": float(f),
        "k": float(k),
        "pattern": pattern_name_for_fk(f, k),
        "mean_A": mean_A,
        "mean_B": mean_B,
        "std_A": std_A,
        "std_B": std_B,
        "pattern_contrast": contrast,
        "spectral_slope": spectral_slope_field(A),
        "time_to_steady": time_to_steady(A_t),
        "n_timesteps_used": int(A.shape[0]),
        "spatial_resolution_used": int(A.shape[-1]),
    }


def extract_gray_scott_file(
    url: str,
    split: str,
    time_stride: int = 8,
    space_stride: int = 4,
    max_traj: int | None = None,
) -> list[dict]:
    """Extract Gray–Scott features from one HDF5 (A/B concentration fields)."""
    file_stem = Path(url.split("?")[0]).name
    rows: list[dict] = []
    with fsspec.open(url, "rb") as fo:
        with h5py.File(fo, "r") as f:
            f_val = _read_scalar(f, ["f", "F", "feed"])
            k_val = _read_scalar(f, ["k", "K", "kill"])
            if f_val is None or k_val is None:
                m = _FK_RE.search(file_stem)
                if m:
                    f_val, k_val = float(m.group("f")), float(m.group("k"))
                else:
                    # Try matching regime from filename tokens
                    f_val, k_val = 0.03, 0.062
                    for name, ff, kk in GRAY_SCOTT_REGIMES:
                        if name.lower() in file_stem.lower():
                            f_val, k_val = ff, kk
                            break

            t0 = f["t0_fields"]
            # Prefer explicit A/B; else first two scalar fields
            if "A" in t0 and "B" in t0:
                A_ds, B_ds = t0["A"], t0["B"]
            elif "concentration_A" in t0:
                A_ds, B_ds = t0["concentration_A"], t0["concentration_B"]
            else:
                keys = [k for k in t0.keys() if isinstance(t0[k], h5py.Dataset)]
                if len(keys) < 2:
                    raise KeyError(f"No A/B fields in {file_stem}; keys={list(t0.keys())}")
                A_ds, B_ds = t0[keys[0]], t0[keys[1]]

            n_traj = A_ds.shape[0]
            n_use = n_traj if max_traj is None else min(n_traj, max_traj)
            for i in range(n_use):
                A_i = np.asarray(A_ds[i, ::time_stride, ::space_stride, ::space_stride])
                B_i = np.asarray(B_ds[i, ::time_stride, ::space_stride, ::space_stride])
                rows.append(
                    features_gray_scott_traj(A_i, B_i, f_val, k_val, split, file_stem, i)
                )
    return rows


def _estimate_maze_width(density: np.ndarray) -> int:
    """Estimate path width (px) from a density snapshot; fallback to mid-range."""
    # Walls ~1e6, paths ~3 — threshold mid-log
    flat = density.reshape(-1)
    if flat.size < 16:
        return 10
    log_d = np.log10(np.clip(flat, 1e-6, None))
    is_path = log_d < 3.0  # path-like
    path_frac = float(is_path.mean())
    # Wider paths → higher path fraction; map roughly onto [6, 16]
    width = int(np.clip(round(6 + path_frac * 20), 6, 16))
    # Snap to even widths used in generation
    even = [6, 8, 10, 12, 14, 16]
    return min(even, key=lambda w: abs(w - width))


def _estimate_n_sources(pressure0: np.ndarray, threshold_frac: float = 0.5) -> int:
    """Count local high-pressure blobs in the first frame."""
    p = pressure0
    if p.ndim == 3:
        p = p[0]
    thr = float(p.min() + threshold_frac * (p.max() - p.min()))
    if p.max() - p.min() < 1e-8:
        return 1
    mask = p > thr
    # Crude connected-component count via flood-like dilation labels
    from scipy import ndimage

    labeled, n = ndimage.label(mask)
    return int(np.clip(n, 1, 6))


def features_acoustic_traj(
    pressure: np.ndarray,
    velocity: np.ndarray | None,
    density: np.ndarray | None,
    maze_width: int,
    n_sources: int,
    split: str,
    file_stem: str,
    traj_idx: int,
) -> dict:
    mean_abs_p = float(np.mean(np.abs(pressure)))
    pe = float(np.mean(pressure**2))
    if velocity is not None:
        vx, vy = velocity[..., 0], velocity[..., 1]
        ke = float(0.5 * np.mean(vx**2 + vy**2))
        slope = spectral_slope_ke(vx, vy)
    else:
        ke = float("nan")
        slope = spectral_slope_field(pressure)
    if density is not None:
        dens = density[0] if density.ndim == 3 else density
        wall_frac = float((dens > 100.0).mean())
    else:
        wall_frac = 0.5
    p_t = np.mean(np.abs(pressure), axis=(-2, -1))
    return {
        "split": split,
        "file": file_stem,
        "traj_idx": int(traj_idx),
        "replicate": int(traj_idx),
        "maze_width": int(maze_width),
        "n_sources": int(n_sources),
        "mean_abs_pressure": mean_abs_p,
        "pressure_energy": pe,
        "kinetic_energy": ke,
        "wall_fraction": wall_frac,
        "spectral_slope": slope,
        "time_to_steady": time_to_steady(p_t),
        "n_timesteps_used": int(pressure.shape[0]),
        "spatial_resolution_used": int(pressure.shape[-1]),
    }


def extract_acoustic_file(
    url: str,
    split: str,
    time_stride: int = 4,
    space_stride: int = 8,
    max_traj: int | None = None,
) -> list[dict]:
    """Extract acoustic_scattering_maze features from one HDF5."""
    file_stem = Path(url.split("?")[0]).name
    rows: list[dict] = []
    with fsspec.open(url, "rb") as fo:
        with h5py.File(fo, "r") as f:
            t0 = f["t0_fields"]
            # Pressure field name variants
            p_key = next(
                (k for k in ("pressure", "p", "Pressure") if k in t0),
                None,
            )
            if p_key is None:
                keys = [k for k in t0.keys() if isinstance(t0[k], h5py.Dataset)]
                p_key = keys[0]
            p_ds = t0[p_key]

            dens_ds = None
            for k in ("density", "rho", "material_density"):
                if k in t0:
                    dens_ds = t0[k]
                    break
            # Constant fields sometimes live under a different group
            if dens_ds is None and "constant_fields" in f:
                cf = f["constant_fields"]
                for k in ("density", "rho", "material_density"):
                    if k in cf:
                        dens_ds = cf[k]
                        break

            vel_ds = None
            if "t1_fields" in f and "velocity" in f["t1_fields"]:
                vel_ds = f["t1_fields"]["velocity"]

            width_attr = _read_scalar(f, ["maze_width", "path_width", "width"])
            n_src_attr = _read_scalar(f, ["n_sources", "n_source", "num_sources"])

            n_traj = p_ds.shape[0]
            n_use = n_traj if max_traj is None else min(n_traj, max_traj)
            for i in range(n_use):
                p_i = np.asarray(p_ds[i, ::time_stride, ::space_stride, ::space_stride])
                dens_i = None
                if dens_ds is not None:
                    d_raw = dens_ds[i] if dens_ds.ndim >= 4 else dens_ds[()]
                    dens_i = np.asarray(d_raw)
                    if dens_i.ndim == 3:
                        dens_i = dens_i[::time_stride, ::space_stride, ::space_stride]
                    elif dens_i.ndim == 2:
                        dens_i = dens_i[::space_stride, ::space_stride]
                v_i = None
                if vel_ds is not None:
                    v_i = np.asarray(
                        vel_ds[i, ::time_stride, ::space_stride, ::space_stride, :]
                    )

                if width_attr is not None:
                    maze_width = int(np.clip(round(width_attr), 6, 16))
                elif dens_i is not None:
                    d0 = dens_i[0] if dens_i.ndim == 3 else dens_i
                    maze_width = _estimate_maze_width(d0)
                else:
                    maze_width = 10

                if n_src_attr is not None:
                    n_sources = int(np.clip(round(n_src_attr), 1, 6))
                else:
                    try:
                        n_sources = _estimate_n_sources(p_i[0])
                    except Exception:  # noqa: BLE001
                        n_sources = 3

                rows.append(
                    features_acoustic_traj(
                        p_i, v_i, dens_i, maze_width, n_sources, split, file_stem, i
                    )
                )
    return rows


def coverage_order(files: list[str]) -> list[str]:
    """
    Reorder files to maximize early (alpha, zeta) factorial coverage.

    Round-robin across zeta levels, preferring extreme alphas first so a
    interrupted run still spans both factors for one-/two-way ANOVA.
    """
    buckets: dict[float, list[tuple[float, str]]] = {}
    unmatched: list[str] = []
    for url in files:
        a, z = parse_alpha_zeta(url)
        if a is None or z is None:
            unmatched.append(url)
            continue
        buckets.setdefault(z, []).append((a, url))

    for z in buckets:
        # Extreme |alpha| first, then mid levels
        buckets[z].sort(key=lambda t: (-abs(t[0]), t[0]))

    ordered: list[str] = []
    zetas = sorted(buckets.keys())
    while any(buckets[z] for z in zetas):
        for z in zetas:
            if buckets[z]:
                ordered.append(buckets[z].pop(0)[1])
    ordered.extend(unmatched)
    return ordered


def _write_checkpoint(rows: list[dict], out_path: Path) -> pd.DataFrame:
    """Persist current rows (overwrite) so long HF runs survive interruption."""
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


def _extract_one_file(
    url: str,
    split: str,
    time_stride: int,
    space_stride: int,
    max_traj_per_file: int | None,
    dataset_id: str = "active_matter",
) -> tuple[str, list[dict], float, str | None]:
    """Worker helper: return (stem, rows, elapsed_s, error_or_None)."""
    stem = Path(url.split("?")[0]).name
    t0 = time.time()
    try:
        if dataset_id == "gray_scott":
            rows = extract_gray_scott_file(
                url, split, time_stride, space_stride, max_traj_per_file
            )
        elif dataset_id == "acoustic_scattering":
            rows = extract_acoustic_file(
                url, split, time_stride, space_stride, max_traj_per_file
            )
        else:
            rows = extract_from_file(
                url,
                split=split,
                time_stride=time_stride,
                space_stride=space_stride,
                max_traj=max_traj_per_file,
            )
        for r in rows:
            r["synthetic"] = False
            r["injected_anomaly"] = False
        return stem, rows, time.time() - t0, None
    except Exception as exc:  # noqa: BLE001
        return stem, [], time.time() - t0, str(exc)


def _hf_splits_for(dataset_id: str) -> dict[str, str]:
    if dataset_id == "gray_scott":
        return GRAY_SCOTT_HF_SPLIT
    if dataset_id == "acoustic_scattering":
        return ACOUSTIC_HF_SPLIT
    return DEFAULT_HF_SPLIT


def _factor_cols(dataset_id: str) -> tuple[str, str]:
    spec = get_dataset(dataset_id)
    return spec.factor_a, spec.factor_b


def run_extraction(
    splits: list[str],
    out_path: Path,
    time_stride: int,
    space_stride: int,
    max_files: int | None,
    max_traj_per_file: int | None,
    base_override: str | None,
    alphas: list[float] | None = None,
    zetas: list[float] | None = None,
    resume: bool = False,
    workers: int = 1,
    coverage_first: bool = True,
    dataset_id: str = "active_matter",
) -> pd.DataFrame:
    """
    Stream The Well HDF5s and write a features parquet for the chosen dataset.
    """
    import threading
    from concurrent.futures import ProcessPoolExecutor, as_completed

    factor_a, factor_b = _factor_cols(dataset_id)
    hf_map = _hf_splits_for(dataset_id)

    all_rows: list[dict] = []
    done_stems: set[str] = set()
    if resume and out_path.exists():
        prev = pd.read_parquet(out_path)
        if "synthetic" in prev.columns:
            prev = prev.loc[~prev["synthetic"].fillna(False)].copy()
        all_rows = prev.to_dict(orient="records")
        done_stems = set(prev["file"].astype(str).unique())
        print(
            f"Resuming with {len(all_rows)} rows from {len(done_stems)} files in {out_path}",
            flush=True,
        )

    lock = threading.Lock()
    workers = max(1, int(workers))

    for split in splits:
        base = base_override or hf_map[split]
        files = list_hdf5_files(base)
        if dataset_id == "active_matter":
            files = select_stratified_files(files, alphas=alphas, zetas=zetas)
        if max_files is not None:
            files = files[:max_files]
        pending = [u for u in files if Path(u.split("?")[0]).name not in done_stems]
        if coverage_first and dataset_id == "active_matter":
            pending = coverage_order(pending)
        print(
            f"[{dataset_id}/{split}] {len(files)} matched files from {base} "
            f"({len(pending)} pending, strides t={time_stride} x={space_stride}, "
            f"workers={workers})",
            flush=True,
        )

        def _commit(stem: str, rows: list[dict], elapsed: float) -> None:
            with lock:
                all_rows.extend(rows)
                done_stems.add(stem)
                _write_checkpoint(all_rows, out_path)
                cells = (
                    pd.DataFrame(all_rows).groupby([factor_a, factor_b]).ngroups
                    if all_rows and factor_a in all_rows[0]
                    else "?"
                )
                print(
                    f"  OK {stem}: {len(rows)} traj in {elapsed:.1f}s "
                    f"(checkpoint {len(all_rows)} rows, cells={cells})",
                    flush=True,
                )

        if workers == 1:
            for url in tqdm(pending, desc=f"extract:{dataset_id}:{split}"):
                stem, rows, elapsed, err = _extract_one_file(
                    url, split, time_stride, space_stride, max_traj_per_file, dataset_id
                )
                if err:
                    print(f"  FAIL {url}: {err}", flush=True)
                else:
                    _commit(stem, rows, elapsed)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        _extract_one_file,
                        url,
                        split,
                        time_stride,
                        space_stride,
                        max_traj_per_file,
                        dataset_id,
                    ): url
                    for url in pending
                }
                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"extract:{dataset_id}:{split}",
                ):
                    url = futures[fut]
                    try:
                        stem, rows, elapsed, err = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        print(f"  FAIL {url}: {exc}", flush=True)
                        continue
                    if err:
                        print(f"  FAIL {url}: {err}", flush=True)
                    else:
                        _commit(stem, rows, elapsed)

    if not all_rows:
        raise RuntimeError("No features extracted. Check network / paths.")
    df = _write_checkpoint(all_rows, out_path)
    print(
        f"Wrote {len(df)} rows -> {out_path} "
        f"({factor_a} levels={df[factor_a].nunique()}, "
        f"{factor_b} levels={df[factor_b].nunique()}, "
        f"cells={df.groupby([factor_a, factor_b]).ngroups})",
        flush=True,
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=str,
        default="active_matter",
        choices=list(DATASET_IDS),
        help="Which Well dataset to extract (default: active_matter).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate a physics-plausible synthetic feature table (no download).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        choices=["train", "valid", "test"],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output parquet (default: catalog path for --dataset).",
    )
    parser.add_argument("--time-stride", type=int, default=4)
    parser.add_argument("--space-stride", type=int, default=8)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-traj-per-file", type=int, default=None)
    parser.add_argument(
        "--base",
        type=str,
        default=None,
        help="Override data root (local folder or hf:// URI). If set, --splits are ignored for path.",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=None,
        help="Optional alpha filter (active_matter only).",
    )
    parser.add_argument(
        "--zetas",
        type=float,
        nargs="+",
        default=None,
        help="Optional zeta filter (active_matter only).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip file stems already present in --out and append new rows.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent HDF5 streams (network-bound; 3–4 is usually a good start).",
    )
    parser.add_argument("--n-replicates", type=int, default=5, help="Synthetic only.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = args.out or default_out_path(args.dataset)

    if args.synthetic:
        # Sensible replicate defaults per dataset when user left default 5
        n_rep = args.n_replicates
        if args.dataset == "gray_scott" and n_rep == 5:
            n_rep = 8
        if args.dataset == "acoustic_scattering" and n_rep == 5:
            n_rep = 4
        df = generate_synthetic_for_dataset(
            args.dataset, n_replicates=n_rep, seed=args.seed, splits=args.splits
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        # Keep legacy active_matter path in sync
        if args.dataset == "active_matter":
            legacy = Path("data/features.parquet")
            legacy.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(legacy, index=False)
        print(f"Wrote {len(df)} synthetic rows -> {out_path}")
        return

    run_extraction(
        splits=args.splits,
        out_path=out_path,
        time_stride=args.time_stride,
        space_stride=args.space_stride,
        max_files=args.max_files,
        max_traj_per_file=args.max_traj_per_file,
        base_override=args.base,
        alphas=args.alphas,
        zetas=args.zetas,
        resume=args.resume,
        workers=args.workers,
        coverage_first=True,
        dataset_id=args.dataset,
    )


if __name__ == "__main__":
    main()
