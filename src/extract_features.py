"""
Extract scalar physics features from The Well active_matter trajectories.

Streams HDF5 files from Hugging Face (or a local mirror), downsamples in
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


_ALPHA_ZETA_RE = re.compile(
    r"zeta_(?P<zeta>-?\d+(?:\.\d+)?)_alpha_(?P<alpha>-?\d+(?:\.\d+)?)",
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
) -> tuple[str, list[dict], float, str | None]:
    """Worker helper: return (stem, rows, elapsed_s, error_or_None)."""
    stem = Path(url.split("?")[0]).name
    t0 = time.time()
    try:
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
) -> pd.DataFrame:
    """
    Stream The Well active_matter HDF5s and write features.parquet.

    Efficiency notes
    ----------------
    - Train is a complete 5 alpha × 9 zeta factorial (45 files, typically 3–4 traj).
    - Prefer aggressive --time-stride / --space-stride over skipping factor cells.
    - --workers > 1 streams several HDF5s concurrently (network-bound speedup).
    - --resume skips file stems already in out_path (safe restarts after network blips).
    - Each successful file is checkpointed to out_path immediately.
    - For offline re-runs, download a local mirror once and pass --base data/raw/train.
    """
    import threading
    from concurrent.futures import ProcessPoolExecutor, as_completed

    all_rows: list[dict] = []
    done_stems: set[str] = set()
    if resume and out_path.exists():
        prev = pd.read_parquet(out_path)
        # Keep only real rows when resuming a mixed/partial table
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
        base = base_override or DEFAULT_HF_SPLIT[split]
        files = list_hdf5_files(base)
        files = select_stratified_files(files, alphas=alphas, zetas=zetas)
        if max_files is not None:
            files = files[:max_files]
        pending = [u for u in files if Path(u.split("?")[0]).name not in done_stems]
        if coverage_first:
            pending = coverage_order(pending)
        print(
            f"[{split}] {len(files)} matched files from {base} "
            f"({len(pending)} pending, strides t={time_stride} x={space_stride}, "
            f"workers={workers}, coverage_first={coverage_first})",
            flush=True,
        )

        def _commit(stem: str, rows: list[dict], elapsed: float) -> None:
            with lock:
                all_rows.extend(rows)
                done_stems.add(stem)
                _write_checkpoint(all_rows, out_path)
                print(
                    f"  OK {stem}: {len(rows)} traj in {elapsed:.1f}s "
                    f"(checkpoint {len(all_rows)} rows, cells="
                    f"{pd.DataFrame(all_rows).groupby(['alpha','zeta']).ngroups})",
                    flush=True,
                )

        if workers == 1:
            for url in tqdm(pending, desc=f"extract:{split}"):
                stem, rows, elapsed, err = _extract_one_file(
                    url, split, time_stride, space_stride, max_traj_per_file
                )
                if err:
                    print(f"  FAIL {url}: {err}", flush=True)
                else:
                    _commit(stem, rows, elapsed)
        else:
            Executor = ProcessPoolExecutor
            with Executor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        _extract_one_file,
                        url,
                        split,
                        time_stride,
                        space_stride,
                        max_traj_per_file,
                    ): url
                    for url in pending
                }
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"extract:{split}"):
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
    n_alpha = df["alpha"].nunique()
    n_zeta = df["zeta"].nunique()
    print(
        f"Wrote {len(df)} rows -> {out_path} "
        f"(alpha levels={n_alpha}, zeta levels={n_zeta}, "
        f"cells={df.groupby(['alpha','zeta']).ngroups})",
        flush=True,
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--out", type=Path, default=Path("data/features.parquet"))
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
        help="Optional alpha filter (e.g. --alphas -1 -5). Default: all levels.",
    )
    parser.add_argument(
        "--zetas",
        type=float,
        nargs="+",
        default=None,
        help="Optional zeta filter (e.g. --zetas 1 9 17). Default: all levels.",
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

    if args.synthetic:
        df = generate_synthetic_features(
            n_replicates=args.n_replicates, seed=args.seed, splits=args.splits
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out, index=False)
        print(f"Wrote {len(df)} synthetic rows -> {args.out}")
        return

    run_extraction(
        splits=args.splits,
        out_path=args.out,
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
    )


if __name__ == "__main__":
    main()
