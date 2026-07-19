# The Well · active_matter Statistical Lab

Statistical analysis and an interactive ANOVA dashboard for physics simulations from **[The Well](https://github.com/PolymathicAI/the_well)** (`active_matter` dataset).

## What this project does

1. **Extracts** scalar physics features from each trajectory (nematic order, kinetic energy, enstrophy, divergence residual, spectral slope, …) plus control factors `alpha` (dipole) and `zeta` (alignment).
2. **Quantifies** how much initial circumstances matter with **one- / two-way ANOVA**, effect sizes (η², ω²), Tukey HSD, and pairwise t-tests.
3. **Validates** approximate physics laws (concentration ≈ 1, ∇·u ≈ 0, isotropic→nematic transition vs `zeta`).
4. **Flags anomalies** within each `(alpha, zeta)` cell.
5. Ships an **interactive Streamlit app** with a teaching ANOVA sandbox (sliders for mean difference / within-group dispersion → live F, p, SSE, η² + evidence verdict).

## Project layout

```
phyScintific/
├── app/streamlit_app.py      # Dashboard (3 tabs)
├── data/features.parquet     # Analysis-ready feature table (prefer real)
├── notebooks/analysis.ipynb  # Narrative statistical analysis
├── src/
│   ├── extract_features.py   # HF stream / synthetic feature builder
│   └── stats.py              # Shared ANOVA / t-test / anomaly helpers
├── requirements.txt
└── README.md
```

## Environment (important)

This machine’s default Python may be **3.14**. `the_well` / PyTorch need **3.11 or 3.12**. A `.venv` on Python 3.12 is expected:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

## Data

### Preferred: real features from Hugging Face

`data/features.parquet` should contain **real** The Well `active_matter` trajectories whenever possible. The train split is a complete **5 α × 9 ζ** factorial (**45** HDF5 files). Most cells have **3** trajectories; a few files in the HF release contain 1–2 or 4 trajectories, so a full extract is typically ~130–140 rows rather than a rigid 135.

Each remote file is ~740 MB. Streaming with space/time strides is the efficient path (no full local mirror required):

```powershell
# Full factorial (recommended). Checkpoints after every file; safe to re-run with --resume.
python -u -m src.extract_features --splits train --time-stride 8 --space-stride 16 --max-traj-per-file 3 --workers 4 --out data/features.parquet

# Resume after a network interruption
python -u -m src.extract_features --splits train --time-stride 8 --space-stride 16 --max-traj-per-file 3 --workers 4 --out data/features.parquet --resume

# Stratified subset if you only need a smart span of both factors
python -m src.extract_features --splits train --alphas -1 -5 --time-stride 8 --space-stride 16 --workers 4
python -m src.extract_features --splits train --zetas 9 --time-stride 8 --space-stride 16 --workers 4 --resume
```

**Efficiency plan used by this project**

1. Stream all **45** train cells via `fsspec` + `h5py` (avoids the Windows `WellDataset` path bug).
2. Aggressive but documented strides: `--time-stride 8 --space-stride 16`, capped at 3 traj/file.
3. Concurrent streams with `--workers 4` (network-bound; typically 3–4× wall-time speedup).
4. Checkpoint every file into `data/features.parquet` so partial progress is usable for ANOVA.
5. Optional local mirror for faster re-runs (large disk):

```powershell
# After downloading files under data/raw/train/*.hdf5
python -m src.extract_features --base data/raw/train --out data/features.parquet
```

Smoke test (few files):

```powershell
python -m src.extract_features --splits train --max-files 3 --max-traj-per-file 2 --out data/features_real_sample.parquet
```

### Fallback only: synthetic demo table

Use synthetic data **only** when HF is unavailable. It matches the real factorial design but is not physical truth:

```powershell
python -m src.extract_features --synthetic --out data/features.parquet
```

Design: `alpha ∈ {-1,-2,-3,-4,-5}`, `zeta ∈ {1,3,…,17}`, 5 replicates per cell (225 rows). The Streamlit header labels the table as synthetic when the `synthetic` column is true.

## Run the analysis notebook

```powershell
.\.venv\Scripts\Activate.ps1
jupyter notebook notebooks/analysis.ipynb
```

## Run the Streamlit dashboard

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app/streamlit_app.py
```

### Tabs

| Tab | Content |
|-----|---------|
| **ANOVA sandbox** | Sliders for #groups, n, mean difference, within-group SD → live boxplots, SSB/SSW bars, F, p, SSE, η², and a verdict (“Evidence for a difference” / “Without visible differences”). |
| **Real-data ANOVA** | One-way / two-way ANOVA and pairwise t-tests on `features.parquet`. |
| **Physics & anomalies** | Conservation / incompressibility / phase-transition checks + MAD/IQR/z-score anomaly table. |

## Feature dictionary

| Column | Meaning |
|--------|---------|
| `alpha`, `zeta`, `L` | Control / initial-circumstance parameters |
| `mean_concentration`, `std_concentration` | Conservation / mixing |
| `nematic_order_S`, `nematic_order_S_final` | Orientation order parameter (phase transition) |
| `kinetic_energy`, `enstrophy` | Flow intensity |
| `div_u_rms` | Incompressibility residual |
| `spectral_slope` | KE spectrum log–log slope |
| `time_to_steady` | Fraction of time to order-parameter plateau |
| `synthetic` | `False` for real Well data; `True` for demo generator |

## Citation

If you use The Well / `active_matter` data, please cite the Well paper and the active-matter source paper (see [dataset card](https://huggingface.co/datasets/polymathic-ai/active_matter)).
