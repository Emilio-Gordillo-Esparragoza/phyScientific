# The Well · Statistical Laboratory

Statistical analysis and an interactive ANOVA dashboard for physics simulations from **[The Well](https://github.com/PolymathicAI/the_well)**, plus a pedagogical **Feynman Lost Lecture** lab on elliptical orbits.

Supported ensembles (sidebar picker):

| Lab id | Hugging Face dataset | Analysis mode | Factors |
|--------|----------------------|---------------|---------|
| `active_matter` | `polymathic-ai/active_matter` | ANOVA (sandbox + real-data) | `alpha` × `zeta` |
| `gray_scott` | `polymathic-ai/gray_scott_reaction_diffusion` | **(f, k) phase diagram** + metrics vs params | `f` × `k` (6 regimes) |
| `acoustic_scattering` | `polymathic-ai/acoustic_scattering_maze` | **Response surfaces / interaction plots** | `maze_width` × `n_sources` |
| `planetary_motion` | pedagogical synthetic (not The Well) | **Feynman hodograph** elementary + formal proof | `eccentricity` × `angular_momentum` |

ANOVA panels appear **only** for `active_matter`. Gray–Scott and acoustic use exploratory phase-diagram and multiparameter views instead. `planetary_motion` is a geometric / celestial lab (Feynman’s Lost Lecture).

**Live demo:** [https://lecturelab.onrender.com/](https://lecturelab.onrender.com/)  
*(Render free tier may cold-start for ~30–60s after idle.)*

Repository: [Emilio-Gordillo-Esparragoza/LectureLab](https://github.com/Emilio-Gordillo-Esparragoza/LectureLab)

**License:** [Apache License 2.0](LICENSE) — free to use, modify, and redistribute for education and research; contributions welcome under the same terms.

## What this project does

1. **Extracts** scalar physics features from each trajectory plus dataset-specific control factors.
2. **Quantifies** how much initial circumstances matter with **one- / two-way ANOVA**, effect sizes (η², ω²), Tukey HSD, and pairwise t-tests.
3. **Validates** approximate physics / sanity checks (catalog-driven per dataset).
4. **Flags anomalies** within each circumstance cell.
5. Ships an **interactive Streamlit app** with a hamburger sidebar to switch datasets, plus a teaching ANOVA sandbox.
6. **`planetary_motion`:** interactive dual-plane (position + velocity) demonstration that inverse-square gravity + equal areas imply elliptical orbits, with a formal Runge–Lenz companion, notebook, and docs.

## UI: dataset sidebar

The lab board keeps the same palette and chart-paper grid. Use the **three-line control (top-left)** to open the sidebar and select `active_matter`, `gray_scott`, `acoustic_scattering`, or `planetary_motion`. Branding, factors, response lists, and physics cards update from [`src/dataset_catalog.py`](src/dataset_catalog.py).

## Mathematical documentation

Precise write-ups of **all math used in the repo** (what, why, assumptions, citations):

- [`docs/README.md`](docs/README.md) — index by lab  
- [`docs/bibliography.md`](docs/bibliography.md) — Fisher, Newton, Maxwell, Goodstein, Turing, …  
- Celestial: [`docs/celestial/feynman_elementary.md`](docs/celestial/feynman_elementary.md), [`docs/celestial/feynman_formal.md`](docs/celestial/feynman_formal.md)  
- Notebook: [`notebooks/feynman_lost_lecture.ipynb`](notebooks/feynman_lost_lecture.ipynb)
## Findings (active_matter)

Results below are from the committed real feature table [`data/features.parquet`](data/features.parquet) (also mirrored as [`data/features_active_matter.parquet`](data/features_active_matter.parquet)): **134** trajectories, **45/45** α×ζ factorial cells, `synthetic = 0`. Features were computed while streaming The Well train split from Hugging Face with `--time-stride 8 --space-stride 16` (analysis-ready table, not full-resolution fields).

`gray_scott`, `acoustic_scattering`, and `planetary_motion` ship with **synthetic** demo tables so the sidebar works offline; replace the Well extracts with real data when ready (see Data below). `planetary_motion` is always synthetic / pedagogical.

### Experimental design (active_matter)

- **System:** continuum theory of rod-like active particles in a Stokes fluid (`active_matter`).
- **Factors (initial / control circumstances):** dipole strength `alpha ∈ {-1, -2, -3, -4, -5}` and alignment `zeta ∈ {1, 3, …, 17}` (`beta` fixed at 0.8 in the source ensemble).
- **Responses used for inference:** primarily nematic order `nematic_order_S` (orientation-tensor magnitude), plus kinetic energy, enstrophy, concentration statistics, discrete divergence RMS, spectral slope, and time-to-steady.

### How much do initial circumstances matter?

Two-way ANOVA `nematic_order_S ~ C(alpha) * C(zeta)`:

| Term | F | p | partial η² |
|------|--:|--:|----------:|
| `zeta` (alignment) | ≈ 325 | ≈ 2×10⁻⁶² | **≈ 0.97** |
| `alpha` (dipole) | ≈ 1.5 | ≈ 0.22 | ≈ 0.06 |
| `alpha × zeta` | ≈ 3.6 | ≈ 1×10⁻⁶ | ≈ 0.56 |

One-way ANOVA on `zeta` alone: F ≈ 195, η² ≈ **0.93** — strong evidence for differences across alignment levels.

**Takeaway:** In this ensemble, **alignment (`zeta`) is the primary knob** for the isotropic→nematic-like rise in order. **Dipole strength (`alpha`) does not clearly shift mean order on its own**; it matters mainly by **modulating how strongly `zeta` acts** (significant interaction). Changing “initial circumstances” is therefore not uniform across parameters: `zeta` carries almost all of the between-group structure for `S`.

Reproduce interactively in the app tab **Real-data ANOVA** (two-way) or in [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb).

### Physics-law checks

| Check | Result | Interpretation |
|-------|--------|----------------|
| Concentration conservation | mean concentration = **1.0000** (target 1) | PASS — consistent with periodic BCs and `c≡1` initialization |
| Phase transition signature | Spearman(`S`, `zeta`) ρ ≈ **0.86**, p ≈ 4×10⁻⁴⁰ | PASS — order rises with alignment |
| Near-incompressibility | discrete `div_u_rms` elevated on the coarse analysis grid | Documented **downsampling caveat** — not treated as a failure of the Stokes sims; finer `--space-stride` reduces this residual |

### Anomalies

A within-cell MAD screen on `nematic_order_S` flags on the order of **~12 / 134** trajectories as atypical relative to their `(alpha, zeta)` neighbors. These are **quality / outlier notes** (unusual random initializations or extract noise), not automatic claims that the underlying physics is broken. See the **Physics & anomalies** tab.

### Methods note

All headline numbers refer to the stride-reduced feature table used for statistics and the live demo. Full-resolution HDF5 fields remain on Hugging Face; re-run extraction with smaller strides for higher-fidelity residuals if needed.

## Project layout

```
LectureLab/
├── app/streamlit_app.py      # Dashboard (sidebar + panels per analysis_mode)
├── data/
│   ├── features.parquet                 # active_matter (legacy path)
│   ├── features_active_matter.parquet   # canonical active_matter
│   ├── features_gray_scott.parquet
│   ├── features_acoustic_scattering.parquet
│   └── features_planetary_motion.parquet
├── docs/                     # Math resources (stats, physics, celestial)
│   ├── README.md
│   ├── bibliography.md
│   ├── statistics/
│   ├── physics/
│   └── celestial/
├── notebooks/
│   ├── analysis.ipynb
│   ├── gray_scott_analysis.ipynb
│   ├── acoustic_scattering_analysis.ipynb
│   └── feynman_lost_lecture.ipynb
├── scripts/
│   ├── build_analysis_notebooks.py
│   └── build_feynman_notebook.py
├── src/
│   ├── dataset_catalog.py    # Dataset registry (factors, checks, copy)
│   ├── extract_features.py   # HF stream / synthetic feature builder
│   ├── orbit_feynman.py      # Kepler / hodograph geometry
│   └── stats.py              # Shared ANOVA / t-test / anomaly helpers
├── requirements.txt          # Full local stack (extraction + app)
├── requirements-app.txt      # Slim runtime deps (Render / demo)
├── render.yaml               # Render Blueprint
└── README.md
```
## Environment (important)

Default system Python may be **3.14**. Extraction needs **3.11 or 3.12**. A `.venv` on Python 3.12 is expected:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

For **running only the dashboard** (no extraction):

```powershell
pip install -r requirements-app.txt
streamlit run app/streamlit_app.py
```

## Data

### Preferred: real features from Hugging Face

`data/features_active_matter.parquet` (and legacy `data/features.parquet`) should contain **real** The Well `active_matter` trajectories whenever possible. The train split is a complete **5 α × 9 ζ** factorial (**45** HDF5 files). Most cells have **3** trajectories; a few files in the HF release contain 1–2 or 4 trajectories, so a full extract is typically ~130–140 rows rather than a rigid 135.

Each remote file is large. Streaming with space/time strides is the efficient path (no full local mirror required):

```powershell
# active_matter — full factorial (recommended)
python -u -m src.extract_features --dataset active_matter --splits train --time-stride 8 --space-stride 16 --max-traj-per-file 3 --workers 4

# gray_scott — six (f, k) regimes (sparse design; not a full f×k grid)
python -u -m src.extract_features --dataset gray_scott --splits train --time-stride 16 --space-stride 4 --max-traj-per-file 8 --workers 4

# acoustic_scattering_maze — width × n_sources (factors derived from fields/attrs)
python -u -m src.extract_features --dataset acoustic_scattering --splits train --time-stride 4 --space-stride 8 --max-traj-per-file 3 --workers 4

# Resume after a network interruption
python -u -m src.extract_features --dataset active_matter --splits train --time-stride 8 --space-stride 16 --max-traj-per-file 3 --workers 4 --resume
```

Default output paths come from the catalog (`data/features_<id>.parquet`).

**Efficiency plan used by this project (active_matter)**

1. Stream all **45** train cells via `fsspec` + `h5py` (avoids the Windows `WellDataset` path bug).
2. Aggressive but documented strides: `--time-stride 8 --space-stride 16`, capped at 3 traj/file.
3. Concurrent streams with `--workers 4`.
4. Checkpoint every file into the output parquet.

**Design notes for the other ensembles**

- **gray_scott:** only six discrete `(f, k)` pairs (Gliders, Bubbles, Maze, Worms, Spirals, Spots). Two-way ANOVA still runs on observed cells; interpret interaction carefully.
- **acoustic_scattering:** uses `acoustic_scattering_maze`. `maze_width` and `n_sources` are read from scalars when present, otherwise estimated from density / initial pressure.

### Fallback: synthetic demo tables

```powershell
python -m src.extract_features --dataset active_matter --synthetic
python -m src.extract_features --dataset gray_scott --synthetic
python -m src.extract_features --dataset acoustic_scattering --synthetic
python -m src.extract_features --dataset planetary_motion --synthetic
```

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

Panels depend on the selected dataset:

| Dataset | Panel 1 | Panel 2 | Panel 3 |
|---------|---------|---------|---------|
| **active_matter** | ANOVA sandbox | Real-data ANOVA | Physics & anomalies |
| **gray_scott** | F×k phase diagram | Pattern metrics vs params | Physics & anomalies |
| **acoustic_scattering** | Geometry × sources | Response & interactions | Physics & anomalies |

Open the **sidebar** (hamburger, top-left) to switch ensembles.

Notebooks: [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb) (active_matter ANOVA), [`notebooks/gray_scott_analysis.ipynb`](notebooks/gray_scott_analysis.ipynb), [`notebooks/acoustic_scattering_analysis.ipynb`](notebooks/acoustic_scattering_analysis.ipynb).
## Deploy on Render

The public demo is a **Render** Web Service (Streamlit needs a long-running process; Vercel/Supabase alone are not suitable without a rewrite).

### One-click from Blueprint

1. Push this repo to GitHub (already: `Emilio-Gordillo-Esparragoza/LectureLab`).
2. Open [Render Dashboard](https://dashboard.render.com) → **New** → **Blueprint**.
3. Connect the `LectureLab` repository.
4. Render reads [`render.yaml`](render.yaml): builds with `requirements-app.txt`, starts Streamlit on `$PORT`.
5. After the first deploy, the service URL is typically `https://lecturelab.onrender.com` (confirm in the Render UI).

To replace an older `physcientific` service: delete it in the Render dashboard (or CLI), then create a new Web Service / Blueprint named `lecturelab` from this repo so the public URL becomes `https://lecturelab.onrender.com`.

### Manual Web Service

- **Runtime:** Python 3.12  
- **Build command:** `pip install -r requirements-app.txt`  
- **Start command:** `streamlit run app/streamlit_app.py --server.port $PORT --server.address 0.0.0.0 --browser.gatherUsageStats false`  
- **Health check path:** `/`

`data/features_*.parquet` files are committed so the live app does **not** download The Well at runtime. Free tier instances sleep when idle; the first request after sleep can take about a minute.

## Feature dictionary

### active_matter

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

### gray_scott

| Column | Meaning |
|--------|---------|
| `f`, `k`, `pattern` | Feed / kill rates and named regime label |
| `mean_A`, `mean_B`, `std_A`, `std_B` | Species concentration stats |
| `pattern_contrast` | `std_A + std_B` (pattern strength proxy) |
| `spectral_slope`, `time_to_steady` | Structure / equilibration |

### acoustic_scattering

| Column | Meaning |
|--------|---------|
| `maze_width`, `n_sources` | Path width (px) and initial source count |
| `mean_abs_pressure`, `pressure_energy` | Pressure field intensity |
| `kinetic_energy`, `wall_fraction` | Flow energy / dense-wall occupancy |
| `spectral_slope`, `time_to_steady` | Structure / equilibration |

## Citation

If you use The Well data, please cite the Well paper and the relevant dataset cards:

- [active_matter](https://huggingface.co/datasets/polymathic-ai/active_matter)
- [gray_scott_reaction_diffusion](https://huggingface.co/datasets/polymathic-ai/gray_scott_reaction_diffusion)
- [acoustic_scattering_maze](https://huggingface.co/datasets/polymathic-ai/acoustic_scattering_maze)
