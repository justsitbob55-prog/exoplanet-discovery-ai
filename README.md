# Exoplanet Discovery AI

A premium, deep-space Gradio research dashboard for exploring NASA light curves, preprocessing photometric signals, extracting scientific features, and screening transit-like candidates. The project emphasizes transparent, evidence-based reporting: **AI predictions are not confirmed exoplanets**, and no model metrics are fabricated.

## Highlights

- **Premium Gradio UI:** dark navy/midnight theme with cosmic gradients, glassmorphism cards, cyan/blue/violet accents, responsive layouts, and scientific visualizations.
- **NASA data integration:** search and download TESS/Kepler mission light curves via Lightkurve (network-dependent), with graceful error handling.
- **Light curve analysis:** CSV/FITS/local ingestion, preprocessing and normalization, and extraction of statistical, variability, periodogram, autocorrelation, and frequency-domain features.
- **Transit detection:** Box Least Squares (BLS) screening with AI-predicted candidate outputs and confidence reporting (unverified).
- **Model transparency:** AI Model Insights shows honest Random Forest status and only real, artifact-backed evaluation metrics when available.
- **Local persistence:** analysis history and candidate records stored in SQLite (`data/exoplanet_discovery.db`) and exportable to CSV.

## Application modules

1. **Home Dashboard** — mission overview, real dataset/detection/model statistics, recent analysis activity, and workflow guidance.
2. **NASA Data Explorer** — target search, archive product listing, light-curve download/preview, local upload, and synthetic demo mode.
3. **Light Curve Analysis** — preprocessing, feature extraction, and visualizations (raw, processed, Lomb–Scargle periodogram).
4. **Exoplanet Detection** — BLS transit screening and AI-predicted candidate results.
5. **AI Model Insights** — Random Forest status and actual held-out evaluation metrics (when genuine trained artifacts exist).
6. **Candidate Explorer** — filter and export stored candidate records.
7. **About & Methodology** — scientific workflow and limitations.

## Installation

### Requirements
- Python 3.12+
- A virtual environment is recommended
- Internet access for NASA archive search/download (TESS/Kepler). Offline/demo analysis remains available.

### Setup (Windows PowerShell)

```powershell
cd "C:\Users\Sitij\Downloads\Exoplanet - Discovery"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -e .
```

`requirements.txt` contains `-e .` for package installation (useful for deployment environments such as Hugging Face Spaces). Development and test tools are listed in `requirements-dev.txt`.

## Running the application

### Gradio research dashboard (primary)

```powershell
python -m app.gradio_app
```

Open: `http://localhost:7860`

Hugging Face Spaces entry point: `app.py` (exposes `app` from `app.gradio_app`).

#### Render archive requests

When `RENDER=true` is present, MAST network requests use a 10-second connect timeout and a 90-second read/request timeout. This bounds stalled NASA requests without changing the existing local timeout behavior. The archive cache uses the system temporary directory by default on Render; set `EXOPLANET_ARCHIVE_CACHE_DIR` to use another writable directory.

To adjust the Render timeout, set `EXOPLANET_MAST_TIMEOUT_SECONDS` to an integer from 1 to 600. If not set, Render uses 90 seconds and local runs retain the Astroquery default. Redeploy the service after applying the code changes; the start command remains `python -m app.gradio_app`.

### Optional FastAPI + React interface (existing, separate)

The repository also includes an existing FastAPI backend and React frontend. They are separate from the Gradio dashboard and are not required to use the primary interface.

```powershell
# Terminal 1
python -m uvicorn app.api.main:app --reload --port 8000

# Terminal 2
cd app/frontend
npm install
npm run dev
```

## Architecture

```text
app/
  gradio_app.py          # Primary Gradio dashboard (UI, styling, workflow wiring)
  app.py (root)          # Spaces/entry export for Gradio app
  api/                   # Existing FastAPI endpoints (upload/preprocess/retrieval)
  frontend/              # Existing React frontend (separate from Gradio)

src/exoplanet_discovery/
  ingestion/             # CSV/FITS/lightkurve loading, archive search/download, demo data
  preprocessing/         # Cleaning, normalization, detrending/smoothing, sampling
  features/              # Scientific feature extraction (25 stable light-curve features)
  detection/             # BLS transit screening and candidate construction
  training/              # Baseline training utilities (RF/SVM/GB) and model persistence
  evaluation/            # Binary classification metrics helpers
  common/                # Models, persistence, logging, config/constants

configs/                 # Configuration (YAML/defaults)
data/                    # Offline demo, exports, and local SQLite results
models/checkpoints/      # Model artifacts (created when genuine training is performed)
tests/                   # Automated test suite
```

### Core pipeline

1. **Ingest**: local CSV/FITS, NASA TESS/Kepler archive download, or synthetic demo light curve.
2. **Preprocess**: validate/clean, robust normalization and detrending, bounded sampling for large cadences.
3. **Extract features**: per-curve statistical/variability/periodicity features (`feature_names()`).
4. **Detect**: BLS transit screening; blend with trained Random Forest probability only when a genuine trained checkpoint is available.
5. **Report**: store analysis/candidate history and present transparent, non-confirmatory results in the dashboard.

## Configuration

- Default config: `configs/default_config.yaml`
- Typed config models: `src/config.py` and `src/exoplanet_discovery/config.py`

Key areas include data paths/validation, preprocessing, feature extraction, detection thresholds, and training settings (split/seed/checkpoint directory).

## Genuine Kepler KOI training workflow (reproducible, resumable)

This project includes a script to build a genuine labelled training dataset from official NASA Exoplanet Archive Kepler Objects of Interest (KOI) dispositions and actual Kepler light curves:

- **Script:** `scripts/train_kepler_koi_random_forest.py`
- **Label source/provenance:** NASA Exoplanet Archive KOI cumulative table (`koi_disposition`), queried via TAP. Positive = `CONFIRMED` (1), Negative = `FALSE POSITIVE` (0). `CANDIDATE` and `NOT DISPOSITIONED` records are excluded; kepids with conflicting dispositions are excluded and recorded.
- **Light curves:** actual Kepler mission light curves retrieved via the existing Lightkurve ingestion pipeline (`download_light_curve`, with cache/download directory support), then processed through the existing preprocessing and feature extraction modules (`preprocess_light_curve`, `extract_features`, 25-feature schema).
- **Leakage prevention:** one sample per consistent kepid (duplicate KOI rows for the same star/disposition collapsed), train/test split at **star/target group level** (`kepid`) so no group appears in both splits.
- **Reproducibility/resume:** KOI label snapshot, processing records, progress, and Lightkurve cache are stored under `data/external/koi/`, `data/processed/koi/`, and `data/cache/lightkurve/`; reruns resume from saved per-target records.
- **Artifacts:** trained model/metadata in `models/checkpoints/baseline_random_forest.json` and `baseline_random_forest.pkl`; evaluation report in `models/evaluation/kepler_koi_random_forest_evaluation.json`. AI Model Insights displays artifact-backed status, genuine held-out metrics, confusion matrix, sample/class counts, provenance, and methodology when present.

Example (balanced, resumable run):

```powershell
.\.venv\Scripts\python.exe scripts\train_kepler_koi_random_forest.py --max-per-class 50 --max-attempts 2 --seed 42 --test-size 0.2
```

Use `--train-only` to retrain/evaluate from existing processed records without additional downloads. Processing time and success depend on NASA/MAST archive availability and network conditions.

## AI Model Insights and limitations

- **Artifact-backed reporting:** when genuine trained checkpoints exist, AI Model Insights shows actual held-out Accuracy, Precision, Recall, F1-score, confusion matrix, and ROC-AUC (where appropriate), along with sample/class counts, split/provenance, and methodology. **No labels, samples, or metrics are fabricated.**
- **Current state in this workspace:** a genuine KOI-labelled Kepler training/evaluation run has been executed and saved real artifacts from the successfully processed, resumable subset available at run time (see `data/processed/koi/` and `models/evaluation/kepler_koi_random_forest_evaluation.json`). Because the processed sample size can be small depending on download/processing outcomes, reported metrics may be **insufficient for reliable evaluation**; expand the processed dataset (resume the script) before drawing performance conclusions.
- **Scientific limitations:**
  - AI-predicted candidates are **unverified** and require independent follow-up; they are never presented as confirmed exoplanets.
  - NASA archive search/download depends on external network/service availability and may fail for some targets.
  - Synthetic demo data is for demonstration only and is never used as genuine training data.
  - Large light curves are downsampled/bounded for plotting/screening performance, which may affect sensitivity.
  - KOI dispositions and cumulative catalog composition are heterogeneous; genuine evaluation should be interpreted in that context and improved with larger, carefully curated processed samples.

## Testing and quality checks

Run the full test suite (recommended in the project virtual environment):

```powershell
python -m pytest -q
```

Run a specific file (example):

```powershell
python -m pytest tests/test_new_pipeline.py -q
```

Optional checks:

```powershell
python -m ruff check src tests app
python -m mypy src
```

## Development notes

- Keep functionality evidence-based; do not generate synthetic labels or simulated performance metrics.
- Preserve working ingestion, preprocessing, feature extraction, detection, and navigation behavior when modifying the UI.
- The Gradio dashboard is the primary interface for this project demonstration.

## References

- [NASA TESS Mission](https://tess.mit.edu/)
- [NASA Kepler Mission](https://www.nasa.gov/mission_pages/kepler/overview/)
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [Lightkurve](https://lightkurve.readthedocs.io/)
- [Astropy](https://www.astropy.org/)
- [Gradio](https://www.gradio.app/)
