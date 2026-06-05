# Darija ASR Workbench

Darija ASR Workbench is a local research and labeling application for Moroccan Darija speech recognition. It provides a Flask backend for model inference, dataset sampling, transcription, evaluation, labeling, accepted spelling variants, and batch analysis, plus a React/Vite frontend for working with audio samples.

## Features

- Run multiple Darija ASR models from the backend registry.
- Browse configured Hugging Face datasets and play dataset audio.
- Record or upload audio and transcribe it.
- Compare model output against references with WER/CER and word-level diffs.
- Mark spelling variants as accepted when the model output is equivalent but written differently.
- Label uploaded audio or relabel dataset samples.
- Run batch analysis over samples, including optional noise injection.
- Download model and dataset assets with a one-time setup script.

## Project Layout

```text
backend/                 Flask API and ASR runtime
frontend/                React/Vite workbench UI
scripts/                 Runtime setup and Hugging Face download tools
models/                  Local model/cache directory, ignored by Git
datasets/                Local dataset/cache directory, ignored by Git
data/                    Recorded labels, manifests, accepted variants, ignored by Git
uploads/                 Uploaded audio, ignored by Git
logs/                    Runtime logs, ignored by Git
```

## Setup

Use Python 3.10 or 3.11. On Windows, create and activate a virtual environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install backend and frontend dependencies, create runtime folders, and build the frontend:

```powershell
python scripts/setup_runtime.py --full
```

If you prefer manual commands:

```powershell
pip install -r requirements.txt
cd frontend
npm install
npm run build
cd ..
```

## Download Models And Datasets

List available downloadable models:

```powershell
python scripts/download_hf_assets.py --list-model-catalog
```

Download the models and datasets used by the web app:

```powershell
python scripts/setup_runtime.py --download web
```

Download the broader research catalog:

```powershell
python scripts/setup_runtime.py --download all
```

For private or gated Hugging Face assets, copy `.env.example` to `.env` and set `HF_TOKEN`.

## Run

Build the frontend first, then start Flask:

```powershell
npm --prefix frontend run build
python backend/app.py
```

Open:

```text
http://127.0.0.1:5000
```

For frontend-only development:

```powershell
npm --prefix frontend run dev
```

The frontend expects the Flask backend API to be available on the same origin in production.

## GitHub Notes

The repository is configured to avoid committing:

- `.venv/`
- `node_modules/`
- `frontend/dist/`
- downloaded models and datasets
- uploaded/recorded audio
- logs and local caches
- large local files such as installers, archives, media, and documents

Before pushing, check what Git will include:

```powershell
git status --short
git add .gitignore README.md requirements.txt .env.example backend frontend scripts models/.gitkeep datasets/.gitkeep data/.gitkeep uploads/.gitkeep logs/.gitkeep
git status --short
```

## Production Notes

The built-in Flask server is for local development and research. For deployment, run the Flask app behind a production WSGI server and keep model caches, uploaded audio, and manifests on persistent storage outside the Git repository.
