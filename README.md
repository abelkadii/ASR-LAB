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

By default this project keeps Hugging Face assets on `E:\speechRecon-cache` so large model and dataset downloads do not fill the system drive. To move an existing Windows Hugging Face cache from `C:\Users\<you>\.cache\huggingface` to E and update `.env`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/move_hf_cache_to_e.ps1
```

If an earlier download placed `models--...` folders directly under `E:\speechRecon-cache\huggingface`, normalize the cache layout:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/normalize_hf_cache_layout.ps1
```

List available downloadable models:

```powershell
python scripts/download_hf_assets.py --list-model-catalog
```

Plan the full model download with estimated size and time:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_models_to_e.ps1 -DownloadMbps 100
```

Download all registered and experimental models to E:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_models_to_e.ps1 -DownloadMbps 100 -Run
```

Download and then warm-load supported models one at a time:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_models_to_e.ps1 -DownloadMbps 100 -Run -WarmLoad
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

## Run With Docker

The Docker setup builds the React frontend inside the image and runs the Flask
app on port `5000`. Runtime state stays outside the image through mounted
volumes, so labels, uploads, downloaded models, datasets, and Hugging Face cache
survive rebuilds.

From a fresh clone with Docker Desktop running, start the workbench with:

```powershell
docker compose up --build
```

Open `http://127.0.0.1:5000`. The API starts without downloading a model during
container startup. The first transcription loads the selected model into the
persistent Hugging Face cache; the first dataset sample request downloads that
dataset into its persistent cache. These first requests can take several
minutes depending on model size and connection speed.

To download the web catalog before opening the workbench, run this once:

```powershell
docker compose run --build --rm darija-asr python scripts/setup_runtime.py --download web --dataset-max-samples 50
docker compose up -d
```

This downloads the web models and the first 50 rows of each configured web
dataset into the same volumes used by the app. Omit `--dataset-max-samples 50`
to fetch the full configured dataset splits. The full web model catalog can
take several gigabytes. For a gated Hugging Face repository, set
`SPEECHRECON_HF_TOKEN` in the environment or in `.env.docker` before running
the download command.

Prerequisite on Windows: Docker Desktop must be running with a Linux container
backend. The normal setup is Docker Desktop + WSL2. If Docker says the Desktop
engine cannot start, install WSL first:

```powershell
wsl --install
```

Restart Windows if the installer asks for it, then open Docker Desktop again.

Build and start with repo-local volumes:

```powershell
docker compose up --build
```

Open:

```text
http://127.0.0.1:5000
```

To keep Docker runtime data on `E:` instead of inside the repo, copy the Docker
environment template and edit the paths if needed:

```powershell
Copy-Item .env.docker.example .env.docker
docker compose --env-file .env.docker up --build
```

If you need gated Hugging Face assets in Docker, set
`SPEECHRECON_HF_TOKEN` in `.env.docker`. The Docker variable is intentionally
not named `HF_TOKEN` so Compose does not accidentally import the token from the
normal local `.env` file.

Stop the container:

```powershell
docker compose down
```

Notes:

- The default image installs CPU PyTorch so it works without NVIDIA Docker
  setup. It can still use 32 GB system RAM, but GPU acceleration would need a
  separate CUDA image/runtime setup.
- The frontend is served by Flask from `frontend/dist` inside the container.
- Do not bake large models or datasets into the image; keep them in the mounted
  cache volumes.
- Model loading is lazy by default so the API becomes available before large
  model downloads finish. Set `SPEECHRECON_PRELOAD_DEFAULT_MODEL=1` if you want
  the default model loaded before the API starts serving requests.

For frontend-only development:

```powershell
npm --prefix frontend run dev
```

The frontend expects the Flask backend API to be available on the same origin in production.

## Run The Full Registry Benchmark

Start the Flask backend first, then use the registry benchmark runner from a second terminal.
The script runs each backend model against each backend dataset with `min(available samples, 100)`
samples, uses the backend analysis cache, logs progress and ETA, and keeps going if a model or
dataset pair fails.

Preview the planned work without running inference:

```powershell
python scripts/run_registry_benchmark.py --dry-run
```

Run the fast UI-stat benchmark. This is the recommended mode for populating model and dataset
cards/charts without waiting for a full research benchmark:

```powershell
python scripts/run_registry_benchmark.py
```

It uses the `ui` profile by default: 10 then 25 samples per model/dataset pair, with early stops
for very inaccurate or slow pairs. The backend cache means any samples already computed by a
previous run are reused.

Resume after an interrupted run:

```powershell
python scripts/run_registry_benchmark.py --resume
```

Run a larger but still bounded profile:

```powershell
python scripts/run_registry_benchmark.py --profile balanced --resume
```

Only run the full 100-sample matrix when you explicitly want a long benchmark:

```powershell
python scripts/run_registry_benchmark.py --profile full --resume
```

Outputs:

- `data/registry_benchmark_results.json`
- `logs/registry_benchmark.log`

You can narrow the run while testing:

```powershell
python scripts/run_registry_benchmark.py --only-models whisper-medium-darija --only-datasets hafsata-darija-asr
```

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
