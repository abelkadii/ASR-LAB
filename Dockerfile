# syntax=docker/dockerfile:1.7

FROM node:20-bookworm-slim AS frontend-builder

WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
# api.tsx imports ../../../backend/dataset_metadata.json at build time.
# Inside this stage, /frontend/../../../backend resolves to /backend.
COPY backend/dataset_metadata.json /backend/dataset_metadata.json
# ModelsWorkspace imports ../../../../../data/registry_benchmark_results.json.
# Inside this stage, /frontend/../../../../../data resolves to /data.
COPY data/registry_benchmark_results.json /data/registry_benchmark_results.json
RUN npm run build


FROM python:3.10-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FLASK_HOST=0.0.0.0 \
    FLASK_PORT=5000 \
    FLASK_DEBUG=0 \
    SPEECHRECON_CACHE_ROOT=/cache \
    SPEECHRECON_MODEL_DIR=/cache/models \
    SPEECHRECON_DATASET_DIR=/cache/datasets \
    HF_HOME=/cache/huggingface \
    HF_HUB_CACHE=/cache/huggingface/hub \
    TRANSFORMERS_CACHE=/cache/huggingface/hub \
    HF_DATASETS_CACHE=/cache/huggingface/datasets

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt docker-requirements.txt ./

# The local Windows setup uses the CUDA PyTorch index. The default container is
# CPU-first so it can run on any Docker Desktop installation without requiring
# NVIDIA container runtime support.
RUN pip install --index-url https://pypi.org/simple \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        "torch==2.3.1+cpu" \
        "torchaudio==2.3.1+cpu" \
    && grep -vE '^(--extra-index-url|torch|torchaudio)' requirements.txt > /tmp/requirements-app.txt \
    && pip install --constraint docker-requirements.txt -r /tmp/requirements-app.txt

COPY backend/ ./backend/
COPY scripts/ ./scripts/
COPY README.md ./
COPY --from=frontend-builder /frontend/dist ./frontend/dist

RUN mkdir -p \
      /app/data \
      /app/uploads \
      /app/logs \
      /cache/models \
      /cache/datasets \
      /cache/huggingface

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
  CMD curl -fsS http://127.0.0.1:5000/models >/dev/null || exit 1

CMD ["python", "-m", "backend.app"]
