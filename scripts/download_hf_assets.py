from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT_DIR / "logs"


@dataclass(frozen=True)
class ModelAsset:
    id: str
    hf_path: str
    source: str
    framework: str
    required_for: str
    size_gb: float | None = None
    cpu_fit: str = "unknown"
    notes: str = ""
    allow_patterns: tuple[str, ...] | None = None
    ignore_patterns: tuple[str, ...] | None = None


@dataclass(frozen=True)
class DatasetAsset:
    id: str
    hf_path: str
    split: str
    source: str
    required_for: str
    size_gb: float | None = None
    notes: str = ""


WEB_MODELS = [
    ModelAsset(
        id="boumehdi-wav2vec2-xlsr-darija",
        hf_path="boumehdi/wav2vec2-large-xlsr-moroccan-darija",
        source="backend/app.py AVAILABLE_MODELS",
        framework="ctc-transformers",
        required_for="web",
        size_gb=3.50,
        cpu_fit="good",
        notes="Core model is about 1.2 GB, full repo metadata/cache is about 3.5 GB.",
    ),
    ModelAsset(
        id="wav2vec2-dvoice-darija",
        hf_path="speechbrain/asr-wav2vec2-dvoice-darija",
        source="backend/app.py AVAILABLE_MODELS",
        framework="speechbrain",
        required_for="web,benchmark",
        size_gb=1.19,
        cpu_fit="good",
        notes="SpeechBrain DVoice Darija model; card reports about 18.28 test WER on DVoice.",
    ),
    ModelAsset(
        id="whisper-medium-darija",
        hf_path="ychafiqui/whisper-medium-darija",
        source="backend/app.py AVAILABLE_MODELS",
        framework="transformers",
        required_for="web,benchmark",
        size_gb=2.85,
        cpu_fit="ok-slow",
        notes="Whisper medium Darija model; feasible on 32 GB RAM but slower on CPU.",
    ),
    ModelAsset(
        id="whisper-turbo-darija-adapter",
        hf_path="anaszil/whisper-large-v3-turbo-darija",
        source="backend/app.py AVAILABLE_MODELS / files/registry/models.py",
        framework="peft-adapter",
        required_for="web-optional,benchmark",
        size_gb=0.05,
        cpu_fit="adapter-needs-base",
        notes="LoRA adapter; requires openai/whisper-large-v3-turbo base.",
    ),
    ModelAsset(
        id="whisper-large-v3-turbo-base",
        hf_path="openai/whisper-large-v3-turbo",
        source="base_model for anaszil/whisper-large-v3-turbo-darija",
        framework="transformers-base",
        required_for="web-optional,benchmark",
        size_gb=1.51,
        cpu_fit="ok-slow",
        notes="Base model required by the anaszil Whisper Turbo Darija LoRA adapter.",
    ),
]


EXPERIMENTAL_ASR_MODELS = [
    ModelAsset(
        id="ychafiqui-whisper-small-darija",
        hf_path="ychafiqui/whisper-small-darija",
        source="Hugging Face Moroccan Darija ASR search",
        framework="transformers-whisper",
        required_for="experimental-cpu",
        size_gb=0.90,
        cpu_fit="good",
        notes="Whisper small fine-tuned on Darija Speech to Text; reported WER 48.274 on its eval set.",
        allow_patterns=("*.json", "*.txt", "*.model", "*.safetensors", "preprocessor_config.json", "tokenizer*"),
    ),
    ModelAsset(
        id="ychafiqui-whisper-medium-darija",
        hf_path="ychafiqui/whisper-medium-darija",
        source="Hugging Face Moroccan Darija ASR search",
        framework="transformers-whisper",
        required_for="experimental-cpu",
        size_gb=2.85,
        cpu_fit="ok-slow",
        notes="Whisper medium fine-tuned for Darija; stronger than small but slower on CPU.",
        allow_patterns=("*.json", "*.txt", "*.model", "*.safetensors", "preprocessor_config.json", "tokenizer*"),
    ),
    ModelAsset(
        id="kandirresearch-whisper-small-darija",
        hf_path="KandirResearch/Whisper-Small-Darija",
        source="Hugging Face Moroccan Darija ASR search",
        framework="transformers-whisper",
        required_for="experimental-cpu",
        size_gb=0.90,
        cpu_fit="good",
        notes="Whisper small fine-tuned on Lyte/DarijaTTS-clean; research model, training was interrupted/resumed.",
        allow_patterns=("*.json", "*.txt", "*.model", "*.safetensors", "preprocessor_config.json", "tokenizer*"),
    ),
    ModelAsset(
        id="talocreations-whisper-darija-finetuned",
        hf_path="TaloCreations/whisper-darija-finetuned",
        source="Hugging Face Moroccan Darija ASR search",
        framework="transformers-whisper",
        required_for="experimental-cpu",
        size_gb=0.90,
        cpu_fit="good",
        notes="Whisper small-sized Darija model fine-tuned from giannitto/whisper-morocco-model.",
        allow_patterns=("*.json", "*.txt", "*.model", "*.safetensors", "preprocessor_config.json", "tokenizer*"),
    ),
    ModelAsset(
        id="marialab-whisper-small-dr-ar-mish",
        hf_path="Marialab/whisper-small-dr-ar-mish",
        source="Hugging Face Darija ASR/search",
        framework="transformers-whisper",
        required_for="experimental-cpu",
        size_gb=0.90,
        cpu_fit="good",
        notes="Whisper small model trained on Darija-C; card mentions translation/BLEU, so validate ASR behavior.",
        allow_patterns=("*.json", "*.txt", "*.model", "*.safetensors", "preprocessor_config.json", "tokenizer*"),
    ),
    ModelAsset(
        id="hafsati-whisper-base-darija-merged-diarized",
        hf_path="EtMmohammedHafsati/whisper-base-darija-merged-diarized",
        source="Hugging Face moroccan-arabic ASR listing",
        framework="transformers-whisper",
        required_for="experimental-cpu",
        size_gb=0.28,
        cpu_fit="excellent",
        notes="Very small Whisper base Darija variant. Root model only; checkpoints ignored.",
        allow_patterns=("*.json", "*.txt", "*.model", "model.safetensors", "preprocessor_config.json", "tokenizer*"),
        ignore_patterns=("checkpoint-*/*",),
    ),
    ModelAsset(
        id="hafsati-whisper-small-darija-merged-diarized",
        hf_path="EtMmohammedHafsati/whisper-small-darija-merged-diarized",
        source="Hugging Face moroccan-arabic ASR listing",
        framework="transformers-whisper",
        required_for="experimental-cpu",
        size_gb=0.90,
        cpu_fit="good",
        notes="Whisper small Darija variant. Root model only; checkpoints ignored.",
        allow_patterns=("*.json", "*.txt", "*.model", "model.safetensors", "preprocessor_config.json", "tokenizer*"),
        ignore_patterns=("checkpoint-*/*",),
    ),
    ModelAsset(
        id="atlasia-moulsot-v03",
        hf_path="atlasia/moulsot.v0.3",
        source="AtlasIA MoulSot/Qwen3-ASR model card",
        framework="qwen3-asr-external-runtime",
        required_for="experimental-runtime",
        size_gb=3.81,
        cpu_fit="possible-slow",
        notes="Qwen3-ASR 1.7B fine-tuned on MoulSot; needs qwen-asr/Transformers runtime isolated from the main app.",
        allow_patterns=("*.json", "*.txt", "*.safetensors", "preprocessor_config.json", "tokenizer*", "vocab.json", "merges.txt"),
    ),
    ModelAsset(
        id="qwen3-asr-06b-base",
        hf_path="Qwen/Qwen3-ASR-0.6B",
        source="Base Qwen3-ASR model for external runtime experiments",
        framework="qwen3-asr-external-runtime",
        required_for="experimental-runtime",
        size_gb=1.75,
        cpu_fit="possible-slow",
        notes="Base multilingual Qwen3-ASR 0.6B. Not Darija-tuned but useful baseline if isolated runtime is added.",
        allow_patterns=("*.json", "*.txt", "*.safetensors", "preprocessor_config.json", "tokenizer*", "vocab.json", "merges.txt"),
    ),
    ModelAsset(
        id="anaszil-whisper-large-v3-turbo-darija-lora",
        hf_path="anaszil/whisper-large-v3-turbo-darija",
        source="Hugging Face Moroccan Darija LoRA ASR",
        framework="peft-whisper-adapter",
        required_for="experimental-adapter",
        size_gb=0.05,
        cpu_fit="adapter-needs-base",
        notes="Small LoRA adapter, but requires openai/whisper-large-v3-turbo base (~1.51 GB). Already supported experimentally.",
    ),
    ModelAsset(
        id="hafsa0-whisper-large-v3-darija-lora",
        hf_path="Hafsa0/whisper-large-v3-darija-lora",
        source="Hugging Face moroccan-arabic ASR listing",
        framework="peft-whisper-adapter",
        required_for="experimental-adapter",
        size_gb=0.02,
        cpu_fit="adapter-needs-base",
        notes="Tiny LoRA adapter for Whisper large-v3 family. Needs separate PEFT loader/base compatibility check.",
    ),
    ModelAsset(
        id="hammale-mms-darija-final",
        hf_path="HAMMALE/mms-darija-finetuned",
        source="Hugging Face moroccan-arabic model listing",
        framework="transformers-mms-ctc",
        required_for="experimental-large",
        size_gb=3.60,
        cpu_fit="ok-slow",
        notes="MMS/Wav2Vec2-style Darija model. Full repo is ~197 GB due to checkpoints; root model only by default.",
        allow_patterns=("*.json", "*.txt", "*.safetensors", "preprocessor_config.json", "tokenizer*", "vocab.json"),
        ignore_patterns=("checkpoint-*/*",),
    ),
]


BENCHMARK_MODELS = [
    asset
    for asset in WEB_MODELS
    if asset.hf_path
    in {
        "speechbrain/asr-wav2vec2-dvoice-darija",
        "ychafiqui/whisper-medium-darija",
        "anaszil/whisper-large-v3-turbo-darija",
        "openai/whisper-large-v3-turbo",
    }
]


WEB_DATASETS = [
    DatasetAsset(
        id="dvoice-v2-cleaned",
        hf_path="BounharAbdelaziz/Dvoice-v2-cleaned",
        split="train",
        source="backend/app.py AVAILABLE_DATASETS / files/registry/datasets.py",
        required_for="web,benchmark",
        size_gb=0.02,
    ),
    DatasetAsset(
        id="hafsata-darija-asr",
        hf_path="HAFSATA/darija-asr-monolingual",
        split="train",
        source="backend/app.py AVAILABLE_DATASETS / files/registry/datasets.py",
        required_for="web,benchmark",
        size_gb=0.98,
    ),
    DatasetAsset(
        id="mgb5-moroccan",
        hf_path="ArabicSpeech/MGB-5",
        split="train",
        source="backend/app.py AVAILABLE_DATASETS / files/registry/datasets.py",
        required_for="web,benchmark",
        size_gb=6.20,
    ),
]


BENCHMARK_DATASETS = WEB_DATASETS


DISCOVERED_DATASETS = [
    DatasetAsset(
        id="darija-wiki-audio",
        hf_path="atlasia/Moroccan-Darija-Wiki-Audio-Dataset",
        split="train",
        source="files/registry/datasets.py inactive definition / Hugging Face search",
        required_for="optional-discovered",
        size_gb=10.22,
    ),
    DatasetAsset(
        id="snousnou-darija-asr",
        hf_path="Snousnou/Moroccan-Darija-ASR",
        split="train",
        source="files/registry/datasets.py inactive definition / Hugging Face search",
        required_for="optional-discovered",
        size_gb=0.01,
    ),
    DatasetAsset(
        id="mohamedmou-dataset-darija",
        hf_path="mohamedmou/DATASET-darija",
        split="train",
        source="Hugging Face search: Darija ASR audio dataset",
        required_for="optional-discovered",
        size_gb=0.37,
    ),
    DatasetAsset(
        id="mohamedmou-moroccan-darija-asr",
        hf_path="mohamedmou/moroccan-darija-asr-dataset",
        split="train",
        source="Hugging Face search: Moroccan Darija ASR dataset",
        required_for="optional-discovered",
        size_gb=None,
        notes="Size not measured locally yet.",
    ),
    DatasetAsset(
        id="afyfbadreddine-darija-asr-clean",
        hf_path="afyfbadreddine77/darija-asr-clean",
        split="train",
        source="Hugging Face search: Darija ASR clean audio dataset",
        required_for="optional-discovered",
        size_gb=5.09,
    ),
    DatasetAsset(
        id="atlasia-moulsot-preview",
        hf_path="atlasia/moulsot-preview",
        split="train",
        source="Hugging Face AtlasIA dataset listing",
        required_for="optional-discovered",
        size_gb=0.79,
    ),
    DatasetAsset(
        id="atlasia-moulsot-full",
        hf_path="atlasia/MoulSot-Full",
        split="test",
        source="Hugging Face search: MoulSot Moroccan Darija audio",
        required_for="optional-discovered-large",
        size_gb=308.02,
        notes="Very large full dataset. Download intentionally requires --include-discovered-datasets or --discovered-datasets-only.",
    ),
]


def load_env_file() -> None:
    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def format_size(size_gb: float | None) -> str:
    return f"{size_gb:.2f} GB" if size_gb is not None else "unknown"


def format_duration(hours: float | None) -> str:
    if hours is None:
        return "unknown"
    minutes = hours * 60
    if minutes < 1:
        return "<1 min"
    if minutes < 90:
        return f"{minutes:.0f} min"
    return f"{hours:.1f} hr"


def estimate_download_hours(size_gb: float | None, mbps: float) -> float | None:
    if size_gb is None or mbps <= 0:
        return None
    gib_to_gigabits = 1024 * 8 / 1000
    return size_gb * gib_to_gigabits / mbps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-time Hugging Face cache warmer for speechRecon models and datasets."
    )
    parser.add_argument(
        "--scope",
        choices=["web", "benchmark", "all"],
        default="all",
        help="Config set to download. Default: all.",
    )
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="Download model repositories only.",
    )
    parser.add_argument(
        "--datasets-only",
        action="store_true",
        help="Download dataset splits only.",
    )
    parser.add_argument(
        "--include-discovered-datasets",
        action="store_true",
        help="Also download optional public Darija ASR/audio datasets found outside the active app registry.",
    )
    parser.add_argument(
        "--include-experimental-models",
        action="store_true",
        help="Also include the experimental Darija/Arabic ASR model catalog.",
    )
    parser.add_argument(
        "--experimental-models-only",
        action="store_true",
        help="Download only models from the experimental ASR catalog.",
    )
    parser.add_argument(
        "--model-id",
        action="append",
        default=[],
        help="Download only a specific model id. May be repeated. Use --list-model-catalog to see ids.",
    )
    parser.add_argument(
        "--list-model-catalog",
        action="store_true",
        help="Print known model ids, repo names, estimated sizes, and CPU notes, then exit.",
    )
    parser.add_argument(
        "--full-model-repo",
        action="store_true",
        help="Ignore recommended allow/ignore patterns and download full model repositories, including checkpoints.",
    )
    parser.add_argument(
        "--discovered-datasets-only",
        action="store_true",
        help="Download only the optional discovered datasets.",
    )
    parser.add_argument(
        "--discover-hf",
        action="store_true",
        help="Search Hugging Face for more candidate datasets and log them. Does not download search results.",
    )
    parser.add_argument(
        "--discover-limit",
        type=int,
        default=20,
        help="Maximum Hugging Face search results per query when using --discover-hf.",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.getenv("HF_HUB_CACHE") or (str(Path(os.getenv("HF_HOME")) / "hub") if os.getenv("HF_HOME") else None),
        help="Optional Hugging Face Hub model cache directory. Defaults to HF_HUB_CACHE or HF_HOME/hub.",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        default=os.getenv("HF_DATASETS_CACHE"),
        help="Optional Hugging Face datasets cache directory. Defaults to HF_DATASETS_CACHE.",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="Optional Hugging Face token. Defaults to HF_TOKEN.",
    )
    parser.add_argument(
        "--dataset-max-samples",
        type=int,
        default=None,
        help="Optional cap for dataset rows to materialize. Omit for full configured split.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be downloaded without downloading anything.",
    )
    parser.add_argument(
        "--download-mbps",
        type=float,
        default=100.0,
        help="Connection speed used for rough time estimates. Default: 100 Mbps.",
    )
    return parser.parse_args()


def setup_logging() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"download_hf_assets_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    return log_path


def select_assets(scope: str) -> tuple[list[ModelAsset], list[DatasetAsset]]:
    if scope == "web":
        return dedupe_models(WEB_MODELS), dedupe_datasets(WEB_DATASETS)
    if scope == "benchmark":
        return dedupe_models(BENCHMARK_MODELS), dedupe_datasets(BENCHMARK_DATASETS)
    return (
        dedupe_models([*WEB_MODELS, *BENCHMARK_MODELS]),
        dedupe_datasets([*WEB_DATASETS, *BENCHMARK_DATASETS]),
    )


def all_model_catalog() -> list[ModelAsset]:
    return dedupe_models([*WEB_MODELS, *BENCHMARK_MODELS, *EXPERIMENTAL_ASR_MODELS])


def print_model_catalog() -> None:
    assets = all_model_catalog()
    print("Known ASR model catalog")
    print("-" * 120)
    print(f"{'id':42} {'size':>8} {'cpu':18} {'framework':28} repo")
    print("-" * 120)
    for asset in assets:
        size = format_size(asset.size_gb)
        print(f"{asset.id:42} {size:>8} {asset.cpu_fit:18} {asset.framework:28} {asset.hf_path}")
        if asset.notes:
            print(f"  notes: {asset.notes}")
    print("-" * 120)
    print("Tip: use --model-id <id> --models-only to download one model.")


def log_download_plan(models: list[ModelAsset], datasets: list[DatasetAsset], download_mbps: float) -> None:
    known_model_gb = sum(asset.size_gb or 0 for asset in models)
    known_dataset_gb = sum(asset.size_gb or 0 for asset in datasets)
    unknown_models = sum(1 for asset in models if asset.size_gb is None)
    unknown_datasets = sum(1 for asset in datasets if asset.size_gb is None)
    known_total_gb = known_model_gb + known_dataset_gb
    logging.info(
        "estimate known_total=%s models=%s datasets=%s unknown_models=%s unknown_datasets=%s at=%s Mbps rough_time=%s",
        format_size(known_total_gb),
        format_size(known_model_gb),
        format_size(known_dataset_gb),
        unknown_models,
        unknown_datasets,
        download_mbps,
        format_duration(estimate_download_hours(known_total_gb, download_mbps)),
    )

    for asset in models:
        logging.info(
            "PLAN model id=%s repo=%s size=%s cpu_fit=%s time_at_%sMbps=%s required_for=%s",
            asset.id,
            asset.hf_path,
            format_size(asset.size_gb),
            asset.cpu_fit,
            download_mbps,
            format_duration(estimate_download_hours(asset.size_gb, download_mbps)),
            asset.required_for,
        )
        if asset.notes:
            logging.info("PLAN model notes id=%s %s", asset.id, asset.notes)
    for asset in datasets:
        logging.info(
            "PLAN dataset id=%s repo=%s split=%s size=%s time_at_%sMbps=%s required_for=%s",
            asset.id,
            asset.hf_path,
            asset.split,
            format_size(asset.size_gb),
            download_mbps,
            format_duration(estimate_download_hours(asset.size_gb, download_mbps)),
            asset.required_for,
        )
        if asset.notes:
            logging.info("PLAN dataset notes id=%s %s", asset.id, asset.notes)


def dedupe_models(assets: Iterable[ModelAsset]) -> list[ModelAsset]:
    seen: set[str] = set()
    result: list[ModelAsset] = []
    for asset in assets:
        if asset.hf_path not in seen:
            seen.add(asset.hf_path)
            result.append(asset)
    return result


def dedupe_datasets(assets: Iterable[DatasetAsset]) -> list[DatasetAsset]:
    seen: set[tuple[str, str]] = set()
    result: list[DatasetAsset] = []
    for asset in assets:
        key = (asset.hf_path, asset.split)
        if key not in seen:
            seen.add(key)
            result.append(asset)
    return result


def download_model(asset: ModelAsset, cache_dir: str | None, token: str | None, full_repo: bool = False) -> None:
    from huggingface_hub import snapshot_download

    logging.info(
        "MODEL start id=%s repo=%s framework=%s size_gb=%s cpu_fit=%s source=%s",
        asset.id,
        asset.hf_path,
        asset.framework,
        asset.size_gb,
        asset.cpu_fit,
        asset.source,
    )
    allow_patterns = None if full_repo else asset.allow_patterns
    ignore_patterns = None if full_repo else asset.ignore_patterns
    if allow_patterns or ignore_patterns:
        logging.info(
            "MODEL filtered id=%s allow=%s ignore=%s",
            asset.id,
            allow_patterns or "<default>",
            ignore_patterns or "<default>",
        )
    path = snapshot_download(
        repo_id=asset.hf_path,
        repo_type="model",
        cache_dir=cache_dir,
        token=token,
        resume_download=True,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )
    logging.info("MODEL done  id=%s cache_path=%s", asset.id, path)


def download_dataset(
    asset: DatasetAsset,
    cache_dir: str | None,
    token: str | None,
    max_samples: int | None,
) -> None:
    from datasets import load_dataset

    logging.info(
        "DATASET start id=%s repo=%s split=%s source=%s",
        asset.id,
        asset.hf_path,
        asset.split,
        asset.source,
    )
    split = asset.split
    if max_samples is not None:
        split = f"{asset.split}[:{max_samples}]"
        logging.info("DATASET capped id=%s max_samples=%s split_expr=%s", asset.id, max_samples, split)

    try:
        ds = load_dataset(
            asset.hf_path,
            split=split,
            cache_dir=cache_dir,
            token=token,
        )
    except ValueError as exc:
        if "Unknown split" not in str(exc):
            raise
        builder = load_dataset(
            asset.hf_path,
            cache_dir=cache_dir,
            token=token,
        )
        available_splits = list(builder.keys())
        if not available_splits:
            raise
        fallback_split = available_splits[0]
        if max_samples is not None:
            fallback_split = f"{fallback_split}[:{max_samples}]"
        logging.warning(
            "DATASET split fallback id=%s preferred=%s available=%s using=%s",
            asset.id,
            asset.split,
            available_splits,
            fallback_split,
        )
        ds = load_dataset(
            asset.hf_path,
            split=fallback_split,
            cache_dir=cache_dir,
            token=token,
        )
    logging.info("DATASET done  id=%s rows=%s columns=%s", asset.id, len(ds), list(ds.column_names))


def main() -> int:
    load_env_file()
    args = parse_args()
    log_path = setup_logging()

    if args.list_model_catalog:
        print_model_catalog()
        return 0

    if args.models_only and args.datasets_only:
        logging.error("Choose only one of --models-only or --datasets-only.")
        return 2
    if args.discovered_datasets_only and args.models_only:
        logging.error("Choose only one of --discovered-datasets-only or --models-only.")
        return 2
    if args.experimental_models_only and args.datasets_only:
        logging.error("Choose only one of --experimental-models-only or --datasets-only.")
        return 2

    models, datasets = select_assets(args.scope)
    if args.include_experimental_models:
        models = dedupe_models([*models, *EXPERIMENTAL_ASR_MODELS])
    if args.experimental_models_only:
        models = dedupe_models(EXPERIMENTAL_ASR_MODELS)
        datasets = []
    if args.include_discovered_datasets:
        datasets = dedupe_datasets([*datasets, *DISCOVERED_DATASETS])
    if args.discovered_datasets_only:
        models = []
        datasets = dedupe_datasets(DISCOVERED_DATASETS)
    if args.datasets_only:
        models = []
    if args.models_only:
        datasets = []
    if args.model_id:
        wanted = set(args.model_id)
        catalog = {asset.id: asset for asset in all_model_catalog()}
        missing = sorted(wanted - set(catalog.keys()))
        if missing:
            logging.error("Unknown --model-id values: %s", ", ".join(missing))
            logging.error("Run with --list-model-catalog to see valid ids.")
            return 2
        models = [catalog[model_id] for model_id in args.model_id]
        datasets = []

    logging.info("speechRecon Hugging Face asset download")
    logging.info(
        "scope=%s model_cache_dir=%s dataset_cache_dir=%s log=%s",
        args.scope,
        args.cache_dir or "<default>",
        args.dataset_cache_dir or "<default>",
        log_path,
    )
    logging.info("models=%s datasets=%s", len(models), len(datasets))

    log_download_plan(models, datasets, args.download_mbps)

    if args.dry_run:
        if args.discover_hf:
            discover_hf_datasets(args.discover_limit)
        logging.info("Dry run complete. Nothing downloaded.")
        return 0

    if args.discover_hf:
        discover_hf_datasets(args.discover_limit)

    failures = 0
    for asset in models:
        try:
            download_model(asset, args.cache_dir, args.token, full_repo=args.full_model_repo)
        except Exception:
            failures += 1
            logging.exception("MODEL failed id=%s repo=%s", asset.id, asset.hf_path)

    for asset in datasets:
        try:
            download_dataset(asset, args.dataset_cache_dir, args.token, args.dataset_max_samples)
        except Exception:
            failures += 1
            logging.exception("DATASET failed id=%s repo=%s", asset.id, asset.hf_path)

    if failures:
        logging.error("Finished with %s failure(s). See log: %s", failures, log_path)
        return 1

    logging.info("All downloads completed successfully. See log: %s", log_path)
    return 0


def discover_hf_datasets(limit: int) -> None:
    from huggingface_hub import HfApi

    queries = [
        "darija asr",
        "moroccan darija audio",
        "moroccan arabic asr",
        "darija speech",
    ]
    api = HfApi()
    seen: set[str] = set()

    logging.info("HF discovery start. These are candidate search results only.")
    for query in queries:
        logging.info("HF discovery query=%r limit=%s", query, limit)
        try:
            results = api.list_datasets(search=query, limit=limit)
            for info in results:
                dataset_id = getattr(info, "id", None)
                if not dataset_id or dataset_id in seen:
                    continue
                seen.add(dataset_id)
                tags = getattr(info, "tags", None) or []
                downloads = getattr(info, "downloads", None)
                logging.info(
                    "HF candidate dataset=%s downloads=%s tags=%s",
                    dataset_id,
                    downloads,
                    tags[:8],
                )
        except Exception:
            logging.exception("HF discovery failed for query=%r", query)
    logging.info("HF discovery done. Review candidates before adding them to the download list.")


if __name__ == "__main__":
    raise SystemExit(main())
