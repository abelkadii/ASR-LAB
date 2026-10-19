import torch
import torchaudio
from transformers import (
    AutoProcessor, AutoModelForCTC, pipeline, 
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

# Optional PEFT import - may fail due to version conflicts
try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError as e:
    HAS_PEFT = False
    print(f"Warning: PEFT not available ({e}). LoRA models will be skipped.")

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, make_response, g
import librosa
import numpy as np
import io
import os
import tempfile
import soundfile as sf
from scipy.io import wavfile
from werkzeug.utils import secure_filename
import psutil
import time
import traceback
import difflib
import re
import json
import uuid
import hashlib
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from jiwer import wer as compute_wer, cer as compute_cer
import sys
from unittest.mock import MagicMock

# Set CUDA memory optimization before importing anything else
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# Mock out optional SpeechBrain dependencies on Windows
for mod in ['k2', 'flair', 'speechbrain.integrations.k2_fsa', 'speechbrain.integrations.nlp']:
    sys.modules[mod] = MagicMock()

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file(PROJECT_ROOT / ".env")

app = Flask(__name__, template_folder=str(APP_DIR / 'templates'))
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = str(PROJECT_ROOT / 'uploads')
RECORDED_DATASET_DIR = PROJECT_ROOT / "data" / "recorded_dataset"
RECORDED_MANIFEST = RECORDED_DATASET_DIR / "manifest.jsonl"
LABEL_MANIFEST = RECORDED_DATASET_DIR / "labels.jsonl"
WORD_VARIANTS_MANIFEST = RECORDED_DATASET_DIR / "accepted_word_variants.jsonl"
CUSTOM_DATASETS_DIR = PROJECT_ROOT / "data" / "custom_datasets"
FRONTEND_DIST_DIR = PROJECT_ROOT / "frontend" / "dist"
CACHE_ROOT = Path(os.getenv("SPEECHRECON_CACHE_ROOT", PROJECT_ROOT)).expanduser()
MODEL_CACHE_DIR = Path(os.getenv("SPEECHRECON_MODEL_DIR", CACHE_ROOT / "models")).expanduser()
DATASET_CACHE_DIR = Path(os.getenv("SPEECHRECON_DATASET_DIR", CACHE_ROOT / "datasets")).expanduser()
DATASET_METADATA_PATH = APP_DIR / "dataset_metadata.json"
ANALYSIS_CACHE_PATH = PROJECT_ROOT / "data" / "analysis_cache.json"
ANALYSIS_RUNS_PATH = PROJECT_ROOT / "data" / "analysis_runs.jsonl"
REGISTRY_BENCHMARK_RESULTS_PATH = PROJECT_ROOT / "data" / "registry_benchmark_results.json"

os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR / "hf-cache"))
os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HUB_CACHE"])
os.environ.setdefault("HF_DATASETS_CACHE", str(DATASET_CACHE_DIR / "hf-datasets"))


def log_event(message: str, **fields) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    request_id = getattr(g, "request_id", "-") if request else "-"
    details = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
    suffix = f" {details}" if details else ""
    print(f"[{timestamp}] [{request_id}] {message}{suffix}", flush=True)


@app.before_request
def log_request_start():
    g.request_id = uuid.uuid4().hex[:8]
    g.request_start = time.perf_counter()
    if request.path == "/progress":
        return
    log_event(
        "REQUEST start",
        method=request.method,
        path=request.path,
        query=request.query_string.decode("utf-8")[:160] or None,
        content_length=request.content_length,
    )


@app.after_request
def log_request_end(response):
    if request.path != "/progress":
        elapsed_ms = int((time.perf_counter() - getattr(g, "request_start", time.perf_counter())) * 1000)
        log_event("REQUEST done", status=response.status_code, elapsed_ms=elapsed_ms)
    return response

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
RECORDED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
CUSTOM_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATASET_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Define available models
AVAILABLE_MODELS = {
    "boumehdi/wav2vec2-large-xlsr-moroccan-darija": {
        "name": "Wav2Vec2 XLSR Moroccan Darija (CTC)",
        "hf_path": "boumehdi/wav2vec2-large-xlsr-moroccan-darija",
        "framework": "ctc-transformers",
        "language_hint": None,
        "min_vram_gb": 2.0
    },
    "wav2vec2-dvoice-darija": {
        "name": "WAV2Vec2 DVoice Darija (CTC)",
        "hf_path": "speechbrain/asr-wav2vec2-dvoice-darija",
        "framework": "speechbrain",
        "language_hint": None,
        "min_vram_gb": 1.5
    },
    "whisper-medium-darija": {
        "name": "Whisper Medium Darija",
        "hf_path": "ychafiqui/whisper-medium-darija",
        "framework": "transformers",
        "language_hint": "ar",
        "min_vram_gb": 2.5
    },
    "ychafiqui-whisper-small-darija": {
        "name": "Whisper Small Darija (YChafiqui)",
        "hf_path": "ychafiqui/whisper-small-darija",
        "framework": "transformers",
        "language_hint": "ar",
        "min_vram_gb": 1.0,
        "force_cpu": True,
    },
    "kandirresearch-whisper-small-darija": {
        "name": "Whisper Small Darija (Kandir Research)",
        "hf_path": "KandirResearch/Whisper-Small-Darija",
        "framework": "transformers",
        "language_hint": "ar",
        "min_vram_gb": 1.0,
        "force_cpu": True,
    },
    "talocreations-whisper-darija-finetuned": {
        "name": "Whisper Darija Fine-Tuned (TaloCreations)",
        "hf_path": "TaloCreations/whisper-darija-finetuned",
        "framework": "transformers",
        "language_hint": "ar",
        "min_vram_gb": 1.0,
        "force_cpu": True,
    },
    "marialab-whisper-small-dr-ar-mish": {
        "name": "Whisper Small Darija/Arabic (Marialab)",
        "hf_path": "Marialab/whisper-small-dr-ar-mish",
        "framework": "transformers",
        "language_hint": "ar",
        "min_vram_gb": 1.0,
        "force_cpu": True,
    },
    "hafsati-whisper-base-darija-merged-diarized": {
        "name": "Whisper Base Darija Diarized (Hafsati)",
        "hf_path": "EtMmohammedHafsati/whisper-base-darija-merged-diarized",
        "framework": "transformers",
        "language_hint": "ar",
        "min_vram_gb": 0.5,
        "force_cpu": True,
    },
    "hafsati-whisper-small-darija-merged-diarized": {
        "name": "Whisper Small Darija Diarized (Hafsati)",
        "hf_path": "EtMmohammedHafsati/whisper-small-darija-merged-diarized",
        "framework": "transformers",
        "language_hint": "ar",
        "min_vram_gb": 1.0,
        "force_cpu": True,
    },
    "hammale-mms-darija-final": {
        "name": "MMS Darija Fine-Tuned (HAMMALE)",
        "hf_path": "HAMMALE/mms-darija-finetuned",
        "framework": "ctc-transformers",
        "language_hint": None,
        "min_vram_gb": 3.0,
        "force_cpu": True,
    },
}

# Only add PEFT model if PEFT is available
if HAS_PEFT:
    AVAILABLE_MODELS["whisper-turbo-darija"] = {
        "name": "Whisper Turbo Darija (LoRA)",
        "hf_path": "anaszil/whisper-large-v3-turbo-darija",
        "framework": "peft+transformers",
        "base_model": "openai/whisper-large-v3-turbo",
        "language_hint": "ar",
        "min_vram_gb": 3.0,
        "force_cpu": True
    }

# Define available datasets
AVAILABLE_DATASETS = {
    "dvoice-v2-cleaned": {
        "name": "DVoice v2 Cleaned",
        "hf_path": "BounharAbdelaziz/Dvoice-v2-cleaned",
        "split": "train",
        "audio_column": "audio",
        "transcript_column": "transcription_darija_ar",
        "language": "Moroccan Darija",
        "task": "ASR",
        "source_type": "huggingface",
        "license": "Unknown",
        "sample_count": None,
        "size_gb": None,
        "hours": None,
        "sampling_rate": None,
        "status": "configured",
        "notes": "Cleaned DVoice v2 Darija speech dataset. Exact size is reported as unknown until inspected/downloaded locally.",
    },
    "hafsata-darija-asr": {
        "name": "HAFSATA Darija ASR",
        "hf_path": "HAFSATA/darija-asr-monolingual",
        "split": "train",
        "audio_column": "audio",
        "transcript_column": "sentence",
        "language": "Moroccan Darija",
        "task": "ASR",
        "source_type": "huggingface",
        "license": "Unknown",
        "sample_count": None,
        "size_gb": None,
        "hours": None,
        "sampling_rate": None,
        "status": "configured",
        "notes": "Monolingual Darija ASR dataset with train/validation/test parquet splits.",
    },
    "mgb5-moroccan": {
        "name": "MGB-5 Moroccan Arabic",
        "hf_path": "ArabicSpeech/MGB-5",
        "split": "train",
        "audio_column": "audio",
        "transcript_column": "sentence",
        "language": "Moroccan Arabic",
        "task": "ASR",
        "source_type": "huggingface",
        "license": "Unknown",
        "sample_count": None,
        "size_gb": None,
        "hours": None,
        "sampling_rate": None,
        "status": "configured",
        "notes": "MGB-5 Moroccan Arabic broadcast speech dataset. Validate license before redistribution.",
    },
    "darija-wiki-audio": {
        "name": "AtlasIA Darija Wiki Audio",
        "hf_path": "atlasia/Moroccan-Darija-Wiki-Audio-Dataset",
        "split": "train",
        "audio_column": "audio",
        "transcript_column": "transcript",
        "language": "Moroccan Darija",
        "task": "ASR / speech corpus",
        "source_type": "huggingface",
        "license": "Unknown",
        "sample_count": None,
        "size_gb": None,
        "hours": None,
        "sampling_rate": None,
        "status": "optional",
        "notes": "AtlasIA Moroccan Darija wiki audio corpus.",
    },
    "snousnou-darija-asr": {
        "name": "Snousnou Moroccan Darija ASR",
        "hf_path": "Snousnou/Moroccan-Darija-ASR",
        "split": "train",
        "audio_column": "audio",
        "transcript_column": "transcription",
        "language": "Moroccan Darija",
        "task": "ASR",
        "source_type": "huggingface",
        "license": "Unknown",
        "sample_count": None,
        "size_gb": None,
        "hours": None,
        "sampling_rate": None,
        "status": "optional",
        "notes": "Moroccan Darija ASR dataset discovered during model/dataset research.",
    },
    "mohamedmou-dataset-darija": {
        "name": "Mohamedmou DATASET Darija",
        "hf_path": "mohamedmou/DATASET-darija",
        "split": "train",
        "audio_column": "audio",
        "transcript_column": "sentence",
        "language": "Moroccan Darija",
        "task": "ASR",
        "source_type": "huggingface",
        "license": "Unknown",
        "sample_count": None,
        "size_gb": None,
        "hours": None,
        "sampling_rate": None,
        "status": "optional",
        "notes": "Darija audio/text dataset from Hugging Face.",
    },
    # "mohamedmou-moroccan-darija-asr": {
    #     "name": "Mohamedmou Moroccan Darija ASR",
    #     "hf_path": "mohamedmou/moroccan-darija-asr-dataset",
    #     "split": "train",
    #     "audio_column": "audio",
    #     "transcript_column": "sentence"
    # },
    "afyfbadreddine-darija-asr-clean": {
        "name": "Darija ASR Clean",
        "hf_path": "afyfbadreddine77/darija-asr-clean",
        "split": "train",
        "audio_column": "audio",
        "transcript_column": "sentence",
        "language": "Moroccan Darija",
        "task": "ASR",
        "source_type": "huggingface",
        "license": "Unknown",
        "sample_count": None,
        "size_gb": None,
        "hours": None,
        "sampling_rate": None,
        "status": "optional",
        "notes": "Clean Darija ASR dataset from Hugging Face.",
    },
    "atlasia-moulsot-preview": {
        "name": "AtlasIA MoulSot Preview",
        "hf_path": "atlasia/moulsot-preview",
        "split": "train",
        "audio_column": "audio",
        "transcript_column": "transcript",
        "language": "Moroccan Darija",
        "task": "ASR / speech corpus",
        "source_type": "huggingface",
        "license": "Unknown",
        "sample_count": None,
        "size_gb": None,
        "hours": None,
        "sampling_rate": None,
        "status": "optional",
        "notes": "AtlasIA MoulSot preview subset.",
    },
    "atlasia-moulsot-full": {
        "name": "AtlasIA MoulSot Full",
        "hf_path": "atlasia/MoulSot-Full",
        "split": "test",
        "audio_column": "audio",
        "transcript_column": "transcript",
        "language": "Moroccan Darija",
        "task": "ASR / speech corpus",
        "source_type": "huggingface",
        "license": "Unknown",
        "sample_count": None,
        "size_gb": None,
        "hours": None,
        "sampling_rate": None,
        "status": "optional-large",
        "notes": "Large AtlasIA MoulSot dataset. Configured split is test because this dataset does not expose train in the current loader.",
    },
    "recorded-audio": {
        "name": "Recorded / Uploaded Audio",
        "type": "local-recordings",
        "audio_column": "audio",
        "transcript_column": "reference",
        "language": "User provided",
        "task": "ASR labeling",
        "source_type": "local",
        "license": "User provided",
        "sample_count": 0,
        "size_gb": 0,
        "hours": None,
        "sampling_rate": 16000,
        "status": "local",
        "notes": "Audio recorded or uploaded through the app and stored under data/recorded_dataset.",
    }
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# Model cache
model_cache = {}
processor_cache = {}
dataset_cache = {}
word_variant_cache = None
analysis_result_cache = None

AUDIO_COLUMN_CANDIDATES = ["audio", "speech", "input_values", "file", "path"]
TRANSCRIPT_COLUMN_CANDIDATES = [
    "sentence",
    "transcription",
    "transcription_darija_ar",
    "text",
    "transcript",
    "normalized_text",
    "reference",
]

# Progress tracking
progress_data = {
    'progress': 0,
    'status': '',
    'start_time': None,
    'processing': False
}

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def get_system_resources():
    """Get current system resource usage"""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info['gpu_available'] = True
            gpu_info['gpu_name'] = torch.cuda.get_device_name(0)
            gpu_info['gpu_vram_mb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024**2))
            gpu_info['gpu_used_mb'] = round((torch.cuda.memory_allocated() / (1024**2)))
        else:
            gpu_info['gpu_available'] = False
        
        return {
            'cpu_percent': round(cpu_percent, 1),
            'memory_percent': round(memory.percent, 1),
            'memory_available_mb': round(memory.available / (1024**2), 1),
            'memory_total_mb': round(memory.total / (1024**2), 1),
            **gpu_info
        }
    except Exception as e:
        return {
            'cpu_percent': 0,
            'memory_percent': 0,
            'memory_available_mb': 0,
            'memory_total_mb': 0,
            'error': str(e)
        }

def load_model(model_id):
    """Load model and processor, using cache if available"""
    global device
    if model_id not in model_cache:
        load_started = time.perf_counter()
        try:
            model_config = AVAILABLE_MODELS[model_id]
            framework = model_config.get("framework", "transformers")
            hf_path = model_config["hf_path"]
            lang_hint = model_config.get("language_hint")
            target_device = "cpu" if model_config.get("force_cpu") else device
            trust_remote_code = bool(model_config.get("trust_remote_code"))
            log_event(
                "MODEL load start",
                model_id=model_id,
                framework=framework,
                hf_path=hf_path,
                device=target_device,
                force_cpu=bool(model_config.get("force_cpu")),
            )
            
            if framework == "speechbrain":
                fn = _load_speechbrain_model(model_id, hf_path, target_device)
                model_cache[model_id] = ("speechbrain", fn)
                processor_cache[model_id] = None
                
            elif framework == "peft+transformers":
                fn = _load_peft_transformers_model(model_id, hf_path, model_config.get("base_model"), lang_hint, target_device)
                model_cache[model_id] = ("peft", fn)
                processor_cache[model_id] = None
                
            elif framework == "ctc-transformers":
                fn = _load_ctc_transformers_model(model_id, hf_path, target_device)
                model_cache[model_id] = ("ctc-transformers", fn)
                processor_cache[model_id] = None

            else:  # transformers
                fn = _load_transformers_model(model_id, hf_path, lang_hint, target_device, trust_remote_code)
                model_cache[model_id] = ("transformers", fn)
                processor_cache[model_id] = None
            
            log_event("MODEL load done", model_id=model_id, elapsed_s=f"{time.perf_counter() - load_started:.2f}")
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                log_event("MODEL cuda_oom_fallback", model_id=model_id)
                device = "cpu"
                # Retry loading on CPU
                model_config = AVAILABLE_MODELS[model_id]
                framework = model_config.get("framework", "transformers")
                hf_path = model_config["hf_path"]
                lang_hint = model_config.get("language_hint")
                
                if framework == "speechbrain":
                    fn = _load_speechbrain_model(model_id, hf_path, device)
                    model_cache[model_id] = ("speechbrain", fn)
                elif framework == "peft+transformers":
                    fn = _load_peft_transformers_model(model_id, hf_path, model_config.get("base_model"), lang_hint, device)
                    model_cache[model_id] = ("peft", fn)
                elif framework == "ctc-transformers":
                    fn = _load_ctc_transformers_model(model_id, hf_path, device)
                    model_cache[model_id] = ("ctc-transformers", fn)
                else:
                    fn = _load_transformers_model(model_id, hf_path, lang_hint, device, bool(model_config.get("trust_remote_code")))
                    model_cache[model_id] = ("transformers", fn)
                processor_cache[model_id] = None
            else:
                log_event("MODEL load failed", model_id=model_id, error=repr(e))
                traceback.print_exc()
                raise
        except Exception:
            log_event("MODEL load failed", model_id=model_id)
            traceback.print_exc()
            raise
    else:
        framework, _ = model_cache[model_id]
        log_event("MODEL cache hit", model_id=model_id, framework=framework)
    
    framework, fn = model_cache[model_id]
    return fn

def _load_transformers_model(model_id: str, hf_path: str, language_hint: str, device: str, trust_remote_code: bool = False):
    """Load a standard Whisper model using transformers pipeline"""
    log_event("TRANSFORMERS loader start", model_id=model_id, hf_path=hf_path, device=device)
    
    dtype = torch.float16 if device == "cuda" else torch.float32
    dev = 0 if device == "cuda" else -1

    try:
        asr = pipeline(
            task="automatic-speech-recognition",
            model=hf_path,
            device=dev,
            torch_dtype=dtype,
            chunk_length_s=None,
            trust_remote_code=trust_remote_code,
        )
    except OSError:
        log_event("TRANSFORMERS processor fallback", model_id=model_id, hf_path=hf_path)
        model = WhisperForConditionalGeneration.from_pretrained(
            hf_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
        ).to(device)
        processor = _load_whisper_processor_with_fallback(hf_path, language_hint)
        asr = pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=dev,
            chunk_length_s=None,
        )
    
    generate_kwargs = {"task": "transcribe"}
    if language_hint:
        generate_kwargs["language"] = language_hint
    
    def transcribe(audio_array: np.ndarray, sample_rate: int) -> str:
        infer_started = time.perf_counter()
        duration_s = len(audio_array) / float(sample_rate or 16000)
        log_event("INFERENCE start", model_id=model_id, framework="transformers", duration_s=f"{duration_s:.2f}", sample_rate=sample_rate)
        text = run_asr_pipeline(
            asr,
            audio_array,
            sample_rate,
            generate_kwargs,
            model_id=model_id,
        )
        if not text and language_hint:
            text = run_asr_pipeline(
                asr,
                audio_array,
                sample_rate,
                {"task": "transcribe"},
                model_id=model_id,
            )
        log_event("INFERENCE done", model_id=model_id, elapsed_s=f"{time.perf_counter() - infer_started:.2f}", chars=len(text))
        return text
    
    return transcribe

def _load_whisper_processor_with_fallback(hf_path: str, language_hint: str | None):
    try:
        return WhisperProcessor.from_pretrained(
            hf_path,
            language=language_hint or "Arabic",
            task="transcribe",
        )
    except OSError:
        pass

    config = {}
    processor_config = {}
    try:
        from huggingface_hub import snapshot_download

        snapshot_path = Path(snapshot_download(
            repo_id=hf_path,
            repo_type="model",
            cache_dir=os.getenv("HF_HUB_CACHE"),
            local_files_only=True,
        ))
        config_path = snapshot_path / "config.json"
        processor_config_path = snapshot_path / "processor_config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
        if processor_config_path.exists():
            processor_config = json.loads(processor_config_path.read_text(encoding="utf-8"))
    except Exception:
        snapshot_path = None

    base_processor = config.get("_name_or_path") or "openai/whisper-large-v3-turbo"
    feature_config = processor_config.get("feature_extractor", {})
    feature_config.pop("feature_extractor_type", None)
    feature_extractor = WhisperFeatureExtractor(**feature_config) if feature_config else None

    tokenizer = None
    if snapshot_path is not None:
        try:
            tokenizer = WhisperTokenizer.from_pretrained(
                str(snapshot_path),
                language=language_hint or "Arabic",
                task="transcribe",
                local_files_only=True,
            )
        except Exception:
            tokenizer = None

    if feature_extractor is None or tokenizer is None:
        base_processor_obj = WhisperProcessor.from_pretrained(
            base_processor,
            language=language_hint or "Arabic",
            task="transcribe",
        )
        feature_extractor = feature_extractor or base_processor_obj.feature_extractor
        tokenizer = tokenizer or base_processor_obj.tokenizer

    return WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

def run_asr_pipeline(asr, audio_array: np.ndarray, sample_rate: int, generate_kwargs: dict, model_id: str) -> str:
    audio = np.asarray(audio_array, dtype=np.float32)
    payloads = [
        ("array", {"array": audio, "sampling_rate": sample_rate}),
        ("raw", {"raw": audio, "sampling_rate": sample_rate}),
        ("ndarray", audio),
    ]
    fallback_kwargs = dict(generate_kwargs or {})
    fallback_kwargs.pop("language", None)
    attempts = [
        ("requested", generate_kwargs or {}),
        ("without_language", fallback_kwargs),
        ("no_generate_kwargs", None),
    ]
    last_error = None

    for payload_name, payload in payloads:
        for attempt_name, kwargs in attempts:
            try:
                if kwargs is None:
                    result = asr(payload)
                else:
                    result = asr(payload, generate_kwargs=kwargs)
                return result["text"].strip()
            except ValueError as exc:
                last_error = exc
                message = str(exc)
                retryable_payload = (
                    "needs to contain a \"raw\" key" in message
                    or "needs to contain an \"array\" key" in message
                    or "numpy array" in message
                )
                retryable_generation = "generation config is outdated" in message or "forced_decoder_ids" in message
                if not retryable_payload and not retryable_generation:
                    raise
                log_event(
                    "INFERENCE retry",
                    model_id=model_id,
                    payload=payload_name,
                    attempt=attempt_name,
                    reason=message[:120],
                )

    if last_error:
        raise last_error
    return ""

def _load_ctc_transformers_model(model_id: str, hf_path: str, device: str):
    """Load a standard Transformers CTC ASR model."""
    log_event("CTC loader start", model_id=model_id, hf_path=hf_path, device=device)

    processor = AutoProcessor.from_pretrained(hf_path)
    model = AutoModelForCTC.from_pretrained(hf_path).to(device)
    model.eval()

    def transcribe(audio_array: np.ndarray, sample_rate: int) -> str:
        infer_started = time.perf_counter()
        duration_s = len(audio_array) / float(sample_rate or 16000)
        log_event("INFERENCE start", model_id=model_id, framework="ctc-transformers", duration_s=f"{duration_s:.2f}", sample_rate=sample_rate)
        audio = np.asarray(audio_array, dtype=np.float32)
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        inputs = processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(device)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            logits = model(input_values, attention_mask=attention_mask).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(predicted_ids)[0].strip()
        log_event("INFERENCE done", model_id=model_id, elapsed_s=f"{time.perf_counter() - infer_started:.2f}", chars=len(text))
        return text

    return transcribe

def _load_peft_transformers_model(model_id: str, hf_path: str, base_model: str, language_hint: str, device: str):
    """Load a PEFT + Transformers model (LoRA adapter)"""
    if not HAS_PEFT:
        raise ImportError("PEFT library not available. Cannot load LoRA model.")
    
    log_event("PEFT loader start", model_id=model_id, hf_path=hf_path, base_model=base_model, device=device)
    
    use_cuda = device == "cuda"
    dtype = torch.float16 if use_cuda else torch.float32
    dev = 0 if use_cuda else -1
    
    try:
        # Load base model
        log_event("PEFT base load start", model_id=model_id, base_model=base_model)
        base = WhisperForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        
        # Load and merge LoRA adapter
        log_event("PEFT adapter merge start", model_id=model_id, adapter=hf_path)
        model = PeftModel.from_pretrained(base, hf_path)
        model = model.merge_and_unload()
    except Exception:
        log_event("PEFT merge failed", model_id=model_id)
        traceback.print_exc()
        raise
    
    # Clear GPU memory before moving
    if use_cuda:
        torch.cuda.empty_cache()
    
    # Move to device
    if use_cuda:
        model = model.to(dtype=torch.float16)
        model = model.to("cuda")
    else:
        model = model.to(device)
    
    # Create processor
    processor = WhisperProcessor.from_pretrained(
        base_model,
        language=language_hint or "Arabic",
        task="transcribe",
    )
    
    asr = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=None,
        device=dev,
    )
    
    generate_kwargs = {"task": "transcribe"}
    if language_hint:
        generate_kwargs["language"] = language_hint
    
    def transcribe(audio_array: np.ndarray, sample_rate: int) -> str:
        infer_started = time.perf_counter()
        duration_s = len(audio_array) / float(sample_rate or 16000)
        log_event("INFERENCE start", model_id=model_id, framework="peft+transformers", duration_s=f"{duration_s:.2f}", sample_rate=sample_rate)
        text = run_asr_pipeline(
            asr,
            audio_array,
            sample_rate,
            generate_kwargs,
            model_id=model_id,
        )
        if not text and language_hint:
            text = run_asr_pipeline(
                asr,
                audio_array,
                sample_rate,
                {"task": "transcribe"},
                model_id=model_id,
            )
        log_event("INFERENCE done", model_id=model_id, elapsed_s=f"{time.perf_counter() - infer_started:.2f}", chars=len(text))
        return text
    
    return transcribe

def _load_speechbrain_model(model_id: str, hf_path: str, device: str):
    """Load a SpeechBrain EncoderASR model (CTC)"""
    log_event("SPEECHBRAIN loader start", model_id=model_id, hf_path=hf_path, device=device)
    
    try:
        from speechbrain.inference.ASR import EncoderASR
    except ImportError:
        from speechbrain.pretrained import EncoderASR
    
    run_opts = {"device": device}
    asr_model = EncoderASR.from_hparams(
        source=hf_path,
        savedir=f"pretrained_models/{model_id}",
        run_opts=run_opts,
    )
    
    def transcribe(audio_array: np.ndarray, sample_rate: int) -> str:
        infer_started = time.perf_counter()
        duration_s = len(audio_array) / float(sample_rate or 16000)
        log_event("INFERENCE start", model_id=model_id, framework="speechbrain", duration_s=f"{duration_s:.2f}", sample_rate=sample_rate)
        waveform = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=16000,
            )
            waveform = resampler(waveform)
        
        waveform = waveform.to(asr_model.device)
        lengths = torch.tensor([1.0]).to(asr_model.device)
        
        transcriptions = asr_model.transcribe_batch(waveform, lengths)
        if not transcriptions:
            return ""
        first = transcriptions[0]
        if isinstance(first, (list, tuple)):
            first = first[0] if first else ""
        text = str(first).strip()
        log_event("INFERENCE done", model_id=model_id, elapsed_s=f"{time.perf_counter() - infer_started:.2f}", chars=len(text))
        return text
    
    return transcribe

def load_audio(audio_data, target_sr: int = 16000):
    """Load audio from bytes with multiple fallback methods"""
    tmp_path = None
    try:
        # Method 1: Try scipy (best for BytesIO WAV files)
        try:
            audio_bytes = io.BytesIO(audio_data)
            sr, audio = wavfile.read(audio_bytes)
            audio = normalize_audio_array(audio)
            
            # Resample if needed
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except Exception as scipy_error:
            print(f"Scipy failed: {scipy_error}, trying librosa...")
            
            # Method 2: Use temporary file for librosa (most compatible)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name
            
            audio, sr = librosa.load(tmp_path, sr=target_sr, mono=True)
            return normalize_audio_array(audio), target_sr
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        audio = normalize_audio_array(audio)
        return audio, target_sr
    except Exception as e:
        raise Exception(f"Failed to load audio with all methods: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

def normalize_audio_array(audio):
    """Convert decoded audio to mono finite float32 samples in the range models expect."""
    original_dtype = getattr(audio, "dtype", None)
    audio = np.asarray(audio)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if np.issubdtype(audio.dtype, np.integer):
        max_value = float(np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / max_value
    else:
        audio = audio.astype(np.float32)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        # Browser/dataset paths should already be [-1, 1]. This guards bad WAV paths.
        if peak > 2.0:
            audio = audio / peak

    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    print(
        f"Audio decoded: dtype={original_dtype}, samples={len(audio)}, "
        f"peak={peak:.4f}, duration={len(audio) / 16000:.2f}s"
    )
    return audio

def transcribe_audio(audio_data, model_id=None):
    """Transcribe audio data using specified model"""
    global progress_data, device
    
    try:
        start_time = time.time()
        perf_start = time.perf_counter()
        progress_data['processing'] = True
        progress_data['start_time'] = start_time
        progress_data['progress'] = 0
        progress_data['status'] = 'Loading audio...'
        
        # Clear GPU memory before starting
        clear_gpu_memory()
        
        if model_id is None:
            model_id = list(AVAILABLE_MODELS.keys())[0]
        
        if model_id not in AVAILABLE_MODELS:
            return f"Error: Unknown model '{model_id}'"

        log_event("TRANSCRIBE upload start", model_id=model_id, bytes=len(audio_data))
        
        # Load model
        progress_data['progress'] = 10
        progress_data['status'] = f"Loading model: {AVAILABLE_MODELS[model_id]['name']}"
        step_started = time.perf_counter()
        transcribe_fn = load_model(model_id)
        log_event("TRANSCRIBE model ready", model_id=model_id, elapsed_s=f"{time.perf_counter() - step_started:.2f}")
        
        # Load audio
        progress_data['progress'] = 25
        progress_data['status'] = "Processing audio..."
        step_started = time.perf_counter()
        audio, sr = load_audio(audio_data, target_sr=16000)
        log_event("TRANSCRIBE audio ready", model_id=model_id, elapsed_s=f"{time.perf_counter() - step_started:.2f}", samples=len(audio), sample_rate=sr)
        
        # Transcribe
        progress_data['progress'] = 60
        progress_data['status'] = "Transcribing..."
        step_started = time.perf_counter()
        transcription = transcribe_fn(audio, sr)
        log_event("TRANSCRIBE inference ready", model_id=model_id, elapsed_s=f"{time.perf_counter() - step_started:.2f}", chars=len(transcription or ""))
        log_inference_result(model_id, "upload", transcription)
        
        progress_data['progress'] = 100
        progress_data['status'] = "Complete!"
        progress_data['processing'] = False
        log_event("TRANSCRIBE upload done", model_id=model_id, total_s=f"{time.perf_counter() - perf_start:.2f}")
        
        # Clean up
        clear_gpu_memory()
        
        return transcription
    except Exception as e:
        progress_data['processing'] = False
        progress_data['status'] = f"Error: {e}"
        log_event("TRANSCRIBE upload failed", model_id=model_id, error=repr(e))
        clear_gpu_memory()
        return f"Error: {str(e)}"

def is_error_result(value):
    return isinstance(value, str) and value.startswith("Error:")

def log_inference_result(model_id, source, text):
    preview = (text or "").replace("\n", " ")[:120]
    print(f"Inference result: model={model_id}, source={source}, chars={len(text or '')}, text={ascii(preview)}")

def custom_dataset_id_from_name(name):
    base = secure_filename((name or "").strip().lower().replace(" ", "-"))
    base = re.sub(r"-+", "-", base).strip("-_")
    return f"custom-{base or uuid.uuid4().hex[:8]}"

def custom_dataset_dir(dataset_id):
    safe_id = secure_filename(dataset_id or "")
    if not safe_id.startswith("custom-"):
        raise ValueError("Custom dataset ids must start with custom-")
    return CUSTOM_DATASETS_DIR / safe_id

def custom_dataset_manifest(dataset_id):
    return custom_dataset_dir(dataset_id) / "manifest.jsonl"

def custom_dataset_meta_path(dataset_id):
    return custom_dataset_dir(dataset_id) / "dataset.json"

def load_custom_dataset_configs():
    configs = {}
    for meta_path in sorted(CUSTOM_DATASETS_DIR.glob("custom-*/dataset.json")):
        try:
            with meta_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue
        dataset_id = payload.get("id") or meta_path.parent.name
        configs[dataset_id] = {
            "name": payload.get("name") or dataset_id,
            "type": "custom",
            "audio_column": "audio",
            "transcript_column": "reference",
            "language": payload.get("language") or "Moroccan Darija",
            "task": "ASR labeling",
            "source_type": "custom",
            "license": payload.get("license") or "User provided",
            "sample_count": 0,
            "size_gb": 0,
            "hours": None,
            "sampling_rate": 16000,
            "status": "custom",
            "notes": payload.get("notes") or "Editable custom dataset created in the workbench.",
        }
    return configs

def sync_custom_datasets():
    AVAILABLE_DATASETS.update(load_custom_dataset_configs())

def create_custom_dataset(name, notes="", language="Moroccan Darija"):
    dataset_id = custom_dataset_id_from_name(name)
    base_id = dataset_id
    suffix = 2
    while custom_dataset_meta_path(dataset_id).exists() or dataset_id in AVAILABLE_DATASETS:
        dataset_id = f"{base_id}-{suffix}"
        suffix += 1
    dataset_dir = custom_dataset_dir(dataset_id)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": dataset_id,
        "name": name or dataset_id,
        "notes": notes or "",
        "language": language or "Moroccan Darija",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with custom_dataset_meta_path(dataset_id).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    custom_dataset_manifest(dataset_id).touch(exist_ok=True)
    sync_custom_datasets()
    dataset_cache.clear()
    return dataset_id, AVAILABLE_DATASETS[dataset_id]

def local_dataset_paths(dataset_id):
    if dataset_id == "recorded-audio":
        return RECORDED_DATASET_DIR, RECORDED_MANIFEST
    if dataset_id.startswith("custom-"):
        return custom_dataset_dir(dataset_id), custom_dataset_manifest(dataset_id)
    raise ValueError("Dataset is not editable local storage")

def save_local_sample(audio_data, reference, source_name="recording", target_dataset_id="recorded-audio"):
    """Save uploaded/recorded audio as a local or custom dataset sample."""
    try:
        sync_custom_datasets()
        dataset_dir, manifest_path = local_dataset_paths(target_dataset_id)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        audio, sr = load_audio(audio_data, target_sr=16000)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_source = Path(secure_filename(source_name or "recording") or "recording").stem
        stem = f"{timestamp}_{safe_source}"
        wav_path = dataset_dir / f"{stem}.wav"
        sf.write(wav_path, audio, sr, format="WAV")
        record = {
            "file": wav_path.name,
            "reference": reference or "",
            "source": source_name or "",
            "dataset_id": target_dataset_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        dataset_cache.clear()
        print(f"Saved local sample: {wav_path}")
        return record
    except Exception:
        print("Failed to save local sample:")
        traceback.print_exc()
        return None

def save_recorded_sample(audio_data, reference, source_name="recording"):
    """Backward-compatible default local dataset save."""
    return save_local_sample(audio_data, reference, source_name=source_name, target_dataset_id="recorded-audio")

def save_labeled_audio(audio_data, label, source_name="labeled-audio", notes="", target_dataset_id="recorded-audio"):
    """Save human-labeled audio as part of the local recorded dataset."""
    record = save_local_sample(audio_data, label, source_name=source_name, target_dataset_id=target_dataset_id)
    if not record:
        return None

    label_record = {
        **record,
        "label": label or "",
        "notes": notes or "",
        "label_type": "human",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    dataset_dir, _ = local_dataset_paths(target_dataset_id)
    label_manifest = dataset_dir / "labels.jsonl"
    with label_manifest.open("a", encoding="utf-8") as f:
        f.write(json.dumps(label_record, ensure_ascii=False) + "\n")
    if target_dataset_id == "recorded-audio":
        with LABEL_MANIFEST.open("a", encoding="utf-8") as f:
            f.write(json.dumps(label_record, ensure_ascii=False) + "\n")
    return label_record

def save_labeled_sample(dataset_id, sample_index, label, sample_limit=50, notes="", target_dataset_id="recorded-audio"):
    """Copy a selected dataset sample into the local labeled dataset with a human label."""
    sample = get_dataset_sample(dataset_id, int(sample_index), max_samples=int(sample_limit))
    audio, sr = sample_to_audio(sample)
    audio_buffer = audio_to_wav_buffer(audio, sr).read()
    source = f"{dataset_id}_{sample_index}_human_label.wav"
    return save_labeled_audio(audio_buffer, label, source_name=source, notes=notes, target_dataset_id=target_dataset_id)

def load_word_variants():
    """Load accepted spelling variants as normalized reference/hypothesis word pairs."""
    global word_variant_cache
    if word_variant_cache is not None:
        return word_variant_cache

    variants = set()
    if WORD_VARIANTS_MANIFEST.exists():
        import json
        with WORD_VARIANTS_MANIFEST.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ref = normalize_eval_text(row.get("reference_word", ""))
                hyp = normalize_eval_text(row.get("hypothesis_word", ""))
                if ref and hyp:
                    variants.add((ref, hyp))
    word_variant_cache = variants
    return variants

def save_word_variant(reference_word, hypothesis_word, source=None):
    """Persist a word pair that should count as correct despite spelling differences."""
    global word_variant_cache
    ref = normalize_eval_text(reference_word)
    hyp = normalize_eval_text(hypothesis_word)
    if not ref or not hyp:
        raise ValueError("Both reference_word and hypothesis_word are required")

    row = {
        "reference_word": ref,
        "hypothesis_word": hyp,
        "source": source or "",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with WORD_VARIANTS_MANIFEST.open("a", encoding="utf-8") as f:
        import json
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    word_variant_cache = None
    return row

def load_local_samples(dataset_id="recorded-audio", max_samples=100):
    try:
        dataset_dir, manifest_path = local_dataset_paths(dataset_id)
    except ValueError:
        return []
    if not manifest_path.exists():
        return []
    rows = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            wav_path = dataset_dir / record.get("file", "")
            if not wav_path.exists():
                continue
            rows.append({
                "index": len(rows),
                "source_index": len(rows),
                "audio_path": str(wav_path),
                "sampling_rate": 16000,
                "reference": record.get("reference") or record.get("source") or wav_path.stem,
                "created_at": record.get("created_at", ""),
            })
            if len(rows) >= max_samples:
                break
    return rows

def load_recorded_samples(max_samples=100):
    return load_local_samples("recorded-audio", max_samples=max_samples)

def directory_size_bytes(path: Path):
    if not path.exists():
        return 0
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            try:
                total += file_path.stat().st_size
            except OSError:
                pass
    return total

def local_dataset_metadata(dataset_id="recorded-audio"):
    try:
        dataset_dir, _ = local_dataset_paths(dataset_id)
    except ValueError:
        return {}
    samples = load_local_samples(dataset_id, max_samples=100000)
    size_bytes = directory_size_bytes(dataset_dir)
    total_seconds = 0.0
    for sample in samples:
        audio_path = sample.get("audio_path")
        if not audio_path:
            continue
        try:
            info = sf.info(audio_path)
            total_seconds += float(info.frames) / float(info.samplerate or 16000)
        except Exception:
            continue
    return {
        "sample_count": len(samples),
        "size_bytes": size_bytes,
        "size_gb": round(size_bytes / (1024 ** 3), 4),
        "hours": round(total_seconds / 3600.0, 3) if total_seconds else None,
    }

def recorded_dataset_metadata():
    return local_dataset_metadata("recorded-audio")

def load_collected_dataset_metadata():
    if not DATASET_METADATA_PATH.exists():
        return {}
    try:
        with DATASET_METADATA_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload.get("datasets", {})
    except Exception:
        print("Failed to read dataset metadata:")
        traceback.print_exc()
        return {}

def load_registry_benchmark_results():
    if not REGISTRY_BENCHMARK_RESULTS_PATH.exists():
        return {}
    try:
        with REGISTRY_BENCHMARK_RESULTS_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        print("Failed to read registry benchmark results:")
        traceback.print_exc()
        return {}

def benchmark_stats_to_api(stats, details=None):
    if not isinstance(stats, dict):
        return None
    payload = {
        "runs": stats.get("runs", 0),
        "samples": stats.get("samples", 0),
        "avg_wer": stats.get("avg_wer"),
        "avg_cer": stats.get("avg_cer"),
        "avg_elapsed_seconds": stats.get("avg_elapsed_seconds"),
        "cache_hits": stats.get("cache_hits", 0),
        "cache_misses": stats.get("cache_misses", 0),
        "best": stats.get("best"),
        "worst": stats.get("worst"),
    }
    if isinstance(details, dict):
        payload.update(details)
    return payload

def model_benchmark_details(benchmark_payload):
    results = benchmark_payload.get("results", [])
    model_stats = benchmark_payload.get("model_stats", {})
    ranked = sorted(
        [
            (model_id, stats)
            for model_id, stats in model_stats.items()
            if isinstance(stats, dict) and int(stats.get("samples") or 0) > 0 and stats.get("avg_wer") is not None
        ],
        key=lambda item: (float(item[1].get("avg_wer")), float(item[1].get("avg_cer") or 1.0)),
    )
    ranks = {model_id: index + 1 for index, (model_id, _) in enumerate(ranked)}
    details = {}
    for model_id, stats in model_stats.items():
        rows = [
            row for row in results
            if row.get("model_id") == model_id
            and row.get("status", "ok") != "failed"
            and int(row.get("count") or 0) > 0
            and row.get("avg_wer") is not None
        ]
        rows = sorted(rows, key=lambda row: (float(row.get("avg_wer")), float(row.get("avg_cer") or 1.0)))
        buckets = {"excellent": 0, "good": 0, "fair": 0, "poor": 0, "failed": 0}
        for row in rows:
            wer = float(row.get("avg_wer"))
            if wer < 0.35:
                buckets["excellent"] += 1
            elif wer < 0.55:
                buckets["good"] += 1
            elif wer < 0.8:
                buckets["fair"] += 1
            else:
                buckets["poor"] += 1
        buckets["failed"] = len([
            row for row in results
            if row.get("model_id") == model_id and (row.get("status") == "failed" or row.get("error"))
        ])
        details[model_id] = {
            "rank": ranks.get(model_id),
            "rank_total": len(ranked),
            "dataset_scores": [
                {
                    "dataset_id": row.get("dataset_id"),
                    "count": row.get("count", 0),
                    "avg_wer": row.get("avg_wer"),
                    "avg_cer": row.get("avg_cer"),
                    "avg_elapsed_seconds": row.get("avg_elapsed_seconds"),
                    "cache_hits": row.get("cache_hits", 0),
                    "cache_misses": row.get("cache_misses", 0),
                    "stopped": row.get("stopped"),
                }
                for row in rows
            ],
            "distribution": buckets,
        }
    return details

def dataset_to_api(dataset_id, config, collected_metadata=None):
    live = local_dataset_metadata(dataset_id) if dataset_id == "recorded-audio" or dataset_id.startswith("custom-") else {}
    collected = (collected_metadata or {}).get(dataset_id, {})
    sample_count = live.get("sample_count", collected.get("sample_count", config.get("sample_count")))
    size_bytes = live.get("size_bytes", collected.get("local_size_bytes") or collected.get("repo_size_bytes"))
    size_gb = live.get("size_gb", collected.get("local_size_gb") or collected.get("repo_size_gb") or config.get("size_gb"))
    license_name = collected.get("license") or config.get("license", "Unknown")
    return {
        "id": dataset_id,
        "name": config.get("name", dataset_id),
        "hf_path": config.get("hf_path", ""),
        "split": config.get("split", ""),
        "audio_column": config.get("audio_column", ""),
        "transcript_column": config.get("transcript_column", ""),
        "language": config.get("language", "Unknown"),
        "task": config.get("task", "ASR"),
        "source_type": config.get("source_type", config.get("type", "huggingface")),
        "license": license_name,
        "sample_count": sample_count,
        "size_gb": size_gb,
        "size_bytes": size_bytes,
        "hours": live.get("hours", config.get("hours")),
        "sampling_rate": config.get("sampling_rate"),
        "status": config.get("status", "configured"),
        "notes": config.get("notes", ""),
        "downloads": collected.get("downloads"),
        "likes": collected.get("likes"),
        "repo_file_count": collected.get("repo_file_count"),
        "local_size_gb": collected.get("local_size_gb"),
        "available_splits": collected.get("available_splits"),
        "split_rows": collected.get("split_rows"),
        "metadata_collected_at": collected.get("collected_at"),
        "metadata_error": collected.get("hf_metadata_error"),
        "available": dataset_id == "recorded-audio" or dataset_id.startswith("custom-") or HAS_DATASETS,
    }

def load_dataset_samples(dataset_id: str, max_samples: int = 100):
    """Load samples from a dataset"""
    sync_custom_datasets()
    cache_key = f"{dataset_id}:{max_samples}"
    if cache_key in dataset_cache:
        log_event("DATASET cache hit", dataset_id=dataset_id, limit=max_samples, count=len(dataset_cache[cache_key]))
        return dataset_cache[cache_key]

    if dataset_id == "recorded-audio" or dataset_id.startswith("custom-"):
        log_event("DATASET local load start", dataset_id=dataset_id, limit=max_samples)
        samples = load_local_samples(dataset_id, max_samples=max_samples)
        dataset_cache[cache_key] = samples
        log_event("DATASET local load done", dataset_id=dataset_id, count=len(samples))
        return samples

    if not HAS_DATASETS:
        return None
    
    try:
        load_started = time.perf_counter()
        dataset_config = AVAILABLE_DATASETS[dataset_id]
        log_event("DATASET load start", dataset_id=dataset_id, limit=max_samples, hf_path=dataset_config.get("hf_path"), split=dataset_config.get("split"))
        
        # Try loading with different transcript column names if needed
        ds = load_dataset_with_split_fallback(dataset_config["hf_path"], dataset_config["split"])
        
        audio_col = find_column(ds.column_names, dataset_config["audio_column"], AUDIO_COLUMN_CANDIDATES)
        if audio_col is None:
            raise ValueError(f"No audio column found. Available columns: {ds.column_names}")

        # Try to find the transcript column
        transcript_col = find_column(
            ds.column_names,
            dataset_config["transcript_column"],
            TRANSCRIPT_COLUMN_CANDIDATES,
        )
        if transcript_col is None:
            raise ValueError(f"No transcript column found. Available columns: {ds.column_names}")
        
        samples = []
        for i, item in enumerate(ds):
            if i >= max_samples:
                break
            
            audio = item.get(audio_col)
            transcript = item.get(transcript_col)
            if audio is None or transcript is None:
                continue
            
            # Handle audio dict format
            if isinstance(audio, dict) and "array" in audio:
                audio_array = audio["array"]
                sr = audio["sampling_rate"]
            else:
                audio_array = audio
                sr = dataset_config.get("sampling_rate", 16000)
            
            samples.append({
                "index": len(samples),
                "source_index": i,
                "audio_array": audio_array.tolist() if isinstance(audio_array, np.ndarray) else audio_array,
                "sampling_rate": sr,
                "reference": str(transcript) if transcript else ""
            })
        
        dataset_cache[cache_key] = samples
        log_event("DATASET load done", dataset_id=dataset_id, limit=max_samples, count=len(samples), elapsed_s=f"{time.perf_counter() - load_started:.2f}")
        return samples
    except Exception as e:
        log_event("DATASET load failed", dataset_id=dataset_id, error=repr(e))
        return None

def get_dataset_sample(dataset_id: str, sample_index: int, max_samples: int = 100):
    log_event("DATASET sample request", dataset_id=dataset_id, sample_index=sample_index, limit=max_samples)
    samples = load_dataset_samples(dataset_id, max_samples=max_samples)
    if samples is None:
        raise ValueError("Failed to load dataset")
    if sample_index < 0 or sample_index >= len(samples):
        raise IndexError(f"Sample {sample_index} is outside loaded range 0-{len(samples) - 1}")
    return samples[sample_index]

def sample_to_audio(sample):
    started = time.perf_counter()
    if sample.get("audio_path"):
        audio, sr = sf.read(sample["audio_path"], dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = np.mean(audio, axis=1)
        normalized = normalize_audio_array(audio)
        log_event("DATASET sample audio decoded", source="file", samples=len(normalized), sample_rate=int(sr), elapsed_s=f"{time.perf_counter() - started:.2f}")
        return normalized, int(sr)
    normalized = normalize_audio_array(sample["audio_array"])
    sr = int(sample.get("sampling_rate", 16000))
    log_event("DATASET sample audio decoded", source="memory", samples=len(normalized), sample_rate=sr, elapsed_s=f"{time.perf_counter() - started:.2f}")
    return normalized, sr

def audio_to_wav_buffer(audio, sr):
    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(audio, dtype=np.float32), int(sr), format="WAV")
    buffer.seek(0)
    return buffer

def read_analysis_cache():
    global analysis_result_cache
    if analysis_result_cache is not None:
        return analysis_result_cache
    if not ANALYSIS_CACHE_PATH.exists():
        analysis_result_cache = {}
        return analysis_result_cache
    try:
        payload = json.loads(ANALYSIS_CACHE_PATH.read_text(encoding="utf-8"))
        analysis_result_cache = payload.get("items", payload if isinstance(payload, dict) else {})
    except Exception:
        print("Failed to read analysis cache:")
        traceback.print_exc()
        analysis_result_cache = {}
    return analysis_result_cache


def write_analysis_cache():
    if analysis_result_cache is None:
        return
    ANALYSIS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1,
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "items": analysis_result_cache,
    }
    temp_path = ANALYSIS_CACHE_PATH.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(ANALYSIS_CACHE_PATH)


def stable_json_hash(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def analysis_cache_key(model_id, dataset_id, sample, add_noise, snr_db):
    reference = sample.get("reference", "")
    identity = {
        "schema": 1,
        "model_id": model_id,
        "dataset_id": dataset_id,
        "sample_index": sample.get("source_index", sample.get("index")),
        "index": sample.get("index"),
        "audio_path": sample.get("audio_path", ""),
        "created_at": sample.get("created_at", ""),
        "reference_hash": stable_json_hash(reference),
        "noise": bool(add_noise),
        "snr_db": round(float(snr_db), 2) if add_noise else None,
    }
    return stable_json_hash(identity)


def analysis_seed_from_key(cache_key):
    return int(cache_key[:8], 16)


def save_analysis_run(run_payload):
    ANALYSIS_RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    created_at = datetime.utcnow().isoformat() + "Z"
    run_id = uuid.uuid4().hex[:12]
    saved = {
        **run_payload,
        "run_id": run_id,
        "created_at": created_at,
    }
    with ANALYSIS_RUNS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(saved, ensure_ascii=False) + "\n")
    return saved


def load_saved_analysis_runs(limit=30):
    if not ANALYSIS_RUNS_PATH.exists():
        return []
    rows = []
    lines = ANALYSIS_RUNS_PATH.read_text(encoding="utf-8").splitlines()
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append(record)
        if len(rows) >= limit:
            break
    return rows


def summarize_saved_analysis_run(record):
    return {
        "run_id": record.get("run_id"),
        "created_at": record.get("created_at"),
        "model_ids": record.get("model_ids", []),
        "dataset_ids": record.get("dataset_ids", []),
        "settings": record.get("settings", {}),
        "summary": record.get("summary", {}),
    }


def add_noise_to_audio(audio, snr_db=20.0, seed=None):
    """Add Gaussian noise at the requested signal-to-noise ratio."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio

    signal_power = float(np.mean(np.square(audio)))
    if signal_power <= 1e-12:
        return audio

    snr_linear = 10 ** (float(snr_db) / 10.0)
    noise_power = signal_power / snr_linear
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, np.sqrt(noise_power), size=audio.shape).astype(np.float32)
    return np.clip(audio + noise, -1.0, 1.0).astype(np.float32)

def normalize_eval_text(text):
    text = str(text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text

def tokenize_words(text):
    return normalize_eval_text(text).split()

def align_sequences(reference_items, hypothesis_items, accepted_pairs=None):
    matcher = difflib.SequenceMatcher(a=reference_items, b=hypothesis_items, autojunk=False)
    rows = []
    summary = Counter()
    accepted_pairs = accepted_pairs or set()

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        ref_chunk = reference_items[i1:i2]
        hyp_chunk = hypothesis_items[j1:j2]
        span = max(len(ref_chunk), len(hyp_chunk))

        if tag == "equal":
            for offset in range(span):
                rows.append({
                    "type": "equal",
                    "reference": ref_chunk[offset],
                    "hypothesis": hyp_chunk[offset],
                    "reference_index": i1 + offset,
                    "hypothesis_index": j1 + offset,
                })
                summary["correct"] += 1
        elif tag == "delete":
            for offset, item in enumerate(ref_chunk):
                rows.append({
                    "type": "deletion",
                    "reference": item,
                    "hypothesis": "",
                    "reference_index": i1 + offset,
                    "hypothesis_index": None,
                })
                summary["deletion"] += 1
        elif tag == "insert":
            for offset, item in enumerate(hyp_chunk):
                rows.append({
                    "type": "insertion",
                    "reference": "",
                    "hypothesis": item,
                    "reference_index": None,
                    "hypothesis_index": j1 + offset,
                })
                summary["insertion"] += 1
        else:
            for offset in range(span):
                ref_item = ref_chunk[offset] if offset < len(ref_chunk) else ""
                hyp_item = hyp_chunk[offset] if offset < len(hyp_chunk) else ""
                if ref_item and hyp_item and (ref_item, hyp_item) in accepted_pairs:
                    row_type = "accepted"
                elif ref_item and hyp_item:
                    row_type = "substitution"
                elif ref_item:
                    row_type = "deletion"
                else:
                    row_type = "insertion"
                rows.append({
                    "type": row_type,
                    "reference": ref_item,
                    "hypothesis": hyp_item,
                    "reference_index": i1 + offset if ref_item else None,
                    "hypothesis_index": j1 + offset if hyp_item else None,
                })
                summary[row_type] += 1

    return rows, dict(summary)

def summarize_word_patterns(metrics_list):
    word_stats = defaultdict(lambda: {"seen": 0, "correct": 0, "errors": 0})
    substitution_pairs = Counter()
    char_substitutions = Counter()

    for metrics in metrics_list:
        for row in metrics.get("word_diff", []):
            ref_word = row.get("reference", "")
            hyp_word = row.get("hypothesis", "")
            row_type = row.get("type")

            if ref_word:
                stat = word_stats[ref_word]
                stat["seen"] += 1
                if row_type in {"equal", "accepted"}:
                    stat["correct"] += 1
                else:
                    stat["errors"] += 1
            if row_type == "substitution" and ref_word and hyp_word:
                substitution_pairs[(ref_word, hyp_word)] += 1

        for row in metrics.get("char_diff", []):
            if row.get("type") == "substitution":
                char_substitutions[(row.get("reference", ""), row.get("hypothesis", ""))] += 1

    easiest = []
    hardest = []
    for word, stat in word_stats.items():
        accuracy = stat["correct"] / stat["seen"] if stat["seen"] else 0.0
        item = {
            "word": word,
            "seen": stat["seen"],
            "correct": stat["correct"],
            "errors": stat["errors"],
            "accuracy": accuracy,
        }
        if stat["seen"] >= 1:
            easiest.append(item)
            hardest.append(item)

    easiest.sort(key=lambda item: (-item["accuracy"], -item["seen"], item["word"]))
    hardest.sort(key=lambda item: (item["accuracy"], -item["seen"], item["word"]))

    return {
        "easiest_words": easiest[:20],
        "hardest_words": hardest[:20],
        "word_substitutions": [
            {"reference": ref, "hypothesis": hyp, "count": count}
            for (ref, hyp), count in substitution_pairs.most_common(20)
        ],
        "char_substitutions": [
            {"reference": ref, "hypothesis": hyp, "count": count}
            for (ref, hyp), count in char_substitutions.most_common(20)
        ],
    }

def find_column(columns, preferred, candidates):
    """Find a dataset column by preferred name, then common ASR aliases."""
    if preferred in columns:
        return preferred
    for column in candidates:
        if column in columns:
            return column
    lower_columns = {column.lower(): column for column in columns}
    for column in [preferred, *candidates]:
        match = lower_columns.get(column.lower())
        if match:
            return match
    return None

def load_dataset_with_split_fallback(hf_path, preferred_split):
    """Load a split, retrying with the first available split if the preferred one is absent."""
    try:
        return load_dataset(hf_path, split=preferred_split)
    except ValueError as e:
        message = str(e)
        if "Unknown split" not in message:
            raise

        builder = load_dataset(hf_path)
        available_splits = list(builder.keys())
        if not available_splits:
            raise

        fallback_split = available_splits[0]
        print(
            f"Warning: {hf_path} has no split '{preferred_split}'. "
            f"Using '{fallback_split}' instead. Available: {available_splits}"
        )
        return builder[fallback_split]

def calculate_metrics(reference: str, hypothesis: str):
    """Calculate WER/CER plus word and character diffs."""
    try:
        normalized_reference = normalize_eval_text(reference)
        normalized_hypothesis = normalize_eval_text(hypothesis)
        wer_score = compute_wer(normalized_reference, normalized_hypothesis)
        cer_score = compute_cer(normalized_reference, normalized_hypothesis)

        ref_words = tokenize_words(reference)
        hyp_words = tokenize_words(hypothesis)
        accepted_pairs = load_word_variants()
        word_diff, word_summary = align_sequences(ref_words, hyp_words, accepted_pairs=accepted_pairs)
        char_diff, char_summary = align_sequences(list(normalized_reference), list(normalized_hypothesis))
        adjusted_word_errors = (
            word_summary.get("substitution", 0)
            + word_summary.get("deletion", 0)
            + word_summary.get("insertion", 0)
        )
        adjusted_wer = adjusted_word_errors / len(ref_words) if ref_words else (0.0 if not hyp_words else 1.0)
        word_errors = [
            {
                "position": row.get("reference_index") if row.get("reference_index") is not None else row.get("hypothesis_index"),
                "reference": row.get("reference", ""),
                "hypothesis": row.get("hypothesis", ""),
                "type": row.get("type"),
            }
            for row in word_diff
            if row.get("type") not in {"equal", "accepted"}
        ]
        
        return {
            "wer": min(1.0, adjusted_wer),  # Cap at 1.0
            "raw_wer": min(1.0, wer_score),
            "adjusted_wer": min(1.0, adjusted_wer),
            "cer": min(1.0, cer_score),  # Cap at 1.0
            "reference": reference,
            "hypothesis": hypothesis,
            "normalized_reference": normalized_reference,
            "normalized_hypothesis": normalized_hypothesis,
            "word_diff": word_diff,
            "char_diff": char_diff,
            "word_summary": word_summary,
            "char_summary": char_summary,
            "word_errors": word_errors,
            "reference_word_count": len(ref_words),
            "hypothesis_word_count": len(hyp_words),
            "reference_char_count": len(normalized_reference),
            "hypothesis_char_count": len(normalized_hypothesis),
            "accepted_variant_count": word_summary.get("accepted", 0),
        }
    except Exception as e:
        return {
            "wer": 0.0,
            "cer": 0.0,
            "reference": reference,
            "hypothesis": hypothesis,
            "normalized_reference": normalize_eval_text(reference),
            "normalized_hypothesis": normalize_eval_text(hypothesis),
            "word_diff": [],
            "char_diff": [],
            "word_summary": {},
            "char_summary": {},
            "word_errors": [],
            "error": str(e)
        }

# Keep the API responsive while the container starts. Models load on first use;
# set SPEECHRECON_PRELOAD_DEFAULT_MODEL=1 to warm the default model at startup.
print(f"Device: {device}")
default_model_id = list(AVAILABLE_MODELS.keys())[0]
if os.getenv("SPEECHRECON_PRELOAD_DEFAULT_MODEL", "0").lower() in {"1", "true", "yes", "on"}:
    print("Loading default model...")
    try:
        load_model(default_model_id)
    except Exception as e:
        print(f"Warning: Could not preload default model: {e}")

@app.route('/')
def index():
    react_index = FRONTEND_DIST_DIR / "index.html"
    if react_index.exists():
        response = make_response(send_file(react_index))
        response.headers["Cache-Control"] = "no-store, max-age=0"
        return response
    return render_template('index.html')

@app.route('/assets/<path:filename>')
def frontend_assets(filename):
    return send_from_directory(FRONTEND_DIST_DIR / "assets", filename)

@app.route('/models', methods=['GET'])
def get_models():
    """Get list of available models"""
    benchmark_payload = load_registry_benchmark_results()
    model_stats = benchmark_payload.get("model_stats", {})
    model_details = model_benchmark_details(benchmark_payload)
    models_list = [
        {
            'id': model_id,
            'name': config['name'],
            'framework': config.get('framework', 'transformers'),
            'min_vram_gb': config.get('min_vram_gb', 0),
            'benchmark': benchmark_stats_to_api(model_stats.get(model_id), model_details.get(model_id)),
        }
        for model_id, config in AVAILABLE_MODELS.items()
    ]
    return jsonify({
        'models': models_list,
        'default': default_model_id,
        'benchmark_generated_at': benchmark_payload.get("generated_at"),
        'benchmark_progress': benchmark_payload.get("progress"),
    })

@app.route('/datasets', methods=['GET'])
def get_datasets():
    """Get list of available datasets"""
    sync_custom_datasets()
    collected_metadata = load_collected_dataset_metadata()
    benchmark_payload = load_registry_benchmark_results()
    dataset_stats = benchmark_payload.get("dataset_stats", {})
    datasets_list = [
        {
            **dataset_to_api(dataset_id, config, collected_metadata),
            "benchmark": benchmark_stats_to_api(dataset_stats.get(dataset_id)),
        }
        for dataset_id, config in AVAILABLE_DATASETS.items()
    ]
    return jsonify({
        'datasets': datasets_list,
        'available': HAS_DATASETS,
        'benchmark_generated_at': benchmark_payload.get("generated_at"),
        'benchmark_progress': benchmark_payload.get("progress"),
    })

@app.route('/registry-benchmark', methods=['GET'])
def registry_benchmark():
    """Return the latest registry benchmark summary produced by scripts/run_registry_benchmark.py."""
    payload = load_registry_benchmark_results()
    if not payload:
        return jsonify({"success": False, "error": "Registry benchmark results not found"}), 404
    return jsonify({
        "success": True,
        "generated_at": payload.get("generated_at"),
        "settings": payload.get("settings"),
        "progress": payload.get("progress"),
        "model_stats": payload.get("model_stats", {}),
        "dataset_stats": payload.get("dataset_stats", {}),
        "results": payload.get("results", []),
    })

@app.route('/custom-datasets', methods=['POST'])
def create_custom_dataset_endpoint():
    """Create an editable local dataset for labeling and copied samples."""
    payload = request.get_json(silent=True) or {}
    name = str(payload.get('name') or '').strip()
    notes = str(payload.get('notes') or '').strip()
    language = str(payload.get('language') or 'Moroccan Darija').strip()
    if not name:
        return jsonify({'success': False, 'error': 'Dataset name is required'}), 400
    try:
        dataset_id, config = create_custom_dataset(name, notes=notes, language=language)
        return jsonify({
            'success': True,
            'dataset': dataset_to_api(dataset_id, config, load_collected_dataset_metadata())
        })
    except Exception as e:
        log_event("ENDPOINT custom-datasets failed", error=repr(e))
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/dataset-samples', methods=['GET'])
def get_dataset_samples():
    """Get samples from a dataset"""
    sync_custom_datasets()
    dataset_id = request.args.get('dataset_id')
    limit = request.args.get('limit', default=50, type=int)
    
    if not dataset_id or dataset_id not in AVAILABLE_DATASETS:
        return jsonify({'error': 'Invalid dataset'}), 400
    
    if not HAS_DATASETS and not (dataset_id == "recorded-audio" or dataset_id.startswith("custom-")):
        return jsonify({'error': 'datasets library not available'}), 400
    
    try:
        endpoint_started = time.perf_counter()
        log_event("ENDPOINT dataset-samples start", dataset_id=dataset_id, limit=limit)
        samples = load_dataset_samples(dataset_id, max_samples=limit)
        if samples is None:
            return jsonify({'error': 'Failed to load dataset'}), 500
        log_event("ENDPOINT dataset-samples done", dataset_id=dataset_id, count=len(samples), elapsed_s=f"{time.perf_counter() - endpoint_started:.2f}")
        return jsonify({
            'dataset_id': dataset_id,
            'count': len(samples),
            'samples': [
                {
                    'index': sample['index'],
                    'source_index': sample.get('source_index'),
                    'sampling_rate': sample.get('sampling_rate', 16000),
                    'reference': sample.get('reference', ''),
                    'created_at': sample.get('created_at', ''),
                    'audio_url': f"/dataset-audio?dataset_id={dataset_id}&index={sample['index']}&limit={limit}",
                }
                for sample in samples
            ]
        })
    except Exception as e:
        log_event("ENDPOINT dataset-samples failed", dataset_id=dataset_id, error=repr(e))
        return jsonify({'error': str(e)}), 500

@app.route('/dataset-audio', methods=['GET'])
def get_dataset_audio():
    sync_custom_datasets()
    dataset_id = request.args.get('dataset_id')
    sample_index = request.args.get('index', type=int)
    limit = request.args.get('limit', default=50, type=int)

    if not dataset_id or dataset_id not in AVAILABLE_DATASETS or sample_index is None:
        return jsonify({'error': 'Invalid dataset audio request'}), 400

    try:
        endpoint_started = time.perf_counter()
        log_event("ENDPOINT dataset-audio start", dataset_id=dataset_id, sample_index=sample_index, limit=limit)
        sample = get_dataset_sample(dataset_id, sample_index, max_samples=limit)
        audio, sr = sample_to_audio(sample)
        log_event("ENDPOINT dataset-audio ready", dataset_id=dataset_id, sample_index=sample_index, elapsed_s=f"{time.perf_counter() - endpoint_started:.2f}", sample_rate=sr, samples=len(audio))
        return send_file(
            audio_to_wav_buffer(audio, sr),
            mimetype="audio/wav",
            download_name=f"{dataset_id}-{sample_index}.wav",
        )
    except Exception as e:
        log_event("ENDPOINT dataset-audio failed", dataset_id=dataset_id, sample_index=sample_index, error=repr(e))
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/progress', methods=['GET'])
def get_progress():
    """Get transcription progress and resources"""
    resources = get_system_resources()
    
    eta = None
    if progress_data['processing'] and progress_data['start_time']:
        elapsed = time.time() - progress_data['start_time']
        if progress_data['progress'] > 0:
            total_time = (elapsed / progress_data['progress']) * 100
            eta = max(0, int(total_time - elapsed))
    
    return jsonify({
        'progress': progress_data['progress'],
        'status': progress_data['status'],
        'processing': progress_data['processing'],
        'eta_seconds': eta,
        'resources': resources
    })

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """AJAX endpoint for transcription"""
    log_event("ENDPOINT transcribe hit")
    if 'audio' not in request.files:
        log_event("ENDPOINT transcribe rejected", reason="no_audio_file")
        return jsonify({'success': False, 'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'success': False, 'error': 'No audio file selected'}), 400
    
    model_id = request.form.get('model_id', default_model_id)
    reference_text = request.form.get('reference', '').strip()
    target_dataset_id = request.form.get('target_dataset_id', 'recorded-audio').strip() or 'recorded-audio'
    sync_custom_datasets()

    if model_id not in AVAILABLE_MODELS:
        return jsonify({'success': False, 'error': 'Invalid model'}), 400
    if target_dataset_id != '__session__' and (
        target_dataset_id not in AVAILABLE_DATASETS or not (target_dataset_id == 'recorded-audio' or target_dataset_id.startswith('custom-'))
    ):
        return jsonify({'success': False, 'error': 'Invalid target dataset'}), 400
    
    try:
        endpoint_started = time.perf_counter()
        log_event("ENDPOINT transcribe upload read", filename=audio_file.filename, model_id=model_id)
        audio_data = audio_file.read()
        log_event("ENDPOINT transcribe upload ready", filename=audio_file.filename, bytes=len(audio_data), model_id=model_id)
        transcription = transcribe_audio(audio_data, model_id)
        if is_error_result(transcription):
            return jsonify({'success': False, 'error': transcription}), 500
        if target_dataset_id != '__session__':
            save_local_sample(audio_data, reference_text or transcription, audio_file.filename, target_dataset_id=target_dataset_id)
        metrics = calculate_metrics(reference_text, transcription) if reference_text else None
        log_event("ENDPOINT transcribe done", model_id=model_id, elapsed_s=f"{time.perf_counter() - endpoint_started:.2f}", chars=len(transcription or ""))
        return jsonify({
            'success': True,
            'transcription': transcription,
            'reference': reference_text,
            'metrics': metrics,
            'model_id': model_id,
        })
    except Exception as e:
        log_event("ENDPOINT transcribe failed", model_id=model_id, error=repr(e))
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/compare', methods=['POST'])
def compare():
    """Compare transcription with reference text and calculate metrics"""
    reference_text = request.form.get('reference', '')
    model_id = request.form.get('model_id', default_model_id)
    
    if not reference_text:
        return jsonify({'success': False, 'error': 'No reference text provided'}), 400
    
    try:
        dataset_id = request.form.get('dataset_id')
        sample_index = request.form.get('sample_index', type=int)
        sample_limit = request.form.get('sample_limit', default=50, type=int)

        if dataset_id and sample_index is not None:
            sample = get_dataset_sample(dataset_id, sample_index, max_samples=sample_limit)
            transcribe_fn = load_model(model_id)
            audio, sr = sample_to_audio(sample)
            hypothesis = transcribe_fn(audio, sr)
            log_inference_result(model_id, f"dataset:{dataset_id}:{sample_index}", hypothesis)
        else:
            if 'audio' not in request.files:
                return jsonify({'success': False, 'error': 'No audio file provided'}), 400
            audio_file = request.files['audio']
            audio_data = audio_file.read()
            hypothesis = transcribe_audio(audio_data, model_id)

        if is_error_result(hypothesis):
            return jsonify({'success': False, 'error': hypothesis}), 500
        
        # Calculate metrics
        metrics = calculate_metrics(reference_text, hypothesis)
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'model_id': model_id
        })
    except Exception as e:
        print("Compare failed:")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/transcribe-sample', methods=['POST'])
def transcribe_sample():
    """Transcribe an existing dataset or recorded sample."""
    global progress_data
    sync_custom_datasets()
    payload = request.get_json(silent=True) or request.form
    model_id = payload.get('model_id', default_model_id)
    dataset_id = payload.get('dataset_id')

    try:
        sample_index = int(payload.get('sample_index'))
        sample_limit = int(payload.get('sample_limit', 50))
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': 'sample_index is required'}), 400

    if model_id not in AVAILABLE_MODELS:
        return jsonify({'success': False, 'error': 'Invalid model'}), 400
    if not dataset_id or dataset_id not in AVAILABLE_DATASETS:
        return jsonify({'success': False, 'error': 'Invalid dataset'}), 400

    try:
        endpoint_started = time.perf_counter()
        progress_data['processing'] = True
        progress_data['start_time'] = time.time()
        progress_data['progress'] = 5
        progress_data['status'] = f"Loading sample {sample_index + 1} from {dataset_id}..."
        log_event(
            "ENDPOINT transcribe-sample start",
            model_id=model_id,
            dataset_id=dataset_id,
            sample_index=sample_index,
            sample_limit=sample_limit,
        )
        step_started = time.perf_counter()
        sample = get_dataset_sample(dataset_id, sample_index, max_samples=sample_limit)
        log_event("ENDPOINT transcribe-sample loaded sample", elapsed_s=f"{time.perf_counter() - step_started:.2f}")
        progress_data['progress'] = 20
        progress_data['status'] = "Preparing sample audio..."
        step_started = time.perf_counter()
        audio, sr = sample_to_audio(sample)
        log_event("ENDPOINT transcribe-sample audio ready", elapsed_s=f"{time.perf_counter() - step_started:.2f}", samples=len(audio), sample_rate=sr)
        progress_data['progress'] = 35
        progress_data['status'] = f"Loading model: {AVAILABLE_MODELS[model_id]['name']}"
        step_started = time.perf_counter()
        transcribe_fn = load_model(model_id)
        log_event("ENDPOINT transcribe-sample model ready", elapsed_s=f"{time.perf_counter() - step_started:.2f}")
        progress_data['progress'] = 65
        progress_data['status'] = "Transcribing dataset sample..."
        step_started = time.perf_counter()
        hypothesis = transcribe_fn(audio, sr)
        log_event("ENDPOINT transcribe-sample inference ready", elapsed_s=f"{time.perf_counter() - step_started:.2f}", chars=len(hypothesis or ""))
        log_inference_result(model_id, f"sample:{dataset_id}:{sample_index}", hypothesis)
        reference = sample.get("reference", "")
        metrics = calculate_metrics(reference, hypothesis) if reference else None
        progress_data['progress'] = 100
        progress_data['status'] = "Complete!"
        progress_data['processing'] = False
        log_event("ENDPOINT transcribe-sample done", elapsed_s=f"{time.perf_counter() - endpoint_started:.2f}", model_id=model_id)
        return jsonify({
            'success': True,
            'dataset_id': dataset_id,
            'sample_index': sample_index,
            'reference': reference,
            'transcription': hypothesis,
            'metrics': metrics,
            'model_id': model_id,
        })
    except Exception as e:
        progress_data['processing'] = False
        progress_data['status'] = f"Error: {e}"
        log_event("ENDPOINT transcribe-sample failed", model_id=model_id, dataset_id=dataset_id, sample_index=sample_index, error=repr(e))
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/label-audio', methods=['POST'])
def label_audio():
    """Save uploaded audio with a human-provided transcript label."""
    sync_custom_datasets()
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No audio file provided'}), 400

    label = request.form.get('label', '').strip()
    notes = request.form.get('notes', '').strip()
    target_dataset_id = request.form.get('target_dataset_id', 'recorded-audio').strip() or 'recorded-audio'
    if target_dataset_id not in AVAILABLE_DATASETS or not (target_dataset_id == 'recorded-audio' or target_dataset_id.startswith('custom-')):
        return jsonify({'success': False, 'error': 'Invalid target dataset'}), 400
    if not label:
        return jsonify({'success': False, 'error': 'Label text is required'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'success': False, 'error': 'No audio file selected'}), 400

    try:
        record = save_labeled_audio(audio_file.read(), label, source_name=audio_file.filename, notes=notes, target_dataset_id=target_dataset_id)
        if not record:
            return jsonify({'success': False, 'error': 'Could not save labeled audio'}), 500
        return jsonify({'success': True, 'record': record})
    except Exception as e:
        print("Label audio failed:")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/label-sample', methods=['POST'])
def label_sample():
    """Copy a dataset sample into local storage with a human-provided transcript label."""
    sync_custom_datasets()
    payload = request.get_json(silent=True) or {}
    dataset_id = payload.get('dataset_id')
    label = str(payload.get('label', '')).strip()
    notes = str(payload.get('notes', '')).strip()
    target_dataset_id = str(payload.get('target_dataset_id') or 'recorded-audio').strip()

    if not dataset_id or dataset_id not in AVAILABLE_DATASETS:
        return jsonify({'success': False, 'error': 'Invalid dataset'}), 400
    if target_dataset_id not in AVAILABLE_DATASETS or not (target_dataset_id == 'recorded-audio' or target_dataset_id.startswith('custom-')):
        return jsonify({'success': False, 'error': 'Invalid target dataset'}), 400
    if not label:
        return jsonify({'success': False, 'error': 'Label text is required'}), 400

    try:
        sample_index = int(payload.get('sample_index'))
        sample_limit = int(payload.get('sample_limit', 50))
        record = save_labeled_sample(dataset_id, sample_index, label, sample_limit=sample_limit, notes=notes, target_dataset_id=target_dataset_id)
        if not record:
            return jsonify({'success': False, 'error': 'Could not save labeled sample'}), 500
        return jsonify({'success': True, 'record': record})
    except Exception as e:
        print("Label sample failed:")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/word-variants', methods=['GET', 'POST'])
def word_variants():
    """List or create accepted reference/hypothesis word spelling variants."""
    if request.method == 'GET':
        rows = [
            {"reference_word": ref, "hypothesis_word": hyp}
            for ref, hyp in sorted(load_word_variants())
        ]
        return jsonify({'success': True, 'variants': rows, 'count': len(rows)})

    payload = request.get_json(silent=True) or {}
    try:
        row = save_word_variant(
            payload.get('reference_word', ''),
            payload.get('hypothesis_word', ''),
            source=payload.get('source', ''),
        )
        return jsonify({'success': True, 'variant': row})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/analysis-runs', methods=['GET'])
def analysis_runs():
    """Return saved analysis runs so the frontend can reopen earlier results."""
    limit = min(max(int(request.args.get("limit", 30) or 30), 1), 100)
    rows = [summarize_saved_analysis_run(row) for row in load_saved_analysis_runs(limit=limit)]
    return jsonify({"success": True, "runs": rows})


@app.route('/analysis-runs/<run_id>', methods=['GET'])
def analysis_run_detail(run_id):
    for record in load_saved_analysis_runs(limit=500):
        if record.get("run_id") == run_id:
            return jsonify(record)
    return jsonify({"success": False, "error": "Analysis run not found"}), 404


@app.route('/analyze-batch', methods=['POST'])
def analyze_batch():
    """Run one or more models over one or more dataset sample groups and summarize errors."""
    sync_custom_datasets()
    payload = request.get_json(silent=True) or {}
    model_ids = payload.get("model_ids") or [payload.get('model_id', default_model_id)]
    dataset_ids = payload.get("dataset_ids") or [payload.get('dataset_id')]
    model_ids = [model_id for model_id in model_ids if model_id]
    dataset_ids = [dataset_id for dataset_id in dataset_ids if dataset_id]
    limit = min(max(int(payload.get('limit', 10) or 10), 1), 100)
    sample_pool = min(max(int(payload.get('sample_pool', max(limit, 50)) or 50), limit), 300)
    add_noise = bool(payload.get('add_noise', False))
    snr_db = float(payload.get('snr_db', 20.0) or 20.0)
    mix_datasets = bool(payload.get("mix_datasets", False))

    invalid_models = [model_id for model_id in model_ids if model_id not in AVAILABLE_MODELS]
    invalid_datasets = [dataset_id for dataset_id in dataset_ids if dataset_id not in AVAILABLE_DATASETS]
    if invalid_models:
        return jsonify({'success': False, 'error': f"Invalid model(s): {', '.join(invalid_models)}"}), 400
    if invalid_datasets or not dataset_ids:
        return jsonify({'success': False, 'error': f"Invalid dataset(s): {', '.join(invalid_datasets) or 'none'}"}), 400

    try:
        analysis_started = time.perf_counter()
        log_event(
            "ANALYSIS start",
            models=len(model_ids),
            datasets=len(dataset_ids),
            limit=limit,
            sample_pool=sample_pool,
            mix=mix_datasets,
        )

        dataset_samples = {}
        for dataset_id in dataset_ids:
            samples = load_dataset_samples(dataset_id, max_samples=sample_pool)
            if samples is None:
                return jsonify({'success': False, 'error': f'Failed to load dataset: {dataset_id}'}), 500
            dataset_samples[dataset_id] = samples

        runs = []
        for model_id in model_ids:
            if mix_datasets:
                mixed_samples = []
                for dataset_id, samples in dataset_samples.items():
                    for sample in samples[:limit]:
                        mixed_samples.append((dataset_id, sample))
                runs.append(run_analysis_batch_safe(
                    model_id=model_id,
                    dataset_id="mixed",
                    dataset_name="Mixed datasets",
                    sample_items=mixed_samples[: max(limit, limit * len(dataset_ids))],
                    sample_pool=sample_pool,
                    add_noise=add_noise,
                    snr_db=snr_db,
                ))
            else:
                for dataset_id in dataset_ids:
                    runs.append(run_analysis_batch_safe(
                        model_id=model_id,
                        dataset_id=dataset_id,
                        dataset_name=AVAILABLE_DATASETS[dataset_id].get("name", dataset_id),
                        sample_items=[(dataset_id, sample) for sample in dataset_samples[dataset_id][:limit]],
                        sample_pool=sample_pool,
                        add_noise=add_noise,
                        snr_db=snr_db,
                    ))

        successful_runs = [
            run for run in runs
            if run.get("status") != "failed" and run.get("summary", {}).get("count", 0) > 0
        ]
        failed_runs = [run for run in runs if run.get("status") == "failed"]
        all_results = [row for run in successful_runs for row in run["results"]]

        rankings = sorted(
            [
                {
                    "model_id": run["model_id"],
                    "dataset_id": run["dataset_id"],
                    "status": run.get("status", "ok"),
                    "error": run.get("error"),
                    "avg_wer": run["summary"]["avg_wer"],
                    "avg_cer": run["summary"]["avg_cer"],
                    "count": run["summary"]["count"],
                    "avg_elapsed_seconds": run["summary"]["avg_elapsed_seconds"],
                }
                for run in successful_runs
            ],
            key=lambda row: (row["avg_wer"], row["avg_cer"], row["avg_elapsed_seconds"]),
        )
        rankings.extend(
            {
                "model_id": run["model_id"],
                "dataset_id": run["dataset_id"],
                "status": "failed",
                "error": run.get("error"),
                "avg_wer": 1.0,
                "avg_cer": 1.0,
                "count": 0,
                "avg_elapsed_seconds": 0.0,
            }
            for run in failed_runs
        )
        combined_metrics = [
            {
                "word_diff": [],
                "char_diff": [],
                "word_errors": [{"reference": row.get("reference", ""), "hypothesis": row.get("hypothesis", ""), "type": "sample"}],
            }
            for row in all_results
        ]

        cache_hits = sum(int(run["summary"].get("cache_hits", 0)) for run in runs)
        cache_misses = sum(int(run["summary"].get("cache_misses", 0)) for run in runs)
        response_payload = {
            "success": True,
            "model_id": model_ids[0],
            "dataset_id": dataset_ids[0],
            "model_ids": model_ids,
            "dataset_ids": dataset_ids,
            "settings": {
                "limit": limit,
                "sample_pool": sample_pool,
                "add_noise": add_noise,
                "snr_db": snr_db,
                "mix_datasets": mix_datasets,
            },
            "summary": {
                "count": len(all_results),
                "runs": len(runs),
                "successful_runs": len(successful_runs),
                "failed_runs": len(failed_runs),
                "avg_wer": float(np.mean([row["wer"] for row in all_results])) if all_results else 0.0,
                "avg_cer": float(np.mean([row["cer"] for row in all_results])) if all_results else 0.0,
                "avg_duration_seconds": float(np.mean([row["duration_seconds"] for row in all_results])) if all_results else 0.0,
                "avg_elapsed_seconds": float(np.mean([row["elapsed_seconds"] for row in all_results])) if all_results else 0.0,
                "best_run": rankings[0] if rankings else None,
                "worst_run": rankings[-1] if rankings else None,
                "rankings": rankings,
                "elapsed_seconds": round(time.perf_counter() - analysis_started, 2),
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
            },
            "runs": runs,
            "results": all_results,
        }
        saved = save_analysis_run(response_payload)
        log_event("ANALYSIS saved", run_id=saved["run_id"], cache_hits=cache_hits, cache_misses=cache_misses)
        return jsonify(saved)
    except Exception as e:
        log_event("ANALYSIS failed", error=repr(e))
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


def run_analysis_batch_safe(model_id, dataset_id, dataset_name, sample_items, sample_pool, add_noise, snr_db):
    try:
        return run_analysis_batch(
            model_id=model_id,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            sample_items=sample_items,
            sample_pool=sample_pool,
            add_noise=add_noise,
            snr_db=snr_db,
        )
    except Exception as exc:
        log_event("ANALYSIS run failed; continuing", model_id=model_id, dataset_id=dataset_id, error=repr(exc))
        traceback.print_exc()
        return make_failed_analysis_run(model_id, dataset_id, dataset_name, exc)


def make_failed_analysis_run(model_id, dataset_id, dataset_name, exc):
    message = str(exc) or repr(exc)
    return {
        "model_id": model_id,
        "model_name": AVAILABLE_MODELS.get(model_id, {}).get("name", model_id),
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "status": "failed",
        "error": message,
        "summary": {
            "count": 0,
            "avg_wer": 1.0,
            "avg_cer": 1.0,
            "avg_duration_seconds": 0.0,
            "avg_elapsed_seconds": 0.0,
            "best_sample": None,
            "worst_sample": None,
            "elapsed_seconds": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "error": message,
        },
        "results": [],
    }


def run_analysis_batch(model_id, dataset_id, dataset_name, sample_items, sample_pool, add_noise, snr_db):
    run_started = time.perf_counter()
    log_event("ANALYSIS run start", model_id=model_id, dataset_id=dataset_id, samples=len(sample_items))
    transcribe_fn = None
    result_cache = read_analysis_cache()
    cache_dirty = False
    cache_hits = 0
    cache_misses = 0
    results = []
    metric_rows = []
    durations = []

    for order, (source_dataset_id, sample) in enumerate(sample_items):
        audio, sr = sample_to_audio(sample)
        reference = sample.get("reference", "")
        cache_key = analysis_cache_key(model_id, source_dataset_id, sample, add_noise, snr_db)
        if add_noise:
            audio = add_noise_to_audio(audio, snr_db=snr_db, seed=analysis_seed_from_key(cache_key))
        duration = len(audio) / float(sr or 16000)
        cached = result_cache.get(cache_key)
        if cached:
            cache_hits += 1
            hypothesis = cached.get("hypothesis", "")
            elapsed = 0.0
            cached_elapsed = float(cached.get("elapsed_seconds", 0.0) or 0.0)
            log_event("ANALYSIS cache hit", model_id=model_id, dataset_id=source_dataset_id, index=sample.get("index"))
        else:
            cache_misses += 1
            if transcribe_fn is None:
                transcribe_fn = load_model(model_id)
            started = time.perf_counter()
            hypothesis = transcribe_fn(audio, sr)
            elapsed = time.perf_counter() - started
            cached_elapsed = None
            result_cache[cache_key] = {
                "model_id": model_id,
                "dataset_id": source_dataset_id,
                "index": sample.get("index"),
                "source_index": sample.get("source_index"),
                "hypothesis": hypothesis,
                "elapsed_seconds": round(elapsed, 4),
                "duration_seconds": round(duration, 4),
                "created_at": datetime.utcnow().isoformat() + "Z",
                "noise": bool(add_noise),
                "snr_db": round(float(snr_db), 2) if add_noise else None,
            }
            cache_dirty = True
            log_event("ANALYSIS cache miss stored", model_id=model_id, dataset_id=source_dataset_id, index=sample.get("index"))
        metrics = calculate_metrics(reference, hypothesis)
        metric_rows.append(metrics)
        durations.append(duration)
        result = {
            "run_order": order,
            "model_id": model_id,
            "dataset_id": source_dataset_id,
            "index": sample.get("index"),
            "source_index": sample.get("source_index"),
            "reference": reference,
            "hypothesis": hypothesis,
            "wer": metrics["wer"],
            "cer": metrics["cer"],
            "duration_seconds": round(duration, 2),
            "elapsed_seconds": round(elapsed, 2),
            "cached_elapsed_seconds": round(cached_elapsed, 2) if cached_elapsed is not None else None,
            "cache_hit": bool(cached),
            "cache_key": cache_key,
            "rtf": round(elapsed / duration, 2) if duration else None,
            "word_errors": len(metrics.get("word_errors", [])),
            "audio_url": f"/dataset-audio?dataset_id={source_dataset_id}&index={sample.get('index')}&limit={sample_pool}",
        }
        results.append(result)
        log_inference_result(model_id, f"analysis:{source_dataset_id}:{sample.get('index')}", hypothesis)

    if cache_dirty:
        write_analysis_cache()

    if not results:
        summary = {
            "count": 0,
            "avg_wer": 0.0,
            "avg_cer": 0.0,
            "avg_duration_seconds": 0.0,
            "avg_elapsed_seconds": 0.0,
            "best_sample": None,
            "worst_sample": None,
            "elapsed_seconds": round(time.perf_counter() - run_started, 2),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
        }
    else:
        summary = {
            "count": len(results),
            "avg_wer": float(np.mean([row["wer"] for row in results])),
            "avg_cer": float(np.mean([row["cer"] for row in results])),
            "avg_duration_seconds": float(np.mean(durations)) if durations else 0.0,
            "avg_elapsed_seconds": float(np.mean([row["elapsed_seconds"] for row in results])),
            "best_sample": min(results, key=lambda row: (row["wer"], row["cer"])),
            "worst_sample": max(results, key=lambda row: (row["wer"], row["cer"])),
            "elapsed_seconds": round(time.perf_counter() - run_started, 2),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            **summarize_word_patterns(metric_rows),
        }
    log_event(
        "ANALYSIS run done",
        model_id=model_id,
        dataset_id=dataset_id,
        elapsed_s=summary["elapsed_seconds"],
        cache_hits=cache_hits,
        cache_misses=cache_misses,
    )
    return {
        "model_id": model_id,
        "model_name": AVAILABLE_MODELS[model_id].get("name", model_id),
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "summary": summary,
        "results": results,
    }

if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0").lower() in {"1", "true", "yes", "on"}

    print("\n" + "="*60)
    print("Starting Speech Recognition Web App")
    print("="*60)
    print(f"Open your browser and go to: http://{host}:{port}")
    print("="*60 + "\n")

    app.run(debug=debug, use_reloader=False, host=host, port=port)
