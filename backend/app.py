import torch
import torchaudio
from transformers import (
    AutoProcessor, AutoModelForCTC, pipeline, 
    WhisperForConditionalGeneration, WhisperProcessor
)

# Optional PEFT import - may fail due to version conflicts
try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError as e:
    HAS_PEFT = False
    print(f"Warning: PEFT not available ({e}). LoRA models will be skipped.")

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
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

app = Flask(__name__, template_folder=str(APP_DIR / 'templates'))
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = str(PROJECT_ROOT / 'uploads')
RECORDED_DATASET_DIR = PROJECT_ROOT / "data" / "recorded_dataset"
RECORDED_MANIFEST = RECORDED_DATASET_DIR / "manifest.jsonl"
LABEL_MANIFEST = RECORDED_DATASET_DIR / "labels.jsonl"
WORD_VARIANTS_MANIFEST = RECORDED_DATASET_DIR / "accepted_word_variants.jsonl"
FRONTEND_DIST_DIR = PROJECT_ROOT / "frontend" / "dist"
MODEL_CACHE_DIR = PROJECT_ROOT / "models"
DATASET_CACHE_DIR = PROJECT_ROOT / "datasets"

os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR / "hf-cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(MODEL_CACHE_DIR / "transformers"))
os.environ.setdefault("HF_DATASETS_CACHE", str(DATASET_CACHE_DIR / "hf-datasets"))

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
RECORDED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
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
        "transcript_column": "transcription_darija_ar"
    },
    "hafsata-darija-asr": {
        "name": "HAFSATA Darija ASR",
        "hf_path": "HAFSATA/darija-asr-monolingual",
        "split": "train",
        "audio_column": "audio",
        "transcript_column": "sentence"
    },
    "mgb5-moroccan": {
        "name": "MGB-5 Moroccan Arabic",
        "hf_path": "ArabicSpeech/MGB-5",
        "split": "train",
        "audio_column": "audio",
        "transcript_column": "sentence"
    },
    "darija-wiki-audio": {
        "name": "AtlasIA Darija Wiki Audio",
        "hf_path": "atlasia/Moroccan-Darija-Wiki-Audio-Dataset",
        "split": "train",
        "audio_column": "audio",
        "transcript_column": "transcript"
    },
    "snousnou-darija-asr": {
        "name": "Snousnou Moroccan Darija ASR",
        "hf_path": "Snousnou/Moroccan-Darija-ASR",
        "split": "train",
        "audio_column": "audio",
        "transcript_column": "transcription"
    },
    "mohamedmou-dataset-darija": {
        "name": "Mohamedmou DATASET Darija",
        "hf_path": "mohamedmou/DATASET-darija",
        "split": "train",
        "audio_column": "audio",
        "transcript_column": "sentence"
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
        "transcript_column": "sentence"
    },
    "atlasia-moulsot-preview": {
        "name": "AtlasIA MoulSot Preview",
        "hf_path": "atlasia/moulsot-preview",
        "split": "train",
        "audio_column": "audio",
        "transcript_column": "transcript"
    },
    "atlasia-moulsot-full": {
        "name": "AtlasIA MoulSot Full",
        "hf_path": "atlasia/MoulSot-Full",
        "split": "test",
        "audio_column": "audio",
        "transcript_column": "transcript"
    },
    "recorded-audio": {
        "name": "Recorded / Uploaded Audio",
        "type": "local-recordings",
        "audio_column": "audio",
        "transcript_column": "reference"
    }
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# Model cache
model_cache = {}
processor_cache = {}
dataset_cache = {}
word_variant_cache = None

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
        print(f"Loading model: {model_id}...")
        try:
            model_config = AVAILABLE_MODELS[model_id]
            framework = model_config.get("framework", "transformers")
            hf_path = model_config["hf_path"]
            lang_hint = model_config.get("language_hint")
            target_device = "cpu" if model_config.get("force_cpu") else device
            trust_remote_code = bool(model_config.get("trust_remote_code"))
            
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
            
            print(f"OK {model_id} loaded successfully!")
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("WARNING: CUDA out of memory, falling back to CPU...")
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
                print(f"Failed to load model {model_id}:")
                traceback.print_exc()
                raise
        except Exception:
            print(f"Failed to load model {model_id}:")
            traceback.print_exc()
            raise
    
    framework, fn = model_cache[model_id]
    return fn

def _load_transformers_model(model_id: str, hf_path: str, language_hint: str, device: str, trust_remote_code: bool = False):
    """Load a standard Whisper model using transformers pipeline"""
    print(f"Loading Whisper model: {hf_path} on {device}...")
    
    dtype = torch.float16 if device == "cuda" else torch.float32
    dev = 0 if device == "cuda" else -1
    
    asr = pipeline(
        task="automatic-speech-recognition",
        model=hf_path,
        device=dev,
        torch_dtype=dtype,
        chunk_length_s=None,
        trust_remote_code=trust_remote_code,
    )
    
    generate_kwargs = {"task": "transcribe"}
    if language_hint:
        generate_kwargs["language"] = language_hint
    
    def transcribe(audio_array: np.ndarray, sample_rate: int) -> str:
        result = asr(
            {"array": audio_array.astype(np.float32), "sampling_rate": sample_rate},
            generate_kwargs=generate_kwargs,
        )
        text = result["text"].strip()
        if not text and language_hint:
            result = asr(
                {"array": audio_array.astype(np.float32), "sampling_rate": sample_rate},
                generate_kwargs={"task": "transcribe"},
            )
            text = result["text"].strip()
        return text
    
    return transcribe

def _load_ctc_transformers_model(model_id: str, hf_path: str, device: str):
    """Load a standard Transformers CTC ASR model."""
    print(f"Loading CTC model: {hf_path} on {device}...")

    processor = AutoProcessor.from_pretrained(hf_path)
    model = AutoModelForCTC.from_pretrained(hf_path).to(device)
    model.eval()

    def transcribe(audio_array: np.ndarray, sample_rate: int) -> str:
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
        return processor.batch_decode(predicted_ids)[0].strip()

    return transcribe

def _load_peft_transformers_model(model_id: str, hf_path: str, base_model: str, language_hint: str, device: str):
    """Load a PEFT + Transformers model (LoRA adapter)"""
    if not HAS_PEFT:
        raise ImportError("PEFT library not available. Cannot load LoRA model.")
    
    print(f"Loading PEFT model: {hf_path} on {device}...")
    
    use_cuda = device == "cuda"
    dtype = torch.float16 if use_cuda else torch.float32
    dev = 0 if use_cuda else -1
    
    try:
        # Load base model
        print("Loading base model...")
        base = WhisperForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        
        # Load and merge LoRA adapter
        print("Loading and merging LoRA adapter...")
        model = PeftModel.from_pretrained(base, hf_path)
        model = model.merge_and_unload()
    except Exception:
        print("PEFT merge failed:")
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
        result = asr(
            {"array": audio_array.astype(np.float32), "sampling_rate": sample_rate},
            generate_kwargs=generate_kwargs,
        )
        text = result["text"].strip()
        if not text and language_hint:
            result = asr(
                {"array": audio_array.astype(np.float32), "sampling_rate": sample_rate},
                generate_kwargs={"task": "transcribe"},
            )
            text = result["text"].strip()
        return text
    
    return transcribe

def _load_speechbrain_model(model_id: str, hf_path: str, device: str):
    """Load a SpeechBrain EncoderASR model (CTC)"""
    print(f"Loading SpeechBrain model: {hf_path} on {device}...")
    
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
        return str(first).strip()
    
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
        
        # Load model
        progress_data['progress'] = 10
        progress_data['status'] = f"Loading model: {AVAILABLE_MODELS[model_id]['name']}"
        transcribe_fn = load_model(model_id)
        
        # Load audio
        progress_data['progress'] = 25
        progress_data['status'] = "Processing audio..."
        audio, sr = load_audio(audio_data, target_sr=16000)
        
        # Transcribe
        progress_data['progress'] = 60
        progress_data['status'] = "Transcribing..."
        transcription = transcribe_fn(audio, sr)
        log_inference_result(model_id, "upload", transcription)
        
        progress_data['progress'] = 100
        progress_data['status'] = "Complete!"
        progress_data['processing'] = False
        
        # Clean up
        clear_gpu_memory()
        
        return transcription
    except Exception as e:
        progress_data['processing'] = False
        clear_gpu_memory()
        return f"Error: {str(e)}"

def is_error_result(value):
    return isinstance(value, str) and value.startswith("Error:")

def log_inference_result(model_id, source, text):
    preview = (text or "").replace("\n", " ")[:120]
    print(f"Inference result: model={model_id}, source={source}, chars={len(text or '')}, text={ascii(preview)}")

def save_recorded_sample(audio_data, reference, source_name="recording"):
    """Save uploaded/recorded audio as a local dataset sample."""
    try:
        audio, sr = load_audio(audio_data, target_sr=16000)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_source = Path(secure_filename(source_name or "recording") or "recording").stem
        stem = f"{timestamp}_{safe_source}"
        wav_path = RECORDED_DATASET_DIR / f"{stem}.wav"
        sf.write(wav_path, audio, sr, format="WAV")
        record = {
            "file": wav_path.name,
            "reference": reference or "",
            "source": source_name or "",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        with RECORDED_MANIFEST.open("a", encoding="utf-8") as f:
            import json
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        dataset_cache.clear()
        print(f"Saved recorded sample: {wav_path}")
        return record
    except Exception:
        print("Failed to save recorded sample:")
        traceback.print_exc()
        return None

def save_labeled_audio(audio_data, label, source_name="labeled-audio", notes=""):
    """Save human-labeled audio as part of the local recorded dataset."""
    record = save_recorded_sample(audio_data, label, source_name=source_name)
    if not record:
        return None

    label_record = {
        **record,
        "label": label or "",
        "notes": notes or "",
        "label_type": "human",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with LABEL_MANIFEST.open("a", encoding="utf-8") as f:
        import json
        f.write(json.dumps(label_record, ensure_ascii=False) + "\n")
    return label_record

def save_labeled_sample(dataset_id, sample_index, label, sample_limit=50, notes=""):
    """Copy a selected dataset sample into the local labeled dataset with a human label."""
    sample = get_dataset_sample(dataset_id, int(sample_index), max_samples=int(sample_limit))
    audio, sr = sample_to_audio(sample)
    audio_buffer = audio_to_wav_buffer(audio, sr).read()
    source = f"{dataset_id}_{sample_index}_human_label.wav"
    return save_labeled_audio(audio_buffer, label, source_name=source, notes=notes)

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

def load_recorded_samples(max_samples=100):
    if not RECORDED_MANIFEST.exists():
        return []
    import json
    rows = []
    with RECORDED_MANIFEST.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            wav_path = RECORDED_DATASET_DIR / record.get("file", "")
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

def load_dataset_samples(dataset_id: str, max_samples: int = 100):
    """Load samples from a dataset"""
    cache_key = f"{dataset_id}:{max_samples}"
    if cache_key in dataset_cache:
        return dataset_cache[cache_key]

    if dataset_id == "recorded-audio":
        samples = load_recorded_samples(max_samples=max_samples)
        dataset_cache[cache_key] = samples
        return samples

    if not HAS_DATASETS:
        return None
    
    try:
        dataset_config = AVAILABLE_DATASETS[dataset_id]
        print(f"Loading dataset: {dataset_id}...")
        
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
        return samples
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def get_dataset_sample(dataset_id: str, sample_index: int, max_samples: int = 100):
    samples = load_dataset_samples(dataset_id, max_samples=max_samples)
    if samples is None:
        raise ValueError("Failed to load dataset")
    if sample_index < 0 or sample_index >= len(samples):
        raise IndexError(f"Sample {sample_index} is outside loaded range 0-{len(samples) - 1}")
    return samples[sample_index]

def sample_to_audio(sample):
    if sample.get("audio_path"):
        audio, sr = sf.read(sample["audio_path"], dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = np.mean(audio, axis=1)
        return normalize_audio_array(audio), int(sr)
    return normalize_audio_array(sample["audio_array"]), int(sample.get("sampling_rate", 16000))

def audio_to_wav_buffer(audio, sr):
    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(audio, dtype=np.float32), int(sr), format="WAV")
    buffer.seek(0)
    return buffer

def add_noise_to_audio(audio, snr_db=20.0):
    """Add Gaussian noise at the requested signal-to-noise ratio."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio

    signal_power = float(np.mean(np.square(audio)))
    if signal_power <= 1e-12:
        return audio

    snr_linear = 10 ** (float(snr_db) / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0.0, np.sqrt(noise_power), size=audio.shape).astype(np.float32)
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

# Load the default model on startup
print(f"Device: {device}")
print("Loading default model...")
default_model_id = list(AVAILABLE_MODELS.keys())[0]
try:
    load_model(default_model_id)
except Exception as e:
    print(f"Warning: Could not preload default model: {e}")

@app.route('/')
def index():
    react_index = FRONTEND_DIST_DIR / "index.html"
    if react_index.exists():
        return send_file(react_index)
    return render_template('index.html')

@app.route('/assets/<path:filename>')
def frontend_assets(filename):
    return send_from_directory(FRONTEND_DIST_DIR / "assets", filename)

@app.route('/models', methods=['GET'])
def get_models():
    """Get list of available models"""
    models_list = [
        {
            'id': model_id,
            'name': config['name'],
            'framework': config.get('framework', 'transformers'),
            'min_vram_gb': config.get('min_vram_gb', 0)
        }
        for model_id, config in AVAILABLE_MODELS.items()
    ]
    return jsonify({
        'models': models_list,
        'default': default_model_id
    })

@app.route('/datasets', methods=['GET'])
def get_datasets():
    """Get list of available datasets"""
    datasets_list = [
        {
            'id': dataset_id,
            'name': config['name']
        }
        for dataset_id, config in AVAILABLE_DATASETS.items()
    ]
    return jsonify({
        'datasets': datasets_list,
        'available': HAS_DATASETS
    })

@app.route('/dataset-samples', methods=['GET'])
def get_dataset_samples():
    """Get samples from a dataset"""
    dataset_id = request.args.get('dataset_id')
    limit = request.args.get('limit', default=50, type=int)
    
    if not dataset_id or dataset_id not in AVAILABLE_DATASETS:
        return jsonify({'error': 'Invalid dataset'}), 400
    
    if not HAS_DATASETS:
        return jsonify({'error': 'datasets library not available'}), 400
    
    try:
        samples = load_dataset_samples(dataset_id, max_samples=limit)
        if samples is None:
            return jsonify({'error': 'Failed to load dataset'}), 500
        
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
        print(f"Error getting dataset samples: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/dataset-audio', methods=['GET'])
def get_dataset_audio():
    dataset_id = request.args.get('dataset_id')
    sample_index = request.args.get('index', type=int)
    limit = request.args.get('limit', default=50, type=int)

    if not dataset_id or dataset_id not in AVAILABLE_DATASETS or sample_index is None:
        return jsonify({'error': 'Invalid dataset audio request'}), 400

    try:
        sample = get_dataset_sample(dataset_id, sample_index, max_samples=limit)
        audio, sr = sample_to_audio(sample)
        return send_file(
            audio_to_wav_buffer(audio, sr),
            mimetype="audio/wav",
            download_name=f"{dataset_id}-{sample_index}.wav",
        )
    except Exception as e:
        print("Dataset audio failed:")
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
    print("Transcribe endpoint hit")
    if 'audio' not in request.files:
        print("Transcribe rejected: no audio file")
        return jsonify({'success': False, 'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'success': False, 'error': 'No audio file selected'}), 400
    
    # Get model selection from frontend
    model_id = request.form.get('model_id', default_model_id)
    
    try:
        print(f"Transcribing upload: filename={audio_file.filename!r}, model={model_id}")
        audio_data = audio_file.read()
        transcription = transcribe_audio(audio_data, model_id)
        if is_error_result(transcription):
            return jsonify({'success': False, 'error': transcription}), 500
        save_recorded_sample(audio_data, transcription, audio_file.filename)
        return jsonify({'success': True, 'transcription': transcription})
    except Exception as e:
        print("Transcribe failed:")
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
        sample = get_dataset_sample(dataset_id, sample_index, max_samples=sample_limit)
        audio, sr = sample_to_audio(sample)
        transcribe_fn = load_model(model_id)
        hypothesis = transcribe_fn(audio, sr)
        log_inference_result(model_id, f"sample:{dataset_id}:{sample_index}", hypothesis)
        reference = sample.get("reference", "")
        metrics = calculate_metrics(reference, hypothesis) if reference else None
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
        print("Sample transcription failed:")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/label-audio', methods=['POST'])
def label_audio():
    """Save uploaded audio with a human-provided transcript label."""
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No audio file provided'}), 400

    label = request.form.get('label', '').strip()
    notes = request.form.get('notes', '').strip()
    if not label:
        return jsonify({'success': False, 'error': 'Label text is required'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'success': False, 'error': 'No audio file selected'}), 400

    try:
        record = save_labeled_audio(audio_file.read(), label, source_name=audio_file.filename, notes=notes)
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
    payload = request.get_json(silent=True) or {}
    dataset_id = payload.get('dataset_id')
    label = str(payload.get('label', '')).strip()
    notes = str(payload.get('notes', '')).strip()

    if not dataset_id or dataset_id not in AVAILABLE_DATASETS:
        return jsonify({'success': False, 'error': 'Invalid dataset'}), 400
    if not label:
        return jsonify({'success': False, 'error': 'Label text is required'}), 400

    try:
        sample_index = int(payload.get('sample_index'))
        sample_limit = int(payload.get('sample_limit', 50))
        record = save_labeled_sample(dataset_id, sample_index, label, sample_limit=sample_limit, notes=notes)
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

@app.route('/analyze-batch', methods=['POST'])
def analyze_batch():
    """Run a model over a group of dataset samples and summarize errors."""
    payload = request.get_json(silent=True) or {}
    model_id = payload.get('model_id', default_model_id)
    dataset_id = payload.get('dataset_id')
    limit = min(max(int(payload.get('limit', 10) or 10), 1), 50)
    sample_pool = min(max(int(payload.get('sample_pool', max(limit, 50)) or 50), limit), 200)
    add_noise = bool(payload.get('add_noise', False))
    snr_db = float(payload.get('snr_db', 20.0) or 20.0)

    if model_id not in AVAILABLE_MODELS:
        return jsonify({'success': False, 'error': 'Invalid model'}), 400
    if not dataset_id or dataset_id not in AVAILABLE_DATASETS:
        return jsonify({'success': False, 'error': 'Invalid dataset'}), 400

    try:
        samples = load_dataset_samples(dataset_id, max_samples=sample_pool)
        if samples is None:
            return jsonify({'success': False, 'error': 'Failed to load dataset'}), 500

        transcribe_fn = load_model(model_id)
        results = []
        metric_rows = []
        durations = []

        for sample in samples[:limit]:
            audio, sr = sample_to_audio(sample)
            if add_noise:
                audio = add_noise_to_audio(audio, snr_db=snr_db)
            duration = len(audio) / float(sr or 16000)
            started = time.time()
            hypothesis = transcribe_fn(audio, sr)
            elapsed = time.time() - started
            reference = sample.get("reference", "")
            metrics = calculate_metrics(reference, hypothesis)
            metric_rows.append(metrics)
            durations.append(duration)
            results.append({
                "index": sample.get("index"),
                "source_index": sample.get("source_index"),
                "reference": reference,
                "hypothesis": hypothesis,
                "wer": metrics["wer"],
                "cer": metrics["cer"],
                "duration_seconds": round(duration, 2),
                "elapsed_seconds": round(elapsed, 2),
                "word_errors": len(metrics.get("word_errors", [])),
                "audio_url": f"/dataset-audio?dataset_id={dataset_id}&index={sample.get('index')}&limit={sample_pool}",
            })
            log_inference_result(model_id, f"batch:{dataset_id}:{sample.get('index')}", hypothesis)

        if not results:
            return jsonify({'success': False, 'error': 'No samples available for analysis'}), 400

        avg_wer = float(np.mean([row["wer"] for row in results]))
        avg_cer = float(np.mean([row["cer"] for row in results]))
        best = min(results, key=lambda row: (row["wer"], row["cer"]))
        worst = max(results, key=lambda row: (row["wer"], row["cer"]))
        patterns = summarize_word_patterns(metric_rows)

        return jsonify({
            "success": True,
            "model_id": model_id,
            "dataset_id": dataset_id,
            "settings": {
                "limit": limit,
                "sample_pool": sample_pool,
                "add_noise": add_noise,
                "snr_db": snr_db,
            },
            "summary": {
                "count": len(results),
                "avg_wer": avg_wer,
                "avg_cer": avg_cer,
                "avg_duration_seconds": float(np.mean(durations)) if durations else 0.0,
                "best_sample": best,
                "worst_sample": worst,
                **patterns,
            },
            "results": results,
        })
    except Exception as e:
        print("Batch analysis failed:")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

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
