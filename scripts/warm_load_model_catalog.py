from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import torch

from download_hf_assets import all_model_catalog, load_env_file


ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Warm-load downloaded ASR models one at a time to verify the local cache."
    )
    parser.add_argument(
        "--include-experimental-models",
        action="store_true",
        help="Include the experimental model catalog.",
    )
    parser.add_argument(
        "--model-id",
        action="append",
        default=[],
        help="Warm-load only a specific model id. May be repeated.",
    )
    return parser.parse_args()


def unload(*objects: object) -> None:
    del objects
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def local_snapshot_path(repo_id: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        cache_dir=os.getenv("HF_HUB_CACHE"),
        local_files_only=True,
    )


def warm_load_transformers_ctc(repo_id: str) -> None:
    from transformers import AutoModelForCTC, AutoProcessor

    snapshot_path = local_snapshot_path(repo_id)
    processor = AutoProcessor.from_pretrained(snapshot_path, local_files_only=True)
    model = AutoModelForCTC.from_pretrained(snapshot_path, local_files_only=True)
    unload(model, processor)


def warm_load_whisper(repo_id: str) -> None:
    from transformers import (
        WhisperFeatureExtractor,
        WhisperForConditionalGeneration,
        WhisperProcessor,
        WhisperTokenizer,
    )

    snapshot_path = local_snapshot_path(repo_id)
    try:
        processor = WhisperProcessor.from_pretrained(snapshot_path, local_files_only=True)
    except OSError:
        config_path = Path(snapshot_path) / "config.json"
        base_processor_repo = "openai/whisper-large-v3-turbo"
        if config_path.exists():
            config_payload = json.loads(config_path.read_text(encoding="utf-8"))
            base_processor_repo = config_payload.get("_name_or_path") or base_processor_repo
        try:
            base_processor_path = local_snapshot_path(base_processor_repo)
        except Exception:
            base_processor_path = local_snapshot_path("openai/whisper-large-v3-turbo")

        processor_config = Path(snapshot_path) / "processor_config.json"
        feature_config = {}
        if processor_config.exists():
            payload = json.loads(processor_config.read_text(encoding="utf-8"))
            feature_config = payload.get("feature_extractor", {})
            feature_config.pop("feature_extractor_type", None)
        feature_extractor = (
            WhisperFeatureExtractor(**feature_config)
            if feature_config
            else WhisperFeatureExtractor.from_pretrained(base_processor_path, local_files_only=True)
        )
        try:
            tokenizer = WhisperTokenizer.from_pretrained(snapshot_path, local_files_only=True)
        except Exception:
            tokenizer = WhisperTokenizer.from_pretrained(base_processor_path, local_files_only=True)
        processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    model = WhisperForConditionalGeneration.from_pretrained(snapshot_path, local_files_only=True)
    unload(model, processor)


def warm_load_speechbrain(repo_id: str, model_id: str) -> None:
    try:
        from speechbrain.inference.ASR import EncoderASR
    except ImportError:
        from speechbrain.pretrained import EncoderASR

    snapshot_path = local_snapshot_path(repo_id)
    savedir = ROOT_DIR / "pretrained_models" / model_id
    model = EncoderASR.from_hparams(
        source=snapshot_path,
        savedir=str(savedir),
        run_opts={"device": "cpu"},
    )
    unload(model)


def main() -> int:
    load_env_file()
    args = parse_args()

    catalog = all_model_catalog()
    if not args.include_experimental_models:
        catalog = [asset for asset in catalog if not asset.required_for.startswith("experimental")]

    if args.model_id:
        wanted = set(args.model_id)
        by_id = {asset.id: asset for asset in all_model_catalog()}
        missing = sorted(wanted - set(by_id.keys()))
        if missing:
            print(f"[warm-load] Unknown model id(s): {', '.join(missing)}")
            return 2
        catalog = [by_id[model_id] for model_id in args.model_id]

    print(f"[warm-load] HF_HOME={os.getenv('HF_HOME', '<default>')}")
    failures = 0
    for asset in catalog:
        print(f"[warm-load] {asset.id} ({asset.framework})")
        try:
            if asset.framework in {"ctc-transformers", "transformers-mms-ctc"}:
                warm_load_transformers_ctc(asset.hf_path)
            elif asset.framework in {"transformers", "transformers-whisper", "transformers-base"}:
                warm_load_whisper(asset.hf_path)
            elif asset.framework == "speechbrain":
                warm_load_speechbrain(asset.hf_path, asset.id)
            else:
                print(f"[warm-load] skipped loader check for {asset.id}: external/adapter runtime")
                continue
            print(f"[warm-load] ok {asset.id}")
        except Exception as exc:
            failures += 1
            print(f"[warm-load] failed {asset.id}: {exc}")

    if failures:
        print(f"[warm-load] Finished with {failures} failure(s).")
        return 1
    print("[warm-load] All supported loaders verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
