"""Collect lightweight dataset metadata for the web UI.

This script avoids downloading dataset rows. It combines local disk inspection
with Hugging Face repository metadata and writes a JSON file consumed by Flask.
"""

from __future__ import annotations

import argparse
import ast
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_APP = PROJECT_ROOT / "backend" / "app.py"
DEFAULT_OUTPUT = PROJECT_ROOT / "backend" / "dataset_metadata.json"
DATASET_CACHE_DIR = PROJECT_ROOT / "datasets"
RECORDED_DATASET_DIR = PROJECT_ROOT / "data" / "recorded_dataset"


def parse_available_datasets() -> dict[str, dict[str, Any]]:
    tree = ast.parse(BACKEND_APP.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "AVAILABLE_DATASETS":
                    return ast.literal_eval(node.value)
    raise RuntimeError(f"AVAILABLE_DATASETS not found in {BACKEND_APP}")


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for file_path in path.rglob("*"):
        if not file_path.is_file():
            continue
        try:
            total += file_path.stat().st_size
        except OSError:
            pass
    return total


def matching_cache_size(hf_path: str) -> int:
    if not hf_path:
        return 0
    parts = [part.lower() for part in hf_path.replace("\\", "/").split("/") if part]
    if not parts or not DATASET_CACHE_DIR.exists():
        return 0

    total = 0
    for path in DATASET_CACHE_DIR.rglob("*"):
        if not path.is_dir():
            continue
        name = path.name.lower()
        if all(part in name for part in parts) or any(part in name for part in parts):
            total += directory_size_bytes(path)
    return total


def recorded_stats() -> dict[str, Any]:
    manifest = RECORDED_DATASET_DIR / "manifest.jsonl"
    sample_count = 0
    if manifest.exists():
        with manifest.open("r", encoding="utf-8") as handle:
            sample_count = sum(1 for line in handle if line.strip())
    return {
        "sample_count": sample_count,
        "local_size_bytes": directory_size_bytes(RECORDED_DATASET_DIR),
        "metadata_source": "local",
    }


def hf_metadata(hf_path: str, token: str | None = None) -> dict[str, Any]:
    if not hf_path:
        return {}

    result: dict[str, Any] = {
        "metadata_source": "huggingface",
        "hf_metadata_error": "",
    }

    try:
        from huggingface_hub import HfApi

        info = HfApi(token=token).dataset_info(hf_path, files_metadata=True)
        siblings = getattr(info, "siblings", None) or []
        file_sizes = [
            getattr(sibling, "size", None)
            for sibling in siblings
            if isinstance(getattr(sibling, "size", None), int)
        ]
        card_data = getattr(info, "cardData", None) or {}

        result.update(
            {
                "downloads": getattr(info, "downloads", None),
                "likes": getattr(info, "likes", None),
                "last_modified": iso_or_none(getattr(info, "lastModified", None)),
                "repo_file_count": len(siblings),
                "repo_size_bytes": sum(file_sizes) if file_sizes else None,
                "license": pick_license(card_data),
                "pretty_name": card_data.get("pretty_name") or card_data.get("dataset_name"),
                "tags": getattr(info, "tags", None) or [],
            }
        )
    except Exception as exc:
        result["hf_metadata_error"] = str(exc)

    try:
        from datasets import get_dataset_config_names, get_dataset_infos, get_dataset_split_names

        configs = get_dataset_config_names(hf_path, token=token)
        result["configs"] = configs
        config_name = configs[0] if configs else None
        try:
          infos = get_dataset_infos(hf_path, token=token)
        except TypeError:
          infos = get_dataset_infos(hf_path)
        info_values = list(infos.values()) if isinstance(infos, dict) else []
        if info_values:
            dataset_info = info_values[0]
            result["description"] = clean_text(getattr(dataset_info, "description", "") or "")
            result["features"] = list(getattr(dataset_info, "features", {}) or {})
            splits = getattr(dataset_info, "splits", None)
            if splits:
                split_rows = {}
                total_rows = 0
                for split_name, split_info in splits.items():
                    rows = getattr(split_info, "num_examples", None)
                    if isinstance(rows, int):
                        split_rows[split_name] = rows
                        total_rows += rows
                result["split_rows"] = split_rows
                result["sample_count"] = total_rows or None
        if config_name:
            try:
                result["available_splits"] = get_dataset_split_names(hf_path, config_name, token=token)
            except TypeError:
                result["available_splits"] = get_dataset_split_names(hf_path, config_name)
    except Exception as exc:
        existing = result.get("hf_metadata_error")
        result["hf_metadata_error"] = f"{existing}; datasets: {exc}".strip("; ")

    return result


def pick_license(card_data: dict[str, Any]) -> str | None:
    value = card_data.get("license")
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if item)
    return None


def clean_text(value: str) -> str:
    return " ".join(value.split())[:500]


def iso_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def bytes_to_gb(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / (1024**3), 4)


def collect(include_remote: bool, token: str | None = None) -> dict[str, Any]:
    registry = parse_available_datasets()
    datasets: dict[str, Any] = {}

    for dataset_id, config in registry.items():
        hf_path = config.get("hf_path", "")
        local_size = directory_size_bytes(RECORDED_DATASET_DIR) if dataset_id == "recorded-audio" else matching_cache_size(hf_path)
        entry: dict[str, Any] = {
            "id": dataset_id,
            "name": config.get("name", dataset_id),
            "hf_path": hf_path,
            "split": config.get("split", ""),
            "audio_column": config.get("audio_column", ""),
            "transcript_column": config.get("transcript_column", ""),
            "local_size_bytes": local_size,
            "local_size_gb": bytes_to_gb(local_size),
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }

        if dataset_id == "recorded-audio":
            entry.update(recorded_stats())
        elif include_remote:
            entry.update(hf_metadata(hf_path, token=token))

        repo_size = entry.get("repo_size_bytes")
        entry["repo_size_gb"] = bytes_to_gb(repo_size) if isinstance(repo_size, int) else None
        datasets[dataset_id] = entry

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_count": len(datasets),
        "cache_root": str(DATASET_CACHE_DIR),
        "datasets": datasets,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect dataset metadata for the web UI.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="JSON path to write.")
    parser.add_argument("--local-only", action="store_true", help="Skip Hugging Face metadata calls.")
    parser.add_argument("--token", default=None, help="Optional Hugging Face token for gated datasets.")
    args = parser.parse_args()

    payload = collect(include_remote=not args.local_only, token=args.token)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {payload['dataset_count']} dataset records to {output}")
    for dataset_id, data in payload["datasets"].items():
        rows = data.get("sample_count")
        repo_size = data.get("repo_size_gb")
        local_size = data.get("local_size_gb")
        print(f"- {dataset_id}: rows={rows or 'n/a'} repo_gb={repo_size or 'n/a'} local_gb={local_size or 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
