from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
LOG_DIR = ROOT_DIR / "logs"
DEFAULT_OUTPUT = DATA_DIR / "registry_benchmark_results.json"
DEFAULT_LOG = LOG_DIR / "registry_benchmark.log"


@dataclass
class BenchmarkTask:
    model_id: str
    dataset_id: str
    limit: int
    sample_pool: int


PROFILE_STAGES = {
    "ui": (10, 25),
    "balanced": (10, 25, 50),
    "full": (100,),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def setup_logging(log_path: Path, verbose: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def api_url(base_url: str, path: str, query: dict[str, Any] | None = None) -> str:
    url = base_url.rstrip("/") + path
    if query:
        url += "?" + urllib.parse.urlencode(query)
    return url


def request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: int = 60,
    query: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        api_url(base_url, path, query),
        data=body,
        method=method,
        headers=headers,
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {path}: {raw[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach backend at {base_url}: {exc}") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Backend returned non-JSON response from {path}: {raw[:1000]}") from exc


def load_models(base_url: str, timeout: int) -> list[dict[str, Any]]:
    payload = request_json(base_url, "/models", timeout=timeout)
    models = payload.get("models", payload if isinstance(payload, list) else [])
    if not isinstance(models, list):
        raise RuntimeError("/models did not return a model list")
    return [model for model in models if isinstance(model, dict) and model.get("id")]


def load_datasets(base_url: str, timeout: int) -> list[dict[str, Any]]:
    payload = request_json(base_url, "/datasets", timeout=timeout)
    datasets = payload.get("datasets", payload if isinstance(payload, list) else [])
    if not isinstance(datasets, list):
        raise RuntimeError("/datasets did not return a dataset list")
    return [dataset for dataset in datasets if isinstance(dataset, dict) and dataset.get("id")]


def dataset_limit(dataset: dict[str, Any], cap: int) -> int:
    sample_count = dataset.get("sample_count") or dataset.get("rows")
    try:
        sample_count_int = int(sample_count)
    except (TypeError, ValueError):
        sample_count_int = cap
    return max(1, min(sample_count_int, cap))


def build_tasks(
    models: list[dict[str, Any]],
    datasets: list[dict[str, Any]],
    *,
    cap: int,
    sample_pool: int,
    only_models: set[str],
    only_datasets: set[str],
    skip_recorded_empty: bool,
) -> list[BenchmarkTask]:
    selected_models = [
        model for model in models
        if not only_models or str(model["id"]) in only_models
    ]
    selected_datasets = [
        dataset for dataset in datasets
        if not only_datasets or str(dataset["id"]) in only_datasets
    ]

    tasks: list[BenchmarkTask] = []
    for dataset in selected_datasets:
        limit = dataset_limit(dataset, cap)
        if skip_recorded_empty and dataset.get("id") == "recorded-audio":
            raw_count = dataset.get("sample_count") or dataset.get("rows") or 0
            try:
                if int(raw_count) <= 0:
                    logging.info("Skipping recorded-audio because it has no samples yet")
                    continue
            except (TypeError, ValueError):
                pass
        for model in selected_models:
            tasks.append(
                BenchmarkTask(
                    model_id=str(model["id"]),
                    dataset_id=str(dataset["id"]),
                    limit=limit,
                    sample_pool=max(limit, sample_pool),
                )
            )
    return tasks


def run_task(
    base_url: str,
    task: BenchmarkTask,
    *,
    timeout: int,
    add_noise: bool,
    snr_db: float,
) -> dict[str, Any]:
    payload = {
        "model_ids": [task.model_id],
        "dataset_ids": [task.dataset_id],
        "limit": task.limit,
        "sample_pool": task.sample_pool,
        "add_noise": add_noise,
        "snr_db": snr_db,
        "mix_datasets": False,
    }
    return request_json(
        base_url,
        "/analyze-batch",
        method="POST",
        payload=payload,
        timeout=timeout,
    )


def summarize_pair(response: dict[str, Any], task: BenchmarkTask, elapsed: float) -> dict[str, Any]:
    run = None
    runs = response.get("runs")
    if isinstance(runs, list) and runs:
        run = runs[0]

    summary = run.get("summary", {}) if isinstance(run, dict) else {}
    return {
        "model_id": task.model_id,
        "dataset_id": task.dataset_id,
        "limit": task.limit,
        "sample_pool": task.sample_pool,
        "status": run.get("status", "ok") if isinstance(run, dict) else ("ok" if response.get("success") else "failed"),
        "run_id": response.get("run_id"),
        "count": summary.get("count", 0),
        "avg_wer": summary.get("avg_wer"),
        "avg_cer": summary.get("avg_cer"),
        "avg_duration_seconds": summary.get("avg_duration_seconds"),
        "avg_elapsed_seconds": summary.get("avg_elapsed_seconds"),
        "elapsed_seconds": round(elapsed, 2),
        "cache_hits": summary.get("cache_hits", 0),
        "cache_misses": summary.get("cache_misses", 0),
        "best_sample": summary.get("best_sample"),
        "worst_sample": summary.get("worst_sample"),
        "error": run.get("error") if isinstance(run, dict) else response.get("error"),
    }


def stage_limits(max_limit: int, profile: str) -> list[int]:
    stages = PROFILE_STAGES[profile]
    clipped = sorted({min(max_limit, stage) for stage in stages if stage > 0})
    return clipped or [max_limit]


def desired_resume_count(task: BenchmarkTask, profile: str) -> int:
    return stage_limits(task.limit, profile)[-1]


def should_stop_early(row: dict[str, Any], args: argparse.Namespace, stage: int, stage_elapsed: float) -> str | None:
    if row.get("status") == "failed" or row.get("error"):
        return "failed"
    if stage < args.early_stop_min:
        return None
    avg_wer = row.get("avg_wer")
    avg_elapsed = row.get("avg_elapsed_seconds")
    if avg_wer is not None and float(avg_wer) >= args.early_stop_wer:
        return f"high_wer>={args.early_stop_wer}"
    if avg_elapsed is not None and float(avg_elapsed) >= args.max_avg_elapsed:
        return f"slow_avg_sample>={args.max_avg_elapsed}s"
    if stage_elapsed >= args.max_stage_seconds:
        return f"slow_stage>={args.max_stage_seconds}s"
    return None


def write_results(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def aggregate_stats(results: list[dict[str, Any]]) -> dict[str, Any]:
    model_rows: dict[str, list[dict[str, Any]]] = {}
    dataset_rows: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        if row.get("status") == "failed":
            continue
        if row.get("avg_wer") is None or row.get("avg_cer") is None:
            continue
        model_rows.setdefault(str(row.get("model_id")), []).append(row)
        dataset_rows.setdefault(str(row.get("dataset_id")), []).append(row)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(rows)
        sample_count = sum(int(row.get("count") or 0) for row in rows)
        cache_hits = sum(int(row.get("cache_hits") or 0) for row in rows)
        cache_misses = sum(int(row.get("cache_misses") or 0) for row in rows)
        avg_wer = sum(float(row["avg_wer"]) for row in rows) / count if count else None
        avg_cer = sum(float(row["avg_cer"]) for row in rows) / count if count else None
        avg_elapsed = sum(float(row.get("avg_elapsed_seconds") or 0) for row in rows) / count if count else None
        best = min(rows, key=lambda row: (float(row["avg_wer"]), float(row["avg_cer"]))) if rows else None
        worst = max(rows, key=lambda row: (float(row["avg_wer"]), float(row["avg_cer"]))) if rows else None
        return {
            "runs": count,
            "samples": sample_count,
            "avg_wer": avg_wer,
            "avg_cer": avg_cer,
            "avg_elapsed_seconds": avg_elapsed,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "best": {
                "model_id": best.get("model_id"),
                "dataset_id": best.get("dataset_id"),
                "avg_wer": best.get("avg_wer"),
                "avg_cer": best.get("avg_cer"),
            } if best else None,
            "worst": {
                "model_id": worst.get("model_id"),
                "dataset_id": worst.get("dataset_id"),
                "avg_wer": worst.get("avg_wer"),
                "avg_cer": worst.get("avg_cer"),
            } if worst else None,
        }

    return {
        "model_stats": {model_id: summarize(rows) for model_id, rows in sorted(model_rows.items())},
        "dataset_stats": {dataset_id: summarize(rows) for dataset_id, rows in sorted(dataset_rows.items())},
    }


def load_existing_results(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logging.warning("Could not read existing output file %s; starting fresh", path)
        return {}
    rows = payload.get("results", [])
    existing = {}
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and row.get("model_id") and row.get("dataset_id"):
                existing[(str(row["model_id"]), str(row["dataset_id"]))] = row
    return existing


def parse_csv_filter(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run backend models against backend datasets for UI stats or full "
            "benchmarking. Uses the backend analysis cache and keeps going after errors."
        )
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:5000", help="Running Flask backend URL.")
    parser.add_argument("--profile", choices=("ui", "balanced", "full"), default="ui", help="ui=10/25 samples with early stops, balanced=10/25/50, full=100.")
    parser.add_argument("--cap", type=int, default=100, help="Maximum samples per dataset/model pair.")
    parser.add_argument("--sample-pool", type=int, default=100, help="Dataset sample pool loaded by the backend.")
    parser.add_argument("--timeout", type=int, default=3600, help="Per-request timeout in seconds.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Aggregate output JSON path.")
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG, help="Log file path.")
    parser.add_argument("--only-models", help="Comma-separated model ids to include.")
    parser.add_argument("--only-datasets", help="Comma-separated dataset ids to include.")
    parser.add_argument("--add-noise", action="store_true", help="Run with backend noise injection enabled.")
    parser.add_argument("--snr-db", type=float, default=20.0, help="SNR for noise injection.")
    parser.add_argument("--resume", action="store_true", help="Skip model/dataset pairs already present in the output file.")
    parser.add_argument("--include-empty-recorded", action="store_true", help="Do not skip recorded-audio when it has zero samples.")
    parser.add_argument("--early-stop-min", type=int, default=20, help="Minimum stage size before early stop rules can skip a pair.")
    parser.add_argument("--early-stop-wer", type=float, default=0.92, help="Stop expanding a pair once WER is this bad or worse.")
    parser.add_argument("--max-avg-elapsed", type=float, default=12.0, help="Stop expanding a pair if average inference seconds/sample is this slow.")
    parser.add_argument("--max-stage-seconds", type=float, default=240.0, help="Stop expanding a pair if a stage request takes this long.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned tasks without calling /analyze-batch.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    if args.cap < 1 or args.cap > 100:
        parser.error("--cap must be between 1 and 100 because the backend /analyze-batch endpoint caps at 100.")

    setup_logging(args.log_file, args.verbose)
    logging.info("Registry benchmark runner started")
    logging.info("Backend: %s", args.base_url)
    logging.info("Profile: %s stages=%s", args.profile, PROFILE_STAGES[args.profile])

    models = load_models(args.base_url, args.timeout)
    datasets = load_datasets(args.base_url, args.timeout)
    tasks = build_tasks(
        models,
        datasets,
        cap=args.cap,
        sample_pool=args.sample_pool,
        only_models=parse_csv_filter(args.only_models),
        only_datasets=parse_csv_filter(args.only_datasets),
        skip_recorded_empty=not args.include_empty_recorded,
    )

    existing = load_existing_results(args.output) if args.resume else {}
    if args.resume and existing:
        tasks = [
            task for task in tasks
            if int(existing.get((task.model_id, task.dataset_id), {}).get("count") or 0) < desired_resume_count(task, args.profile)
        ]

    logging.info("Models discovered: %s", len(models))
    logging.info("Datasets discovered: %s", len(datasets))
    logging.info("Tasks planned: %s", len(tasks))
    logging.info("Output: %s", args.output)
    logging.info("Log: %s", args.log_file)

    for task in tasks:
        logging.info(
            "PLAN model=%s dataset=%s stages=%s",
            task.model_id,
            task.dataset_id,
            stage_limits(task.limit, args.profile),
        )

    if args.dry_run:
        logging.info("Dry run complete; no benchmark requests sent.")
        return 0

    started_at = time.perf_counter()
    results = list(existing.values())
    failures = 0

    output_payload = {
        "generated_at": utc_now(),
        "base_url": args.base_url,
        "settings": {
            "cap": args.cap,
            "profile": args.profile,
            "stages": PROFILE_STAGES[args.profile],
            "sample_pool": args.sample_pool,
            "add_noise": args.add_noise,
            "snr_db": args.snr_db,
            "resume": args.resume,
        },
        "models": models,
        "datasets": datasets,
        "results": results,
        **aggregate_stats(results),
    }
    write_results(args.output, output_payload)

    total = len(tasks)
    for index, task in enumerate(tasks, start=1):
        task_started = time.perf_counter()
        completed = index - 1
        avg = (task_started - started_at) / completed if completed else None
        eta = avg * (total - completed) if avg is not None else None
        stages = stage_limits(task.limit, args.profile)
        logging.info(
            "[%s/%s] START model=%s dataset=%s stages=%s eta=%s",
            index,
            total,
            task.model_id,
            task.dataset_id,
            stages,
            format_duration(eta),
        )

        try:
            stage_rows = []
            stop_reason = None
            row = None
            for stage in stages:
                staged_task = BenchmarkTask(
                    model_id=task.model_id,
                    dataset_id=task.dataset_id,
                    limit=stage,
                    sample_pool=max(stage, task.sample_pool),
                )
                stage_started = time.perf_counter()
                logging.info(
                    "[%s/%s] STAGE model=%s dataset=%s limit=%s",
                    index,
                    total,
                    task.model_id,
                    task.dataset_id,
                    stage,
                )
                response = run_task(
                    args.base_url,
                    staged_task,
                    timeout=args.timeout,
                    add_noise=args.add_noise,
                    snr_db=args.snr_db,
                )
                stage_elapsed = time.perf_counter() - stage_started
                row = summarize_pair(response, staged_task, time.perf_counter() - task_started)
                stage_rows.append({
                    "limit": stage,
                    "count": row.get("count"),
                    "avg_wer": row.get("avg_wer"),
                    "avg_cer": row.get("avg_cer"),
                    "avg_elapsed_seconds": row.get("avg_elapsed_seconds"),
                    "cache_hits": row.get("cache_hits"),
                    "cache_misses": row.get("cache_misses"),
                    "elapsed_seconds": round(stage_elapsed, 2),
                    "status": row.get("status"),
                    "error": row.get("error"),
                })
                stop_reason = should_stop_early(row, args, stage, stage_elapsed)
                if stop_reason:
                    break
            if row is None:
                raise RuntimeError("No stages were run")
            elapsed = time.perf_counter() - task_started
            row["profile"] = args.profile
            row["stages"] = stage_rows
            row["stopped_reason"] = stop_reason or "completed_profile"
            if row.get("status") == "failed" or row.get("error"):
                failures += 1
                logging.error(
                    "[%s/%s] FAILED model=%s dataset=%s elapsed=%s error=%s",
                    index,
                    total,
                    task.model_id,
                    task.dataset_id,
                    format_duration(elapsed),
                    row.get("error"),
                )
            else:
                logging.info(
                    "[%s/%s] DONE model=%s dataset=%s count=%s wer=%s cer=%s cache=%s/%s stopped=%s elapsed=%s",
                    index,
                    total,
                    task.model_id,
                    task.dataset_id,
                    row.get("count"),
                    row.get("avg_wer"),
                    row.get("avg_cer"),
                    row.get("cache_hits"),
                    row.get("cache_misses"),
                    row.get("stopped_reason"),
                    format_duration(elapsed),
                )
            results.append(row)
        except Exception as exc:
            failures += 1
            elapsed = time.perf_counter() - task_started
            logging.exception(
                "[%s/%s] ERROR model=%s dataset=%s elapsed=%s",
                index,
                total,
                task.model_id,
                task.dataset_id,
                format_duration(elapsed),
            )
            results.append(
                {
                    "model_id": task.model_id,
                    "dataset_id": task.dataset_id,
                    "limit": task.limit,
                    "sample_pool": task.sample_pool,
                    "status": "failed",
                    "elapsed_seconds": round(elapsed, 2),
                    "error": str(exc),
                }
            )

        output_payload = {
            "generated_at": utc_now(),
            "base_url": args.base_url,
            "settings": {
                "cap": args.cap,
                "profile": args.profile,
                "stages": PROFILE_STAGES[args.profile],
                "sample_pool": args.sample_pool,
                "add_noise": args.add_noise,
                "snr_db": args.snr_db,
                "resume": args.resume,
            },
            "progress": {
                "completed": index,
                "planned": total,
                "failures": failures,
                "elapsed_seconds": round(time.perf_counter() - started_at, 2),
            },
            "models": models,
            "datasets": datasets,
            "results": results,
            **aggregate_stats(results),
        }
        write_results(args.output, output_payload)

    total_elapsed = time.perf_counter() - started_at
    logging.info(
        "Registry benchmark complete tasks=%s failures=%s elapsed=%s output=%s",
        total,
        failures,
        format_duration(total_elapsed),
        args.output,
    )
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
