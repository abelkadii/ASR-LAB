from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(command: list[str], cwd: Path = ROOT) -> None:
    print(f"[setup] {' '.join(command)}")
    subprocess.run(command, cwd=cwd, check=True)


def ensure_dirs() -> None:
    for relative in ("uploads", "data", "logs", "models", "datasets"):
        path = ROOT / relative
        path.mkdir(parents=True, exist_ok=True)
        keep = path / ".gitkeep"
        keep.touch(exist_ok=True)


def write_env_if_missing() -> None:
    env_path = ROOT / ".env"
    example = ROOT / ".env.example"
    if not env_path.exists() and example.exists():
        env_path.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
        print("[setup] Created .env from .env.example")


def install_python_deps() -> None:
    run([PYTHON, "-m", "pip", "install", "-r", "requirements.txt"])


def install_frontend_deps() -> None:
    run(["npm.cmd" if os.name == "nt" else "npm", "install"], cwd=ROOT / "frontend")


def build_frontend() -> None:
    run(["npm.cmd" if os.name == "nt" else "npm", "run", "build"], cwd=ROOT / "frontend")


def download_assets(selection: str, max_samples: int | None) -> None:
    command = [
        PYTHON,
        str(ROOT / "scripts" / "download_hf_assets.py"),
        "--scope",
        selection,
        "--cache-dir",
        str(ROOT / "models" / "hf-cache"),
    ]
    if max_samples is not None:
        command.extend(["--dataset-max-samples", str(max_samples)])
    run(command)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Darija ASR Workbench runtime.")
    parser.add_argument("--install-python", action="store_true", help="Install backend Python requirements.")
    parser.add_argument("--install-frontend", action="store_true", help="Install frontend npm dependencies.")
    parser.add_argument("--build-frontend", action="store_true", help="Build frontend/dist for Flask to serve.")
    parser.add_argument("--download", choices=("web", "all"), help="Download Hugging Face models/datasets.")
    parser.add_argument("--dataset-max-samples", type=int, help="Limit downloaded dataset samples when supported.")
    parser.add_argument("--full", action="store_true", help="Install Python deps, install frontend deps, and build frontend.")
    args = parser.parse_args()

    ensure_dirs()
    write_env_if_missing()

    if args.full or args.install_python:
        install_python_deps()
    if args.full or args.install_frontend:
        install_frontend_deps()
    if args.full or args.build_frontend:
        build_frontend()
    if args.download:
        download_assets(args.download, args.dataset_max_samples)

    print("[setup] Done.")
    print("[setup] Start backend with: python backend/app.py")


if __name__ == "__main__":
    main()
