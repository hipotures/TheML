from __future__ import annotations

import gzip
import shutil
import subprocess
import zipfile
from collections.abc import Callable
from pathlib import Path

from tml.core.errors import TmlError


def download_competition_data(slug: str, data_dir: Path, progress: Callable[[str], None] | None = None) -> None:
    if shutil.which("kaggle") is None:
        raise TmlError(
            "Kaggle CLI is not installed. Install/configure it, then run: "
            f"uv run tml init project {slug} download=true"
        )
    data_dir.mkdir(parents=True, exist_ok=True)
    _progress(progress, f"Downloading Kaggle data for {slug}...")
    result = subprocess.run(
        ["kaggle", "competitions", "download", "-c", slug, "-p", str(data_dir)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        if "ModuleNotFoundError" in message and "kaggle" in message:
            raise TmlError(
                "Kaggle CLI is installed but its Python package is not importable. "
                "Install it in the environment used by the `kaggle` executable, or "
                "put a working Kaggle CLI earlier on PATH. Then rerun: "
                f"uv run tml init project {slug}"
            )
        raise TmlError(
            "Kaggle download failed. Check Kaggle credentials at ~/.kaggle/kaggle.json "
            f"and competition access. Kaggle CLI said: {_one_line(message)}"
        )
    for archive in data_dir.glob("*.zip"):
        _progress(progress, f"Unzipping {archive.name}...")
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(data_dir)
            extracted = [data_dir / name for name in zip_file.namelist()]
        for path in extracted:
            if path.is_file():
                _progress(progress, f"Compressing {path.name} -> {path.name}.gz...")
                _gzip_file(path)


def submit_competition_file(slug: str, submission_path: Path, message: str, upload_path: Path) -> object:
    if not submission_path.is_file():
        raise TmlError(f"Submission file does not exist: {submission_path}")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ModuleNotFoundError as exc:
        raise TmlError("Kaggle Python package is not installed in this environment.") from exc
    api = KaggleApi()
    api.authenticate()
    if upload_path != submission_path:
        shutil.copy2(submission_path, upload_path)
    return api.competition_submit(str(upload_path), message, slug, quiet=False)


def _one_line(message: str) -> str:
    return " ".join(line.strip() for line in message.splitlines() if line.strip())


def _gzip_file(path: Path) -> Path:
    if path.suffix == ".gz":
        return path
    gz_path = path.with_name(path.name + ".gz")
    with path.open("rb") as source, gzip.open(gz_path, "wb") as target:
        shutil.copyfileobj(source, target)
    path.unlink()
    return gz_path


def _progress(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
