from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

from tml.core.errors import TmlError


def download_competition_data(slug: str, data_dir: Path) -> None:
    if shutil.which("kaggle") is None:
        raise TmlError(
            "Kaggle CLI is not installed. Install/configure it, then run: "
            f"uv run tml init project {slug} download=true"
        )
    data_dir.mkdir(parents=True, exist_ok=True)
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
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(data_dir)


def _one_line(message: str) -> str:
    return " ".join(line.strip() for line in message.splitlines() if line.strip())
