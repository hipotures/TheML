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
        raise TmlError(
            "Kaggle download failed. Check Kaggle credentials at ~/.kaggle/kaggle.json "
            f"and competition access. Kaggle CLI said: {message}"
        )
    for archive in data_dir.glob("*.zip"):
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(data_dir)
