from __future__ import annotations

import csv
import shutil
import subprocess
from collections.abc import Callable

from tml.core.errors import TmlError


DEFAULT_PROJECT_PAGES = ("abstract", "Evaluation", "data-description")


def fetch_competition_pages(
    slug: str,
    *,
    page_names: tuple[str, ...] = DEFAULT_PROJECT_PAGES,
    progress: Callable[[str], None] | None = None,
) -> dict[str, str]:
    if shutil.which("kaggle") is None:
        raise TmlError("Kaggle CLI is not installed. Cannot fetch competition pages.")
    pages: dict[str, str] = {}
    for page_name in page_names:
        _progress(progress, f"Fetching Kaggle page: {page_name}...")
        result = subprocess.run(
            [
                "kaggle",
                "competitions",
                "pages",
                slug,
                "--page-name",
                page_name,
                "--content",
                "-v",
                "-q",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip()
            raise TmlError(f"Kaggle pages fetch failed for {page_name}: {_one_line(message)}")
        content = _parse_page_csv(result.stdout, page_name)
        if content:
            pages[page_name] = content
    return pages


def _parse_page_csv(output: str, requested_page: str) -> str:
    rows = list(csv.DictReader(output.splitlines()))
    for row in rows:
        if str(row.get("name") or "").strip() == requested_page:
            return str(row.get("content") or "").strip()
    return ""


def _one_line(message: str) -> str:
    return " ".join(line.strip() for line in message.splitlines() if line.strip())


def _progress(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
