from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tml.utils.atomic import atomic_write_text
from tml.utils.yaml_io import write_yaml

from .result import ExecutionResult


RESULT_PREFIX = "TML_RESULT_JSON:"


def run_python_script(script: Path, work_dir: Path, timeout_seconds: int = 900) -> ExecutionResult:
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=work_dir,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return ExecutionResult(
            status="failed",
            returncode=124,
            stdout=stdout,
            stderr=stderr,
            error=f"Timed out after {timeout_seconds} seconds",
        )

    metric = None
    maximize = None
    for line in completed.stdout.splitlines():
        if not line.startswith(RESULT_PREFIX):
            continue
        try:
            payload = json.loads(line[len(RESULT_PREFIX) :])
        except json.JSONDecodeError:
            continue
        raw_metric = payload.get("metric")
        metric = float(raw_metric) if isinstance(raw_metric, (int, float)) else None
        raw_maximize = payload.get("maximize")
        maximize = bool(raw_maximize) if isinstance(raw_maximize, bool) else None
    status = "ok" if completed.returncode == 0 else "failed"
    return ExecutionResult(
        status=status,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        metric=metric,
        maximize=maximize,
        error=None if completed.returncode == 0 else f"Process exited with {completed.returncode}",
    )


def write_attempt_result(attempt_dir: Path, result: ExecutionResult) -> None:
    atomic_write_text(attempt_dir / "stdout.log", result.stdout)
    atomic_write_text(attempt_dir / "stderr.log", result.stderr)
    payload = {
        "status": result.status,
        "returncode": result.returncode,
        "metric": result.metric,
        "maximize": result.maximize,
        "error": result.error,
    }
    if result.status == "ok":
        write_yaml(attempt_dir / "result.yaml", payload)
    else:
        write_yaml(attempt_dir / "failed.yaml", payload)
