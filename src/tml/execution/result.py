from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    returncode: int
    stdout: str
    stderr: str
    metric: float | None = None
    maximize: bool | None = None
    error: str | None = None
