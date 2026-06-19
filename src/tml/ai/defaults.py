from __future__ import annotations

from typing import Any


MODEL_ROLE_DEFAULTS: dict[str, dict[str, Any]] = {
    "metadata": {"timeout_seconds": 30},
    "hypothesis": {"timeout_seconds": 120},
    "code": {"timeout_seconds": 900},
    "review": {"timeout_seconds": 300},
    "bugfix": {"timeout_seconds": 600},
}
