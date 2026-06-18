from __future__ import annotations

import secrets
from datetime import datetime


def timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def run_id() -> str:
    return f"{timestamp_id()}-{secrets.token_hex(4)}"


def node_id(step: int) -> str:
    return f"{timestamp_id()}-{secrets.token_hex(4)}-{step}"


def hypothesis_id(number: int) -> str:
    return f"{number:06d}"
