from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class AiRequest:
    role: str
    model: str
    prompt: str


@dataclass(frozen=True)
class AiResponse:
    text: str
    metadata: dict[str, object]


class AiClient(Protocol):
    def call(self, request: AiRequest) -> AiResponse:
        """Return a model response for an archived request."""
