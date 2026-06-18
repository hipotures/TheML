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


@dataclass(frozen=True)
class ModelSpec:
    raw: str
    provider: str
    model: str | None = None
    reasoning_effort: str | None = None
    provider_config: dict[str, object] | None = None


class AiClient(Protocol):
    def call(self, request: AiRequest) -> AiResponse:
        """Return a model response for an archived request."""
