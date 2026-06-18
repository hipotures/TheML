from __future__ import annotations

from .client import AiClient, AiRequest, AiResponse
from .codex import CodexAiClient
from .mock import MockAiClient


def client_for_model(model: str) -> AiClient:
    if model == "mock":
        return MockAiClient()
    if model.startswith("codex") or model.startswith("gpt"):
        return CodexAiClient()
    raise ValueError(f"Unknown AI model/backend {model!r}")


__all__ = ["AiClient", "AiRequest", "AiResponse", "MockAiClient", "CodexAiClient", "client_for_model"]
