from __future__ import annotations

from .client import AiClient, AiRequest, AiResponse, ModelSpec
from .codex import CodexAiClient
from .models import resolve_model_spec
from .mock import MockAiClient
from .provider_mock import MockProviderAiClient


def client_for_model(model: str, providers: dict[str, object] | None = None) -> AiClient:
    spec = resolve_model_spec(model, providers)
    if spec.provider == "mock":
        return MockAiClient()
    kind = str((spec.provider_config or {}).get("kind") or "")
    if spec.provider == "codex" or kind == "codex_cli":
        return CodexAiClient(spec)
    if kind in {"openai", "openai_compatible"}:
        return MockProviderAiClient(spec)
    return MockProviderAiClient(spec)


__all__ = [
    "AiClient",
    "AiRequest",
    "AiResponse",
    "ModelSpec",
    "MockAiClient",
    "CodexAiClient",
    "client_for_model",
    "resolve_model_spec",
]
