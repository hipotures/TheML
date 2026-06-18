from __future__ import annotations

from .client import AiClient, AiRequest, AiResponse, ModelSpec
from .codex import CodexAiClient
from .invocation import ModelInvocation, ProviderInvocationError, ProviderResult, run_model_invocation
from .models import resolve_model_spec
from .mock import MockAiClient
from .provider_mock import MockProviderAiClient


def client_for_model(model: str, providers: dict[str, object] | None = None) -> AiClient:
    spec = resolve_model_spec(model, providers)
    return client_for_spec(spec)


def client_for_spec(spec: ModelSpec) -> AiClient:
    if spec.provider == "mock":
        return MockAiClient()
    kind = str((spec.provider_config or {}).get("kind") or "")
    if spec.provider == "codex" or kind in {"codex_app_server", "codex_cli", "codex_sdk"}:
        return CodexAiClient(spec)
    if kind in {"openai", "openai_compatible"}:
        return MockProviderAiClient(spec)
    return MockProviderAiClient(spec)


__all__ = [
    "AiClient",
    "AiRequest",
    "AiResponse",
    "ModelSpec",
    "ModelInvocation",
    "ProviderInvocationError",
    "ProviderResult",
    "MockAiClient",
    "CodexAiClient",
    "client_for_spec",
    "client_for_model",
    "resolve_model_spec",
    "run_model_invocation",
]
