from __future__ import annotations

from .client import AiRequest, AiResponse, ModelSpec
from .mock import MockAiClient


class MockProviderAiClient:
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    def call(self, request: AiRequest) -> AiResponse:
        response = MockAiClient().call(request)
        metadata = {
            **response.metadata,
            "provider": self.spec.provider,
            "provider_kind": (self.spec.provider_config or {}).get("kind", "mock"),
            "model": self.spec.model,
            "reasoning_effort": self.spec.reasoning_effort,
        }
        return AiResponse(text=response.text, metadata=metadata)
