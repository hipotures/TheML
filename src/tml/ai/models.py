from __future__ import annotations

from typing import Any

from .client import ModelSpec


VALID_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


def resolve_model_spec(raw: str, providers: dict[str, Any] | None = None) -> ModelSpec:
    value = str(raw or "mock").strip()
    provider_configs = providers or {}
    if value == "mock":
        return ModelSpec(raw=value, provider="mock", provider_config={"kind": "mock"})

    parts = value.split(":")
    if len(parts) == 1:
        provider = _legacy_provider(parts[0])
        if provider == "mock":
            return ModelSpec(raw=value, provider="mock", provider_config={"kind": "mock"})
        provider_config = _provider_config(provider, provider_configs)
        if value == provider and provider_config:
            raise ValueError(
                f"Model spec {value!r} selects a provider but does not name a model. "
                f"Use {provider}:<model>[:effort], for example {provider}:gpt-5.4:low."
            )
        return ModelSpec(
            raw=value,
            provider=provider,
            model=parts[0],
            provider_config=provider_config,
        )
    if len(parts) not in {2, 3}:
        raise ValueError(f"Invalid model spec {value!r}. Expected provider:model[:effort].")

    provider, model = parts[0].strip(), parts[1].strip()
    effort = parts[2].strip() if len(parts) == 3 else None
    if not provider:
        raise ValueError(f"Invalid empty provider in model spec {value!r}.")
    if not model:
        raise ValueError(f"Invalid empty model in model spec {value!r}.")
    if effort and effort not in VALID_REASONING_EFFORTS:
        raise ValueError(
            f"Invalid reasoning effort {effort!r}. Expected one of: {', '.join(sorted(VALID_REASONING_EFFORTS))}."
        )
    return ModelSpec(
        raw=value,
        provider=provider,
        model=model,
        reasoning_effort=effort,
        provider_config=_provider_config(provider, provider_configs),
    )


def _legacy_provider(model: str) -> str:
    if model == "mock":
        return "mock"
    if model == "codex" or model.startswith("gpt") or model.startswith("o"):
        return "codex"
    return model


def _provider_config(provider: str, providers: dict[str, Any]) -> dict[str, object]:
    config = providers.get(provider, {}) if isinstance(providers, dict) else {}
    return config if isinstance(config, dict) else {}
