from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from tml.utils.atomic import atomic_write_json, atomic_write_text

from .client import AiRequest, AiResponse, ModelSpec
from .models import resolve_model_spec


@dataclass(frozen=True)
class ModelInvocation:
    role: str
    model: str
    prompt: str
    template_id: str | None = None
    template_path: str | None = None
    template_hash: str | None = None
    rendered_prompt_hash: str | None = None
    messages: list[dict[str, str]] | None = None
    project_dir: str = "."
    cwd: Path | None = None
    sandbox: str | None = None
    output_schema: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    progress: Callable[[str], None] | None = field(default=None, repr=False, compare=False)
    runtime_artifact_dir: Path | None = field(default=None, repr=False, compare=False)
    runtime_response_prefix: str | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class ProviderResult:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: Any | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    stdout: str | None = None
    stderr: str | None = None


class InvocationCapableClient:
    def invoke(self, invocation: ModelInvocation, spec: ModelSpec) -> ProviderResult:
        raise NotImplementedError


class ProviderInvocationError(RuntimeError):
    def __init__(self, message: str, result: ProviderResult) -> None:
        super().__init__(message)
        self.result = result


def run_model_invocation(
    invocation: ModelInvocation,
    *,
    artifact_dir: Path,
    providers: dict[str, object] | None = None,
    role_options: dict[str, object] | None = None,
    response_prefix: str | None = None,
) -> AiResponse:
    spec = resolve_model_spec(invocation.model, providers)
    role_config = dict(role_options or {})
    if role_config:
        spec = replace(
            spec,
            provider_config={**(spec.provider_config or {}), **role_config},
            role_config=role_config,
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if invocation.runtime_artifact_dir is None:
        invocation = replace(
            invocation,
            runtime_artifact_dir=artifact_dir,
            runtime_response_prefix=response_prefix,
        )
    _write_request_artifacts(artifact_dir, invocation, spec, response_prefix=response_prefix)

    from . import client_for_spec

    client = client_for_spec(spec)
    try:
        if hasattr(client, "invoke"):
            result = client.invoke(invocation, spec)  # type: ignore[attr-defined]
        else:
            legacy = client.call(
                AiRequest(role=invocation.role, model=spec.model or invocation.model, prompt=invocation.prompt)
            )
            result = ProviderResult(text=legacy.text, metadata=legacy.metadata)
    except ProviderInvocationError as exc:
        _write_response_artifacts(artifact_dir, exc.result, response_prefix=response_prefix)
        raise
    except Exception as exc:
        result = ProviderResult(
            text="",
            metadata={"status": "error", "error": str(exc), "error_type": type(exc).__name__},
            raw={"error": str(exc), "error_type": type(exc).__name__},
        )
        _write_response_artifacts(artifact_dir, result, response_prefix=response_prefix)
        raise

    _write_response_artifacts(artifact_dir, result, response_prefix=response_prefix)
    return AiResponse(text=result.text, metadata=_response_metadata(result))


def _write_request_artifacts(
    artifact_dir: Path,
    invocation: ModelInvocation,
    spec: ModelSpec,
    *,
    response_prefix: str | None,
) -> None:
    prefix = f"{response_prefix}." if response_prefix else ""
    atomic_write_text(artifact_dir / f"{prefix}request.md", invocation.prompt)
    payload: dict[str, Any] = {
        "role": invocation.role,
        "model": invocation.model,
        "provider": spec.provider,
        "provider_kind": (spec.provider_config or {}).get("kind"),
        "resolved_model": spec.model,
        "reasoning_effort": spec.reasoning_effort,
        "role_options": spec.role_config or {},
        "messages": invocation.messages or [{"role": "user", "content": invocation.prompt}],
        "template_id": invocation.template_id,
        "template_path": invocation.template_path,
        "template_hash": invocation.template_hash,
        "rendered_prompt_hash": invocation.rendered_prompt_hash,
        "project_dir": invocation.project_dir,
        "cwd": _artifact_path(invocation.cwd) if invocation.cwd else None,
        "sandbox": invocation.sandbox,
        "output_schema": invocation.output_schema,
        "metadata": invocation.metadata,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    atomic_write_json(artifact_dir / f"{prefix}request.json", payload)


def _write_response_artifacts(
    artifact_dir: Path,
    result: ProviderResult,
    *,
    response_prefix: str | None,
) -> None:
    prefix = f"{response_prefix}." if response_prefix else ""
    atomic_write_text(artifact_dir / f"{prefix}response.md", _pretty_response_text(result.text))
    atomic_write_json(artifact_dir / f"{prefix}response.json", {"text": result.text, **result.metadata})
    if result.raw is not None:
        atomic_write_json(artifact_dir / f"{prefix}provider-raw.json", {"raw": _jsonable(result.raw)})
    if result.events:
        lines = "".join(json.dumps(_jsonable(event), sort_keys=True) + "\n" for event in result.events)
        atomic_write_text(artifact_dir / f"{prefix}events.jsonl", lines)
    if result.stdout:
        atomic_write_text(artifact_dir / f"{prefix}stdout.txt", result.stdout)
    if result.stderr:
        atomic_write_text(artifact_dir / f"{prefix}stderr.txt", result.stderr)


def _response_metadata(result: ProviderResult) -> dict[str, Any]:
    metadata = dict(result.metadata)
    if result.raw is not None:
        metadata["raw_logged"] = True
    if result.events:
        metadata["event_count"] = len(result.events)
    return metadata


def _pretty_response_text(text: str) -> str:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    if not isinstance(parsed, (dict, list)):
        return text
    return json.dumps(parsed, indent=2, sort_keys=False, ensure_ascii=False) + "\n"


def _artifact_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix() or "."
    except ValueError:
        return path.name


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        return _jsonable(asdict(value))
    return repr(value)
