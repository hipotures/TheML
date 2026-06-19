from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import tempfile
from dataclasses import asdict, is_dataclass
from time import perf_counter
from pathlib import Path
from typing import Any

from .client import AiRequest, AiResponse, ModelSpec
from .invocation import ModelInvocation, ProviderInvocationError, ProviderResult
from tml.utils.atomic import atomic_write_text


class CodexAiClient:
    """Small Codex CLI adapter; future app-server clients can implement AiClient."""

    def __init__(self, spec: ModelSpec | None = None) -> None:
        self.spec = spec

    def call(self, request: AiRequest) -> AiResponse:
        if self.spec is not None and self.spec.provider == "codex" and self.spec.raw != (self.spec.model or ""):
            result = self.invoke(
                ModelInvocation(role=request.role, model=request.model, prompt=request.prompt),
                self.spec,
            )
            return AiResponse(text=result.text, metadata=result.metadata)
        return self._call_cli(request)

    def invoke(self, invocation: ModelInvocation, spec: ModelSpec) -> ProviderResult:
        kind = str((spec.provider_config or {}).get("kind") or "codex_sdk")
        if kind == "codex_cli":
            response = self._call_cli(
                AiRequest(role=invocation.role, model=spec.model or invocation.model, prompt=invocation.prompt)
            )
            return ProviderResult(text=response.text, metadata=response.metadata)
        if kind == "codex_app_server":
            return self._call_app_server(invocation, spec)
        return self._call_sdk(invocation, spec)

    def _call_cli(self, request: AiRequest) -> AiResponse:
        with tempfile.TemporaryDirectory(prefix="tml-codex-") as tmp:
            response_path = Path(tmp) / "response.txt"
            result = subprocess.run(
                [
                    "codex",
                    "--ask-for-approval",
                    "never",
                    "exec",
                    "--sandbox",
                    "read-only",
                    "--model",
                    request.model,
                    "--output-last-message",
                    str(response_path),
                    "-",
                ],
                input=request.prompt,
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                message = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(f"Codex backend failed: {message}")
            return AiResponse(
                text=response_path.read_text(encoding="utf-8"),
                metadata={"backend": "codex", "model": request.model},
            )

    def _call_app_server(self, invocation: ModelInvocation, spec: ModelSpec) -> ProviderResult:
        model = spec.model
        if not model:
            raise RuntimeError("Codex model spec must include a model, for example codex:gpt-5.4:high.")

        provider_config = spec.provider_config or {}
        cwd = invocation.cwd or Path.cwd()
        sandbox = _codex_app_server_sandbox(
            invocation.sandbox or str(provider_config.get("sandbox") or "read_only")
        )
        summary = str(provider_config.get("summary") or "concise")
        timeout_seconds = int(provider_config.get("timeout_seconds") or 120)
        isolated_home = _as_bool(provider_config.get("isolated_home"), default=True)
        log_event_deltas = _as_bool(provider_config.get("log_event_deltas"), default=False)
        log_raw_jsonl = _as_bool(provider_config.get("log_raw_jsonl"), default=False)
        log_raw_notifications = _as_bool(provider_config.get("log_raw_notifications"), default=False)
        stderr_text = ""
        raw_lines: list[str] = []
        rpc_responses: list[dict[str, Any]] = []
        notifications: list[dict[str, Any]] = []
        final_chunks: list[str] = []
        usage: Any | None = None
        turn_completed: dict[str, Any] | None = None
        error_message: str | None = None
        final_answer_completed = False
        post_final_deadline: float | None = None
        thread_idle = False

        with tempfile.TemporaryDirectory(prefix="tml-codex-app-server-") as tmp:
            app_server_cmd = _app_server_command(provider_config)
            app_server_env = _app_server_env(cwd, provider_config, isolated_home=isolated_home)
            stderr_path = Path(tmp) / "stderr.txt"
            with stderr_path.open("w+", encoding="utf-8") as stderr_handle:
                _report(invocation, "Starting Codex app-server...")
                proc = subprocess.Popen(
                    app_server_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=stderr_handle,
                    env=app_server_env,
                    text=True,
                    bufsize=1,
                )
                started = perf_counter()
                try:
                    _report(invocation, "Initializing Codex app-server...")
                    _send(proc, _request(0, "initialize", {"clientInfo": _client_info()}))
                    init_response = _read_until_response(
                        proc,
                        0,
                        raw_lines,
                        rpc_responses,
                        notifications,
                        timeout_seconds=timeout_seconds,
                    )
                    _raise_for_rpc_error(init_response, "initialize")
                    _flush_live_artifacts(invocation, raw_lines, notifications, include_deltas=log_event_deltas)
                    _send(proc, {"method": "initialized", "params": {}})

                    _report(invocation, f"Starting Codex thread ({model}, effort={spec.reasoning_effort or 'default'})...")
                    thread_params: dict[str, Any] = {
                        "model": model,
                        "cwd": str(cwd),
                        "sandbox": sandbox,
                        "approvalPolicy": "never",
                        "ephemeral": True,
                        "config": {"model_reasoning_effort": spec.reasoning_effort} if spec.reasoning_effort else None,
                    }
                    _send(proc, _request(1, "thread/start", thread_params))
                    thread_response = _read_until_response(
                        proc,
                        1,
                        raw_lines,
                        rpc_responses,
                        notifications,
                        timeout_seconds=timeout_seconds,
                    )
                    _raise_for_rpc_error(thread_response, "thread/start")
                    _flush_live_artifacts(invocation, raw_lines, notifications, include_deltas=log_event_deltas)
                    thread_id = (
                        thread_response.get("result", {})
                        .get("thread", {})
                        .get("id")
                    )
                    if not thread_id:
                        raise RuntimeError(f"Codex app-server did not return a thread id: {thread_response!r}")

                    _report(invocation, "Sending prompt to Codex...")
                    turn_params: dict[str, Any] = {
                        "threadId": thread_id,
                        "input": [{"type": "text", "text": invocation.prompt}],
                        "model": model,
                        "effort": spec.reasoning_effort,
                        "summary": summary,
                        "cwd": str(cwd),
                        "outputSchema": invocation.output_schema,
                    }
                    _send(proc, _request(2, "turn/start", turn_params))
                    deadline = started + timeout_seconds
                    while perf_counter() < deadline:
                        if post_final_deadline is not None and perf_counter() >= post_final_deadline:
                            break
                        message = _read_message(proc, raw_lines, timeout_seconds=1)
                        if message is None:
                            if proc.poll() is not None:
                                break
                            continue
                        if "id" in message:
                            rpc_responses.append(message)
                            _raise_for_rpc_error(message, str(message.get("id")))
                            continue
                        notifications.append(message)
                        _flush_live_artifacts(invocation, raw_lines, notifications, include_deltas=log_event_deltas)
                        method = message.get("method")
                        params = message.get("params") if isinstance(message.get("params"), dict) else {}
                        _report_event(invocation, method, len(notifications))
                        if method == "item/agentMessage/delta":
                            delta = params.get("delta")
                            if isinstance(delta, str):
                                final_chunks.append(delta)
                        elif method == "thread/tokenUsage/updated":
                            usage = params
                            if final_answer_completed:
                                break
                        elif method == "turn/completed":
                            turn_completed = params
                            break
                        elif method == "item/completed":
                            item = params.get("item") if isinstance(params, dict) else {}
                            if (
                                isinstance(item, dict)
                                and item.get("type") == "agentMessage"
                                and item.get("phase") == "final_answer"
                            ):
                                item_text = item.get("text")
                                if isinstance(item_text, str) and item_text.strip():
                                    final_chunks = [item_text]
                                final_answer_completed = True
                                if usage is not None:
                                    break
                                post_final_deadline = perf_counter() + 2
                        elif method == "thread/status/changed":
                            status = params.get("status") if isinstance(params, dict) else {}
                            thread_idle = isinstance(status, dict) and status.get("type") == "idle"
                            if final_answer_completed and thread_idle and usage is not None:
                                break
                        elif method == "error":
                            raise RuntimeError(f"Codex app-server error notification: {params!r}")
                    if turn_completed is None and not final_answer_completed:
                        error_message = "Codex app-server turn did not complete before timeout."
                finally:
                    if proc.stdin:
                        proc.stdin.close()
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    stderr_handle.flush()
                    stderr_handle.seek(0)
                    stderr_text = stderr_handle.read()

        wall_ms = int((perf_counter() - started) * 1000)
        text = "".join(final_chunks).strip()
        logged_notifications = notifications if log_raw_notifications else _compact_events(notifications, include_deltas=log_event_deltas)
        raw = {
            "rpc_responses": rpc_responses,
            "notifications": logged_notifications,
            "turn_completed": turn_completed,
            "usage": usage,
        }
        if log_raw_jsonl:
            raw["stdout_jsonl"] = raw_lines
        status = _turn_status(turn_completed) or ("completed" if final_answer_completed else None)
        result = ProviderResult(
            text=text,
            metadata={
                "backend": "codex",
                "provider": spec.provider,
                "provider_kind": provider_config.get("kind", "codex_app_server"),
                "model": model,
                "reasoning_effort": spec.reasoning_effort,
                "status": status,
                "error": error_message,
                "wall_ms": wall_ms,
                "usage": usage,
            },
            raw=raw,
            events=logged_notifications,
            stdout="".join(raw_lines) if log_raw_jsonl else None,
            stderr=stderr_text or None,
        )
        if error_message:
            raise ProviderInvocationError(error_message, result)
        _report(invocation, f"Codex completed in {wall_ms} ms.")
        return result

    def _call_sdk(self, invocation: ModelInvocation, spec: ModelSpec) -> ProviderResult:
        try:
            from openai_codex import Codex, Sandbox
        except ImportError as exc:
            raise RuntimeError(
                "Codex SDK backend requires openai-codex. Install it with: uv pip install openai-codex"
            ) from exc

        model = spec.model
        if not model:
            raise RuntimeError("Codex model spec must include a model, for example codex:gpt-5.4:high.")
        effort = spec.reasoning_effort
        sandbox = _codex_sandbox(invocation.sandbox or str((spec.provider_config or {}).get("sandbox") or "read_only"), Sandbox)
        cwd = invocation.cwd or Path.cwd()
        summary = str((spec.provider_config or {}).get("summary") or "concise")
        config = {}
        if effort:
            config["model_reasoning_effort"] = effort

        started = perf_counter()
        with Codex() as codex:
            thread = codex.thread_start(model=model, cwd=str(cwd), sandbox=sandbox, config=config or None)
            result = thread.run(invocation.prompt, model=model, effort=effort, summary=summary)
        wall_ms = int((perf_counter() - started) * 1000)

        metadata = {
            "backend": "codex",
            "provider": spec.provider,
            "provider_kind": (spec.provider_config or {}).get("kind", "codex_sdk"),
            "model": model,
            "reasoning_effort": effort,
            "status": getattr(result, "status", None),
            "duration_ms": getattr(result, "duration_ms", None),
            "wall_ms": wall_ms,
            "started_at": getattr(result, "started_at", None),
            "completed_at": getattr(result, "completed_at", None),
        }
        raw = {
            "status": getattr(result, "status", None),
            "duration_ms": getattr(result, "duration_ms", None),
            "started_at": getattr(result, "started_at", None),
            "completed_at": getattr(result, "completed_at", None),
            "usage": _dump_obj(getattr(result, "usage", None)),
            "items": _dump_obj(getattr(result, "items", None)),
            "final_response": getattr(result, "final_response", ""),
        }
        events = []
        items = raw.get("items")
        if isinstance(items, list):
            events = [{"type": "item", "item": item} for item in items if isinstance(item, dict)]
        return ProviderResult(
            text=str(getattr(result, "final_response", "") or ""),
            metadata={**metadata, "usage": raw["usage"]},
            raw=raw,
            events=events,
        )


def _codex_sandbox(value: str, sandbox_cls: Any) -> Any:
    normalized = value.replace("-", "_")
    if normalized in {"read_only", "readonly"}:
        return sandbox_cls.read_only
    if normalized in {"workspace_write", "workspace"}:
        return sandbox_cls.workspace_write
    if normalized in {"full_access", "danger_full_access", "full"}:
        return sandbox_cls.full_access
    raise RuntimeError(f"Unsupported Codex sandbox {value!r}. Expected read_only, workspace_write, or full_access.")


def _codex_app_server_sandbox(value: str) -> str:
    normalized = value.replace("_", "-")
    if normalized in {"read-only", "readonly"}:
        return "read-only"
    if normalized in {"workspace-write", "workspace"}:
        return "workspace-write"
    if normalized in {"full-access", "danger-full-access", "full"}:
        return "danger-full-access"
    raise RuntimeError(f"Unsupported Codex sandbox {value!r}. Expected read_only, workspace_write, or full_access.")


def _flush_live_artifacts(
    invocation: ModelInvocation,
    raw_lines: list[str],
    notifications: list[dict[str, Any]],
    *,
    include_deltas: bool,
) -> None:
    if invocation.runtime_artifact_dir is None:
        return
    prefix = f"{invocation.runtime_response_prefix}." if invocation.runtime_response_prefix else ""
    if raw_lines and _provider_bool(invocation, "log_raw_jsonl", default=False):
        atomic_write_text(invocation.runtime_artifact_dir / f"{prefix}stdout.txt", "".join(raw_lines))
    if notifications:
        lines = "".join(
            json.dumps(_dump_obj(event), sort_keys=True) + "\n"
            for event in _compact_events(notifications, include_deltas=include_deltas)
        )
        atomic_write_text(invocation.runtime_artifact_dir / f"{prefix}events.jsonl", lines)


def _compact_events(events: list[dict[str, Any]], *, include_deltas: bool) -> list[dict[str, Any]]:
    compacted = []
    for event in events:
        compact = _compact_event(event, include_deltas=include_deltas)
        if compact is not None:
            compacted.append(compact)
    return compacted


def _compact_event(event: dict[str, Any], *, include_deltas: bool) -> dict[str, Any] | None:
    if include_deltas:
        return event
    method = event.get("method")
    params = event.get("params")
    if method == "item/agentMessage/delta" and isinstance(params, dict):
        return None
    if method in {"item/reasoning/summaryTextDelta", "item/reasoning/textDelta"} and isinstance(params, dict):
        return None
    if method in {"item/started", "item/completed"} and isinstance(params, dict):
        compact = dict(params)
        item = compact.get("item")
        if isinstance(item, dict):
            compact["item"] = _compact_item(item)
        return {"method": method, "params": compact}
    return event


def _compact_item(item: dict[str, Any]) -> dict[str, Any]:
    compact = dict(item)
    text = compact.pop("text", None)
    if isinstance(text, str):
        compact["text_chars"] = len(text)
    content = compact.get("content")
    if isinstance(content, list):
        compact["content"] = [_compact_content(part) for part in content]
    return compact


def _compact_content(part: Any) -> Any:
    if not isinstance(part, dict):
        return part
    compact = dict(part)
    text = compact.pop("text", None)
    if isinstance(text, str):
        compact["text_chars"] = len(text)
    return compact


def _provider_bool(invocation: ModelInvocation, key: str, *, default: bool) -> bool:
    value = invocation.metadata.get(key)
    return _as_bool(value, default=default)


def _report(invocation: ModelInvocation, message: str) -> None:
    if invocation.progress is not None:
        invocation.progress(message)


def _report_event(invocation: ModelInvocation, method: Any, count: int) -> None:
    if not isinstance(method, str):
        _report(invocation, f"Codex event {count}...")
        return
    labels = {
        "turn/started": "Codex turn started...",
        "item/agentMessage/delta": "Receiving Codex response...",
        "item/completed": "Codex item completed...",
        "thread/tokenUsage/updated": "Codex usage received...",
        "thread/status/changed": "Codex status updated...",
        "turn/completed": "Codex turn completed...",
    }
    _report(invocation, labels.get(method, f"Codex event {count}: {method}"))


def _client_info() -> dict[str, str]:
    return {"name": "theml", "title": "TheML", "version": "0.1.0"}


def _app_server_command(provider_config: dict[str, Any]) -> list[str]:
    args = ["codex", "app-server"]
    disable_features = provider_config.get("disable_features") or [
        "apps",
        "browser_use",
        "computer_use",
        "multi_agent",
        "plugins",
        "shell_snapshot",
        "skill_mcp_dependency_install",
        "tool_suggest",
        "workspace_dependencies",
    ]
    if isinstance(disable_features, list):
        for feature in disable_features:
            args.extend(["--disable", str(feature)])
    config_overrides = {
        "project_doc_max_bytes": int(provider_config.get("project_doc_max_bytes") or 0),
        "project_doc_fallback_filenames": [],
        "web_search": "disabled",
        "mcp_servers": {},
        "features.skill_mcp_dependency_install": False,
    }
    extra = provider_config.get("config")
    if isinstance(extra, dict):
        config_overrides.update(extra)
    for key, value in config_overrides.items():
        args.extend(["-c", f"{key}={_toml_value(value)}"])
    return args


def _app_server_env(cwd: Path, provider_config: dict[str, Any], *, isolated_home: bool) -> dict[str, str] | None:
    if not isolated_home:
        return None
    home_value = str(provider_config.get("home_dir") or "tmp/app-server")
    codex_home = Path(home_value)
    if not codex_home.is_absolute():
        codex_home = cwd / codex_home
    codex_home.mkdir(parents=True, exist_ok=True)

    source_home = Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser()
    source_auth = source_home / "auth.json"
    target_auth = codex_home / "auth.json"
    if source_auth.exists() and source_auth.resolve() != target_auth.resolve() and not target_auth.exists():
        shutil.copy2(source_auth, target_auth)

    env = os.environ.copy()
    env["CODEX_HOME"] = str(codex_home)
    env["CODEX_SQLITE_HOME"] = str(codex_home)
    return env


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key} = {_toml_value(item)}" for key, item in value.items()) + "}"
    if value is None:
        return '""'
    return json.dumps(str(value))


def _as_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _request(request_id: int, method: str, params: dict[str, Any]) -> dict[str, Any]:
    return {"id": request_id, "method": method, "params": {key: value for key, value in params.items() if value is not None}}


def _send(proc: subprocess.Popen[str], message: dict[str, Any]) -> None:
    if proc.stdin is None:
        raise RuntimeError("Codex app-server stdin is closed.")
    proc.stdin.write(json.dumps(message) + "\n")
    proc.stdin.flush()


def _read_until_response(
    proc: subprocess.Popen[str],
    request_id: int,
    raw_lines: list[str],
    rpc_responses: list[dict[str, Any]],
    notifications: list[dict[str, Any]],
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    deadline = perf_counter() + timeout_seconds
    while perf_counter() < deadline:
        message = _read_message(proc, raw_lines, timeout_seconds=1)
        if message is None:
            if proc.poll() is not None:
                break
            continue
        if message.get("id") == request_id:
            rpc_responses.append(message)
            return message
        if "id" in message:
            rpc_responses.append(message)
        else:
            notifications.append(message)
    raise RuntimeError(f"Timed out waiting for Codex app-server response id={request_id}.")


def _read_message(
    proc: subprocess.Popen[str],
    raw_lines: list[str],
    *,
    timeout_seconds: int,
) -> dict[str, Any] | None:
    if proc.stdout is None:
        raise RuntimeError("Codex app-server stdout is closed.")
    ready, _, _ = select.select([proc.stdout], [], [], timeout_seconds)
    if not ready:
        return None
    line = proc.stdout.readline()
    if not line:
        return None
    raw_lines.append(line)
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Codex app-server returned invalid JSONL: {line!r}") from exc
    return value if isinstance(value, dict) else {"value": value}


def _raise_for_rpc_error(message: dict[str, Any], method: str) -> None:
    if "error" in message:
        raise RuntimeError(f"Codex app-server {method} failed: {message['error']!r}")


def _turn_status(turn_completed: dict[str, Any] | None) -> Any:
    if not isinstance(turn_completed, dict):
        return None
    turn = turn_completed.get("turn")
    if isinstance(turn, dict):
        return turn.get("status")
    return None


def _dump_obj(obj: Any) -> Any:
    if obj is None:
        return None
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, (list, tuple)):
        return [_dump_obj(item) for item in obj]
    if isinstance(obj, dict):
        return {str(key): _dump_obj(value) for key, value in obj.items()}
    return obj
