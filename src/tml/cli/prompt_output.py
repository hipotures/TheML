from __future__ import annotations

from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

from tml.utils.yaml_io import read_yaml


def print_prompt_choices(console: Console) -> None:
    table = Table(title="Available prompts", box=box.SIMPLE_HEAVY)
    table.add_column("Name", style="bold", no_wrap=True)
    table.add_column("Render", no_wrap=True)
    table.add_column("Probe", no_wrap=True)
    table.add_row(
        "metadata",
        "uv run tml prompt render metadata",
        "uv run tml prompt probe metadata",
    )
    table.add_row(
        "hypothesis",
        "uv run tml prompt render hypothesis",
        "uv run tml prompt probe hypothesis",
    )
    table.add_row(
        "code",
        "uv run tml prompt render <hypothesis_id> code",
        "uv run tml prompt probe <hypothesis_id> code",
    )
    console.print(table)


def print_prompt_render_summary(console: Console, path: Path) -> None:
    request_json = path.parent / "request.json"
    metadata = read_yaml(request_json) if request_json.exists() else {}
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    table = Table(title="Prompt rendered", box=box.SIMPLE_HEAVY, show_header=False, pad_edge=False)
    table.add_column("Field", style="bold", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("Output", str(path))
    if isinstance(metadata, dict):
        template_id = metadata.get("template_id")
        template_path = metadata.get("template_path")
        if template_id:
            table.add_row("Template", str(template_id))
        if template_path:
            table.add_row("Template file", str(template_path))
    table.add_row("Size", f"{len(text.encode('utf-8'))} bytes")
    table.add_row("Lines", str(len(text.splitlines())))
    console.print(table)
    console.print(str(path))


def print_prompt_probe_summary(console: Console, out_dir: Path) -> None:
    request = out_dir / "request.md"
    response = out_dir / "response.md"
    request_json = out_dir / "request.json"
    provider_raw = out_dir / "provider-raw.json"
    request_meta = read_yaml(request_json) if request_json.exists() else {}
    provider_meta = read_yaml(provider_raw) if provider_raw.exists() else {}
    raw_provider = provider_meta.get("raw", {}) if isinstance(provider_meta, dict) else {}
    role = _metadata_value(request_meta, "role") or _metadata_value(raw_provider, "role")
    template = _metadata_value(request_meta, "template_id")
    provider = _metadata_value(request_meta, "provider") or _metadata_value(raw_provider, "provider")
    model = _metadata_value(request_meta, "model") or _metadata_value(raw_provider, "model")
    provider_kind = _metadata_value(request_meta, "provider_kind")
    simulated = _metadata_value(raw_provider, "simulated")
    table = Table(title="Prompt probe", box=box.SIMPLE_HEAVY, show_header=False, pad_edge=False)
    table.add_column("Field", style="bold", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("Action", _probe_action(str(role) if role else None))
    if role:
        table.add_row("Role", str(role))
    if template:
        table.add_row("Template", str(template))
    if provider:
        table.add_row("Provider", str(provider))
    if provider_kind and provider_kind != provider:
        table.add_row("Provider kind", str(provider_kind))
    if model:
        table.add_row("Model", str(model))
    if simulated is not None:
        table.add_row("Simulated", str(simulated).lower())
    if request.exists():
        table.add_row("Request", str(request))
    if response.exists():
        table.add_row("Response", str(response))
    table.add_row("Output dir", str(out_dir))
    console.print(table)
    console.print(str(response if response.exists() else out_dir))


def _metadata_value(metadata: object, key: str) -> object | None:
    return metadata.get(key) if isinstance(metadata, dict) else None


def _probe_action(role: str | None) -> str:
    if role == "hypothesis":
        return "Sent hypothesis prompt to configured model"
    if role == "code":
        return "Sent code-generation prompt to configured model"
    if role == "metadata":
        return "Sent metadata prompt to configured model"
    return "Sent prompt to configured model"
