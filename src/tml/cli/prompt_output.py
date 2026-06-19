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
