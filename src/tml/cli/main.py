from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.tree import Tree

from tml.cli.prompt_output import print_prompt_choices, print_prompt_probe_summary, print_prompt_render_summary
from tml.core.config import active_mode, active_profile_id, load_project_config
from tml.core.errors import TmlError
from tml.core.kaggle import download_competition_data
from tml.core.paths import active_project_ref, workspace_root
from tml.core.profiles import load_profile, profile_hash
from tml.core.project import default_download_data, default_project_kind, init_project, use_project
from tml.db.reindex import reindex_project
from tml.db.state import (
    best_score,
    materialization_rows,
    root_counts,
    root_hypothesis_rows,
    root_run_rows,
)
from tml.hypotheses.generate import GeneratedHypothesis, generate_missing_root_hypotheses
from tml.hypotheses.materialize import materialize_missing
from tml.hypotheses.run import run_missing
from tml.prompts.diff import diff_prompt
from tml.prompts.probe import probe_prompt, render_prompt
from tml.utils.yaml_io import read_yaml


console = Console(highlight=False)
app = typer.Typer(no_args_is_help=True)
init_app = typer.Typer(no_args_is_help=True)
project_app = typer.Typer(no_args_is_help=True)
root_app = typer.Typer(no_args_is_help=True)
prompt_app = typer.Typer(no_args_is_help=True)
kaggle_app = typer.Typer(no_args_is_help=True)

ROOT_HYPOTHESIS_COLUMNS = [
    {"key": "id", "label": "ID", "style": "bold", "no_wrap": True},
    {"key": "status", "label": "S", "justify": "center", "no_wrap": True},
    {"key": "created_at", "label": "Created", "no_wrap": True},
    {"key": "model", "label": "Model", "no_wrap": True},
    {"key": "reasoning_tokens", "label": "Res/Tokens", "justify": "right", "no_wrap": True},
    {"key": "duration", "label": "Gen", "no_wrap": True},
    {"key": "summary", "label": "Summary", "overflow": "fold", "min_width": 40, "ratio": 1, "truncate": True},
]

app.add_typer(init_app, name="init")
app.add_typer(project_app, name="project")
app.add_typer(root_app, name="root")
app.add_typer(prompt_app, name="prompt")
app.add_typer(kaggle_app, name="kaggle")

EXTRA = {"allow_extra_args": True, "ignore_unknown_options": True}


@init_app.command("project", context_settings=EXTRA)
def init_project_cmd(ctx: typer.Context, slug: str) -> None:
    try:
        overrides = _overrides(ctx.args)
        root = workspace_root()
        kind = str(overrides.get("kind") or default_project_kind(root) or "kaggle")
        download = _bool(overrides["download"]) if "download" in overrides else default_download_data(root)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("Preparing project...", total=None)

            def report(message: str) -> None:
                progress.update(task, description=message)

            ref = init_project(root, slug, kind, download=download, progress=report)
        _print_init_project_summary(root, ref.path, slug, download=download)
    except Exception as exc:
        _abort(exc)


@project_app.command("use")
def use_project_cmd(slug: str) -> None:
    try:
        ref = use_project(workspace_root(), slug)
        console.print(f"Active project: {ref.slug}")
    except Exception as exc:
        _abort(exc)


@root_app.command("status", context_settings=EXTRA)
def root_status_cmd(ctx: typer.Context) -> None:
    _ = ctx
    try:
        ref = active_project_ref()
        config = load_project_config(ref.path)
        counts = root_counts(ref.path)
        mode = active_mode(config)
        profile_id = active_profile_id(config, mode)
        active_hash = profile_hash(ref.path, mode, profile_id)
        best = best_score(ref.path)
        console.print(f"Active project: {ref.slug}")
        console.print(f"Target ROOT count: {config.get('root', {}).get('target_count', 20)}")
        console.print(f"Active mode: {mode}")
        console.print(f"Active profile: {profile_id} ({active_hash})")
        console.print(f"Hypotheses: {counts['hypotheses']}")
        console.print(f"Materialized: {counts['materialized']}")
        console.print(f"Evaluated: {counts['evaluated']}")
        console.print(f"Incomplete nodes: {counts['incomplete']}")
        console.print(f"Best ROOT score: {best if best is not None else 'n/a'}")
    except Exception as exc:
        _abort(exc)


@root_app.command("generate", context_settings=EXTRA)
def root_generate_cmd(ctx: typer.Context) -> None:
    try:
        ref = active_project_ref()
        _reject_positional(ctx.args, "tml root generate")
        overrides = _overrides(ctx.args)
        _validate_override_keys(overrides, {"count", "json", "json_output"}, "tml root generate")
        count = int(overrides["count"]) if "count" in overrides else None
        json_output = _bool(overrides.get("json", False)) or _bool(overrides.get("json_output", False))
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=json_output,
            disable=json_output,
        ) as progress:
            task = progress.add_task("Generating ROOT hypotheses...", total=None)

            def report(message: str) -> None:
                progress.update(task, description=message)

            created = generate_missing_root_hypotheses(ref.path, count=count, progress=report)
        if json_output:
            _print_generated_hypotheses_json(ref.path, created)
            return
        _print_generated_hypotheses(ref.path, created)
    except Exception as exc:
        _abort(exc)


@root_app.command("materialize", context_settings=EXTRA)
def root_materialize_cmd(ctx: typer.Context) -> None:
    try:
        ref = active_project_ref()
        _reject_positional(ctx.args, "tml root materialize")
        config = load_project_config(ref.path)
        overrides = _overrides(ctx.args)
        _validate_override_keys(overrides, {"mode", "hypothesis", "id"}, "tml root materialize")
        mode = str(overrides.get("mode") or active_mode(config))
        hypothesis_id = overrides.get("hypothesis") or overrides.get("id")
        created = _materialize_with_progress(
            ref.path,
            mode=mode,
            hypothesis_id=str(hypothesis_id) if hypothesis_id else None,
        )
        _print_root_materializations(ref.path, mode=mode, created_count=created, hypothesis_id=str(hypothesis_id) if hypothesis_id else None)
    except Exception as exc:
        _abort(exc)


def _materialize_with_progress(project_dir: Path, *, mode: str, hypothesis_id: str | None) -> int:
    state: dict[str, object] = {
        "message": "Preparing materialization...",
        "timeout": 1,
        "started_at": time.monotonic(),
        "created": 0,
        "error": None,
    }
    lock = threading.Lock()

    def report(message: str, timeout_seconds: int | None = None) -> None:
        with lock:
            state["message"] = message
            state["timeout"] = max(1, int(timeout_seconds or 1))
            state["started_at"] = time.monotonic()

    def run() -> None:
        try:
            created = materialize_missing(project_dir, mode=mode, hypothesis_id=hypothesis_id, progress=report)
            with lock:
                state["created"] = created
        except Exception as exc:  # pragma: no cover - re-raised in caller thread
            with lock:
                state["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed:.0f}/{task.total:.0f}s"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Preparing materialization...", total=1)
        worker.start()
        while worker.is_alive():
            with lock:
                message = str(state["message"])
                timeout_seconds = int(state["timeout"])
                started_at = float(state["started_at"])
            elapsed = min(timeout_seconds, max(0.0, time.monotonic() - started_at))
            progress.update(task, description=message, total=timeout_seconds, completed=elapsed)
            time.sleep(0.25)
        worker.join()
        with lock:
            error = state["error"]
            created = int(state["created"])
            timeout_seconds = int(state["timeout"])
            message = str(state["message"])
            started_at = float(state["started_at"])
        elapsed = min(timeout_seconds, max(0.0, time.monotonic() - started_at))
        progress.update(task, description=message, total=timeout_seconds, completed=elapsed)
    if isinstance(error, Exception):
        raise error
    return created


@root_app.command("run", context_settings=EXTRA)
def root_run_cmd(ctx: typer.Context) -> None:
    try:
        ref = active_project_ref()
        _reject_positional(ctx.args, "tml root run")
        overrides = _overrides(ctx.args)
        mode = str(overrides["mode"]) if "mode" in overrides else None
        config = load_project_config(ref.path)
        active_run_mode = mode or active_mode(config)
        allowed = {"mode", "hypothesis", "id"} | _profile_override_keys(ref.path, active_run_mode)
        _validate_override_keys(overrides, allowed, "tml root run")
        hypothesis_id = overrides.get("hypothesis") or overrides.get("id")
        run_overrides = {key: value for key, value in overrides.items() if key not in {"mode", "hypothesis", "id"}}
        executed_ids = run_missing(
            ref.path,
            mode=mode,
            hypothesis_id=str(hypothesis_id) if hypothesis_id else None,
            profile_overrides=run_overrides,
        )
        _print_root_run_summary(
            ref.path,
            mode=active_run_mode,
            executed_ids=set(executed_ids),
            hypothesis_id=str(hypothesis_id) if hypothesis_id else None,
        )
    except Exception as exc:
        _abort(exc)


@root_app.command("ensure", context_settings=EXTRA)
def root_ensure_cmd(ctx: typer.Context) -> None:
    try:
        ref = active_project_ref()
        _reject_positional(ctx.args, "tml root ensure")
        config = load_project_config(ref.path)
        overrides = _overrides(ctx.args)
        mode = str(overrides.get("mode") or active_mode(config))
        allowed = {"count", "mode", "hypothesis", "id"} | _profile_override_keys(ref.path, mode)
        _validate_override_keys(overrides, allowed, "tml root ensure")
        count = int(overrides["count"]) if "count" in overrides else None
        hypothesis_id = overrides.get("hypothesis") or overrides.get("id")
        generated = generate_missing_root_hypotheses(ref.path, count=count)
        materialized = materialize_missing(
            ref.path,
            mode=mode,
            hypothesis_id=str(hypothesis_id) if hypothesis_id else None,
        )
        run_overrides = {key: value for key, value in overrides.items() if key not in {"count", "mode", "hypothesis", "id"}}
        ran = run_missing(
            ref.path,
            mode=mode,
            hypothesis_id=str(hypothesis_id) if hypothesis_id else None,
            profile_overrides=run_overrides,
        )
        reindex_project(ref.path, ref.db_path)
        console.print(f"Generated: {len(generated)}; materialized: {materialized}; executed: {len(ran)}")
    except Exception as exc:
        _abort(exc)


@app.command("reindex")
def reindex_cmd(scope: str | None = None, run_id: str | None = None) -> None:
    try:
        _ = (scope, run_id)
        ref = active_project_ref()
        counts = reindex_project(ref.path, ref.db_path)
        console.print(
            f"Reindexed {counts['hypotheses']} hypotheses, "
            f"{counts['materializations']} materializations, {counts['nodes']} nodes"
        )
    except Exception as exc:
        _abort(exc)


@kaggle_app.command("download")
def kaggle_download_cmd() -> None:
    try:
        ref = active_project_ref()
        config = load_project_config(ref.path)
        slug = str(config.get("kaggle_slug") or ref.slug)
        data_dir = ref.path / str(config.get("data_dir") or "data")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task(f"Downloading Kaggle data for {slug}...", total=None)

            def report(message: str) -> None:
                progress.update(task, description=message)

            download_competition_data(slug, data_dir, progress=report)
        _print_kaggle_download_summary(ref.root, slug, data_dir)
    except Exception as exc:
        _abort(exc)


@prompt_app.command("render", context_settings=EXTRA)
def prompt_render_cmd(ctx: typer.Context) -> None:
    try:
        ref = active_project_ref()
        positional = _positional(ctx.args)
        if not positional:
            print_prompt_choices(console)
            return
        target, stage = _prompt_target_stage(positional)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("Rendering prompt...", total=None)
            path = render_prompt(
                ref.path,
                target=target,
                stage=stage,
                tmp_root=_tmp_root(),
            )
            progress.update(task, description=f"Prompt rendered: {path}")
        print_prompt_render_summary(console, path)
    except Exception as exc:
        _abort(exc)


@prompt_app.command("probe", context_settings=EXTRA)
def prompt_probe_cmd(ctx: typer.Context) -> None:
    try:
        ref = active_project_ref()
        overrides = _overrides(ctx.args)
        positional = _positional(ctx.args)
        if not positional:
            print_prompt_choices(console)
            return
        tmp = _bool(overrides.get("tmp", True))
        model = str(overrides["model"]) if "model" in overrides else None
        profile_overrides = {key: value for key, value in overrides.items() if key not in {"tmp", "model"}}
        target, stage = _prompt_target_stage(positional)
        path = _probe_with_progress(
            ref.path,
            tmp=tmp,
            target=target,
            stage=stage,
            model_override=model,
            profile_overrides=profile_overrides,
        )
        print_prompt_probe_summary(console, path)
    except Exception as exc:
        _abort(exc)


def _probe_with_progress(
    project_dir: Path,
    *,
    tmp: bool,
    target: str | None,
    stage: str | None,
    model_override: str | None,
    profile_overrides: dict[str, object] | None,
) -> Path:
    state: dict[str, object] = {
        "message": "Preparing prompt probe...",
        "timeout": 1,
        "started_at": time.monotonic(),
        "path": None,
        "error": None,
    }
    lock = threading.Lock()

    def report(message: str, timeout_seconds: int | None = None) -> None:
        with lock:
            state["message"] = message
            state["timeout"] = max(1, int(timeout_seconds or 1))
            state["started_at"] = time.monotonic()

    def run() -> None:
        try:
            path = probe_prompt(
                project_dir,
                tmp=tmp,
                target=target,
                stage=stage,
                model_override=model_override,
                profile_overrides=profile_overrides,
                tmp_root=_tmp_root(),
                progress=report,
            )
            with lock:
                state["path"] = path
        except Exception as exc:  # pragma: no cover - re-raised in caller thread
            with lock:
                state["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed:.0f}/{task.total:.0f}s"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Preparing prompt probe...", total=1)
        worker.start()
        while worker.is_alive():
            with lock:
                message = str(state["message"])
                timeout_seconds = int(state["timeout"])
                started_at = float(state["started_at"])
            elapsed = min(timeout_seconds, max(0.0, time.monotonic() - started_at))
            progress.update(task, description=message, total=timeout_seconds, completed=elapsed)
            time.sleep(0.25)
        worker.join()
        with lock:
            error = state["error"]
            path = state["path"]
            timeout_seconds = int(state["timeout"])
            message = str(state["message"])
            started_at = float(state["started_at"])
        elapsed = min(timeout_seconds, max(0.0, time.monotonic() - started_at))
        progress.update(task, description=message, total=timeout_seconds, completed=elapsed)
    if isinstance(error, Exception):
        raise error
    if not isinstance(path, Path):
        raise TmlError("Prompt probe did not return an output directory.")
    return path


@prompt_app.command("diff", context_settings=EXTRA)
def prompt_diff_cmd(ctx: typer.Context, target: str | None = None, stage: str | None = None) -> None:
    try:
        ref = active_project_ref()
        args = list(ctx.args)
        target = target or (args.pop(0) if args else None)
        stage = stage or (args.pop(0) if args else None)
        if target is None:
            raise TmlError("Missing prompt target. Example: tml prompt diff 1 code")
        label, diff = diff_prompt(ref.path, target, stage or "code")
        console.print(label)
        if diff:
            console.print(diff)
    except Exception as exc:
        _abort(exc)


def _overrides(args: list[str]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for arg in args:
        if "=" not in arg:
            continue
        key, value = arg.split("=", 1)
        parsed[key] = _coerce(value)
    return parsed


def _validate_override_keys(overrides: dict[str, object], allowed: set[str], command: str) -> None:
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        allowed_list = ", ".join(sorted(allowed))
        unknown_list = ", ".join(unknown)
        raise TmlError(f"Unknown parameter for {command}: {unknown_list}. Allowed parameters: {allowed_list}")


def _reject_positional(args: list[str], command: str) -> None:
    positional = _positional(args)
    if positional:
        raise TmlError(f"Unexpected argument for {command}: {positional[0]}")


def _profile_override_keys(project_dir: Path, mode: str) -> set[str]:
    config = load_project_config(project_dir)
    profile_id = active_profile_id(config, mode)
    try:
        profile = load_profile(project_dir, mode, profile_id)
    except Exception:
        profile = {}
    ignored = {"schema_version", "profile_id", "source_profile", "mode"}
    keys = {key for key in profile if key not in ignored}
    keys.update({"preset", "time", "preprocess_timeout"})
    return keys


def _positional(args: list[str]) -> list[str]:
    return [arg for arg in args if "=" not in arg]


def _prompt_target_stage(positional: list[str]) -> tuple[str | None, str | None]:
    if positional == ["metadata"]:
        return "project", "metadata"
    if positional in (["hypothesis"], ["root", "hypothesis"]):
        return None, None
    return (
        positional[0] if len(positional) >= 1 else None,
        positional[1] if len(positional) >= 2 else None,
    )


def _coerce(value: str) -> object:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        return value


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _tmp_root() -> Path:
    return Path("/tmp")


def _print_init_project_summary(root: Path, project_dir: Path, slug: str, *, download: bool) -> None:
    data_dir = project_dir / "data"
    table = Table(title="Project initialized", box=box.SIMPLE_HEAVY, show_header=False, pad_edge=False)
    table.add_column("Field", style="bold", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("Initialized project", _repo_path(project_dir, root))
    table.add_row("Project config", _repo_path(project_dir / "project.yaml", root))
    table.add_row("Task file", _repo_path(project_dir / "task.md", root))
    table.add_row("Data dir", _repo_path(data_dir, root))
    metadata = _metadata_run_summary(project_dir)
    if metadata:
        table.add_row("Metadata model", str(metadata["model"]))
        table.add_row("Metadata result", str(metadata["result"]))
    if download:
        files = _data_files(data_dir)
        suffix = f" ({', '.join(files)})" if files else ""
        table.add_row("Kaggle data", f"downloaded{suffix}")
    else:
        table.add_row("Kaggle data", "skipped (download=false)")
    console.print(table)
    _print_data_tree(root, project_dir)
    console.print(f"[bold]Next:[/bold] uv run tml project use {slug}")


def _print_generated_hypotheses(project_dir: Path, created: list[GeneratedHypothesis]) -> None:
    created_ids = {item.hypothesis_id for item in created}
    _print_existing_root_hypotheses(project_dir, created_ids=created_ids)


def _print_generated_hypotheses_json(project_dir: Path, created: list[GeneratedHypothesis]) -> None:
    created_ids = {item.hypothesis_id for item in created}
    payload = {
        "created": sorted(created_ids),
        "hypotheses": [
            _root_hypothesis_json_row(row, created_ids=created_ids)
            for row in root_hypothesis_rows(project_dir)
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _print_existing_root_hypotheses(project_dir: Path, *, created_ids: set[str]) -> None:
    table = Table(title="Existing ROOT hypotheses", box=box.SIMPLE_HEAVY)
    for column in ROOT_HYPOTHESIS_COLUMNS:
        table.add_column(
            str(column["label"]),
            style=str(column["style"]) if column.get("style") else None,
            justify=str(column["justify"]) if column.get("justify") else "left",
            no_wrap=bool(column.get("no_wrap", False)),
            overflow=str(column["overflow"]) if column.get("overflow") else None,
            min_width=int(column["min_width"]) if column.get("min_width") else None,
            ratio=int(column["ratio"]) if column.get("ratio") else None,
        )
    summary_limit = 30 + max(0, _env_int("TML_WIDE_TERMINAL", 0))
    rows = []
    for db_row in root_hypothesis_rows(project_dir):
        row = _root_hypothesis_row(db_row, created_ids=created_ids)
        if not row:
            continue
        rows.append(
            (
                bool(row["is_new"]),
                _root_hypothesis_table_values(row, summary_limit=summary_limit),
            )
        )
    old_rows = [row for is_new, row in rows if not is_new]
    new_rows = [row for is_new, row in rows if is_new]
    for row in old_rows:
        table.add_row(*row)
    if old_rows and new_rows:
        table.add_row("NEW", *["" for _ in ROOT_HYPOTHESIS_COLUMNS[1:]], style="reverse")
    for row in new_rows:
        table.add_row(*row, style="bold")
    console.print(table)


def _print_root_materializations(
    project_dir: Path,
    *,
    mode: str,
    created_count: int,
    hypothesis_id: str | None,
) -> None:
    target_id = hypothesis_id.zfill(6) if hypothesis_id else None
    table = Table(title=f"ROOT materializations (created: {created_count})", box=box.SIMPLE_HEAVY)
    table.add_column("ID", style="bold", no_wrap=True)
    table.add_column("S", no_wrap=True)
    table.add_column("Mode", no_wrap=True)
    table.add_column("File", no_wrap=True)
    table.add_column("Model", no_wrap=True)
    table.add_column("Res/Tokens", justify="right", no_wrap=True)
    table.add_column("Gen", justify="right", no_wrap=True)
    table.add_column("Summary", overflow="fold", min_width=40, ratio=1)
    summary_limit = 30 + max(0, _env_int("TML_WIDE_TERMINAL", 0))
    for db_row in materialization_rows(project_dir, mode=mode, hypothesis_id=target_id):
        table.add_row(*_root_materialization_row(db_row, summary_limit=summary_limit))
    console.print(table)


def _root_materialization_row(db_row: dict[str, object], *, summary_limit: int) -> list[str]:
    return [
        str(db_row.get("hypothesis_id") or ""),
        "⌘",
        str(db_row.get("mode") or ""),
        str(db_row.get("file") or ""),
        str(db_row.get("model") or ""),
        _token_summary(db_row),
        _seconds_text(db_row.get("generation_seconds")),
        _short_text(str(db_row.get("summary") or ""), summary_limit),
    ]


def _print_root_run_summary(
    project_dir: Path,
    *,
    mode: str,
    executed_ids: set[str],
    hypothesis_id: str | None,
) -> None:
    config = load_project_config(project_dir)
    profile_id = active_profile_id(config, mode)
    table = Table(title=f"ROOT run (executed: {len(executed_ids)})", box=box.SIMPLE_HEAVY)
    table.add_column("ID", style="bold", no_wrap=True)
    table.add_column("S", justify="center", no_wrap=True)
    table.add_column("Created", no_wrap=True)
    table.add_column("Model", no_wrap=True)
    table.add_column("Res/Tokens", justify="right", no_wrap=True)
    table.add_column("Gen", no_wrap=True)
    table.add_column("Run", justify="right", no_wrap=True)
    table.add_column("Score", justify="right", no_wrap=True)
    table.add_column("Node", no_wrap=True)
    table.add_column("Summary", overflow="fold", min_width=36, ratio=1)
    summary_limit = 30 + max(0, _env_int("TML_WIDE_TERMINAL", 0))
    old_rows: list[list[str]] = []
    new_rows: list[list[str]] = []
    for db_row in root_run_rows(project_dir, mode=mode, profile_id=profile_id):
        row = _root_run_row(db_row, summary_limit=summary_limit)
        hypothesis_id_value = str(db_row.get("hypothesis_id") or "")
        if hypothesis_id_value in executed_ids:
            new_rows.append(row)
        else:
            old_rows.append(row)
    for row in old_rows:
        table.add_row(*row)
    if old_rows and new_rows:
        table.add_row(*["" for _ in range(10)], style="reverse")
    for row in new_rows:
        table.add_row(*row, style="bold")
    console.print(table)


def _root_run_row(
    db_row: dict[str, object],
    *,
    summary_limit: int,
) -> list[str]:
    node_status = str(db_row.get("node_status") or "")
    if node_status:
        status = "▶" if node_status == "complete" else "⚠"
        score = _format_score(db_row.get("metric"))
        node = str(db_row.get("node_id") or "")
        run_duration = _seconds_text(db_row.get("run_seconds"))
    elif db_row.get("code_hash"):
        status = "⌘"
        score = ""
        node = ""
        run_duration = ""
    else:
        status = "◇"
        score = ""
        node = ""
        run_duration = ""
    return [
        str(db_row.get("hypothesis_id") or ""),
        status,
        str(db_row.get("created_at") or ""),
        str(db_row.get("model") or ""),
        _token_summary(db_row),
        _seconds_text(db_row.get("generation_seconds")),
        run_duration,
        score,
        node,
        _short_text(str(db_row.get("summary") or ""), summary_limit),
    ]


def _format_score(value: object) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.5f}"
    return ""


def _token_summary(row: dict[str, object]) -> str:
    reasoning = row.get("reasoning_tokens")
    total = row.get("total_tokens")
    if isinstance(reasoning, int) and isinstance(total, int):
        return f"{reasoning}/{total}"
    if isinstance(total, int):
        return str(total)
    return ""


def _seconds_text(value: object) -> str:
    if isinstance(value, int):
        return f"{value}s"
    if isinstance(value, float):
        return f"{round(value)}s"
    return ""


def _root_hypothesis_json_row(db_row: dict[str, object], *, created_ids: set[str]) -> dict[str, object]:
    row = _root_hypothesis_row(db_row, created_ids=created_ids)
    if not row:
        return {}
    return {
        "is_new": bool(row["is_new"]),
        **{str(column["key"]): row.get(str(column["key"]), "") for column in ROOT_HYPOTHESIS_COLUMNS},
    }


def _root_hypothesis_table_values(row: dict[str, object], *, summary_limit: int) -> list[str]:
    values = []
    for column in ROOT_HYPOTHESIS_COLUMNS:
        key = str(column["key"])
        value = str(row.get(key, ""))
        if column.get("truncate"):
            value = _short_text(value, summary_limit)
        values.append(value)
    return values


def _root_hypothesis_row(db_row: dict[str, object], *, created_ids: set[str]) -> dict[str, object]:
    hypothesis_id = str(db_row.get("hypothesis_id") or "")
    return {
        "id": hypothesis_id,
        "is_new": hypothesis_id in created_ids,
        "status": str(db_row.get("status_icon") or ""),
        "created_at": str(db_row.get("created_at") or ""),
        "model": str(db_row.get("model") or ""),
        "reasoning_tokens": _token_summary(db_row),
        "duration": _seconds_text(db_row.get("generation_seconds")),
        "summary": str(db_row.get("summary") or ""),
    }


def _short_text(value: str, limit: int) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        value = _dotenv_value(name)
    if value is None:
        value = _config_env_value(name)
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _dotenv_value(name: str) -> str | None:
    path = workspace_root() / ".env"
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == name:
            return value.strip().strip("\"'")
    return None


def _config_env_value(name: str) -> object | None:
    config = read_yaml(workspace_root() / "tml.yaml")
    env = config.get("env") if isinstance(config, dict) else None
    if not isinstance(env, dict):
        return None
    return env.get(name)


def _metadata_run_summary(project_dir: Path) -> dict[str, str] | None:
    request_path = project_dir / "logs" / "project-metadata" / "request.json"
    response_path = project_dir / "logs" / "project-metadata" / "response.json"
    return _run_summary_from_paths(request_path, response_path)


def _run_summary_from_paths(request_path: Path, response_path: Path) -> dict[str, str] | None:
    if not request_path.exists() or not response_path.exists():
        return None
    try:
        request = read_yaml(request_path)
        response = read_yaml(response_path)
    except Exception:
        return None
    if not isinstance(request, dict) or not isinstance(response, dict):
        return None
    model = str(request.get("model") or response.get("model") or "unknown")
    status = str(response.get("status") or "unknown")
    wall_ms = response.get("wall_ms")
    duration = ""
    result = status
    if isinstance(wall_ms, int):
        duration = f"{round(wall_ms / 1000)}s"
        result = f"{status} in {duration}"
    total_tokens = _run_total_tokens(response)
    reasoning_tokens = _run_reasoning_tokens(response)
    token_summary = ""
    if reasoning_tokens is not None and total_tokens is not None:
        token_summary = f"{reasoning_tokens}/{total_tokens}"
    elif total_tokens is not None:
        token_summary = str(total_tokens)
    return {"model": model, "result": result, "duration": duration, "reasoning_tokens": token_summary}


def _run_total_tokens(response: dict[str, object]) -> int | None:
    total = _run_total_token_usage(response)
    if total is None:
        return None
    total_tokens = total.get("totalTokens")
    return total_tokens if isinstance(total_tokens, int) else None


def _run_reasoning_tokens(response: dict[str, object]) -> int | None:
    total = _run_total_token_usage(response)
    if total is None:
        return None
    reasoning_tokens = total.get("reasoningOutputTokens")
    return reasoning_tokens if isinstance(reasoning_tokens, int) else None


def _run_total_token_usage(response: dict[str, object]) -> dict[str, object] | None:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    token_usage = usage.get("tokenUsage")
    if not isinstance(token_usage, dict):
        return None
    total = token_usage.get("total")
    if not isinstance(total, dict):
        return None
    return total


def _print_kaggle_download_summary(root: Path, slug: str, data_dir: Path) -> None:
    table = Table(title="Kaggle data downloaded", box=box.SIMPLE_HEAVY, show_header=False, pad_edge=False)
    table.add_column("Field", style="bold", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("Competition", slug)
    table.add_row("Data dir", _repo_path(data_dir, root))
    files = _data_files(data_dir)
    table.add_row("Files", ", ".join(files) if files else "none")
    console.print(table)


def _print_data_tree(root: Path, project_dir: Path) -> None:
    data_dir = project_dir / "data"
    files = _data_files(data_dir)
    if not files:
        return
    tree = Tree(_repo_path(project_dir, root))
    data = tree.add("data")
    for filename in files:
        data.add(filename)
    console.print(tree)


def _repo_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _data_files(data_dir: Path) -> list[str]:
    if not data_dir.exists():
        return []
    return sorted(path.name for path in data_dir.iterdir() if path.is_file())


def _abort(exc: Exception) -> None:
    message = str(exc)
    console.print(f"Error: {message}")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
