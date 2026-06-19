from __future__ import annotations

from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

from tml.core.config import active_mode, active_profile_id, load_project_config
from tml.core.errors import TmlError
from tml.core.kaggle import download_competition_data
from tml.core.paths import active_project_ref, workspace_root
from tml.core.profiles import profile_hash
from tml.core.project import default_download_data, default_project_kind, init_project, use_project
from tml.db.reindex import reindex_project
from tml.hypotheses.generate import generate_missing_root_hypotheses
from tml.hypotheses.materialize import materialize_missing
from tml.hypotheses.run import run_missing
from tml.hypotheses.status import filesystem_counts
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
        counts = filesystem_counts(ref.path)
        mode = active_mode(config)
        profile_id = active_profile_id(config, mode)
        active_hash = profile_hash(ref.path, mode, profile_id)
        best = _best_score(ref.path)
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
        overrides = _overrides(ctx.args)
        count = int(overrides["count"]) if "count" in overrides else None
        created = generate_missing_root_hypotheses(ref.path, count=count)
        console.print(f"Generated ROOT hypotheses: {created}")
    except Exception as exc:
        _abort(exc)


@root_app.command("materialize", context_settings=EXTRA)
def root_materialize_cmd(ctx: typer.Context) -> None:
    try:
        ref = active_project_ref()
        config = load_project_config(ref.path)
        overrides = _overrides(ctx.args)
        mode = str(overrides.get("mode") or active_mode(config))
        created = materialize_missing(ref.path, mode=mode)
        console.print(f"Materializations created: {created}")
    except Exception as exc:
        _abort(exc)


@root_app.command("run", context_settings=EXTRA)
def root_run_cmd(ctx: typer.Context) -> None:
    try:
        ref = active_project_ref()
        overrides = _overrides(ctx.args)
        mode = str(overrides["mode"]) if "mode" in overrides else None
        ran = run_missing(ref.path, mode=mode)
        console.print(f"ROOT nodes executed: {ran}")
    except Exception as exc:
        _abort(exc)


@root_app.command("ensure", context_settings=EXTRA)
def root_ensure_cmd(ctx: typer.Context) -> None:
    try:
        ref = active_project_ref()
        config = load_project_config(ref.path)
        overrides = _overrides(ctx.args)
        count = int(overrides["count"]) if "count" in overrides else None
        mode = str(overrides.get("mode") or active_mode(config))
        generated = generate_missing_root_hypotheses(ref.path, count=count)
        materialized = materialize_missing(ref.path, mode=mode)
        ran = run_missing(ref.path, mode=mode)
        reindex_project(ref.path, ref.db_path)
        console.print(f"Generated: {generated}; materialized: {materialized}; executed: {ran}")
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
            _print_prompt_choices()
            return
        target, stage = _prompt_target_stage(positional)
        path = render_prompt(
            ref.path,
            target=target,
            stage=stage,
            tmp_root=_tmp_root(),
        )
        console.print(str(path))
    except Exception as exc:
        _abort(exc)


@prompt_app.command("probe", context_settings=EXTRA)
def prompt_probe_cmd(ctx: typer.Context) -> None:
    try:
        ref = active_project_ref()
        overrides = _overrides(ctx.args)
        positional = _positional(ctx.args)
        tmp = _bool(overrides.get("tmp", True))
        model = str(overrides["model"]) if "model" in overrides else None
        target, stage = _prompt_target_stage(positional)
        if model and model.startswith("codex:"):
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=False,
            ) as progress:
                task = progress.add_task("Rendering prompt...", total=None)

                def report(message: str) -> None:
                    progress.update(task, description=message)

                path = probe_prompt(
                    ref.path,
                    tmp=tmp,
                    target=target,
                    stage=stage,
                    model_override=model,
                    tmp_root=_tmp_root(),
                    progress=report,
                )
        else:
            path = probe_prompt(
                ref.path,
                tmp=tmp,
                target=target,
                stage=stage,
                model_override=model,
                tmp_root=_tmp_root(),
            )
        console.print(str(path))
    except Exception as exc:
        _abort(exc)


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


def _positional(args: list[str]) -> list[str]:
    return [arg for arg in args if "=" not in arg]


def _prompt_target_stage(positional: list[str]) -> tuple[str | None, str | None]:
    if positional == ["metadata"]:
        return "project", "metadata"
    if positional == ["root", "hypothesis"]:
        return None, None
    return (
        positional[0] if len(positional) >= 1 else None,
        positional[1] if len(positional) >= 2 else None,
    )


def _print_prompt_choices() -> None:
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
        "root hypothesis",
        "uv run tml prompt render root hypothesis",
        "uv run tml prompt probe root hypothesis",
    )
    table.add_row(
        "code",
        "uv run tml prompt render <hypothesis_id> code",
        "uv run tml prompt probe <hypothesis_id> code",
    )
    console.print(table)


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


def _metadata_run_summary(project_dir: Path) -> dict[str, str] | None:
    request_path = project_dir / "logs" / "project-metadata" / "request.json"
    response_path = project_dir / "logs" / "project-metadata" / "response.json"
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
    role_options = request.get("role_options") if isinstance(request.get("role_options"), dict) else {}
    timeout = role_options.get("timeout_seconds") if isinstance(role_options, dict) else None
    if timeout is not None:
        model = f"{model} (timeout={timeout}s)"
    status = str(response.get("status") or "unknown")
    wall_ms = response.get("wall_ms")
    result = status
    if isinstance(wall_ms, int):
        result = f"{status} in {wall_ms / 1000:.1f}s"
    total_tokens = _metadata_total_tokens(response)
    if total_tokens is not None:
        result = f"{result}, totalTokens={total_tokens}"
    return {"model": model, "result": result}


def _metadata_total_tokens(response: dict[str, object]) -> int | None:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    token_usage = usage.get("tokenUsage")
    if not isinstance(token_usage, dict):
        return None
    total = token_usage.get("total")
    if not isinstance(total, dict):
        return None
    total_tokens = total.get("totalTokens")
    return total_tokens if isinstance(total_tokens, int) else None


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


def _best_score(project_dir: Path) -> float | None:
    scores: list[float] = []
    for done in (project_dir / "runs").glob("*/artifacts/*/node.done.yaml"):
        payload = read_yaml(done)
        metric = payload.get("metric")
        if isinstance(metric, (int, float)):
            scores.append(float(metric))
    return max(scores) if scores else None


def _abort(exc: Exception) -> None:
    message = str(exc)
    console.print(f"Error: {message}")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
