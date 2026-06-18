from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from tml.core.config import active_mode, active_profile_id, load_project_config
from tml.core.errors import TmlError
from tml.core.kaggle import download_competition_data
from tml.core.paths import active_project_ref, workspace_root
from tml.core.project import default_download_data, default_project_kind, init_project, use_project
from tml.db.reindex import reindex_project
from tml.hypotheses.generate import generate_missing_root_hypotheses
from tml.hypotheses.materialize import materialize_missing
from tml.hypotheses.run import run_missing
from tml.hypotheses.status import filesystem_counts
from tml.prompts.diff import diff_prompt
from tml.prompts.probe import probe_prompt, render_prompt
from tml.utils.hashing import sha256_file
from tml.utils.yaml_io import read_yaml


console = Console()
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
        ref = init_project(root, slug, kind, download=download)
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
        profile_path = ref.path / "profiles" / "root" / f"{profile_id}.yaml"
        profile_hash = sha256_file(profile_path) if profile_path.exists() else "missing"
        best = _best_score(ref.path)
        console.print(f"Active project: {ref.slug}")
        console.print(f"Target ROOT count: {config.get('root', {}).get('target_count', 20)}")
        console.print(f"Active mode: {mode}")
        console.print(f"Active profile: {profile_id} ({profile_hash})")
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
        download_competition_data(slug, data_dir)
        typer.echo(f"Kaggle data: downloaded for {slug}")
        typer.echo(f"Data dir: {_repo_path(data_dir, ref.root)}")
        files = _data_files(data_dir)
        if files:
            typer.echo("Data files: " + ", ".join(files))
    except Exception as exc:
        _abort(exc)


@prompt_app.command("render", context_settings=EXTRA)
def prompt_render_cmd(ctx: typer.Context) -> None:
    try:
        ref = active_project_ref()
        positional = _positional(ctx.args)
        path = render_prompt(
            ref.path,
            target=positional[0] if len(positional) >= 1 else None,
            stage=positional[1] if len(positional) >= 2 else None,
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
        tmp = _bool(overrides.get("tmp", False))
        model = str(overrides["model"]) if "model" in overrides else None
        path = probe_prompt(
            ref.path,
            tmp=tmp,
            target=positional[0] if len(positional) >= 1 else None,
            stage=positional[1] if len(positional) >= 2 else None,
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
    typer.echo(f"Initialized project: {_repo_path(project_dir, root)}")
    typer.echo(f"Project config: {_repo_path(project_dir / 'project.yaml', root)}")
    typer.echo(f"Task file: {_repo_path(project_dir / 'task.md', root)}")
    typer.echo(f"Data dir: {_repo_path(data_dir, root)}")
    if download:
        files = _data_files(data_dir)
        suffix = f" ({', '.join(files)})" if files else ""
        typer.echo(f"Kaggle data: downloaded{suffix}")
    else:
        typer.echo("Kaggle data: skipped (download=false)")
    typer.echo(f"Next: uv run tml project use {slug}")


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
