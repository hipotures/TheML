from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from tml.branches.compose import add_branch
from tml.branches.run import run_missing_branches
from tml.core.paths import active_project_ref
from tml.metamodel.features import build_candidate_frame, write_feature_artifacts
from tml.metamodel.importer import build_meta_dataset, load_records_from_csv
from tml.metamodel.reporting import write_report
from tml.metamodel.search import CandidateSearchConfig, generate_candidate_suggestions
from tml.metamodel.training import predict_with_uncertainty, run_meta_modeling, train_target_from_dataset


meta_app = typer.Typer(no_args_is_help=True, help="Build, train, and report standalone experiment meta-models.")
console = Console(highlight=False)
EXTRA = {"allow_extra_args": True, "ignore_unknown_options": True}


@meta_app.command("dataset", context_settings=EXTRA, help="Build the standalone meta-model dataset from project artifacts.")
def dataset_cmd(ctx: typer.Context) -> None:
    try:
        overrides = _overrides(ctx.args)
        _validate_keys(overrides, {"out"}, "tml meta dataset")
        ref = active_project_ref()
        out = Path(str(overrides["out"])) if "out" in overrides else None
        records, dataset = build_meta_dataset(ref.path, out)
        _, features = write_feature_artifacts(records, dataset.output_dir)
        table = Table(title="Meta-model dataset", box=box.SIMPLE_HEAVY, show_header=False)
        table.add_column("Field", style="bold", no_wrap=True)
        table.add_column("Value", overflow="fold")
        table.add_row("Project", ref.slug)
        table.add_row("Records", str(dataset.record_count))
        table.add_row("CV labels", str(dataset.cv_score_count))
        table.add_row("Public labels", str(dataset.public_score_count))
        table.add_row("Features", str(features.feature_count))
        table.add_row("Dataset CSV", str(dataset.dataset_csv))
        table.add_row("Feature CSV", str(features.feature_csv))
        table.add_row("Metadata JSON", str(dataset.metadata_json))
        console.print(table)
    except Exception as exc:
        _abort(exc)


@meta_app.command("run", context_settings=EXTRA, help="Build dataset, train targets, and write reports.")
def run_cmd(ctx: typer.Context) -> None:
    try:
        overrides = _overrides(ctx.args)
        _validate_keys(overrides, {"profile", "out", "public"}, "tml meta run")
        ref = active_project_ref()
        profile = str(overrides.get("profile") or "quick")
        out = Path(str(overrides["out"])) if "out" in overrides else None
        train_public = _bool(overrides.get("public", True))
        result = run_meta_modeling(ref.path, output_dir=out, profile_name=profile, train_public=train_public)
        report = write_report(result)
        _print_run_summary(ref.slug, result.output_dir, result.target_results, report.report_md, report.report_json)
    except Exception as exc:
        _abort(exc)


@meta_app.command("train", context_settings=EXTRA, help="Train one target from a meta-model dataset CSV.")
def train_cmd(ctx: typer.Context) -> None:
    try:
        overrides = _overrides(ctx.args)
        _validate_keys(overrides, {"target", "profile", "dataset", "out"}, "tml meta train")
        ref = active_project_ref()
        target = str(overrides.get("target") or "cv_score")
        profile = str(overrides.get("profile") or "quick")
        out = Path(str(overrides["out"])) if "out" in overrides else ref.path / "meta_models" / "meta_runs" / f"manual-{target}-{profile}"
        dataset_csv = _resolve_existing_or_project_path(ref.path, overrides["dataset"]) if "dataset" in overrides else None
        if dataset_csv is None:
            _, dataset = build_meta_dataset(ref.path, out / "dataset")
            dataset_csv = dataset.dataset_csv
        result = train_target_from_dataset(ref.path, dataset_csv=dataset_csv, output_dir=out, target=target, profile_name=profile)
        table = Table(title="Meta-model target training", box=box.SIMPLE_HEAVY, show_header=False)
        table.add_column("Field", style="bold", no_wrap=True)
        table.add_column("Value", overflow="fold")
        table.add_row("Target", result.target)
        table.add_row("Status", result.status)
        table.add_row("Examples", str(result.n_examples))
        table.add_row("Model", str(result.model_dir or "n/a"))
        table.add_row("Metrics", str(result.metrics_json))
        table.add_row("Predictions", str(result.validation_predictions_csv or "n/a"))
        console.print(table)
    except Exception as exc:
        _abort(exc)


@meta_app.command("report", context_settings=EXTRA, help="Show report artifact paths for a meta-model run.")
def report_cmd(ctx: typer.Context) -> None:
    try:
        overrides = _overrides(ctx.args)
        _validate_keys(overrides, {"run"}, "tml meta report")
        ref = active_project_ref()
        run_dir = Path(str(overrides.get("run") or ""))
        if not run_dir:
            raise ValueError("Missing run=<path>.")
        run_dir = _resolve_existing_or_project_path(ref.path, run_dir)
        report_md = run_dir / "report.md"
        report_json = run_dir / "report.json"
        if not report_md.exists() or not report_json.exists():
            raise ValueError(f"Report artifacts not found in {run_dir}. Run: uv run tml meta run")
        table = Table(title="Meta-model report", box=box.SIMPLE_HEAVY, show_header=False)
        table.add_column("Field", style="bold", no_wrap=True)
        table.add_column("Value", overflow="fold")
        table.add_row("Markdown", str(report_md))
        table.add_row("JSON", str(report_json))
        console.print(table)
    except Exception as exc:
        _abort(exc)


@meta_app.command("predict", context_settings=EXTRA, help="Predict/advisory for a candidate group set.")
def predict_cmd(
    ctx: typer.Context,
    json_flag: Annotated[bool, typer.Option("--json", help="Print machine-readable JSON.")] = False,
) -> None:
    try:
        overrides = _overrides(ctx.args)
        _validate_keys(overrides, {"run", "target", "groups", "parent", "mode", "profile_id", "json"}, "tml meta predict")
        ref = active_project_ref()
        run_dir = Path(str(overrides.get("run") or ""))
        if not run_dir:
            raise ValueError("Missing run=<path>.")
        run_dir = _resolve_existing_or_project_path(ref.path, run_dir)
        target = str(overrides.get("target") or "cv_score")
        groups = _split_csv(str(overrides.get("groups") or ""))
        if not groups:
            raise ValueError("Missing groups=<id[@revision],...>.")
        dataset_csv = run_dir / "dataset" / "meta_dataset.csv"
        feature_spec = json.loads((run_dir / "dataset" / "feature_spec.json").read_text(encoding="utf-8"))
        feature_columns = [str(column) for column in feature_spec["feature_columns"]]
        records = load_records_from_csv(dataset_csv)
        frame = build_candidate_frame(
            ref.path,
            records,
            groups=groups,
            parent=_optional_str(overrides.get("parent")),
            mode=str(overrides.get("mode") or "autogluon"),
            profile_id=_optional_str(overrides.get("profile_id")),
            feature_columns=feature_columns,
        )
        model_dir = run_dir / "targets" / target / "final" / "AutoGluonModels"
        prediction = predict_with_uncertainty(model_dir, frame)
        payload = {
            "target": target,
            "groups": groups,
            "parent": _optional_str(overrides.get("parent")),
            "model_dir": str(model_dir),
            **prediction,
        }
        if json_flag or _bool(overrides.get("json", False)):
            console.print(json.dumps(payload, indent=2, sort_keys=True))
            return
        table = Table(title="Meta-model advisory prediction", box=box.SIMPLE_HEAVY, show_header=False)
        table.add_column("Field", style="bold", no_wrap=True)
        table.add_column("Value", overflow="fold")
        table.add_row("Target", target)
        table.add_row("Prediction", _first_float(payload.get("prediction")))
        table.add_row("Split std", _first_float(payload.get("split_prediction_std")))
        table.add_row("Split p10", _first_float(payload.get("split_prediction_p10")))
        table.add_row("Split p90", _first_float(payload.get("split_prediction_p90")))
        table.add_row("Split models", str(payload.get("split_model_count")))
        console.print(table)
    except Exception as exc:
        _abort(exc)


@meta_app.command("suggest", context_settings=EXTRA, help="Generate advisory hypothesis-set candidates ranked by the meta-model.")
def suggest_cmd(
    ctx: typer.Context,
    json_flag: Annotated[bool, typer.Option("--json", help="Print machine-readable JSON.")] = False,
) -> None:
    try:
        overrides = _overrides(ctx.args)
        _validate_keys(
            overrides,
            {
                "run",
                "target",
                "top",
                "candidates",
                "beam",
                "depth",
                "exploration",
                "seed",
                "max_groups",
                "include-tested",
                "include_tested",
                "buildable",
                "mode",
                "out",
                "json",
            },
            "tml meta suggest",
        )
        ref = active_project_ref()
        run_dir = Path(str(overrides.get("run") or ""))
        if not run_dir:
            raise ValueError("Missing run=<path>.")
        run_dir = _resolve_existing_or_project_path(ref.path, run_dir)
        target = str(overrides.get("target") or "cv_score")
        output_dir = _resolve_existing_or_project_path(ref.path, overrides["out"]) if "out" in overrides else None
        include_tested = _bool(overrides.get("include-tested", overrides.get("include_tested", False)))
        config = CandidateSearchConfig(
            target=target,
            top=_int(overrides.get("top", 25), "top"),
            candidates=_int(overrides.get("candidates", 2000), "candidates"),
            beam=_int(overrides.get("beam", 50), "beam"),
            depth=_int(overrides.get("depth", 3), "depth"),
            exploration=_float(overrides.get("exploration", 0.10), "exploration"),
            seed=_int(overrides.get("seed", 42), "seed"),
            max_groups=_optional_int(overrides.get("max_groups"), "max_groups"),
            include_tested=include_tested,
            buildable=_bool(overrides.get("buildable", True)),
            mode=str(overrides.get("mode") or "autogluon"),
        )
        result = generate_candidate_suggestions(ref.path, run_dir, config=config, output_dir=output_dir)
        if json_flag or _bool(overrides.get("json", False)):
            console.print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
            return
        _print_suggest_summary(result)
    except Exception as exc:
        _abort(exc)


@meta_app.command("build", context_settings=EXTRA, help="Materialize one meta-model candidate suggestion as BRANCH steps.")
def build_cmd(ctx: typer.Context) -> None:
    try:
        overrides = _overrides(ctx.args)
        _validate_keys(overrides, {"search", "rank", "run", "mode"}, "tml meta build")
        ref = active_project_ref()
        if "search" not in overrides:
            raise ValueError("Missing search=<candidate_search_dir_or_json>.")
        search_json = _candidate_search_json(ref.path, overrides["search"])
        rank = _int(overrides.get("rank", 1), "rank")
        run_after_build = _bool(overrides.get("run", False))
        mode_override = _optional_str(overrides.get("mode"))
        payload = json.loads(search_json.read_text(encoding="utf-8"))
        suggestion = _candidate_suggestion(payload, rank)
        steps = _candidate_branch_steps(suggestion)
        if not steps:
            raise ValueError(f"Suggestion rank={rank} has no branch-add steps to build.")

        created_rows = []
        current_parent: str | None = None
        final_mode = mode_override
        for step_index, step in enumerate(steps, start=1):
            parent_ref = current_parent or step["parent"] or _optional_str(suggestion.get("parent_ref"))
            if not parent_ref or parent_ref == "<BRANCH_FROM_PREVIOUS_STEP>":
                raise ValueError(f"Suggestion rank={rank} step {step_index} has no concrete parent branch.")
            mode = mode_override or step["mode"] or str((payload.get("config") or {}).get("mode") or "autogluon")
            created = add_branch(ref.path, parent_ref=parent_ref, source_ref=step["source"], mode=mode)
            current_parent = created.branch_id
            final_mode = created.parent.mode
            created_rows.append(
                {
                    "step": step_index,
                    "parent": parent_ref,
                    "source": step["source"],
                    "branch": created.branch_id,
                    "file": created.materialization_path,
                }
            )

        executed_ids: list[str] = []
        if run_after_build and current_parent:
            executed_ids = run_missing_branches(
                ref.path,
                mode=final_mode,
                branch_id=current_parent,
                profile_overrides={},
                progress=console.print,
            )
        _print_candidate_build_summary(ref.slug, search_json, suggestion, created_rows, current_parent, executed_ids)
    except Exception as exc:
        _abort(exc)


def _print_run_summary(slug: str, output_dir: Path, targets, report_md: Path, report_json: Path) -> None:
    table = Table(title="Meta-model run", box=box.SIMPLE_HEAVY, show_header=False)
    table.add_column("Field", style="bold", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("Project", slug)
    table.add_row("Output", str(output_dir))
    for result in targets:
        table.add_row(f"{result.target} status", f"{result.status} ({result.n_examples} examples)")
    table.add_row("Report MD", str(report_md))
    table.add_row("Report JSON", str(report_json))
    console.print(table)


def _print_suggest_summary(result) -> None:
    table = Table(title="Meta-model candidate suggestions", box=box.SIMPLE_HEAVY)
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("Pred", justify="right", no_wrap=True)
    table.add_column("Std", justify="right", no_wrap=True)
    table.add_column("Groups", justify="right", no_wrap=True)
    table.add_column("Added", overflow="fold")
    table.add_column("Removed", overflow="fold")
    table.add_column("Nearest", overflow="fold")
    table.add_column("Build", no_wrap=True)
    for item in result.suggestions:
        table.add_row(
            str(item.rank),
            f"{item.prediction:.6f}",
            _first_float(item.split_prediction_std),
            str(item.group_count),
            ",".join(item.added_groups) or "none",
            ",".join(item.removed_groups) or "none",
            f"{item.nearest_ref or 'n/a'} ({item.nearest_jaccard:.3f})",
            f"rank={item.rank}",
        )
    console.print(table)
    artifacts = Table(title="Candidate search artifacts", box=box.SIMPLE_HEAVY, show_header=False)
    artifacts.add_column("Field", style="bold", no_wrap=True)
    artifacts.add_column("Value", overflow="fold")
    artifacts.add_row("Output", str(result.output_dir))
    artifacts.add_row("JSON", str(result.json_path))
    artifacts.add_row("CSV", str(result.csv_path))
    artifacts.add_row("Markdown", str(result.markdown_path))
    artifacts.add_row("Generated", str(result.generated_count))
    artifacts.add_row("Scored", str(result.scored_count))
    artifacts.add_row("Build rank", f"uv run tml meta build search={result.output_dir} rank=<rank>")
    artifacts.add_row("Build and run", f"uv run tml meta build search={result.output_dir} rank=<rank> run=true")
    console.print(artifacts)


def _print_candidate_build_summary(
    slug: str,
    search_json: Path,
    suggestion: dict[str, object],
    created_rows: list[dict[str, object]],
    final_branch: str | None,
    executed_ids: list[str],
) -> None:
    table = Table(title="Meta-model candidate build", box=box.SIMPLE_HEAVY, show_header=False)
    table.add_column("Field", style="bold", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("Project", slug)
    table.add_row("Search", str(search_json))
    table.add_row("Rank", str(suggestion.get("rank") or "n/a"))
    prediction = suggestion.get("prediction")
    table.add_row("Prediction", f"{prediction:.6f}" if isinstance(prediction, int | float) else "n/a")
    table.add_row("Final branch", final_branch or "n/a")
    if final_branch and not executed_ids:
        table.add_row("Run", f"uv run tml branch run branch={final_branch}")
    if executed_ids:
        table.add_row("Executed nodes", ",".join(executed_ids))
    console.print(table)

    steps = Table(title="Materialized BRANCH steps", box=box.SIMPLE_HEAVY)
    steps.add_column("#", justify="right", no_wrap=True)
    steps.add_column("Parent", overflow="fold")
    steps.add_column("Source", overflow="fold")
    steps.add_column("Branch", no_wrap=True)
    for row in created_rows:
        steps.add_row(str(row["step"]), str(row["parent"]), str(row["source"]), str(row["branch"]))
    console.print(steps)


def _candidate_search_json(project_dir: Path, value: object) -> Path:
    path = _resolve_existing_or_project_path(project_dir, value)
    if path.is_dir():
        path = path / "candidate_search.json"
    if not path.exists():
        raise ValueError(f"Candidate search JSON not found: {path}")
    return path


def _candidate_suggestion(payload: dict[str, object], rank: int) -> dict[str, object]:
    suggestions = payload.get("suggestions")
    if not isinstance(suggestions, list):
        raise ValueError("Candidate search JSON does not contain suggestions.")
    for item in suggestions:
        if isinstance(item, dict) and item.get("rank") == rank:
            return item
    raise ValueError(f"Candidate suggestion rank={rank} not found.")


def _candidate_branch_steps(suggestion: dict[str, object]) -> list[dict[str, str]]:
    commands = suggestion.get("branch_add_commands")
    if not isinstance(commands, list):
        return []
    steps = []
    for command in commands:
        fields = _command_fields(str(command))
        source = fields.get("source")
        if not source:
            raise ValueError(f"Invalid branch-add command in suggestion: {command}")
        steps.append({"parent": fields.get("parent") or "", "source": source, "mode": fields.get("mode") or ""})
    return steps


def _command_fields(command: str) -> dict[str, str]:
    fields = {}
    for token in command.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def _overrides(args: list[str]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for arg in args:
        if "=" not in arg:
            continue
        key, value = arg.split("=", 1)
        parsed[key] = _coerce(value)
    return parsed


def _validate_keys(overrides: dict[str, object], allowed: set[str], command: str) -> None:
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"Unknown parameter for {command}: {', '.join(unknown)}")


def _resolve_existing_or_project_path(project_dir: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return project_dir / path


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
    return str(value).strip().lower() == "true"


def _int(value: object, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if parsed <= 0:
        raise ValueError(f"{label} must be positive.")
    return parsed


def _optional_int(value: object, label: str) -> int | None:
    if value is None or value == "":
        return None
    return _int(value, label)


def _float(value: object, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number.") from exc
    if parsed < 0:
        raise ValueError(f"{label} must be non-negative.")
    return parsed


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _first_float(value: object) -> str:
    if isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, int | float):
        return f"{float(value):.6f}"
    return "n/a"


def _abort(exc: Exception) -> None:
    console.print(f"Error: {exc}")
    raise typer.Exit(code=1)
