from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from typer.core import TyperGroup

from tml.branches.algorithms import BranchAlgorithmResult, branch_add_algorithmic
from tml.branches.compose import (
    BranchDeletePlan,
    BranchRebasePlan,
    BranchRebaseTarget,
    CreatedBranch,
    RebasedBranch,
    add_branch,
    branch_delete_plan,
    branch_rebase_plan,
    branch_rebase_targets,
    delete_branch,
    rebase_branch,
)
from tml.branches.grow import BranchGrowPlan, BranchGrowResult, branch_grow, branch_grow_plan
from tml.branches.runtime_state import read_branch_runtime_state
from tml.branches.run import BranchRunPlan, branch_run_plan, run_missing_branches
from tml.cli.prompt_output import print_prompt_choices, print_prompt_probe_summary, print_prompt_render_summary
from tml.core.config import active_mode, active_profile_id, load_project_config
from tml.core.errors import TmlError
from tml.core.kaggle import download_competition_data, list_competition_submissions, submit_competition_file
from tml.core.paths import active_project_ref, workspace_root
from tml.core.profiles import load_profile, profile_hash
from tml.core.project import default_download_data, default_project_kind, init_project, use_project
from tml.db.reindex import reindex_project
from tml.db.state import (
    best_score,
    branch_rows,
    delete_hypothesis_revision,
    materialization_rows,
    mark_submission_submitted,
    revision_status_rows as db_revision_status_rows,
    root_counts,
    root_hypothesis_rows,
    root_revision_overview_rows,
    root_revision_promote_targets,
    root_run_rows,
    run_request_status,
    solution_tree_branch_rows,
    solution_tree_root_rows,
    submission_by_sha_prefix,
    submission_rows,
    sync_submission_remote_rows,
    upsert_hypothesis,
    upsert_materialization,
)
from tml.hypotheses.generate import (
    GeneratedHypothesis,
    RootGenerationPlan,
    generate_missing_root_hypotheses,
    root_generation_plan,
)
from tml.hypotheses.baseline import ensure_root_baseline
from tml.hypotheses.bugfix import RootBugfixPlan, bugfix_failed_materializations, root_bugfix_plan
from tml.hypotheses.materialize import RootMaterializationPlan, materialize_missing, root_materialization_plan
from tml.hypotheses.revise import RootRevisePlan, revise_root_hypothesis, root_revise_plan
from tml.hypotheses.revisions import delete_revision, normalize_hypothesis_id, set_active_materialization
from tml.hypotheses.run import RootRunPlan, root_run_plan, run_missing
from tml.prompts.diff import diff_prompt
from tml.prompts.probe import probe_prompt, render_prompt
from tml.rerun import RerunPlan, rerun_plan, rerun_submission
from tml.utils.yaml_io import read_yaml


console = Console(highlight=False)
DEFAULT_SUBMISSION_TABLE_LIMIT = 20


class TmlGroup(TyperGroup):
    def main(self, args=None, **kwargs):
        requested_args = list(sys.argv[1:] if args is None else args)
        if requested_args == ["help"]:
            _print_help_topics()
            return None
        if requested_args in (["help=metrics"], ["--help", "metrics"]):
            _print_metrics_help()
            return None
        return super().main(args=args, **kwargs)


app = typer.Typer(
    cls=TmlGroup,
    no_args_is_help=True,
    epilog="[bold]Metrics[/bold]\n  Run [cyan]uv run tml help=metrics[/cyan] to show metric aliases.",
)
init_app = typer.Typer(no_args_is_help=True)
project_app = typer.Typer(no_args_is_help=True)
root_app = typer.Typer(no_args_is_help=True)
branch_app = typer.Typer(no_args_is_help=True)
prompt_app = typer.Typer(no_args_is_help=True)
kaggle_app = typer.Typer(no_args_is_help=True)

ROOT_HYPOTHESIS_COLUMNS = [
    {"key": "id", "label": "ID", "style": "bold", "no_wrap": True},
    {"key": "status", "label": "S", "justify": "center", "no_wrap": True},
    {"key": "revision", "label": "Rev", "justify": "right", "no_wrap": True},
    {"key": "score", "label": "Score", "justify": "right", "no_wrap": True},
    {"key": "created_at", "label": "Created", "no_wrap": True},
    {"key": "model", "label": "Model", "no_wrap": True},
    {"key": "reasoning_tokens", "label": "Res/Tokens", "justify": "right", "no_wrap": True},
    {"key": "duration", "label": "Gen", "no_wrap": True},
    {"key": "summary", "label": "Summary", "overflow": "fold", "min_width": 40, "ratio": 1, "truncate": True},
]

METRIC_ALIAS_ROWS = [
    ("ROC-AUC / AUROC / AUC", "area under ROC curve"),
    ("PR-AUC / AUPRC / AP", "area under precision-recall curve / average precision"),
    ("BA / BAcc", "balanced accuracy"),
    ("ACC", "accuracy"),
    ("ERR", "error rate"),
    ("F1", "F1 score"),
    ("P / PPV", "precision / positive predictive value"),
    ("R / TPR / REC", "recall / true positive rate"),
    ("TNR / SP", "specificity / true negative rate"),
    ("FPR", "false positive rate"),
    ("FNR", "false negative rate"),
    ("NPV", "negative predictive value"),
    ("MCC", "Matthews correlation coefficient"),
    ("LL / NLL / CE", "log loss / negative log likelihood / cross entropy"),
    ("Brier / BS", "Brier score"),
    ("MAE", "mean absolute error"),
    ("MSE", "mean squared error"),
    ("RMSE", "root mean squared error"),
    ("R2 / R²", "coefficient of determination"),
    ("MAPE", "mean absolute percentage error"),
    ("SMAPE", "symmetric MAPE"),
    ("NDCG", "normalized discounted cumulative gain"),
    ("MAP", "mean average precision"),
    ("MRR", "mean reciprocal rank"),
]

METRIC_SHORT_SYMBOLS = {
    "roc_auc": "AUC",
    "auroc": "AUC",
    "auc": "AUC",
    "average_precision": "AP",
    "pr_auc": "AP",
    "auprc": "AP",
    "ap": "AP",
    "balanced_accuracy": "BA",
    "balanced_acc": "BA",
    "bacc": "BA",
    "ba": "BA",
    "accuracy": "ACC",
    "acc": "ACC",
    "error_rate": "ERR",
    "err": "ERR",
    "f1": "F1",
    "precision": "P",
    "positive_predictive_value": "PPV",
    "ppv": "PPV",
    "recall": "R",
    "true_positive_rate": "TPR",
    "tpr": "TPR",
    "rec": "R",
    "specificity": "TNR",
    "true_negative_rate": "TNR",
    "tnr": "TNR",
    "sp": "TNR",
    "false_positive_rate": "FPR",
    "fpr": "FPR",
    "false_negative_rate": "FNR",
    "fnr": "FNR",
    "negative_predictive_value": "NPV",
    "npv": "NPV",
    "matthews_correlation_coefficient": "MCC",
    "matthews_corrcoef": "MCC",
    "mcc": "MCC",
    "log_loss": "LL",
    "negative_log_likelihood": "NLL",
    "nll": "NLL",
    "cross_entropy": "CE",
    "ce": "CE",
    "brier_score": "BS",
    "brier_score_loss": "BS",
    "bs": "BS",
    "mean_absolute_error": "MAE",
    "mae": "MAE",
    "mean_squared_error": "MSE",
    "mse": "MSE",
    "root_mean_squared_error": "RMSE",
    "rmse": "RMSE",
    "r2": "R2",
    "r_squared": "R2",
    "mean_absolute_percentage_error": "MAPE",
    "mape": "MAPE",
    "symmetric_mean_absolute_percentage_error": "SMAPE",
    "smape": "SMAPE",
    "normalized_discounted_cumulative_gain": "NDCG",
    "ndcg": "NDCG",
    "mean_average_precision": "MAP",
    "map": "MAP",
    "mean_reciprocal_rank": "MRR",
    "mrr": "MRR",
}

app.add_typer(init_app, name="init", help="Initialize projects and workspace metadata.")
app.add_typer(project_app, name="project", help="Select and inspect project-level settings.")
app.add_typer(root_app, name="root", help="Generate, materialize, fix, and run ROOT hypotheses.")
app.add_typer(branch_app, name="branch", help="Compose, run, inspect, and delete BRANCH artifacts.")
app.add_typer(prompt_app, name="prompt", help="Render, probe, and compare model prompts.")
app.add_typer(kaggle_app, name="kaggle", help="Download data, sync submissions, and submit to Kaggle.")

EXTRA = {"allow_extra_args": True, "ignore_unknown_options": True}


@init_app.command("project", context_settings=EXTRA, help="Create a new project scaffold.")
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


@project_app.command("use", help="Set the active project by slug.")
def use_project_cmd(slug: str) -> None:
    try:
        ref = use_project(workspace_root(), slug)
        console.print(f"Active project: {ref.slug}")
    except Exception as exc:
        _abort(exc)


@app.command("tree", context_settings=EXTRA, help="Show the active project's solution tree.")
def tree_cmd(ctx: typer.Context) -> None:
    try:
        _reject_positional(ctx.args, "tml tree")
        overrides = _overrides(ctx.args)
        ref = active_project_ref()
        ensure_root_baseline(ref.path)
        _validate_override_keys(overrides, {"mode"}, "tml tree")
        config = load_project_config(ref.path)
        mode = _optional_text(overrides.get("mode")) or active_mode(config)
        _print_solution_tree(ref.path, mode=mode)
    except Exception as exc:
        _abort(exc)


@root_app.command(
    "status",
    context_settings=EXTRA,
    help=(
        "Show ROOT progress, counts, and hypothesis table.\n\n"
        "Accepted key=value parameters:\n"
        "  json=true          Print machine-readable JSON.\n"
        "  json_output=true   Alias for json=true.\n\n"
        "Options:\n"
        "  --json             Print machine-readable JSON.\n"
        "  --json-output      Alias for --json."
    ),
)
def root_status_cmd(
    ctx: typer.Context,
    json_flag: Annotated[bool, typer.Option("--json", help="Print machine-readable JSON.")] = False,
    json_output_flag: Annotated[bool, typer.Option("--json-output", help="Alias for --json.")] = False,
) -> None:
    try:
        overrides = _overrides(ctx.args)
        _validate_override_keys(overrides, {"json", "json_output"}, "tml root status")
        json_output = (
            json_flag
            or json_output_flag
            or _bool(overrides.get("json", False))
            or _bool(overrides.get("json_output", False))
        )
        ref = active_project_ref()
        ensure_root_baseline(ref.path)
        config = load_project_config(ref.path)
        counts = root_counts(ref.path)
        mode = active_mode(config)
        profile_id = active_profile_id(config, mode)
        active_hash = profile_hash(ref.path, mode, profile_id)
        best = best_score(ref.path)
        if json_output:
            _print_root_status_json(
                ref.slug,
                config=config,
                mode=mode,
                profile_id=profile_id,
                profile_hash_value=active_hash,
                counts=counts,
                best=best,
                project_dir=ref.path,
            )
            return
        console.print(f"Active project: {ref.slug}")
        console.print(f"Target ROOT count: {config.get('root', {}).get('target_count', 20)}")
        console.print(f"Active mode: {mode}")
        console.print(f"Active profile: {profile_id} ({active_hash[:8]})")
        console.print(f"Hypotheses: {counts['hypotheses']}")
        console.print(f"Materialized: {counts['materialized']}")
        console.print(f"Evaluated: {counts['evaluated']}")
        console.print(f"Incomplete nodes: {counts['incomplete']}")
        console.print(f"Best ROOT score: {f'{best:.6f}' if best is not None else 'n/a'}")
        _print_existing_root_hypotheses(
            ref.path,
            created_ids=set(),
            mode=mode,
            profile_id=profile_id,
            best_score_value=best,
        )
    except Exception as exc:
        _abort(exc)


@root_app.command(
    "generate",
    context_settings=EXTRA,
    help=(
        "Generate missing ROOT hypotheses for the active project.\n\n"
        "Accepted positional arguments:\n"
        "  status            Print the existing ROOT hypothesis table without generating.\n\n"
        "Accepted key=value parameters:\n"
        "  count=<n>          Generate until ROOT hypothesis n exists.\n"
        "  yes=true           Skip the confirmation prompt.\n"
        "  json=true          Print machine-readable JSON; requires yes=true.\n"
        "  json_output=true   Alias for json=true; requires yes=true.\n\n"
        "Codex web search is configured in tml.yaml under providers.codex.web_search."
    ),
)
def root_generate_cmd(ctx: typer.Context) -> None:
    try:
        ref = active_project_ref()
        ensure_root_baseline(ref.path)
        status_only = _command_status_requested(ctx.args, "tml root generate")
        overrides = _overrides(ctx.args)
        _validate_override_keys(overrides, {"count", "json", "json_output", "yes"}, "tml root generate")
        if status_only:
            _print_generated_hypotheses(ref.path, [])
            return
        count = int(overrides["count"]) if "count" in overrides else None
        json_output = _bool(overrides.get("json", False)) or _bool(overrides.get("json_output", False))
        assume_yes = _bool(overrides.get("yes", False))
        if json_output and not assume_yes:
            raise TmlError("tml root generate json=true requires yes=true because confirmation would break JSON output.")
        plan = root_generation_plan(ref.path, count=count)
        if not json_output:
            _print_root_generation_plan(ref.slug, plan)
            if plan.iteration_count == 0:
                console.print("No ROOT hypotheses to generate.")
                _print_generated_hypotheses(ref.path, [])
                return
            if not assume_yes and not Confirm.ask("Start ROOT generation?", default=False, console=console):
                console.print("ROOT generation cancelled.")
                return
        stop_after_current = False
        interrupted_once = False
        previous_sigint = signal.getsignal(signal.SIGINT)

        def handle_sigint(signum: int, frame: object) -> None:
            nonlocal stop_after_current, interrupted_once
            _ = (signum, frame)
            if interrupted_once:
                raise KeyboardInterrupt
            interrupted_once = True
            stop_after_current = True
            console.print("Interrupt received. Finishing current hypothesis, then stopping.")

        signal.signal(signal.SIGINT, handle_sigint)
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

            try:
                created = generate_missing_root_hypotheses(
                    ref.path,
                    count=count,
                    progress=report,
                    stop_requested=lambda: stop_after_current,
                )
            finally:
                signal.signal(signal.SIGINT, previous_sigint)
        if stop_after_current and not json_output:
            console.print("ROOT generation stopped after the current hypothesis.")
        if json_output:
            _print_generated_hypotheses_json(ref.path, created)
            return
        _print_generated_hypotheses(ref.path, created)
    except Exception as exc:
        _abort(exc)


@root_app.command(
    "revise",
    context_settings=EXTRA,
    help=(
        "Create ROOT hypothesis revisions or show revision status.\n\n"
        "Accepted positional arguments:\n"
        "  status            Print revision status. Without id, print all scored revisions.\n\n"
        "  promote|prom      Promote active materializations to best scored revisions.\n\n"
        "  delete|del        Delete one unmaterialized revision.\n\n"
        "Accepted key=value parameters:\n"
        "  hypothesis=<id>    Hypothesis to revise.\n"
        "  id=<id>            Alias for hypothesis=<id>.\n"
        "  count=<N>          Maximum number of new revisions.\n"
        "  rev=<N>            Revision to delete.\n"
        "  revision=<N>       Alias for rev=<N>.\n"
        "  yes=true           Accepted for scripted workflows."
    ),
)
def root_revise_cmd(ctx: typer.Context) -> None:
    try:
        ref = active_project_ref()
        ensure_root_baseline(ref.path)
        positional = _positional(ctx.args)
        delete_requested = positional in (["delete"], ["del"])
        promote_requested = positional in (["promote"], ["prom"])
        status_only = False if delete_requested or promote_requested else _command_status_requested(ctx.args, "tml root revise")
        overrides = _overrides(ctx.args)
        _validate_override_keys(overrides, {"hypothesis", "id", "count", "mode", "revision", "rev", "yes"}, "tml root revise")
        hypothesis_id = _optional_text(overrides.get("hypothesis") or overrides.get("id"))
        hid = normalize_hypothesis_id(hypothesis_id) if hypothesis_id else None
        config = load_project_config(ref.path)
        mode = str(overrides.get("mode") or active_mode(config))
        profile_id = active_profile_id(config, mode)
        if status_only:
            if hid:
                _print_root_revision_overview(ref.path, mode=mode, profile_id=profile_id, hypothesis_ids={hid})
            else:
                _print_root_revision_overview(ref.path, mode=mode, profile_id=profile_id)
            return
        if promote_requested:
            targets = root_revision_promote_targets(ref.path, mode=mode, profile_id=profile_id)
            if hid:
                targets = [row for row in targets if str(row.get("hypothesis_id") or "") == hid]
            if not targets:
                console.print("No ROOT revision active pointers to promote.")
                return
            _print_root_revision_promote_plan(targets)
            if not _bool(overrides.get("yes")) and not Confirm.ask("Promote ROOT revisions?", default=False):
                console.print("No ROOT revisions promoted.")
                return
            promoted = _promote_root_revisions(ref.path, mode=mode, targets=targets)
            console.print(f"Promoted ROOT active materializations: {len(promoted)}")
            _print_root_revision_overview(ref.path, mode=mode, profile_id=profile_id, hypothesis_ids={row["hypothesis_id"] for row in promoted})
            return
        if delete_requested:
            if not hid:
                raise TmlError("Missing required parameter: id=<hypothesis>.")
            revision = _revision_override(overrides)
            if revision is None:
                raise TmlError("Missing required parameter: rev=<revision>.")
            hdir = ref.path / "hypotheses" / hid
            deleted = delete_revision(ref.path, hdir, revision)
            delete_hypothesis_revision(ref.path, hypothesis_id=hid, revision=revision)
            upsert_hypothesis(ref.path, hdir)
            console.print(f"Deleted ROOT hypothesis {hid} revision {revision}: {len(deleted)} files")
            _print_root_revision_overview(ref.path, mode=mode, profile_id=profile_id, hypothesis_ids={hid})
            return
        if not hid:
            _print_root_revision_overview(ref.path, mode=mode, profile_id=profile_id)
            return
        if "count" not in overrides:
            plan = root_revise_plan(ref.path, hypothesis_id=hid, count=0)
            _print_root_revise_plan(ref.slug, plan)
            console.print("No revision created. Pass count=<N> to create revisions.")
            return
        count = int(overrides.get("count") or 0)
        if count < 1:
            raise TmlError("count=<N> must be greater than 0.")
        plan = root_revise_plan(ref.path, hypothesis_id=hid, count=count)
        _print_root_revise_plan(ref.slug, plan)
        created = revise_root_hypothesis(ref.path, hypothesis_id=hid, count=count, progress=console.print)
        console.print(f"Created ROOT revisions: {len(created)}")
        _print_root_revision_overview(ref.path, mode=mode, profile_id=profile_id, hypothesis_ids={hid})
    except Exception as exc:
        _abort(exc)


@root_app.command(
    "mat",
    context_settings=EXTRA,
    help="Alias for `tml root materialize`.",
)
@root_app.command(
    "materialize",
    context_settings=EXTRA,
    help=(
        "Materialize missing ROOT hypothesis code for the active project.\n\n"
        "Accepted positional arguments:\n"
        "  status            Print the ROOT materialization table without materializing.\n\n"
        "Accepted key=value parameters:\n"
        "  mode=<name>        Materialization mode; defaults to the active mode.\n"
        "  hypothesis=<id>    Materialize only one hypothesis.\n"
        "  id=<id>            Alias for hypothesis=<id>.\n"
        "  revision=<N>       Materialize one hypothesis revision.\n"
        "  rev=<N>            Alias for revision=<N>.\n"
        "  version=new        Create the next materialization version, e.g. autogluon-002.py.\n"
        "  version=<number>   Create a specific materialization version, e.g. version=2.\n"
        "  yes=true           Skip the confirmation prompt."
    ),
)
def root_materialize_cmd(ctx: typer.Context) -> None:
    try:
        ref = active_project_ref()
        ensure_root_baseline(ref.path)
        status_only = _command_status_requested(ctx.args, "tml root materialize")
        config = load_project_config(ref.path)
        overrides = _overrides(ctx.args)
        _validate_override_keys(overrides, {"mode", "hypothesis", "id", "revision", "rev", "version", "yes"}, "tml root materialize")
        mode = str(overrides.get("mode") or active_mode(config))
        hypothesis_id = overrides.get("hypothesis") or overrides.get("id")
        hypothesis_id_text = _optional_text(hypothesis_id)
        revision = _revision_override(overrides)
        if status_only:
            _print_root_materializations(ref.path, mode=mode, created_count=None, hypothesis_id=hypothesis_id_text)
            return
        assume_yes = _bool(overrides.get("yes", False))
        version = _optional_text(overrides.get("version"))
        plan = root_materialization_plan(ref.path, mode=mode, hypothesis_id=hypothesis_id_text, version=version, revision=revision)
        _print_root_materialization_plan(ref.slug, plan, hypothesis_id=hypothesis_id_text)
        if plan.iteration_count == 0:
            console.print("No ROOT materializations to create.")
            _print_root_materializations(ref.path, mode=mode, created_count=0, hypothesis_id=hypothesis_id_text)
            return
        if not assume_yes and not Confirm.ask("Start ROOT materialization?", default=False, console=console):
            console.print("ROOT materialization cancelled.")
            return
        created = _materialize_with_progress(
            ref.path,
            mode=mode,
            hypothesis_id=hypothesis_id_text,
            version=version,
            revision=revision,
        )
        shown_targets = {
            (str(hid).zfill(6), str(file_name))
            for hid, file_name in zip(plan.hypothesis_ids, plan.target_files, strict=False)
        }
        _print_root_materializations(
            ref.path,
            mode=mode,
            created_count=created,
            hypothesis_id=hypothesis_id_text,
            only_targets=shown_targets,
        )
    except Exception as exc:
        _abort(exc)


@root_app.command(
    "bugfix",
    context_settings=EXTRA,
    help=(
        "Generate fixed versions for failed ROOT materializations.\n\n"
        "Accepted positional arguments:\n"
        "  status            Print the ROOT materialization table without fixing.\n\n"
        "Accepted key=value parameters:\n"
        "  mode=<name>        Materialization mode; defaults to the active mode.\n"
        "  hypothesis=<id>    Fix only one hypothesis.\n"
        "  id=<id>            Alias for hypothesis=<id>.\n"
        "  revision=<N>       Fix only one hypothesis revision.\n"
        "  rev=<N>            Alias for revision=<N>.\n"
        "  yes=true           Skip the confirmation prompt."
    ),
)
def root_bugfix_cmd(ctx: typer.Context) -> None:
    try:
        ref = active_project_ref()
        ensure_root_baseline(ref.path)
        status_only = _command_status_requested(ctx.args, "tml root bugfix")
        config = load_project_config(ref.path)
        overrides = _overrides(ctx.args)
        _validate_override_keys(overrides, {"mode", "hypothesis", "id", "revision", "rev", "yes"}, "tml root bugfix")
        mode = str(overrides.get("mode") or active_mode(config))
        hypothesis_id = overrides.get("hypothesis") or overrides.get("id")
        hypothesis_id_text = _optional_text(hypothesis_id)
        revision = _revision_override(overrides)
        if status_only:
            _print_root_materializations(ref.path, mode=mode, created_count=None, hypothesis_id=hypothesis_id_text)
            return
        assume_yes = _bool(overrides.get("yes", False))
        plan = root_bugfix_plan(ref.path, mode=mode, hypothesis_id=hypothesis_id_text, revision=revision)
        _print_root_bugfix_plan(ref.slug, plan, hypothesis_id=hypothesis_id_text)
        if plan.iteration_count == 0:
            console.print("No failed ROOT materializations to fix.")
            _print_root_materializations(ref.path, mode=mode, created_count=0, hypothesis_id=hypothesis_id_text)
            return
        if not assume_yes and not Confirm.ask("Start ROOT bugfix?", default=False, console=console):
            console.print("ROOT bugfix cancelled.")
            return
        created = _bugfix_with_progress(
            ref.path,
            mode=mode,
            hypothesis_id=hypothesis_id_text,
            revision=revision,
        )
        _print_root_materializations(ref.path, mode=mode, created_count=created, hypothesis_id=hypothesis_id_text)
    except Exception as exc:
        _abort(exc)


def _materialize_with_progress(project_dir: Path, *, mode: str, hypothesis_id: str | None, version: str | None = None, revision: int | None = None) -> int:
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
            created = materialize_missing(
                project_dir,
                mode=mode,
                hypothesis_id=hypothesis_id,
                version=version,
                revision=revision,
                progress=report,
            )
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


def _bugfix_with_progress(project_dir: Path, *, mode: str, hypothesis_id: str | None, revision: int | None = None) -> int:
    state: dict[str, object] = {
        "message": "Preparing bugfix...",
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
            created = bugfix_failed_materializations(
                project_dir,
                mode=mode,
                hypothesis_id=hypothesis_id,
                revision=revision,
                progress=report,
            )
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
        task = progress.add_task("Preparing bugfix...", total=1)
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


@root_app.command(
    "run",
    context_settings=EXTRA,
    help=(
        "Run missing ROOT materializations for the active project.\n\n"
        "Accepted positional arguments:\n"
        "  status            Print the ROOT run table without executing.\n\n"
        "Accepted key=value parameters:\n"
        "  mode=<name>        Run mode; defaults to the active mode.\n"
        "  hypothesis=<id>    Run only one hypothesis.\n"
        "  id=<id>            Alias for hypothesis=<id>.\n"
        "  revision=<N>       Run one hypothesis revision.\n"
        "  rev=<N>            Alias for revision=<N>.\n"
        "  force=true         Run even if an evaluation already exists for the same code hash.\n"
        "  yes=true           Skip the confirmation prompt.\n"
        "Profile override parameters are accepted for the active mode."
    ),
)
def root_run_cmd(ctx: typer.Context) -> None:
    try:
        ref = active_project_ref()
        ensure_root_baseline(ref.path)
        status_only = _command_status_requested(ctx.args, "tml root run")
        overrides = _overrides(ctx.args)
        mode = str(overrides["mode"]) if "mode" in overrides else None
        config = load_project_config(ref.path)
        active_run_mode = mode or active_mode(config)
        allowed = {"mode", "hypothesis", "id", "revision", "rev", "force", "yes"} | _profile_override_keys(ref.path, active_run_mode)
        _validate_override_keys(overrides, allowed, "tml root run")
        hypothesis_id = overrides.get("hypothesis") or overrides.get("id")
        hypothesis_id_text = _optional_text(hypothesis_id)
        revision = _revision_override(overrides)
        force = _bool(overrides.get("force", False))
        if status_only:
            _print_root_run_request_status(ref.path, mode=active_run_mode, hypothesis_id=hypothesis_id_text)
            _print_root_run_summary(
                ref.path,
                mode=active_run_mode,
                executed_ids=set(),
                hypothesis_id=hypothesis_id_text,
                executed_count=None,
            )
            return
        assume_yes = _bool(overrides.get("yes", False))
        run_overrides = {key: value for key, value in overrides.items() if key not in {"mode", "hypothesis", "id", "revision", "rev", "force", "yes"}}
        plan = root_run_plan(
            ref.path,
            mode=mode,
            hypothesis_id=hypothesis_id_text,
            revision=revision,
            profile_overrides=run_overrides,
            force=force,
        )
        _print_root_run_plan(ref.slug, plan, hypothesis_id=hypothesis_id_text)
        if plan.iteration_count == 0:
            console.print("No ROOT runs to execute.")
            _print_root_run_request_status(ref.path, mode=active_run_mode, hypothesis_id=hypothesis_id_text)
            _print_root_run_summary(
                ref.path,
                mode=active_run_mode,
                executed_ids=set(),
                hypothesis_id=hypothesis_id_text,
                executed_count=0,
            )
            return
        if not assume_yes and not Confirm.ask("Start ROOT run?", default=False, console=console):
            console.print("ROOT run cancelled.")
            return
        executed_ids = run_missing(
            ref.path,
            mode=mode,
            hypothesis_id=hypothesis_id_text,
            revision=revision,
            profile_overrides=run_overrides,
            force=force,
            progress=console.print,
        )
        _print_root_run_request_status(ref.path, mode=active_run_mode, hypothesis_id=hypothesis_id_text)
        _print_root_run_summary(
            ref.path,
            mode=active_run_mode,
            executed_ids=set(executed_ids),
            hypothesis_id=hypothesis_id_text,
            executed_count=len(executed_ids),
        )
    except Exception as exc:
        _abort(exc)


@root_app.command("ensure", context_settings=EXTRA, help="Generate, materialize, and run missing ROOT work.")
def root_ensure_cmd(ctx: typer.Context) -> None:
    try:
        ref = active_project_ref()
        ensure_root_baseline(ref.path)
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
            hypothesis_id=_optional_text(hypothesis_id),
        )
        run_overrides = {key: value for key, value in overrides.items() if key not in {"count", "mode", "hypothesis", "id"}}
        ran = run_missing(
            ref.path,
            mode=mode,
            hypothesis_id=_optional_text(hypothesis_id),
            profile_overrides=run_overrides,
        )
        reindex_project(ref.path, ref.db_path)
        console.print(f"Generated: {len(generated)}; materialized: {materialized}; executed: {len(ran)}")
    except Exception as exc:
        _abort(exc)


@app.command(
    "autocommit",
    context_settings=EXTRA,
    help=(
        "Commit the active project's hypothesis artifacts, branch artifacts, and tml.yaml/tml.conf config.\n\n"
        "Accepted key=value parameters:\n"
        "  message=<text>     Commit message; defaults to the project slug.\n"
        "  yes=true           Skip the confirmation prompt."
    ),
)
def autocommit_cmd(ctx: typer.Context) -> None:
    _autocommit_artifacts(
        ctx,
        command="tml autocommit",
        include_root=True,
        include_branch=True,
        include_config=True,
        default_message_prefix="Commit artifacts",
        scope_label="project",
    )


@root_app.command(
    "autocommit",
    context_settings=EXTRA,
    help=(
        "Commit the active project's ROOT hypothesis artifacts.\n\n"
        "Accepted key=value parameters:\n"
        "  message=<text>     Commit message; defaults to the project slug.\n"
        "  yes=true           Skip the confirmation prompt."
    ),
)
def root_autocommit_cmd(ctx: typer.Context) -> None:
    _autocommit_artifacts(
        ctx,
        command="tml root autocommit",
        include_root=True,
        include_branch=False,
        include_config=False,
        default_message_prefix="Commit ROOT hypotheses",
        scope_label="ROOT",
    )


def _autocommit_artifacts(
    ctx: typer.Context,
    *,
    command: str,
    include_root: bool,
    include_branch: bool,
    include_config: bool,
    default_message_prefix: str,
    scope_label: str,
) -> None:
    try:
        ref = active_project_ref()
        if include_root:
            ensure_root_baseline(ref.path)
        _reject_positional(ctx.args, command)
        overrides = _overrides(ctx.args)
        _validate_override_keys(overrides, {"message", "yes"}, command)
        message = str(overrides.get("message") or f"{default_message_prefix} for {ref.slug}")
        assume_yes = _bool(overrides.get("yes", False))
        commit_paths = _autocommit_paths(ref.path, include_root=include_root, include_branch=include_branch, include_config=include_config)
        if not commit_paths:
            console.print(f"No {scope_label} artifact paths found for {ref.slug}.")
            return
        changed_before = _git_changed_paths(workspace_root(), commit_paths)
        if not changed_before:
            console.print(f"No {scope_label} artifact changes to commit for {ref.slug}.")
            return
        _print_project_autocommit_plan(ref.slug, commit_paths, message, changed_before, scope_label=scope_label)
        if not assume_yes and not Confirm.ask(f"Commit {scope_label} artifacts?", default=False, console=console):
            console.print(f"{scope_label} artifacts autocommit cancelled.")
            return
        _git_add_paths(workspace_root(), commit_paths)
        staged = _git_staged_paths(workspace_root(), commit_paths)
        if not staged:
            console.print(f"No staged {scope_label} artifact changes to commit for {ref.slug}.")
            return
        _git_commit_paths(workspace_root(), commit_paths, message)
        console.print(f"Committed {len(staged)} {scope_label} artifact files for {ref.slug}.")
    except Exception as exc:
        _abort(exc)


def _autocommit_paths(project_dir: Path, *, include_root: bool, include_branch: bool, include_config: bool) -> list[Path]:
    paths: list[Path] = []
    if include_root:
        hypotheses_path = project_dir / "hypotheses"
        if not hypotheses_path.exists():
            raise TmlError(f"Hypotheses directory does not exist: {hypotheses_path}")
        paths.append(hypotheses_path)
    if include_branch:
        branches_path = project_dir / "branches"
        if branches_path.exists():
            paths.append(branches_path)
    if include_config:
        for config_name in ("tml.yaml", "tml.conf"):
            config_path = workspace_root() / config_name
            if config_path.exists():
                paths.append(config_path)
    return paths


@branch_app.command(
    "autocommit",
    context_settings=EXTRA,
    help=(
        "Commit the active project's BRANCH artifacts.\n\n"
        "Accepted key=value parameters:\n"
        "  message=<text>     Commit message; defaults to the project slug.\n"
        "  yes=true           Skip the confirmation prompt."
    ),
)
def branch_autocommit_cmd(ctx: typer.Context) -> None:
    _autocommit_artifacts(
        ctx,
        command="tml branch autocommit",
        include_root=False,
        include_branch=True,
        include_config=False,
        default_message_prefix="Commit BRANCH artifacts",
        scope_label="BRANCH",
    )


@branch_app.command("add", context_settings=EXTRA, help="Compose a new BRANCH from selected sources.")
def branch_add_cmd(
    ctx: typer.Context,
    parent: Annotated[
        str | None,
        typer.Argument(help="Manual mode: parent=<hypothesis|branch|node>. Algorithmic mode: steps=<N>.", metavar="[parent=<ref>|steps=<N>]"),
    ] = None,
    source: Annotated[
        str | None,
        typer.Argument(help="Manual mode: source=<hypothesis|branch|node>. Algorithmic mode: algo=<id>.", metavar="[source=<ref>|algo=<id>]"),
    ] = None,
    mode: Annotated[
        str | None,
        typer.Argument(help="Optional key=value parameter: mode=<name>; defaults to the active mode.", metavar="[mode=<name>|dry_run=true]"),
    ] = None,
) -> None:
    try:
        ref = active_project_ref()
        args = _command_args(ctx, parent, source, mode)
        _reject_positional(args, "tml branch add")
        config = load_project_config(ref.path)
        overrides = _overrides(args)
        _validate_override_keys(overrides, {"parent", "source", "mode", "steps", "algo", "dry-run", "dry_run"}, "tml branch add")
        has_manual = "parent" in overrides or "source" in overrides
        has_algorithm = "steps" in overrides or "algo" in overrides
        if has_manual and has_algorithm:
            raise TmlError("Do not mix parent/source with steps/algo in tml branch add.")
        mode = str(overrides.get("mode") or active_mode(config))
        if has_algorithm:
            if "steps" not in overrides:
                raise TmlError("Missing required parameter: steps=<N>.")
            algo_id = str(overrides.get("algo") or "default")
            dry_run = _bool(overrides.get("dry-run", overrides.get("dry_run", False)))
            try:
                steps = int(overrides["steps"])
            except (TypeError, ValueError) as exc:
                raise TmlError("steps must be a positive integer.") from exc
            result = branch_add_algorithmic(
                ref.path,
                steps=steps,
                algo_id=algo_id,
                mode=mode,
                dry_run=dry_run,
            )
            _print_branch_algorithm_summary(ref.slug, ref.path, result)
            return

        parent_ref = str(overrides.get("parent") or "")
        source_ref = str(overrides.get("source") or "")
        if not parent_ref:
            raise TmlError("Missing required parameter: parent=<hypothesis|branch|node>.")
        if not source_ref:
            raise TmlError("Missing required parameter: source=<hypothesis|branch|node>.")
        created = add_branch(ref.path, parent_ref=parent_ref, source_ref=source_ref, mode=mode)
        _print_branch_add_summary(ref.slug, created)
        _print_branch_status(ref.path, mode=mode, branch_id=created.branch_id, executed_ids=set(), executed_count=None)
    except Exception as exc:
        _abort(exc)


@branch_app.command("grow", context_settings=EXTRA, help="Iteratively run pending BRANCH nodes or add and run one candidate at a time.")
def branch_grow_cmd(
    ctx: typer.Context,
    parameters: Annotated[
        list[str] | None,
        typer.Argument(
            help=(
                "Required key=value parameter: steps=<N>. Optional fixed keys: algo, mode, yes. "
                "Profile override keys are also accepted for the active mode."
            ),
            metavar="steps=<N> [algo=<id>] [mode=<name>] [yes=true] [profile_key=<value>]",
        ),
    ] = None,
) -> None:
    try:
        ref = active_project_ref()
        args = _command_args(ctx, *(parameters or []))
        _reject_positional(args, "tml branch grow")
        overrides = _overrides(args)
        if "parent" in overrides or "source" in overrides:
            raise TmlError("tml branch grow is algorithmic only; do not pass parent/source.")
        if "dry_run" in overrides or "dry-run" in overrides:
            raise TmlError("tml branch grow does not support dry_run.")
        mode = str(overrides["mode"]) if "mode" in overrides else None
        config = load_project_config(ref.path)
        active_run_mode = mode or active_mode(config)
        allowed = {"steps", "algo", "mode", "yes"} | _profile_override_keys(ref.path, active_run_mode)
        _validate_override_keys(overrides, allowed, "tml branch grow")
        if "steps" not in overrides:
            raise TmlError("Missing required parameter: steps=<N>.")
        try:
            steps = int(overrides["steps"])
        except (TypeError, ValueError) as exc:
            raise TmlError("steps must be a positive integer.") from exc
        if steps <= 0:
            raise TmlError("steps must be a positive integer.")
        algo_id = str(overrides.get("algo") or "default")
        assume_yes = _bool(overrides.get("yes", False))
        run_overrides = {key: value for key, value in overrides.items() if key not in {"steps", "algo", "mode", "yes"}}
        plan = branch_grow_plan(
            ref.path,
            steps=steps,
            algo_id=algo_id,
            mode=mode,
            profile_overrides=run_overrides,
        )
        _print_branch_grow_plan(ref.slug, plan)
        if not assume_yes and not Confirm.ask("Start BRANCH grow?", default=False, console=console):
            console.print("BRANCH grow cancelled.")
            return
        result = branch_grow(
            ref.path,
            steps=steps,
            algo_id=algo_id,
            mode=mode,
            profile_overrides=run_overrides,
            progress=console.print,
        )
        _print_branch_grow_summary(result)
    except Exception as exc:
        _abort(exc)


@branch_app.command("status", context_settings=EXTRA, help="Show BRANCH materializations, runs, and scores.")
def branch_status_cmd(ctx: typer.Context) -> None:
    try:
        ref = active_project_ref()
        _reject_positional(ctx.args, "tml branch status")
        config = load_project_config(ref.path)
        overrides = _overrides(ctx.args)
        _validate_override_keys(overrides, {"mode", "branch", "id", "sort", "order"}, "tml branch status")
        mode = str(overrides.get("mode") or active_mode(config))
        branch_id = _optional_text(overrides.get("branch") or overrides.get("id"))
        sort_by = str(overrides.get("sort") or "score")
        sort_order = _optional_text(overrides.get("order"))
        _print_branch_status(
            ref.path,
            mode=mode,
            branch_id=branch_id,
            executed_ids=set(),
            executed_count=None,
            sort_by=sort_by,
            sort_order=sort_order,
        )
    except Exception as exc:
        _abort(exc)


@branch_app.command("run", context_settings=EXTRA, help="Run missing BRANCH materializations.")
def branch_run_cmd(
    ctx: typer.Context,
    parameters: Annotated[
        list[str] | None,
        typer.Argument(
            help=(
                "Optional positional command/status token and key=value parameters. "
                "Accepted command token: status. Accepted fixed keys: mode, branch, id, yes. "
                "Profile override keys are also accepted for the active mode."
            ),
            metavar="[status] [mode=<name>] [branch=<id>] [id=<id>] [yes=true] [profile_key=<value>]",
        ),
    ] = None,
) -> None:
    try:
        ref = active_project_ref()
        args = _command_args(ctx, *(parameters or []))
        status_only = _command_status_requested(args, "tml branch run")
        overrides = _overrides(args)
        mode = str(overrides["mode"]) if "mode" in overrides else None
        config = load_project_config(ref.path)
        active_run_mode = mode or active_mode(config)
        allowed = {"mode", "branch", "id", "yes"} | _profile_override_keys(ref.path, active_run_mode)
        _validate_override_keys(overrides, allowed, "tml branch run")
        branch_id = _optional_text(overrides.get("branch") or overrides.get("id"))
        if status_only:
            _print_branch_status(ref.path, mode=active_run_mode, branch_id=branch_id, executed_ids=set(), executed_count=None)
            return
        assume_yes = _bool(overrides.get("yes", False))
        run_overrides = {key: value for key, value in overrides.items() if key not in {"mode", "branch", "id", "yes"}}
        plan = branch_run_plan(ref.path, mode=mode, branch_id=branch_id, profile_overrides=run_overrides)
        _print_branch_run_plan(ref.slug, plan, branch_id=branch_id)
        if plan.iteration_count == 0:
            console.print("No BRANCH runs to execute.")
            _print_branch_status(ref.path, mode=active_run_mode, branch_id=branch_id, executed_ids=set(), executed_count=0)
            return
        if not assume_yes and not Confirm.ask("Start BRANCH run?", default=False, console=console):
            console.print("BRANCH run cancelled.")
            return
        executed_ids = run_missing_branches(
            ref.path,
            mode=mode,
            branch_id=branch_id,
            profile_overrides=run_overrides,
            progress=console.print,
        )
        _print_branch_status(
            ref.path,
            mode=active_run_mode,
            branch_id=branch_id,
            executed_ids=set(executed_ids),
            executed_count=len(executed_ids),
            only_executed=True,
        )
    except Exception as exc:
        _abort(exc)


@branch_app.command("rebase", context_settings=EXTRA, help="Create a new BRANCH using active materializations from an existing BRANCH.")
def branch_rebase_cmd(ctx: typer.Context) -> None:
    try:
        ref = active_project_ref()
        _reject_positional(ctx.args, "tml branch rebase")
        overrides = _overrides(ctx.args)
        _validate_override_keys(overrides, {"id", "node", "step", "yes"}, "tml branch rebase")
        raw_id = _optional_text(overrides.get("id"))
        raw_node = _optional_text(overrides.get("node"))
        raw_step = _optional_text(overrides.get("step"))
        if sum(value is not None for value in (raw_id, raw_node, raw_step)) != 1:
            raise TmlError("Specify exactly one of id=<branch>, node=<node_id>, or step=<n>.")
        step = _parse_int(raw_step, "step") if raw_step is not None else None
        targets = branch_rebase_targets(ref.path, branch_id=raw_id, node_id=raw_node, step=step)
        if not targets:
            console.print("No matching BRANCH found.")
            return
        if len(targets) > 1:
            _print_branch_rebase_ambiguous_targets(targets)
            return
        target = targets[0]
        plan = branch_rebase_plan(ref.path, branch_id=target.branch_id, target=target)
        _print_branch_rebase_plan(ref.slug, plan)
        assume_yes = _bool(overrides.get("yes", False))
        if not assume_yes and not Confirm.ask("Create rebased BRANCH?", default=False, console=console):
            console.print("BRANCH rebase cancelled.")
            return
        rebased = rebase_branch(ref.path, branch_id=target.branch_id)
        _print_branch_rebase_summary(ref.slug, target, plan, rebased)
        _print_branch_status(ref.path, mode=rebased.mode, branch_id=rebased.branch_id, executed_ids=set(), executed_count=None)
    except Exception as exc:
        _abort(exc)


@branch_app.command("delete", context_settings=EXTRA, help="Delete a BRANCH and optionally its run nodes.")
def branch_delete_cmd(ctx: typer.Context) -> None:
    try:
        ref = active_project_ref()
        _reject_positional(ctx.args, "tml branch delete")
        overrides = _overrides(ctx.args)
        _validate_override_keys(overrides, {"branch", "id", "force", "yes"}, "tml branch delete")
        branch_id = _optional_text(overrides.get("branch") or overrides.get("id"))
        if not branch_id:
            raise TmlError("Missing required parameter: id=<branch>.")
        force = _bool(overrides.get("force", False))
        assume_yes = _bool(overrides.get("yes", False))
        plan = branch_delete_plan(ref.path, branch_id=branch_id, force=force)
        _print_branch_delete_plan(ref.slug, plan)
        if plan.node_count and not force:
            raise TmlError(f"Branch {plan.branch_id} has {plan.node_count} run node(s); use force=true.")
        if not assume_yes and not Confirm.ask("Delete BRANCH?", default=False, console=console):
            console.print("BRANCH delete cancelled.")
            return
        deleted = delete_branch(ref.path, branch_id=branch_id, force=force)
        console.print(f"Deleted branch {deleted.branch_id}.")
        _print_branch_status(ref.path, mode=deleted.mode, branch_id=None, executed_ids=set(), executed_count=None)
    except Exception as exc:
        _abort(exc)


@app.command("reindex", help="Rebuild the active project's local state database.")
def reindex_cmd(scope: str | None = None, run_id: str | None = None) -> None:
    try:
        _ = (scope, run_id)
        ref = active_project_ref()
        counts = reindex_project(ref.path, ref.db_path)
        console.print(
            f"Reindexed {counts['hypotheses']} hypotheses, "
            f"{counts['materializations']} materializations, {counts['nodes']} nodes, "
            f"{counts['submissions']} submissions"
        )
        _sync_kaggle_submissions(ref.path)
    except Exception as exc:
        _abort(exc)


@app.command("submissions", context_settings=EXTRA, help="List local and synced Kaggle submissions.")
@app.command("sub", context_settings=EXTRA, help="Alias for submissions.")
@app.command("subm", context_settings=EXTRA, help="Alias for submissions.")
def submissions_cmd(ctx: typer.Context) -> None:
    try:
        _reject_positional(ctx.args, "tml sub")
        overrides = _overrides(ctx.args)
        _validate_override_keys(overrides, {"limit"}, "tml sub")
        limit = _submission_table_limit(overrides.get("limit"))
        ref = active_project_ref()
        _print_submissions(ref.path, limit=limit)
    except Exception as exc:
        _abort(exc)


@app.command("rerun", context_settings=EXTRA, help="Re-run from an existing submission SHA prefix.")
def rerun_cmd(ctx: typer.Context) -> None:
    try:
        _reject_positional(ctx.args, "tml rerun")
        overrides = _overrides(ctx.args)
        _validate_override_keys(overrides, {"sha", "sha256", "yes"}, "tml rerun")
        sha = str(overrides.get("sha") or overrides.get("sha256") or "")
        if not sha:
            raise TmlError("Missing required parameter: sha=<submission_sha_prefix>.")
        assume_yes = _bool(overrides.get("yes", False))
        ref = active_project_ref()
        plan = rerun_plan(ref.path, sha_prefix=sha)
        _print_rerun_plan(ref.slug, plan)
        if not assume_yes and not Confirm.ask("Start RERUN?", default=False, console=console):
            console.print("RERUN cancelled.")
            return
        node = rerun_submission(ref.path, sha_prefix=sha, progress=console.print)
        console.print(f"RERUN created node {node}.")
        _print_submissions(ref.path)
    except Exception as exc:
        _abort(exc)


def _sync_kaggle_submissions(project_dir: Path) -> None:
    config = load_project_config(project_dir)
    competition = str(config.get("kaggle_slug") or project_dir.name)
    try:
        remote_submissions = list_competition_submissions(competition)
    except Exception as exc:
        console.print(f"Remote Kaggle submissions unavailable: {exc}")
        return
    changed = sync_submission_remote_rows(project_dir, remote_submissions)
    console.print(f"Remote Kaggle submissions visible: {len(remote_submissions)}; synced local rows: {changed}.")


@kaggle_app.command("download", help="Download competition data for the active project.")
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


@kaggle_app.command("sync", help="Sync remote Kaggle submission metadata into local state.")
def kaggle_sync_cmd() -> None:
    try:
        ref = active_project_ref()
        _sync_kaggle_submissions(ref.path)
    except Exception as exc:
        _abort(exc)


@kaggle_app.command("submit", context_settings=EXTRA, help="Submit a local submission artifact to Kaggle.")
def kaggle_submit_cmd(ctx: typer.Context) -> None:
    try:
        _reject_positional(ctx.args, "tml kaggle submit")
        overrides = _overrides(ctx.args)
        _validate_override_keys(overrides, {"sha", "sha256", "dry-run", "dry_run", "force"}, "tml kaggle submit")
        dry_run = _bool(overrides.get("dry-run", overrides.get("dry_run", False)))
        force = _bool(overrides.get("force", False))
        sha = str(overrides.get("sha") or overrides.get("sha256") or "")
        ref = active_project_ref()
        config = load_project_config(ref.path)
        competition = str(config.get("kaggle_slug") or ref.slug)
        row = submission_by_sha_prefix(ref.path, sha)
        _validate_submit_row(row, allow_uploaded=dry_run or force)
        submission_path = ref.path / str(row["submission_path"])
        message = _kaggle_submit_message(row)
        upload_path = submission_path.with_name(_upload_submission_filename(row))
        if dry_run:
            _print_kaggle_submit_dry_run(ref.path, competition, row, submission_path, upload_path, message)
            return
        response = submit_competition_file(competition, submission_path, message, upload_path)
        mark_submission_submitted(
            ref.path,
            node_id=str(row["node_id"]),
            submission_path=str(row["submission_path"]),
            submitted_at=datetime.now(timezone.utc).isoformat(),
            kaggle_message=message,
            kaggle_response=response,
            upload_path=_repo_relative(ref.path, upload_path),
            uploaded_filename=upload_path.name,
        )
        console.print(f"Submitted {str(row['submission_sha256'])[:10]} to {competition}.")
    except Exception as exc:
        _abort(exc)


@prompt_app.command("render", context_settings=EXTRA, help="Render a prompt template for inspection.")
def prompt_render_cmd(ctx: typer.Context) -> None:
    try:
        ref = active_project_ref()
        positional = _positional(ctx.args)
        if not positional:
            print_prompt_choices(console)
            return
        overrides = _overrides(ctx.args)
        _validate_override_keys(overrides, {"hypothesis", "id", "output"}, "tml prompt render")
        target, stage = _prompt_target_stage(positional)
        hypothesis_id = _optional_text(overrides.get("hypothesis") or overrides.get("id"))
        output = _optional_text(overrides.get("output"))
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
                hypothesis_id=hypothesis_id,
                output_path=Path(output) if output else None,
                tmp_root=_tmp_root(),
            )
            progress.update(task, description=f"Prompt rendered: {path}")
        print_prompt_render_summary(console, path)
    except Exception as exc:
        _abort(exc)


@prompt_app.command("probe", context_settings=EXTRA, help="Render and send a prompt probe to a model.")
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


@prompt_app.command("diff", context_settings=EXTRA, help="Show the diff between prompt versions.")
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


def _command_args(ctx: typer.Context, *values: str | None) -> list[str]:
    args = [value for value in values if value is not None]
    args.extend(ctx.args)
    return args


def _validate_override_keys(overrides: dict[str, object], allowed: set[str], command: str) -> None:
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        allowed_list = ", ".join(sorted(allowed))
        unknown_list = ", ".join(unknown)
        raise TmlError(f"Unknown parameter for {command}: {unknown_list}. Allowed parameters: {allowed_list}")


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)


def _parse_int(value: object, name: str) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise TmlError(f"Invalid {name}: {value!r}. Use an integer.") from exc


def _revision_override(overrides: dict[str, object]) -> int | None:
    raw = overrides.get("revision", overrides.get("rev"))
    if raw is None:
        return None
    if str(raw).strip().lower() == "pending":
        raise TmlError("revision=pending is not supported; use a numeric revision.")
    try:
        revision = int(raw)
    except (TypeError, ValueError) as exc:
        raise TmlError(f"Invalid revision: {raw}. Use a numeric revision.") from exc
    if revision < 1:
        raise TmlError("Revision must be >= 1.")
    return revision


def _reject_positional(args: list[str], command: str) -> None:
    positional = _positional(args)
    if positional:
        raise TmlError(f"Unexpected argument for {command}: {positional[0]}")


def _command_status_requested(args: list[str], command: str) -> bool:
    positional = _positional(args)
    if not positional:
        return False
    if positional == ["status"]:
        return True
    raise TmlError(f"Unexpected argument for {command}: {positional[0]}")


def _print_project_autocommit_plan(
    project_slug: str,
    artifact_paths: list[Path],
    message: str,
    changed_paths: list[str],
    *,
    scope_label: str,
) -> None:
    preview = ", ".join(changed_paths[:3])
    if len(changed_paths) > 3:
        preview += f", ... +{len(changed_paths) - 3}"
    table = Table(title=f"{scope_label} artifacts autocommit plan", box=box.SIMPLE_HEAVY, show_header=False, pad_edge=False)
    table.add_column("Parameter", style="bold", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("Project", project_slug)
    table.add_row("Paths", ", ".join(_git_pathspecs(workspace_root(), artifact_paths)))
    table.add_row("Changed files", str(len(changed_paths)))
    for artifact_path in artifact_paths:
        pathspec = _git_pathspec(workspace_root(), artifact_path)
        count = sum(1 for changed_path in changed_paths if changed_path == pathspec or changed_path.startswith(f"{pathspec}/"))
        table.add_row(f"{artifact_path.name} files", str(count))
    table.add_row("Preview", preview)
    table.add_row("Commit message", message)
    console.print(table)


def _git_pathspec(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _git_pathspecs(root: Path, paths: list[Path]) -> list[str]:
    return [_git_pathspec(root, path) for path in paths]


def _git_changed_paths(root: Path, paths: list[Path]) -> list[str]:
    result = _run_git(root, "status", "--porcelain", "--", *_git_pathspecs(root, paths))
    paths: list[str] = []
    for line in result.stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        paths.append(text[3:] if len(text) > 3 else text)
    return paths


def _git_staged_paths(root: Path, paths: list[Path]) -> list[str]:
    result = _run_git(root, "diff", "--cached", "--name-only", "--", *_git_pathspecs(root, paths))
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _git_add_paths(root: Path, paths: list[Path]) -> None:
    _run_git(root, "add", "--", *_git_pathspecs(root, paths))


def _git_commit_paths(root: Path, paths: list[Path], message: str) -> None:
    _run_git(root, "commit", "-m", message, "--", *_git_pathspecs(root, paths))


def _run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise TmlError(f"git {' '.join(args)} failed: {detail}")
    return result


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


def _print_root_generation_plan(project_slug: str, plan: RootGenerationPlan) -> None:
    ids = plan.hypothesis_ids
    if len(ids) > 8:
        id_text = f"{ids[0]}..{ids[-1]}"
    else:
        id_text = ", ".join(ids) if ids else "none"
    table = Table(title="ROOT generation plan", box=box.SIMPLE_HEAVY, show_header=False, pad_edge=False)
    table.add_column("Parameter", style="bold", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("Project", project_slug)
    table.add_row("Role", plan.role)
    table.add_row("Model", plan.model)
    table.add_row("Provider", plan.provider)
    table.add_row("Provider kind", str(plan.provider_kind or "n/a"))
    table.add_row("Resolved model", str(plan.resolved_model or "n/a"))
    table.add_row("Reasoning effort", str(plan.reasoning_effort or "default"))
    table.add_row("Timeout seconds", str(plan.timeout_seconds or "n/a"))
    table.add_row("Sandbox", plan.sandbox)
    table.add_row("Web search", "enabled" if plan.web_search_enabled else "disabled")
    table.add_row("Target ROOT count", str(plan.target))
    table.add_row("Next hypothesis number", str(plan.next_number))
    table.add_row("Iterations to run", str(plan.iteration_count))
    table.add_row("Hypothesis IDs", id_text)
    console.print(table)


def _print_root_revise_plan(project_slug: str, plan: RootRevisePlan) -> None:
    table = Table(title="ROOT revise plan", box=box.SIMPLE_HEAVY, show_header=False, pad_edge=False)
    table.add_column("Parameter", style="bold", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("Project", project_slug)
    table.add_row("Hypothesis", plan.hypothesis_id)
    table.add_row("Role", plan.role)
    table.add_row("Model", plan.model)
    table.add_row("Provider", plan.provider)
    table.add_row("Provider kind", str(plan.provider_kind or "n/a"))
    table.add_row("Resolved model", str(plan.resolved_model or "n/a"))
    table.add_row("Reasoning effort", str(plan.reasoning_effort or "default"))
    table.add_row("Timeout seconds", str(plan.timeout_seconds or "n/a"))
    table.add_row("Sandbox", plan.sandbox)
    table.add_row("Next revision", str(plan.next_revision))
    table.add_row("Max revisions to create", str(plan.count))
    console.print(table)


def _print_root_materialization_plan(
    project_slug: str,
    plan: RootMaterializationPlan,
    *,
    hypothesis_id: str | None,
) -> None:
    id_revisions = [f"{hid}:{rev}" for hid, rev in zip(plan.hypothesis_ids, plan.revisions, strict=False)]
    if len(id_revisions) > 8:
        id_rev_text = f"{id_revisions[0]} ... {id_revisions[-1]}"
    else:
        id_rev_text = ", ".join(id_revisions) if id_revisions else "none"
    if len(plan.target_files) == 1:
        target_text = plan.target_files[0]
    elif plan.target_files:
        target_text = f"{plan.target_files[0]} ... {plan.target_files[-1]}"
    else:
        target_text = "none"
    table = Table(title="ROOT materialization plan", box=box.SIMPLE_HEAVY, show_header=False, pad_edge=False)
    table.add_column("Parameter", style="bold", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("Project", project_slug)
    table.add_row("Role", plan.role)
    table.add_row("Mode", plan.mode)
    table.add_row("Model", plan.model)
    table.add_row("Provider", plan.provider)
    table.add_row("Provider kind", str(plan.provider_kind or "n/a"))
    table.add_row("Resolved model", str(plan.resolved_model or "n/a"))
    table.add_row("Reasoning effort", str(plan.reasoning_effort or "default"))
    table.add_row("Timeout seconds", str(plan.timeout_seconds or "n/a"))
    table.add_row("Sandbox", plan.sandbox)
    table.add_row("Version", str(plan.version or "default"))
    table.add_row("Hypothesis filter", str(hypothesis_id).zfill(6) if hypothesis_id else "all")
    table.add_row("Candidate hypotheses", str(plan.candidate_count))
    table.add_row("Existing materializations", str(plan.existing_count))
    table.add_row("Iterations to run", str(plan.iteration_count))
    table.add_row("Target file", target_text)
    table.add_row("Hypothesis revisions", id_rev_text)
    console.print(table)


def _print_root_bugfix_plan(
    project_slug: str,
    plan: RootBugfixPlan,
    *,
    hypothesis_id: str | None,
) -> None:
    ids = plan.hypothesis_ids
    if len(ids) > 8:
        id_text = f"{ids[0]}..{ids[-1]}"
    else:
        id_text = ", ".join(ids) if ids else "none"
    if len(plan.source_files) == 1 and len(plan.target_files) == 1:
        file_text = f"{plan.source_files[0]} -> {plan.target_files[0]}"
    elif plan.source_files:
        file_text = f"{plan.source_files[0]} -> {plan.target_files[0]} ... {plan.source_files[-1]} -> {plan.target_files[-1]}"
    else:
        file_text = "none"
    table = Table(title="ROOT bugfix plan", box=box.SIMPLE_HEAVY, show_header=False, pad_edge=False)
    table.add_column("Parameter", style="bold", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("Project", project_slug)
    table.add_row("Role", plan.role)
    table.add_row("Mode", plan.mode)
    table.add_row("Model", plan.model)
    table.add_row("Provider", plan.provider)
    table.add_row("Provider kind", str(plan.provider_kind or "n/a"))
    table.add_row("Resolved model", str(plan.resolved_model or "n/a"))
    table.add_row("Reasoning effort", str(plan.reasoning_effort or "default"))
    table.add_row("Timeout seconds", str(plan.timeout_seconds or "n/a"))
    table.add_row("Sandbox", plan.sandbox)
    table.add_row("Hypothesis filter", str(hypothesis_id).zfill(6) if hypothesis_id else "all")
    table.add_row("Failed materializations", str(plan.candidate_count))
    table.add_row("Iterations to run", str(plan.iteration_count))
    table.add_row("Files", file_text)
    table.add_row("Hypothesis IDs", id_text)
    console.print(table)


def _print_root_run_plan(project_slug: str, plan: RootRunPlan, *, hypothesis_id: str | None) -> None:
    ids = plan.hypothesis_ids
    if len(ids) > 8:
        id_text = f"{ids[0]}..{ids[-1]}"
    else:
        id_text = ", ".join(ids) if ids else "none"
    if len(plan.files) > 8:
        file_text = f"{plan.files[0]} ... {plan.files[-1]}"
    else:
        file_text = ", ".join(plan.files) if plan.files else "none"
    table = Table(title="ROOT run plan", box=box.SIMPLE_HEAVY, show_header=False, pad_edge=False)
    table.add_column("Parameter", style="bold", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("Project", project_slug)
    table.add_row("Mode", plan.mode)
    table.add_row("Active profile", plan.profile_id)
    table.add_row("Profile hash", plan.profile_hash)
    table.add_row("Execution timeout seconds", str(plan.execution_timeout_seconds))
    table.add_row("Run", plan.run_id or "new")
    table.add_row("Run path", plan.run_path)
    table.add_row("Next node step", str(plan.next_node_step))
    table.add_row("Hypothesis filter", str(hypothesis_id).zfill(6) if hypothesis_id else "all")
    table.add_row("Candidate materializations", str(plan.candidate_count))
    table.add_row("Already evaluated", str(plan.already_evaluated_count))
    table.add_row("Force", "true" if plan.force else "false")
    table.add_row("Iterations to run", str(plan.iteration_count))
    table.add_row("Files", file_text)
    table.add_row("Hypothesis IDs", id_text)
    console.print(table)


def _print_branch_add_summary(project_slug: str, created: CreatedBranch) -> None:
    table = Table(title="BRANCH materialized", box=box.SIMPLE_HEAVY, show_header=False, pad_edge=False)
    table.add_column("Parameter", style="bold", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("Project", project_slug)
    table.add_row("Branch", created.branch_id)
    table.add_row("Parent", f"{created.parent.source_type}:{created.parent.source_id}")
    table.add_row("Source", f"{created.source.source_type}:{created.source.source_id}")
    table.add_row("Mode", created.parent.mode)
    table.add_row("File", _repo_relative(created.branch_path.parent.parent, created.materialization_path))
    table.add_row("Composition hash", created.composition_hash[:12])
    console.print(table)


def _print_branch_rebase_summary(
    project_slug: str,
    target: BranchRebaseTarget,
    plan: BranchRebasePlan,
    rebased: RebasedBranch,
) -> None:
    title = "BRANCH rebase existing" if rebased.existing else "BRANCH rebased"
    table = Table(title=title, box=box.SIMPLE_HEAVY, show_header=False, pad_edge=False)
    table.add_column("Parameter", style="bold", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("Project", project_slug)
    table.add_row("Source branch", rebased.source_branch_id)
    source_node = target.node_id or plan.source_node_id
    source_step = target.step if target.step is not None else plan.source_step
    if source_node:
        table.add_row("Source node", str(source_node))
    if source_step is not None:
        table.add_row("Source step", str(source_step))
    table.add_row("Original score", _format_score(plan.source_score) or "n/a")
    table.add_row("Branch", rebased.branch_id)
    table.add_row("Mode", rebased.mode)
    table.add_row("File", _repo_relative(rebased.branch_path.parent.parent, rebased.materialization_path))
    table.add_row("Composition hash", rebased.composition_hash[:12])
    table.add_row("Components", str(rebased.total_components))
    table.add_row("Changed components", str(rebased.changed_components))
    console.print(table)


def _print_branch_rebase_plan(project_slug: str, plan: BranchRebasePlan) -> None:
    table = Table(title="BRANCH rebase plan", box=box.SIMPLE_HEAVY, show_header=False, pad_edge=False)
    table.add_column("Parameter", style="bold", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("Project", project_slug)
    table.add_row("Source branch", plan.source_branch_id)
    if plan.source_node_id:
        table.add_row("Source node", plan.source_node_id)
    if plan.source_step is not None:
        table.add_row("Source step", str(plan.source_step))
    table.add_row("Mode", plan.mode)
    table.add_row("Components", str(plan.total_components))
    table.add_row("Components to update", str(len(plan.changed_components)))
    table.add_row("Result", f"existing {plan.existing_branch_id}" if plan.existing_branch_id else "new branch")
    table.add_row("Composition hash", plan.composition_hash[:12])
    console.print(table)

    if not plan.changed_components:
        console.print("No component materializations need updating.")
        return

    change_table = Table(title="Component updates", box=box.SIMPLE_HEAVY)
    change_table.add_column("ID", no_wrap=True)
    change_table.add_column("Role", no_wrap=True)
    change_table.add_column("Old", no_wrap=True)
    change_table.add_column("New", no_wrap=True)
    change_table.add_column("Old score", justify="right", no_wrap=True)
    change_table.add_column("New score", justify="right", no_wrap=True)
    change_table.add_column("Delta", justify="right", no_wrap=True)
    for change in plan.changed_components:
        change_table.add_row(
            change.source_id,
            change.role,
            change.old_file,
            change.new_file,
            _format_score(change.old_score) or "n/a",
            _format_score(change.new_score) or "n/a",
            _format_score_delta(change.old_score, change.new_score),
        )
    console.print(change_table)


def _print_branch_rebase_ambiguous_targets(targets: list[BranchRebaseTarget]) -> None:
    table = Table(title="Ambiguous BRANCH rebase target", box=box.SIMPLE_HEAVY)
    table.add_column("Branch", no_wrap=True)
    table.add_column("Step", justify="right", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Score", justify="right", no_wrap=True)
    table.add_column("Node", no_wrap=True)
    for target in targets:
        table.add_row(
            target.branch_id,
            "" if target.step is None else str(target.step),
            target.status or "",
            _format_score(target.metric),
            target.node_id or "",
        )
    console.print(table)
    console.print("No branch was created. Use id=<branch> or node=<node_id> to select one target.")


def _print_branch_algorithm_summary(project_slug: str, project_dir: Path, result: BranchAlgorithmResult) -> None:
    title = "BRANCH add algorithm dry-run" if result.dry_run else "BRANCH add algorithm"
    table = Table(title=title, box=box.SIMPLE_HEAVY, show_header=False, pad_edge=False)
    table.add_column("Parameter", style="bold", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("Project", project_slug)
    table.add_row("Mode", result.mode)
    table.add_row("Active profile", result.profile_id)
    table.add_row("Algorithm", result.algorithm_id)
    table.add_row("Config", _repo_relative(workspace_root(), result.config_path))
    table.add_row("Steps", str(result.requested_steps))
    table.add_row("Planned" if result.dry_run else "Created", str(len(result.items)))
    table.add_row("Skipped", str(result.skipped_count))
    if result.skip_reasons:
        reasons = ", ".join(f"{key}={value}" for key, value in sorted(result.skip_reasons.items()))
        table.add_row("Skip reasons", reasons)
    table.add_row("Stop reason", result.stop_reason)
    console.print(table)

    item_title = "Planned branches" if result.dry_run else "Created branches"
    item_table = Table(title=item_title, box=box.SIMPLE_HEAVY, show_header=True, pad_edge=False)
    item_table.add_column("Branch", no_wrap=True)
    item_table.add_column("Parent", no_wrap=True)
    item_table.add_column("Source", no_wrap=True)
    item_table.add_column("Parent score", justify="right")
    item_table.add_column("Source score", justify="right")
    item_table.add_column("Hash", no_wrap=True)
    item_table.add_column("File", overflow="fold")
    for item in result.items:
        item_table.add_row(
            item.branch_id,
            item.parent_ref,
            item.source_ref,
            _format_score(item.parent_score),
            _format_score(item.source_score),
            item.composition_hash[:12],
            "" if item.materialization_path is None else _repo_relative(project_dir, item.materialization_path),
        )
    if result.items:
        console.print(item_table)
    else:
        console.print("No branch candidates were created.")


def _print_branch_run_plan(project_slug: str, plan: BranchRunPlan, *, branch_id: str | None) -> None:
    ids = plan.branch_ids
    if len(ids) > 8:
        id_text = f"{ids[0]}..{ids[-1]}"
    else:
        id_text = ", ".join(ids) if ids else "none"
    table = Table(title="BRANCH run plan", box=box.SIMPLE_HEAVY, show_header=False, pad_edge=False)
    table.add_column("Parameter", style="bold", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("Project", project_slug)
    table.add_row("Mode", plan.mode)
    table.add_row("Active profile", plan.profile_id)
    table.add_row("Profile hash", plan.profile_hash)
    table.add_row("Execution timeout seconds", str(plan.execution_timeout_seconds))
    table.add_row("Run", plan.run_id or "new")
    table.add_row("Run path", plan.run_path)
    table.add_row("Next node step", str(plan.next_node_step))
    table.add_row("Branch filter", _normalize_branch_display(branch_id) if branch_id else "all")
    table.add_row("Candidate materializations", str(plan.candidate_count))
    table.add_row("Already evaluated", str(plan.already_evaluated_count))
    table.add_row("Iterations to run", str(plan.iteration_count))
    table.add_row("Branch IDs", id_text)
    console.print(table)


def _print_branch_grow_plan(project_slug: str, plan: BranchGrowPlan) -> None:
    table = Table(title="BRANCH grow plan", box=box.SIMPLE_HEAVY, show_header=False, pad_edge=False)
    table.add_column("Parameter", style="bold", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("Project", project_slug)
    table.add_row("Mode", plan.mode)
    table.add_row("Active profile", plan.profile_id)
    table.add_row("Algorithm", plan.algorithm_id)
    table.add_row("Config", _repo_relative(workspace_root(), plan.config_path))
    table.add_row("Steps", str(plan.requested_steps))
    table.add_row("Pending branch runs", str(plan.pending_branch_runs))
    table.add_row("Execution timeout seconds", str(plan.execution_timeout_seconds))
    console.print(table)


def _print_branch_grow_summary(result: BranchGrowResult) -> None:
    table = Table(title="BRANCH grow result", box=box.SIMPLE_HEAVY, show_header=False, pad_edge=False)
    table.add_column("Parameter", style="bold", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("Mode", result.mode)
    table.add_row("Active profile", result.profile_id)
    table.add_row("Algorithm", result.algorithm_id)
    table.add_row("Steps", str(result.requested_steps))
    table.add_row("Processed", str(result.processed_steps))
    table.add_row("Existing pending", str(result.existing_pending_steps))
    table.add_row("New branches", str(result.new_branch_steps))
    table.add_row("Stop reason", result.stop_reason)
    console.print(table)

    item_table = Table(title="Processed branches", box=box.SIMPLE_HEAVY, show_header=True, pad_edge=False)
    item_table.add_column("Step", justify="right", no_wrap=True)
    item_table.add_column("Action", no_wrap=True)
    item_table.add_column("Branch", no_wrap=True)
    item_table.add_column("Parent", no_wrap=True)
    item_table.add_column("Source", no_wrap=True)
    item_table.add_column("Run", no_wrap=True)
    item_table.add_column("Score", justify="right", no_wrap=True)
    item_table.add_column("Node", no_wrap=True)
    best_metric = _best_numeric(item.metric for item in result.items)
    for item in result.items:
        item_table.add_row(
            str(item.step_index),
            item.action,
            item.branch_id,
            item.parent_ref or "-",
            item.source_ref or "-",
            item.run_status,
            _score_text(item.metric, best=best_metric, style="reverse"),
            item.node_id or "",
        )
    if result.items:
        console.print(item_table)
    else:
        console.print("No BRANCH nodes were processed.")


def _print_rerun_plan(project_slug: str, plan: RerunPlan) -> None:
    table = Table(title="RERUN plan", box=box.SIMPLE_HEAVY, show_header=False, pad_edge=False)
    table.add_column("Parameter", style="bold", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("Project", project_slug)
    table.add_row("Source", f"{plan.source_kind} {plan.source_id}")
    table.add_row("Source sha", plan.source_submission_sha256[:10])
    table.add_row("Source CV", _plan_rank_score_text(plan.source_cv_rank, plan.source_local_score))
    table.add_row("Source public", _plan_rank_score_text(plan.source_public_rank, plan.source_public_score))
    table.add_row("Source node", plan.source_node_id)
    table.add_row("Source run", str(plan.source_run_id or ""))
    table.add_row("Source step", "" if plan.source_step is None else str(plan.source_step))
    table.add_row("Mode", plan.mode)
    table.add_row("Rerun profile", plan.profile_id)
    table.add_row("- time", _plan_seconds_text(plan.profile_time_limit_seconds))
    table.add_row("- preset", str(plan.profile_preset or "n/a"))
    table.add_row("- gpu", _plan_bool_text(plan.profile_use_gpu))
    table.add_row("Profile hash", plan.profile_hash)
    table.add_row("Execution timeout seconds", str(plan.execution_timeout_seconds))
    table.add_row("Aux enabled", "true" if plan.aux_enabled else "false")
    table.add_row("Run", plan.run_id or "new")
    table.add_row("Run path", plan.run_path)
    table.add_row("Next node step", str(plan.next_node_step))
    table.add_row("Materialization", plan.materialization_path)
    table.add_row("Code hash", plan.code_hash[:12])
    console.print(table)


def _plan_rank_score_text(rank: object, score: object) -> str:
    if not isinstance(score, int | float):
        return "n/a"
    if isinstance(rank, int):
        return f"({rank}) {float(score):.5f}"
    return f"(-) {float(score):.5f}"


def _plan_seconds_text(value: object) -> str:
    if not isinstance(value, int):
        return "n/a"
    return f"{value}s"


def _plan_bool_text(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return "n/a"


def _print_branch_delete_plan(project_slug: str, plan: BranchDeletePlan) -> None:
    table = Table(title="BRANCH delete plan", box=box.SIMPLE_HEAVY, show_header=False, pad_edge=False)
    table.add_column("Parameter", style="bold", no_wrap=True)
    table.add_column("Value", overflow="fold")
    table.add_row("Project", project_slug)
    table.add_row("Branch", plan.branch_id)
    table.add_row("Mode", plan.mode)
    table.add_row("Parent", plan.parent_ref)
    table.add_row("Source", plan.source_ref)
    table.add_row("Path", _repo_relative(workspace_root(), plan.path))
    table.add_row("Run nodes", str(plan.node_count))
    table.add_row("Force", "true" if plan.force else "false")
    console.print(table)


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


def _print_existing_root_hypotheses(
    project_dir: Path,
    *,
    created_ids: set[str],
    mode: str | None = None,
    profile_id: str | None = None,
    best_score_value: float | None = None,
) -> None:
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
    db_rows = root_hypothesis_rows(project_dir, mode=mode, profile_id=profile_id)
    best = best_score_value if best_score_value is not None else _best_numeric(row.get("best_score") for row in db_rows)
    rows = []
    for db_row in db_rows:
        row = _root_hypothesis_row(db_row, created_ids=created_ids, best_score_value=best)
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
    displayed_index = 0
    for row in old_rows:
        table.add_row(*row, style=_zebra_style(displayed_index))
        displayed_index += 1
    if old_rows and new_rows:
        table.add_row("NEW", *["" for _ in ROOT_HYPOTHESIS_COLUMNS[1:]], style="reverse")
    for row in new_rows:
        table.add_row(*row, style=_zebra_style(displayed_index, extra="bold"))
        displayed_index += 1
    console.print(table)


def _print_root_status_json(
    project_slug: str,
    *,
    config: dict[object, object],
    mode: str,
    profile_id: str,
    profile_hash_value: str,
    counts: dict[str, int],
    best: float | None,
    project_dir: Path,
) -> None:
    payload = {
        "project": project_slug,
        "target_root_count": config.get("root", {}).get("target_count", 20) if isinstance(config.get("root"), dict) else 20,
        "mode": mode,
        "profile_id": profile_id,
        "profile_hash": profile_hash_value,
        "counts": counts,
        "best_root_score": best,
        "hypotheses": [
            _root_hypothesis_json_row(row, created_ids=set())
            for row in root_hypothesis_rows(project_dir, mode=mode, profile_id=profile_id)
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _print_root_materializations(
    project_dir: Path,
    *,
    mode: str,
    created_count: int | None,
    hypothesis_id: str | None,
    only_targets: set[tuple[str, str]] | None = None,
) -> None:
    target_id = hypothesis_id.zfill(6) if hypothesis_id else None
    title = "ROOT materializations" if created_count is None else f"ROOT materializations (created: {created_count})"
    table = Table(title=title, box=box.SIMPLE_HEAVY)
    table.add_column("ID", style="bold", no_wrap=True)
    table.add_column("Rev", justify="right", no_wrap=True)
    table.add_column("S", justify="center", no_wrap=True, width=1)
    table.add_column("File", no_wrap=True)
    table.add_column("Model", no_wrap=True)
    table.add_column("Res/Tokens", justify="right", no_wrap=True)
    table.add_column("Gen", justify="right", no_wrap=True)
    group_index = -1
    previous_hypothesis_id: str | None = None
    rows = materialization_rows(project_dir, mode=mode, hypothesis_id=target_id)
    for db_row in rows:
        current_hypothesis_id = str(db_row.get("hypothesis_id") or "")
        if only_targets is not None and (current_hypothesis_id, str(db_row.get("file") or "")) not in only_targets:
            continue
        if current_hypothesis_id != previous_hypothesis_id:
            group_index += 1
            previous_hypothesis_id = current_hypothesis_id
        row_style = _zebra_style(group_index)
        table.add_row(*_root_materialization_row(db_row), style=row_style)
    console.print(table)
    if created_count is not None:
        _print_failed_materialization_summary(rows)


def _print_root_revision_status(project_dir: Path, *, hypothesis_id: str, mode: str) -> None:
    rows = db_revision_status_rows(project_dir, hypothesis_id=hypothesis_id, mode=mode)
    table = Table(title=f"ROOT revisions {str(hypothesis_id).zfill(6)}", box=box.SIMPLE_HEAVY)
    table.add_column("Revision", justify="right", no_wrap=True)
    table.add_column("S", justify="center", no_wrap=True)
    table.add_column("Hypothesis file", no_wrap=True)
    table.add_column("Mat file", no_wrap=True)
    table.add_column("Started", no_wrap=True)
    table.add_column("Score", justify="right", no_wrap=True)
    for row in rows:
        materialization_file = str(row.get("materialization_file") or "none")
        table.add_row(
            str(row.get("revision") or ""),
            _revision_status_icon(row),
            Path(str(row.get("hypothesis_file") or "")).name,
            materialization_file,
            _short_datetime(row.get("evaluation_created_at") or row.get("component_created_at")) or "-",
            _format_score(row.get("metric")) or "-",
        )
    console.print(table)


def _print_root_revision_overview(
    project_dir: Path,
    *,
    mode: str,
    profile_id: str,
    hypothesis_ids: set[str] | None = None,
) -> None:
    rows = root_revision_overview_rows(project_dir, mode=mode, profile_id=profile_id)
    if hypothesis_ids is not None:
        rows = [row for row in rows if str(row.get("hypothesis_id") or "") in hypothesis_ids]
    table = Table(title="ROOT revision status", box=box.SIMPLE_HEAVY)
    table.add_column("ID", style="bold", no_wrap=True)
    table.add_column("Rev", justify="right", no_wrap=True)
    table.add_column("S", justify="center", no_wrap=True)
    table.add_column("Score", justify="right", no_wrap=True)
    previous_hypothesis_id: str | None = None
    group_index = -1
    for row in rows:
        hypothesis_id = str(row.get("hypothesis_id") or "")
        if hypothesis_id != previous_hypothesis_id:
            group_index += 1
            previous_hypothesis_id = hypothesis_id
        table.add_row(
            hypothesis_id,
            str(row.get("revision") or ""),
            _revision_overview_icon(row),
            _format_score(row.get("revision_score")) or "-",
            style=_zebra_style(group_index),
        )
    console.print(table)


def _revision_overview_icon(row: dict[str, object]) -> Text:
    active = bool(row.get("is_active_revision"))
    best = bool(row.get("is_best_revision"))
    if active and best:
        return Text("★", style="bold green")
    if active:
        return Text("!", style="bold red")
    if best:
        return Text("▶", style="green")
    return Text("·", style="dim")


def _print_root_revision_promote_plan(rows: list[dict[str, object]]) -> None:
    table = Table(title="ROOT revision promotion plan", box=box.SIMPLE_HEAVY)
    table.add_column("ID", style="bold", no_wrap=True)
    table.add_column("To rev", justify="right", no_wrap=True)
    table.add_column("To file", no_wrap=True)
    table.add_column("Score", justify="right", no_wrap=True)
    table.add_column("From rev", justify="right", no_wrap=True)
    table.add_column("From file", no_wrap=True)
    for index, row in enumerate(rows):
        table.add_row(
            str(row.get("hypothesis_id") or ""),
            str(row.get("revision") or ""),
            str(row.get("materialization_file") or ""),
            _format_score(row.get("metric")) or "-",
            str(row.get("active_revision") or "-"),
            str(row.get("active_file") or "-"),
            style=_zebra_style(index),
        )
    console.print(table)


def _promote_root_revisions(
    project_dir: Path,
    *,
    mode: str,
    targets: list[dict[str, object]],
) -> list[dict[str, object]]:
    promoted: list[dict[str, object]] = []
    for row in targets:
        hid = str(row.get("hypothesis_id") or "")
        file_name = str(row.get("materialization_file") or "")
        if not hid or not file_name:
            continue
        revision = int(row.get("revision") or 1)
        hdir = project_dir / "hypotheses" / hid
        code_path = hdir / "materializations" / file_name
        if not code_path.exists():
            raise TmlError(f"Cannot promote missing materialization file: {code_path}")
        set_active_materialization(hdir, mode, file_name)
        upsert_materialization(project_dir, hdir, mode, code_path, active=True, hypothesis_revision=revision)
        promoted.append(dict(row))
    return promoted


def _revision_status_icon(row: dict[str, object]) -> Text:
    evaluation_status = str(row.get("evaluation_status") or "")
    if evaluation_status:
        return _run_status_text(evaluation_status)
    component_status = str(row.get("component_status") or "")
    if component_status:
        return _run_status_text(component_status)
    if row.get("materialization_file"):
        materialization_status = str(row.get("materialization_status") or "")
        return Text("⌘", style="bold yellow" if materialization_status == "fixed" else "cyan")
    return Text("◇", style="dim")


def _root_materialization_row(db_row: dict[str, object]) -> list[object]:
    active = bool(db_row.get("active"))
    status = str(db_row.get("status") or "")
    return [
        str(db_row.get("hypothesis_id") or ""),
        str(db_row.get("hypothesis_revision") or ""),
        _materialization_status_icon(status, active=active),
        str(db_row.get("file") or ""),
        str(db_row.get("model") or ""),
        _token_summary(db_row),
        _seconds_text(db_row.get("generation_seconds")),
    ]


def _print_failed_materialization_summary(rows: list[dict[str, object]]) -> None:
    failed = [row for row in rows if str(row.get("status") or "") == "failed"]
    if not failed:
        return
    shown = [f"{row.get('hypothesis_id')}/{row.get('file')}" for row in failed[:8]]
    suffix = "" if len(failed) <= len(shown) else f", +{len(failed) - len(shown)} more"
    console.print(f"Failed validation materializations: {len(failed)} ({', '.join(shown)}{suffix})", style="bold red")


def _materialization_status_icon(status: str, *, active: bool) -> Text:
    if status == "failed":
        return Text("!", style="bold red")
    if status == "bug":
        return Text("⚠", style="bold red")
    if status == "fixed":
        return Text("⌘", style="bold yellow")
    if active:
        return Text("⌘", style="green")
    return Text("·", style="dim")


def _zebra_style(index: int, *, extra: str | None = None) -> str | None:
    background = "on grey23" if index % 2 else None
    if extra and background:
        return f"{extra} {background}"
    return extra or background


def _print_root_run_summary(
    project_dir: Path,
    *,
    mode: str,
    executed_ids: set[str],
    hypothesis_id: str | None,
    executed_count: int | None = None,
) -> None:
    config = load_project_config(project_dir)
    profile_id = active_profile_id(config, mode)
    count = len(executed_ids) if executed_count is None and executed_ids else executed_count
    title = "ROOT run" if count is None else f"ROOT run (executed: {count})"
    table = Table(title=title, box=box.SIMPLE_HEAVY)
    table.add_column("ID", style="bold", no_wrap=True)
    table.add_column("Rev", justify="right", no_wrap=True)
    table.add_column("S", justify="center", no_wrap=True)
    table.add_column("Created", no_wrap=True)
    table.add_column("Model", no_wrap=True)
    table.add_column("Res/Tokens", justify="right", no_wrap=True)
    table.add_column("Gen", no_wrap=True)
    table.add_column("Run", justify="right", no_wrap=True)
    table.add_column("Score", justify="right", no_wrap=True)
    table.add_column("Source score", justify="right", no_wrap=True)
    table.add_column("Node", no_wrap=True)
    table.add_column("Summary", overflow="fold", min_width=36, ratio=1)
    summary_limit = 30 + max(0, _env_int("TML_WIDE_TERMINAL", 0))
    old_rows: list[list[object]] = []
    new_rows: list[list[object]] = []
    target_id = hypothesis_id.zfill(6) if hypothesis_id else None
    for db_row in root_run_rows(project_dir, mode=mode, profile_id=profile_id, hypothesis_id=target_id):
        row = _root_run_row(db_row, summary_limit=summary_limit)
        hypothesis_id_value = str(db_row.get("hypothesis_id") or "")
        revision_key = f"{hypothesis_id_value}:{db_row.get('hypothesis_revision') or 1}"
        if revision_key in executed_ids or hypothesis_id_value in executed_ids:
            new_rows.append(row)
        else:
            old_rows.append(row)
    displayed_index = 0
    for row in old_rows:
        table.add_row(*row, style=_zebra_style(displayed_index))
        displayed_index += 1
    for row in new_rows:
        table.add_row(*row, style=_zebra_style(displayed_index, extra="bold"))
        displayed_index += 1
    console.print(table)


def _print_branch_status(
    project_dir: Path,
    *,
    mode: str,
    branch_id: str | None,
    executed_ids: set[str],
    executed_count: int | None,
    sort_by: str = "score",
    sort_order: str | None = None,
    only_executed: bool = False,
) -> None:
    config = load_project_config(project_dir)
    profile_id = active_profile_id(config, mode)
    count = len(executed_ids) if executed_count is None and executed_ids else executed_count
    title = "BRANCH status" if count is None else f"BRANCH run (executed: {count})"
    table = Table(title=title, box=box.SIMPLE_HEAVY)
    table.add_column("ID", style="bold", no_wrap=True)
    table.add_column("S", justify="center", no_wrap=True)
    table.add_column("Parent", no_wrap=True)
    table.add_column("Source", no_wrap=True)
    table.add_column("File", no_wrap=True)
    table.add_column("Run", justify="right", no_wrap=True)
    table.add_column("Score", justify="right", no_wrap=True)
    table.add_column("Node", no_wrap=True)
    table.add_column("Summary", overflow="fold", min_width=36, ratio=1)
    summary_limit = 30 + max(0, _env_int("TML_WIDE_TERMINAL", 0))
    rows = branch_rows(
        project_dir,
        mode=mode,
        profile_id=profile_id,
        branch_id=branch_id,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    if only_executed and executed_ids:
        rows = [row for row in rows if str(row.get("branch_id") or "") in executed_ids]
    displayed_index = 0
    for db_row in rows:
        bid = str(db_row.get("branch_id") or "")
        extra = "bold" if bid in executed_ids else None
        table.add_row(*_branch_status_row(db_row, summary_limit=summary_limit), style=_zebra_style(displayed_index, extra=extra))
        displayed_index += 1
    console.print(table)


def _branch_status_row(db_row: dict[str, object], *, summary_limit: int) -> list[object]:
    node_status = str(db_row.get("node_status") or "")
    if node_status:
        status = _run_status_text(node_status)
        score = _format_score(db_row.get("metric"))
        node = str(db_row.get("node_id") or "")
        run_duration = _seconds_text(db_row.get("run_seconds"))
    elif str(db_row.get("branch_status") or "") == "materialized":
        status = Text("⌘", style="cyan")
        score = ""
        source_score = ""
        node = ""
        run_duration = ""
    else:
        status = Text("◇", style="dim")
        score = ""
        source_score = ""
        node = ""
        run_duration = ""
    return [
        str(db_row.get("branch_id") or ""),
        status,
        str(db_row.get("parent_ref") or ""),
        str(db_row.get("source_ref") or ""),
        str(db_row.get("materialization_file") or ""),
        run_duration,
        score,
        _format_score(db_row.get("source_metric")) or "",
        node,
        _short_text(str(db_row.get("summary") or ""), summary_limit),
    ]


def _print_solution_tree(project_dir: Path, *, mode: str) -> None:
    config = load_project_config(project_dir)
    profile_id = active_profile_id(config, mode)
    root_rows = solution_tree_root_rows(project_dir, mode=mode, profile_id=profile_id)
    branch_db_rows = solution_tree_branch_rows(project_dir, mode=mode, profile_id=profile_id)
    runtime_state = read_branch_runtime_state(project_dir, mode=mode, profile_id=profile_id)
    runtime_branch_id = _normalize_branch_display(runtime_state.branch_id) if runtime_state else ""
    runtime_display_state = "running" if runtime_state and runtime_state.is_running else "stale" if runtime_state else ""
    best_metric = _best_tree_metric([*root_rows, *branch_db_rows])
    root_by_id = {str(row.get("hypothesis_id") or ""): row for row in root_rows}
    baseline = root_by_id.get("000000") or {"hypothesis_id": "000000"}
    tree = Tree(_solution_tree_label("root", baseline, best_metric=best_metric), guide_style="bright_black")

    children: dict[str, list[dict[str, object]]] = {"root:000000": []}
    known_keys = {"root:000000"}
    for row in root_rows:
        hid = str(row.get("hypothesis_id") or "")
        if hid == "000000":
            continue
        key = f"root:{hid}"
        known_keys.add(key)
        children.setdefault("root:000000", []).append({"kind": "root", "key": key, "row": row})
        children.setdefault(key, [])

    branch_entries: list[dict[str, object]] = []
    for row in branch_db_rows:
        bid = _normalize_branch_display(row.get("branch_id"))
        if runtime_display_state and bid == runtime_branch_id:
            row = {**row, "_runtime_state": runtime_display_state}
        key = f"branch:{bid}"
        known_keys.add(key)
        children.setdefault(key, [])
        branch_entries.append({"kind": "branch", "key": key, "row": row})

    for entry in branch_entries:
        row = entry["row"]
        assert isinstance(row, dict)
        parent_key = _solution_tree_parent_key(row)
        if parent_key not in known_keys:
            parent_key = "root:000000"
        children.setdefault(parent_key, []).append(entry)

    def append_children(parent_tree: Tree, parent_key: str, visited: set[str]) -> None:
        for child in sorted(children.get(parent_key, []), key=_solution_tree_sort_key):
            key = str(child["key"])
            if key in visited:
                continue
            row = child["row"]
            assert isinstance(row, dict)
            subtree = parent_tree.add(_solution_tree_label(str(child["kind"]), row, best_metric=best_metric))
            append_children(subtree, key, {*visited, key})

    append_children(tree, "root:000000", {"root:000000"})
    console.print(tree)


def _solution_tree_parent_key(row: dict[str, object]) -> str:
    parent_kind = str(row.get("parent_kind") or "").lower()
    parent_id = str(row.get("parent_id") or row.get("parent_ref") or "").strip()
    if parent_kind == "branch" or parent_id.upper().startswith("B"):
        return f"branch:{_normalize_branch_display(parent_id)}"
    if parent_id.isdigit():
        return f"root:{parent_id.zfill(6)}"
    return f"root:{parent_id}"


def _solution_tree_sort_key(entry: dict[str, object]) -> tuple[int, int, str]:
    kind = str(entry.get("kind") or "")
    row = entry.get("row")
    if not isinstance(row, dict):
        return (9, 0, "")
    if kind == "root":
        hid = str(row.get("hypothesis_id") or "")
        return (0, int(hid) if hid.isdigit() else 0, hid)
    bid = _normalize_branch_display(row.get("branch_id"))
    suffix = bid[1:] if bid.upper().startswith("B") else bid
    return (1, int(suffix) if suffix.isdigit() else 0, bid)


def _solution_tree_label(kind: str, row: dict[str, object], *, best_metric: float | None) -> Text:
    identifier = str(row.get("hypothesis_id") or "") if kind == "root" else _normalize_branch_display(row.get("branch_id"))
    if kind == "branch":
        runtime_state = str(row.get("_runtime_state") or "")
        if runtime_state:
            return _solution_tree_runtime_branch_label(identifier, runtime_state)
    node_status = str(row.get("node_status") or "")
    metric = _metric_value(row.get("metric"))
    runtime_suffix = _tree_runtime_suffix(row)
    if node_status:
        if node_status == "complete" and metric is not None:
            line = Text()
            line.append("▶ ", style="green")
            metric_style = "bold yellow" if best_metric is not None and metric == best_metric else "green"
            line.append(f"{metric:.5f}·{identifier}{runtime_suffix}", style=metric_style)
            return line
        if node_status == "failed":
            return Text(f"⚠ bug·{identifier}{runtime_suffix}", style="bold red")
        if node_status == "started":
            return Text(f"● {identifier}{runtime_suffix}", style="cyan")
        return Text(f"{node_status.upper()}·{identifier}{runtime_suffix}", style="yellow")

    if kind == "branch":
        if str(row.get("branch_status") or "") == "materialized":
            return Text(f"⌘ {identifier}", style="cyan")
        return Text(f"◇ {identifier}", style="dim")

    if row.get("code_hash"):
        materialization_status = str(row.get("materialization_status") or "")
        style = "bold yellow" if materialization_status == "fixed" else "cyan"
        return Text(f"⌘ {identifier}", style=style)
    return Text(f"◇ {identifier}", style="dim")


def _solution_tree_runtime_branch_label(identifier: str, runtime_state: str) -> Text:
    line = Text()
    if runtime_state == "stale":
        line.append("⚠ ", style="bold red")
        line.append(identifier, style="cyan")
        return line
    line.append("⌘ ", style="cyan")
    line.append(identifier, style="reverse cyan")
    return line


def _tree_runtime_suffix(row: dict[str, object]) -> str:
    text = _seconds_text(row.get("run_seconds"))
    return f"·{text}" if text else ""


def _metric_value(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _best_tree_metric(rows: list[dict[str, object]]) -> float | None:
    values = [_metric_value(row.get("metric")) for row in rows]
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _print_root_run_request_status(project_dir: Path, *, mode: str, hypothesis_id: str | None) -> None:
    rows = run_request_status(project_dir, mode, hypothesis_id)
    if not rows:
        return
    status = str(rows[0].get("status") or "")
    hid = str(rows[0].get("hypothesis_id") or (hypothesis_id or "").zfill(6))
    if status == "missing_hypothesis":
        console.print(f"Run skipped: hypothesis {hid} does not exist.")
    elif status == "missing_materialization":
        console.print(f"Run skipped: hypothesis {hid} has no {mode} materialization. Run: uv run tml root materialize id={int(hid)}")


def _print_submissions(project_dir: Path, *, limit: int | None = None) -> None:
    rows = submission_rows(project_dir)
    display_rows = _submission_display_rows(rows)
    shown_rows = display_rows if limit is None else display_rows[:limit]
    title = "Submission candidates"
    if display_rows and len(shown_rows) < len(display_rows):
        title = f"{title} (showing {len(shown_rows)}/{len(display_rows)}; limit=0 for all)"
    table = Table(title=title, box=box.SIMPLE_HEAVY)
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("ID", no_wrap=True)
    table.add_column("CV#", justify="right", no_wrap=True)
    table.add_column("PUB#", justify="right", no_wrap=True)
    table.add_column("cv", justify="right", no_wrap=True)
    table.add_column("public", justify="right", no_wrap=True)
    table.add_column("S", no_wrap=True)
    table.add_column("kind/profile", no_wrap=True)
    table.add_column("time", justify="right", no_wrap=True)
    table.add_column("metric", no_wrap=True)
    table.add_column("run", no_wrap=True, overflow="ellipsis", max_width=38)
    table.add_column("step", justify="right", no_wrap=True)
    table.add_column("date", no_wrap=True)
    table.add_column("sha", no_wrap=True)
    if not rows:
        console.print("No submissions recorded in project database.")
        return

    best_local = _best_numeric(row.get("local_score") for row in rows)
    best_public = _best_numeric(row.get("public_score") for row in rows)
    for marker, row, is_child, group_index in shown_rows:
        local_score = row.get("local_score")
        public_score = row.get("public_score")
        table.add_row(
            marker,
            str(row.get("source_id") or ""),
            _rank_text(row.get("cv_rank"), local_score),
            _rank_text(row.get("public_rank") or row.get("computed_public_rank"), public_score),
            _score_text(local_score, best=best_local, style="bold black on bright_green"),
            _score_text(public_score, best=best_public, style="bold black on bright_cyan"),
            _submit_status_text(row.get("submit_status")),
            _kind_profile_text(row),
            _minutes_text(row.get("run_seconds")),
            _metric_short_text(row.get("metric")),
            _run_short_text(row.get("run_id")),
            _step_text(row.get("step")),
            _date_yyyymmdd(row.get("created_at")),
            str(row.get("submission_sha256") or "")[:10],
            style=_submission_row_style(group_index, is_child=is_child),
        )
    console.print(table)
    _print_submission_actions(rows)


def _submission_table_limit(value: object) -> int | None:
    if value is None:
        return _adaptive_table_limit(default=DEFAULT_SUBMISSION_TABLE_LIMIT)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise TmlError(f"Invalid limit: {value!r}. Use a non-negative integer.") from exc
    if parsed < 0:
        raise TmlError("Invalid limit: use a non-negative integer.")
    return None if parsed == 0 else parsed


def _adaptive_table_limit(*, default: int) -> int:
    terminal_rows = shutil.get_terminal_size(fallback=(80, default)).lines
    if terminal_rows <= 0:
        return default
    return max(default, int(terminal_rows * 0.65))


def _submission_display_rows(rows: list[dict[str, object]]) -> list[tuple[str, dict[str, object], bool, int]]:
    children_by_source: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        source_sha = str(row.get("source_submission_sha256") or "")
        if source_sha:
            children_by_source.setdefault(source_sha, []).append(row)

    display: list[tuple[str, dict[str, object], bool, int]] = []
    child_ids = {id(row) for children in children_by_source.values() for row in children}
    seen_child_ids: set[int] = set()
    parent_index = 0
    for row in rows:
        if id(row) in child_ids:
            continue
        parent_index += 1
        display.append((str(parent_index), row, False, parent_index))
        source_sha = str(row.get("submission_sha256") or "")
        children = children_by_source.get(source_sha, [])
        for child_index, child in enumerate(children, start=1):
            marker = "└─" if child_index == len(children) else "├─"
            display.append((marker, child, True, parent_index))
            seen_child_ids.add(id(child))

    for row in rows:
        if id(row) in seen_child_ids:
            continue
        if not str(row.get("source_submission_sha256") or ""):
            continue
        parent_index += 1
        display.append((str(parent_index), row, False, parent_index))
    return display


def _submission_row_style(group_index: int, *, is_child: bool) -> str:
    if is_child:
        return "on grey11"
    return "on grey23" if group_index % 2 else "on grey11"


def _print_submission_actions(rows: list[dict[str, object]]) -> None:
    ready = [
        row
        for row in rows
        if isinstance(row.get("local_score"), int | float)
        and str(row.get("status") or "") == "complete"
        and str(row.get("submit_status") or "") == "not_submitted"
        and str(row.get("submission_sha256") or "")
    ]
    ready_submit = ready[:5]
    ready_rerun = [row for row in ready if _is_rerunnable_submission(row)][:5]
    if ready_submit:
        console.print()
        console.print("[bold]Ready submit commands:[/bold]")
        for row in ready_submit:
            sha = str(row.get("submission_sha256") or "")[:10]
            console.print(f"uv run tml kaggle submit sha={sha}")

    if ready_rerun:
        console.print()
        console.print("[bold]Ready rerun commands:[/bold]")
        for row in ready_rerun:
            sha = str(row.get("submission_sha256") or "")[:10]
            console.print(f"uv run tml rerun sha={sha}")
    synced_rows = sum(1 for row in rows if str(row.get("remote_status") or ""))
    if synced_rows:
        console.print(f"Remote Kaggle submissions synced local rows: {synced_rows}")
    else:
        console.print("Remote Kaggle submissions not synced. Run: uv run tml kaggle sync")


def _is_rerunnable_submission(row: dict[str, object]) -> bool:
    if str(row.get("kind") or "") == "rerun":
        return False
    return not str(row.get("source_submission_sha256") or "")


def _validate_submit_row(row: dict[str, object], *, allow_uploaded: bool = False) -> None:
    sha = str(row.get("submission_sha256") or "")[:10]
    if str(row.get("status") or "") != "complete":
        raise TmlError(f"Submission {sha} is not submit-ready: run status is {row.get('status')}.")
    submit_status = str(row.get("submit_status") or "not_submitted")
    if not allow_uploaded and submit_status != "not_submitted":
        raise TmlError(f"Submission {sha} is already uploaded; use force=true to upload it again.")
    if not isinstance(row.get("local_score"), int | float):
        raise TmlError(f"Submission {sha} is not submit-ready: missing local score.")


def _kaggle_submit_message(row: dict[str, object]) -> str:
    score = row.get("local_score")
    score_text = "nan" if not isinstance(score, int | float) else f"{float(score):.5f}"
    run_seconds = row.get("run_seconds")
    time_text = f" | time={float(run_seconds) / 60.0:.1f}m" if isinstance(run_seconds, int | float) else ""
    metric_text = f" | metric={row['metric']}" if row.get("metric") else ""
    node_id = str(row.get("node_id") or "")
    node_text = node_id[:8] if node_id else "unknown"
    return (
        f"cv={score_text} | run={row.get('run_id')} | step={row.get('step')} | "
        f"tml_ts={_date_yyyymmdd(row.get('created_at'))} | node={node_text} | "
        f"sha={str(row.get('submission_sha256') or '')[:10]} | algo=AG{metric_text}{time_text}"
    )


def _print_kaggle_submit_dry_run(
    project_dir: Path,
    competition: str,
    row: dict[str, object],
    submission_path: Path,
    upload_path: Path,
    message: str,
) -> None:
    table = Table(title="Kaggle submit dry-run", box=box.SIMPLE_HEAVY, show_header=False)
    table.add_column("Field", style="bold", no_wrap=True)
    table.add_column("Value", overflow="fold")
    exists = submission_path.is_file()
    size = submission_path.stat().st_size if exists else None
    table.add_row("competition", competition)
    table.add_row("submission_path", _repo_relative(project_dir, submission_path))
    table.add_row("upload_path", _repo_relative(project_dir, upload_path))
    table.add_row("upload_filename", upload_path.name)
    table.add_row("file_exists", str(exists).lower())
    table.add_row("file_size_bytes", "" if size is None else str(size))
    table.add_row("message", message)
    table.add_row("sha", str(row.get("submission_sha256") or ""))
    table.add_row("node", str(row.get("node_id") or ""))
    table.add_row("run", str(row.get("run_id") or ""))
    table.add_row("step", str(row.get("step") or ""))
    score = row.get("local_score")
    score_text = "" if not isinstance(score, int | float) else f"{float(score):.5f}"
    table.add_row("local_score", score_text)
    table.add_row("metric", str(row.get("metric") or ""))
    table.add_row("submit_status", str(row.get("submit_status") or ""))
    console.print(table)
    console.print("Dry-run only: no file copied, no Kaggle upload, no database update.")


def _upload_submission_filename(row: dict[str, object]) -> str:
    score = row.get("local_score")
    score_text = "nan" if not isinstance(score, int | float) else f"{float(score):.5f}"
    sha = str(row.get("submission_sha256") or "nohash")[:10]
    node = str(row.get("node_id") or "unknown")[:8]
    step = _step_text(row.get("step")) or "na"
    date = _date_yyyymmdd(row.get("created_at")) or "unknown"
    return f"sub_{date}_step-{step}_node-{node}_sha-{sha}_cv-{score_text}.csv"


def _repo_relative(base_dir: Path, path: Path) -> str:
    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _best_numeric(values) -> float | None:
    parsed: list[float] = []
    for value in values:
        if isinstance(value, int | float):
            parsed.append(float(value))
    return max(parsed) if parsed else None


def _rank_text(value: object, score: object) -> str:
    if not isinstance(score, int | float):
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return ""


def _score_text(value: object, *, best: float | None, style: str) -> str | Text:
    if not isinstance(value, int | float):
        return ""
    text = f"{float(value):.5f}"
    if best is not None and float(value) == best:
        return Text(text, style=style)
    return text


def _submit_status_text(value: object) -> Text:
    text = str(value or "")
    if text == "submitted":
        return Text("✓", style="green")
    if text == "uploaded":
        return Text("U", style="yellow")
    if text == "not_submitted":
        return Text("-", style="dim")
    if text == "failed":
        return Text("✗", style="red")
    return Text((text[:1] or "?").upper())


def _kind_profile_text(row: dict[str, object]) -> str:
    kind = str(row.get("kind") or "")
    profile = str(row.get("profile_id") or "")
    if kind and profile:
        return f"{kind}/{profile}"
    return profile or kind


def _run_short_text(value: object) -> str:
    text = str(value or "")
    return text.rsplit("-", 1)[-1] if "-" in text else text


def _metric_short_text(value: object) -> str:
    text = str(value or "")
    if not text:
        return ""
    return METRIC_SHORT_SYMBOLS.get(_metric_key(text), text)


def _metric_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_").replace("²", "2")


def _minutes_text(value: object) -> str:
    if not isinstance(value, int | float):
        return ""
    return f"{float(value) / 60.0:.1f}m"


def _step_text(value: object) -> str:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return ""
    return str(parsed)


def _date_yyyymmdd(value: object) -> str:
    text = str(value or "")
    return text[:10].replace("-", "") if len(text) >= 10 else text


def _root_run_row(
    db_row: dict[str, object],
    *,
    summary_limit: int,
) -> list[object]:
    node_status = str(db_row.get("node_status") or "")
    if node_status:
        status = _run_status_text(node_status)
        score = _format_score(db_row.get("metric"))
        node = str(db_row.get("node_id") or "")
        run_duration = _seconds_text(db_row.get("run_seconds"))
    elif db_row.get("component_status"):
        status = _run_status_text(str(db_row.get("component_status") or ""))
        score = ""
        node = str(db_row.get("component_node_id") or "")
        run_duration = _seconds_text(db_row.get("component_run_seconds"))
    elif db_row.get("code_hash"):
        materialization_status = str(db_row.get("materialization_status") or "")
        status = Text("⌘", style="bold yellow" if materialization_status == "fixed" else "cyan")
        score = ""
        node = ""
        run_duration = ""
    else:
        status = Text("◇", style="dim")
        score = ""
        node = ""
        run_duration = ""
    return [
        str(db_row.get("hypothesis_id") or ""),
        str(db_row.get("hypothesis_revision") or ""),
        status,
        _short_datetime(db_row.get("created_at")),
        str(db_row.get("model") or ""),
        _token_summary(db_row),
        _seconds_text(db_row.get("generation_seconds")),
        run_duration,
        score,
        node,
        _short_text(str(db_row.get("summary") or ""), summary_limit),
    ]


def _run_status_text(status: str) -> Text:
    if status == "complete":
        return Text("▶", style="green")
    if status == "failed":
        return Text("⚠", style="bold red")
    if status == "started":
        return Text("●", style="cyan")
    return Text(status.upper() or "?", style="yellow")


def _format_score(value: object) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.5f}"
    return ""


def _format_score_delta(old_score: object, new_score: object) -> str:
    if not (isinstance(old_score, int | float) and isinstance(new_score, int | float)):
        return "n/a"
    delta = float(new_score) - float(old_score)
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.5f}"


def _normalize_branch_display(value: object) -> str:
    text = str(value or "").strip()
    if text.upper().startswith("B") and text[1:].isdigit():
        return f"B{int(text[1:]):06d}"
    if text.isdigit():
        return f"B{int(text):06d}"
    return text


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


def _short_datetime(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if len(text) >= 16 and text[4] == "-" and text[7] == "-" and text[10] == "T":
        return text[5:16]
    return text


def _root_hypothesis_json_row(db_row: dict[str, object], *, created_ids: set[str]) -> dict[str, object]:
    row = _root_hypothesis_row(db_row, created_ids=created_ids)
    if not row:
        return {}
    status = row.get("status", "")
    if isinstance(status, Text):
        status = status.plain
    return {
        "is_new": bool(row["is_new"]),
        **{
            str(column["key"]): status if str(column["key"]) == "status" else row.get(str(column["key"]), "")
            for column in ROOT_HYPOTHESIS_COLUMNS
        },
    }


def _root_hypothesis_table_values(row: dict[str, object], *, summary_limit: int) -> list[object]:
    values: list[object] = []
    for column in ROOT_HYPOTHESIS_COLUMNS:
        key = str(column["key"])
        raw_value = row.get(key, "")
        value: object = raw_value
        if column.get("truncate"):
            value = _short_text(str(raw_value), summary_limit)
        values.append(value)
    return values


def _root_hypothesis_row(
    db_row: dict[str, object],
    *,
    created_ids: set[str],
    best_score_value: float | None = None,
) -> dict[str, object]:
    hypothesis_id = str(db_row.get("hypothesis_id") or "")
    return {
        "id": hypothesis_id,
        "is_new": hypothesis_id in created_ids,
        "status": _hypothesis_status_text(str(db_row.get("status_icon") or "")),
        "revision": str(db_row.get("best_revision") or ""),
        "score": _score_text(db_row.get("best_score"), best=best_score_value, style="reverse"),
        "created_at": _short_datetime(db_row.get("created_at")),
        "model": str(db_row.get("model") or ""),
        "reasoning_tokens": _token_summary(db_row),
        "duration": _seconds_text(db_row.get("generation_seconds")),
        "summary": str(db_row.get("summary") or ""),
    }


def _hypothesis_status_text(status: str) -> Text:
    if status == "▶":
        return Text("▶", style="green")
    if status == "⚠":
        return Text("⚠", style="bold red")
    if status == "⌘":
        return Text("⌘", style="yellow")
    if status in {"⊘", "◇"}:
        return Text(status, style="dim")
    return Text(status or "?", style="yellow")


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


def _print_metrics_help() -> None:
    table = Table(title="Metrics", box=box.SIMPLE_HEAVY)
    table.add_column("Aliases", no_wrap=True)
    table.add_column("Meaning", overflow="fold")
    for aliases, meaning in METRIC_ALIAS_ROWS:
        table.add_row(aliases, meaning)
    console.print(table)


def _print_help_topics() -> None:
    table = Table(title="Help topics", box=box.SIMPLE_HEAVY)
    table.add_column("Topic", no_wrap=True)
    table.add_column("Command", no_wrap=True)
    table.add_column("Description", overflow="fold")
    table.add_row("metrics", "uv run tml help=metrics", "Metric aliases and short symbols.")
    console.print(table)


if __name__ == "__main__":
    app()
