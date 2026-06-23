from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tml.branches.algorithms import branch_add_algorithmic_one, load_branch_algorithm
from tml.branches.run import next_pending_branch, run_one_branch
from tml.core.config import active_mode, active_profile_id, load_project_config
from tml.core.errors import TmlError
from tml.db.state import pending_branch_run_candidates
from tml.hypotheses.run import _execution_timeout_seconds


@dataclass(frozen=True)
class BranchGrowPlan:
    mode: str
    profile_id: str
    algorithm_id: str
    requested_steps: int
    pending_branch_runs: int
    execution_timeout_seconds: int
    config_path: Path


@dataclass(frozen=True)
class BranchGrowItem:
    step_index: int
    action: str
    branch_id: str
    parent_ref: str | None
    source_ref: str | None
    parent_score: float | None
    source_score: float | None
    composition_hash: str | None
    run_status: str
    metric: float | None
    node_id: str | None
    run_seconds: int | None


@dataclass(frozen=True)
class BranchGrowResult:
    mode: str
    profile_id: str
    algorithm_id: str
    requested_steps: int
    processed_steps: int
    existing_pending_steps: int
    new_branch_steps: int
    config_path: Path
    items: list[BranchGrowItem]
    stop_reason: str


def branch_grow_plan(
    project_dir: Path,
    *,
    steps: int,
    algo_id: str,
    mode: str | None = None,
    profile_overrides: dict[str, object] | None = None,
) -> BranchGrowPlan:
    if steps <= 0:
        raise TmlError("steps must be a positive integer.")
    config = load_project_config(project_dir)
    active_run_mode = mode or active_mode(config)
    profile_id = active_profile_id(config, active_run_mode)
    algorithm = load_branch_algorithm(project_dir, algo_id)
    pending = pending_branch_run_candidates(project_dir, mode=active_run_mode, profile_id=profile_id)
    return BranchGrowPlan(
        mode=active_run_mode,
        profile_id=profile_id,
        algorithm_id=algorithm.algorithm_id,
        requested_steps=steps,
        pending_branch_runs=len(pending),
        execution_timeout_seconds=_execution_timeout_seconds(profile_overrides),
        config_path=algorithm.path,
    )


def branch_grow(
    project_dir: Path,
    *,
    steps: int,
    algo_id: str,
    mode: str | None = None,
    profile_overrides: dict[str, object] | None = None,
    progress: Callable[[str], None] | None = None,
) -> BranchGrowResult:
    plan = branch_grow_plan(
        project_dir,
        steps=steps,
        algo_id=algo_id,
        mode=mode,
        profile_overrides=profile_overrides,
    )
    items: list[BranchGrowItem] = []
    stop_reason = ""

    for step_index in range(1, steps + 1):
        pending = next_pending_branch(project_dir, mode=plan.mode, profile_id=plan.profile_id)
        if pending is not None:
            branch_id = str(pending["branch_id"])
            _emit(progress, f"BRANCH grow {step_index}/{steps}: run pending {branch_id}")
            run_item = run_one_branch(
                project_dir,
                mode=plan.mode,
                branch_id=branch_id,
                profile_overrides=profile_overrides,
            )
            if run_item is None:
                stop_reason = f"Pending branch disappeared before execution: {branch_id}"
                break
            _emit(progress, _run_progress_line(step_index, steps, run_item.branch_id, run_item.run_status, run_item.metric))
            items.append(
                BranchGrowItem(
                    step_index=step_index,
                    action="run_pending",
                    branch_id=run_item.branch_id,
                    parent_ref=None,
                    source_ref=None,
                    parent_score=None,
                    source_score=None,
                    composition_hash=str(pending.get("composition_hash") or "") or None,
                    run_status=run_item.run_status,
                    metric=run_item.metric,
                    node_id=run_item.node_id,
                    run_seconds=run_item.run_seconds,
                )
            )
            continue

        added = branch_add_algorithmic_one(project_dir, algo_id=algo_id, mode=plan.mode)
        if added is None:
            stop_reason = "No pending branch and no valid algorithmic candidate."
            break

        _emit(progress, f"BRANCH grow {step_index}/{steps}: add {added.branch_id} parent={added.parent_ref} source={added.source_ref}")
        run_item = run_one_branch(
            project_dir,
            mode=plan.mode,
            branch_id=added.branch_id,
            profile_overrides=profile_overrides,
        )
        if run_item is None:
            stop_reason = f"New branch could not be executed: {added.branch_id}"
            break
        _emit(progress, _run_progress_line(step_index, steps, run_item.branch_id, run_item.run_status, run_item.metric))
        items.append(
            BranchGrowItem(
                step_index=step_index,
                action="add_and_run",
                branch_id=run_item.branch_id,
                parent_ref=added.parent_ref,
                source_ref=added.source_ref,
                parent_score=added.parent_score,
                source_score=added.source_score,
                composition_hash=added.composition_hash,
                run_status=run_item.run_status,
                metric=run_item.metric,
                node_id=run_item.node_id,
                run_seconds=run_item.run_seconds,
            )
        )

    if not stop_reason and len(items) >= steps:
        stop_reason = "Requested steps satisfied."

    return BranchGrowResult(
        mode=plan.mode,
        profile_id=plan.profile_id,
        algorithm_id=plan.algorithm_id,
        requested_steps=steps,
        processed_steps=len(items),
        existing_pending_steps=sum(1 for item in items if item.action == "run_pending"),
        new_branch_steps=sum(1 for item in items if item.action == "add_and_run"),
        config_path=plan.config_path,
        items=items,
        stop_reason=stop_reason,
    )


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _run_progress_line(step_index: int, steps: int, branch_id: str, status: str, metric: float | None) -> str:
    score = "" if metric is None else f" score={metric:.5f}"
    return f"BRANCH grow {step_index}/{steps}: {branch_id} -> {status}{score}"
