from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

from tml.branches.compose import add_branch, plan_branch_composition
from tml.core.config import active_mode, active_profile_id, load_project_config, repo_root_for_project
from tml.core.errors import TmlError
from tml.db.state import (
    branch_algorithm_candidate_pairs,
    direct_branch_child_counts,
    existing_branch_composition_hashes,
)
from tml.utils.yaml_io import read_yaml


@dataclass(frozen=True)
class ScoreEpsilon:
    raw: object
    kind: str
    value: float


@dataclass(frozen=True)
class BranchAlgorithmConfig:
    algorithm_id: str
    path: Path
    parent_kinds: list[str]
    source_kinds: list[str]
    max_children: int
    epsilon: ScoreEpsilon
    candidate_limit: int = 500


@dataclass(frozen=True)
class BranchAlgorithmParent:
    parent_ref: str
    parent_score: float


@dataclass(frozen=True)
class BranchAlgorithmItem:
    branch_id: str
    parent_ref: str
    source_ref: str
    parent_score: float | None
    source_score: float | None
    composition_hash: str
    materialization_path: Path | None


@dataclass(frozen=True)
class BranchAlgorithmResult:
    mode: str
    profile_id: str
    algorithm_id: str
    requested_steps: int
    dry_run: bool
    config_path: Path
    items: list[BranchAlgorithmItem]
    skipped_count: int
    skip_reasons: dict[str, int] = field(default_factory=dict)
    stop_reason: str = ""


def branch_add_algorithmic(
    project_dir: Path,
    *,
    steps: int,
    algo_id: str,
    mode: str | None = None,
    dry_run: bool = False,
) -> BranchAlgorithmResult:
    if steps <= 0:
        raise TmlError("steps must be a positive integer.")
    config = load_project_config(project_dir)
    active_run_mode = mode or active_mode(config)
    profile_id = active_profile_id(config, active_run_mode)
    algorithm = load_branch_algorithm(project_dir, algo_id)

    planned_compositions = existing_branch_composition_hashes(project_dir, mode=active_run_mode)
    child_counts = direct_branch_child_counts(project_dir, mode=active_run_mode)
    items: list[BranchAlgorithmItem] = []
    skip_reasons: dict[str, int] = {}
    stop_reason = ""

    while len(items) < steps:
        item, item_skip_reasons, item_stop_reason = _branch_add_algorithmic_next(
            project_dir,
            algorithm=algorithm,
            active_run_mode=active_run_mode,
            profile_id=profile_id,
            planned_compositions=planned_compositions,
            child_counts=child_counts,
            dry_run=dry_run,
            item_index=len(items) + 1,
        )
        for reason, count in item_skip_reasons.items():
            skip_reasons[reason] = skip_reasons.get(reason, 0) + count
        if item is None:
            stop_reason = item_stop_reason
            break
        items.append(item)

    if not stop_reason and len(items) >= steps:
        stop_reason = "Requested steps satisfied."

    return BranchAlgorithmResult(
        mode=active_run_mode,
        profile_id=profile_id,
        algorithm_id=algorithm.algorithm_id,
        requested_steps=steps,
        dry_run=dry_run,
        config_path=algorithm.path,
        items=items,
        skipped_count=sum(skip_reasons.values()),
        skip_reasons=skip_reasons,
        stop_reason=stop_reason,
    )


def branch_add_algorithmic_one(
    project_dir: Path,
    *,
    algo_id: str,
    mode: str | None = None,
    preferred_parent_ref: str | None = None,
    require_preferred_parent: bool = False,
) -> BranchAlgorithmItem | None:
    config = load_project_config(project_dir)
    active_run_mode = mode or active_mode(config)
    profile_id = active_profile_id(config, active_run_mode)
    algorithm = load_branch_algorithm(project_dir, algo_id)
    item, _, _ = _branch_add_algorithmic_next(
        project_dir,
        algorithm=algorithm,
        active_run_mode=active_run_mode,
        profile_id=profile_id,
        planned_compositions=existing_branch_composition_hashes(project_dir, mode=active_run_mode),
        child_counts=direct_branch_child_counts(project_dir, mode=active_run_mode),
        dry_run=False,
        item_index=1,
        preferred_parent_ref=preferred_parent_ref,
        require_preferred_parent=require_preferred_parent,
    )
    return item


def branch_algorithm_top_parent(
    project_dir: Path,
    *,
    algo_id: str,
    mode: str | None = None,
) -> BranchAlgorithmParent | None:
    config = load_project_config(project_dir)
    active_run_mode = mode or active_mode(config)
    profile_id = active_profile_id(config, active_run_mode)
    algorithm = load_branch_algorithm(project_dir, algo_id)
    candidates = branch_algorithm_candidate_pairs(
        project_dir,
        mode=active_run_mode,
        profile_id=profile_id,
        parent_kinds=algorithm.parent_kinds,
        source_kinds=algorithm.source_kinds,
        max_children=algorithm.max_children,
        limit=1,
    )
    if not candidates:
        return None
    candidate = candidates[0]
    parent_score = _optional_float(candidate.get("parent_score"))
    if parent_score is None:
        return None
    return BranchAlgorithmParent(parent_ref=str(candidate["parent_ref"]), parent_score=parent_score)


def load_branch_algorithm(project_dir: Path, algo_id: str) -> BranchAlgorithmConfig:
    text = str(algo_id).strip()
    if not text:
        raise TmlError("Missing required parameter: algo=<id>.")
    if "/" in text or "\\" in text or text in {".", ".."}:
        raise TmlError(f"Invalid branch algorithm id: {algo_id}")

    project_path = project_dir / "profiles" / "branch" / f"{text}.yaml"
    root_path = repo_root_for_project(project_dir) / "profiles" / "branch" / f"{text}.yaml"
    path = project_path if project_path.exists() else root_path
    payload = read_yaml(path)
    if not payload:
        raise TmlError(f"Missing branch algorithm profile: {text}")
    return _parse_branch_algorithm(path, text, payload)


def _parse_branch_algorithm(path: Path, algo_id: str, payload: dict[str, Any]) -> BranchAlgorithmConfig:
    if int(payload.get("schema_version") or 0) != 1:
        raise TmlError(f"Invalid branch algorithm schema_version in {path}.")
    if str(payload.get("kind") or "") != "branch_add_algorithm":
        raise TmlError(f"Invalid branch algorithm kind in {path}.")
    configured_id = str(payload.get("algorithm_id") or algo_id)
    if configured_id != algo_id:
        raise TmlError(f"Branch algorithm id mismatch: requested {algo_id}, profile contains {configured_id}.")

    selection = payload.get("selection") if isinstance(payload.get("selection"), dict) else {}
    limits = payload.get("limits") if isinstance(payload.get("limits"), dict) else {}
    score = payload.get("score") if isinstance(payload.get("score"), dict) else {}

    parent_kinds = _node_kinds(selection.get("parent_kinds"), default=["root", "branch"])
    source_kinds = _node_kinds(selection.get("source_kinds"), default=["root"])
    if "branch" in source_kinds:
        raise TmlError("Branch algorithm v1 supports source_kinds: [root] only.")

    max_children = int(limits.get("max_children") or 10)
    if max_children <= 0:
        raise TmlError("Branch algorithm limits.max_children must be positive.")
    epsilon = parse_score_epsilon(score.get("epsilon"))

    return BranchAlgorithmConfig(
        algorithm_id=configured_id,
        path=path,
        parent_kinds=parent_kinds,
        source_kinds=source_kinds,
        max_children=max_children,
        epsilon=epsilon,
    )


def _branch_add_algorithmic_next(
    project_dir: Path,
    *,
    algorithm: BranchAlgorithmConfig,
    active_run_mode: str,
    profile_id: str,
    planned_compositions: set[str],
    child_counts: dict[tuple[str, str], int],
    dry_run: bool,
    item_index: int,
    preferred_parent_ref: str | None = None,
    require_preferred_parent: bool = False,
) -> tuple[BranchAlgorithmItem | None, dict[str, int], str]:
    skip_reasons: dict[str, int] = {}
    candidates = branch_algorithm_candidate_pairs(
        project_dir,
        mode=active_run_mode,
        profile_id=profile_id,
        parent_kinds=algorithm.parent_kinds,
        source_kinds=algorithm.source_kinds,
        max_children=algorithm.max_children,
        limit=algorithm.candidate_limit,
        parent_ref=preferred_parent_ref if require_preferred_parent else None,
    )
    if not candidates:
        return None, skip_reasons, "No score-ranked candidate pairs found."

    ordered_candidates = _ordered_candidates(
        candidates,
        preferred_parent_ref=preferred_parent_ref,
        require_preferred_parent=require_preferred_parent,
    )
    if not ordered_candidates and preferred_parent_ref and require_preferred_parent:
        return None, skip_reasons, f"No score-ranked candidate pairs found for node {preferred_parent_ref}."

    for candidate in ordered_candidates:
        parent_key = (str(candidate["parent_kind"]), str(candidate["parent_id"]))
        parent_child_count = child_counts.get(parent_key, int(candidate["parent_child_count"] or 0))
        if parent_child_count >= algorithm.max_children:
            _count_skip(skip_reasons, "parent_child_limit")
            continue
        try:
            composition = plan_branch_composition(
                project_dir,
                parent_ref=str(candidate["parent_ref"]),
                source_ref=str(candidate["source_ref"]),
                mode=active_run_mode,
            )
        except TmlError:
            _count_skip(skip_reasons, "invalid_composition")
            continue

        if composition.existing_branch is not None:
            _count_skip(skip_reasons, "existing_composition")
            planned_compositions.add(composition.composition_hash)
            continue
        if composition.composition_hash in planned_compositions:
            _count_skip(skip_reasons, "planned_composition")
            continue

        if dry_run:
            branch_id = f"planned-{item_index}"
            materialization_path = None
        else:
            created = add_branch(
                project_dir,
                parent_ref=str(candidate["parent_ref"]),
                source_ref=str(candidate["source_ref"]),
                mode=active_run_mode,
            )
            branch_id = created.branch_id
            materialization_path = created.materialization_path

        planned_compositions.add(composition.composition_hash)
        child_counts[parent_key] = parent_child_count + 1
        return (
            BranchAlgorithmItem(
                branch_id=branch_id,
                parent_ref=str(candidate["parent_ref"]),
                source_ref=str(candidate["source_ref"]),
                parent_score=_optional_float(candidate.get("parent_score")),
                source_score=_optional_float(candidate.get("source_score")),
                composition_hash=composition.composition_hash,
                materialization_path=materialization_path,
            ),
            skip_reasons,
            "",
        )

    if preferred_parent_ref and require_preferred_parent:
        return None, skip_reasons, f"No valid candidate pair found for node {preferred_parent_ref}."
    return None, skip_reasons, f"No valid candidate pair found in the top {len(candidates)} candidate pairs."


def parse_score_epsilon(value: object) -> ScoreEpsilon:
    if value is None:
        return ScoreEpsilon(raw=value, kind="absolute", value=0.0)
    if isinstance(value, bool):
        raise TmlError("Invalid score.epsilon: boolean values are not supported.")
    if isinstance(value, int | float):
        parsed = float(value)
        _validate_epsilon_number(parsed, value)
        return ScoreEpsilon(raw=value, kind="absolute", value=parsed)
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("%"):
            raw_number = text[:-1].strip()
            parsed = _parse_epsilon_float(raw_number, value)
            _validate_epsilon_number(parsed, value)
            return ScoreEpsilon(raw=value, kind="percent", value=parsed)
        parsed = _parse_epsilon_float(text, value)
        _validate_epsilon_number(parsed, value)
        return ScoreEpsilon(raw=value, kind="absolute", value=parsed)
    raise TmlError(f"Invalid score.epsilon: {value!r}")


def epsilon_delta(reference_score: float, epsilon: ScoreEpsilon) -> float:
    if epsilon.kind == "absolute":
        return epsilon.value
    if epsilon.kind == "percent":
        return abs(reference_score) * (epsilon.value / 100.0)
    raise TmlError(f"Invalid score epsilon kind: {epsilon.kind}")


def score_improves(candidate_score: float, reference_score: float, epsilon: ScoreEpsilon, *, maximize: bool = True) -> bool:
    delta = epsilon_delta(reference_score, epsilon)
    if maximize:
        return candidate_score >= reference_score + delta
    return candidate_score <= reference_score - delta


def _parse_epsilon_float(text: str, raw: object) -> float:
    if not text:
        raise TmlError(f"Invalid score.epsilon: {raw!r}")
    try:
        return float(text)
    except ValueError as exc:
        raise TmlError(f"Invalid score.epsilon: {raw!r}") from exc


def _validate_epsilon_number(value: float, raw: object) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise TmlError(f"Invalid score.epsilon: {raw!r}")


def _ordered_candidates(
    candidates: list[dict[str, Any]],
    *,
    preferred_parent_ref: str | None,
    require_preferred_parent: bool = False,
) -> list[dict[str, Any]]:
    if not preferred_parent_ref:
        return candidates
    preferred = [candidate for candidate in candidates if str(candidate.get("parent_ref") or "") == preferred_parent_ref]
    if require_preferred_parent:
        return preferred
    if not preferred:
        return candidates
    rest = [candidate for candidate in candidates if str(candidate.get("parent_ref") or "") != preferred_parent_ref]
    return preferred + rest


def _node_kinds(value: object, *, default: list[str]) -> list[str]:
    raw_items = value if isinstance(value, list) else default
    kinds: list[str] = []
    for item in raw_items:
        kind = str(item).strip().lower()
        if kind not in {"root", "branch"}:
            raise TmlError(f"Invalid branch algorithm node kind: {item}")
        if kind not in kinds:
            kinds.append(kind)
    if not kinds:
        raise TmlError("Branch algorithm node kind list cannot be empty.")
    return kinds


def _count_skip(skip_reasons: dict[str, int], reason: str) -> None:
    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1


def _optional_float(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None
