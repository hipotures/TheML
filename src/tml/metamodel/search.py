from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import random
from pathlib import Path
from typing import Any

from tml.db.connect import connect
from tml.db.state import ensure_project_db
from tml.metamodel.features import build_candidate_frames
from tml.metamodel.importer import load_records_from_csv
from tml.metamodel.records import MetaRecord
from tml.metamodel.training import predict_with_uncertainty


@dataclass(frozen=True)
class CandidateSearchConfig:
    target: str = "cv_score"
    top: int = 25
    candidates: int = 2000
    beam: int = 50
    depth: int = 3
    exploration: float = 0.10
    seed: int = 42
    max_groups: int | None = None
    include_tested: bool = False
    buildable: bool = True
    mode: str = "autogluon"


@dataclass(frozen=True)
class CandidateSuggestion:
    rank: int
    groups: list[str]
    prediction: float
    split_prediction_std: float | None
    split_prediction_p10: float | None
    split_prediction_p90: float | None
    group_count: int
    added_groups: list[str]
    removed_groups: list[str]
    anchor_ref: str
    anchor_score: float | None
    parent_ref: str | None
    nearest_ref: str | None
    nearest_jaccard: float
    already_tested: bool
    branch_add_commands: list[str]
    generation_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "groups": self.groups,
            "prediction": self.prediction,
            "split_prediction_std": self.split_prediction_std,
            "split_prediction_p10": self.split_prediction_p10,
            "split_prediction_p90": self.split_prediction_p90,
            "group_count": self.group_count,
            "added_groups": self.added_groups,
            "removed_groups": self.removed_groups,
            "anchor_ref": self.anchor_ref,
            "anchor_score": self.anchor_score,
            "parent_ref": self.parent_ref,
            "nearest_ref": self.nearest_ref,
            "nearest_jaccard": self.nearest_jaccard,
            "already_tested": self.already_tested,
            "branch_add_commands": self.branch_add_commands,
            "generation_reason": self.generation_reason,
        }


@dataclass(frozen=True)
class CandidateSearchResult:
    output_dir: Path
    target: str
    config: CandidateSearchConfig
    generated_count: int
    scored_count: int
    suggestions: list[CandidateSuggestion]
    json_path: Path
    csv_path: Path
    markdown_path: Path
    low_confidence: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "tml_meta_candidate_search",
            "target": self.target,
            "config": self.config.__dict__,
            "generated_count": self.generated_count,
            "scored_count": self.scored_count,
            "low_confidence": self.low_confidence,
            "output_dir": str(self.output_dir),
            "json_path": str(self.json_path),
            "csv_path": str(self.csv_path),
            "markdown_path": str(self.markdown_path),
            "suggestions": [suggestion.to_dict() for suggestion in self.suggestions],
        }


@dataclass(frozen=True)
class _Anchor:
    groups: frozenset[str]
    ref: str
    score: float | None
    reason: str


@dataclass(frozen=True)
class _RawCandidate:
    groups: frozenset[str]
    anchor: _Anchor
    reason: str
    heuristic: float


@dataclass(frozen=True)
class _ParentPlan:
    parent_ref: str | None
    parent_groups: frozenset[str]
    missing_groups: list[str]
    commands: list[str]


def generate_candidate_suggestions(
    project_dir: Path,
    run_dir: Path,
    *,
    config: CandidateSearchConfig,
    output_dir: Path | None = None,
) -> CandidateSearchResult:
    _validate_config(config)
    if config.target != "cv_score":
        raise ValueError("Candidate search v1 ranks cv_score only. Use target=cv_score.")
    records = load_records_from_csv(run_dir / "dataset" / "meta_dataset.csv")
    feature_spec = json.loads((run_dir / "dataset" / "feature_spec.json").read_text(encoding="utf-8"))
    feature_columns = [str(column) for column in feature_spec["feature_columns"]]
    model_dir = run_dir / "targets" / config.target / "final" / "AutoGluonModels"
    if not model_dir.exists():
        raise ValueError(f"Meta-model target not found: {model_dir}")

    target_records = [record for record in records if _score(record, config.target) is not None]
    if not target_records:
        raise ValueError(f"No labeled records for target={config.target}.")

    mode = config.mode or _latest_mode(records)
    universe = _hypothesis_universe(project_dir, records, mode=mode)
    tested_sets = {_record_groups(record) for record in records if _record_groups(record)}
    max_groups = config.max_groups or min(25, max((len(groups) for groups in tested_sets), default=1) + 2)
    priors = _historical_priors(records, target=config.target)
    anchors = _anchors(records, target=config.target)
    raw_candidates = _generate_raw_candidates(
        anchors=anchors,
        universe=universe,
        priors=priors,
        tested_sets=tested_sets,
        max_groups=max_groups,
        config=config,
    )

    planned: list[tuple[_RawCandidate, _ParentPlan]] = []
    for candidate in raw_candidates:
        plan = _parent_plan(records, candidate.groups, priors=priors, mode=mode)
        if config.buildable and plan.parent_ref is None:
            continue
        planned.append((candidate, plan))
        if len(planned) >= config.candidates:
            break
    if not planned:
        raise ValueError("No buildable candidate suggestions generated.")

    feature_frame = build_candidate_frames(
        project_dir,
        records,
        candidates=[(sorted(candidate.groups), plan.parent_ref) for candidate, plan in planned],
        mode=mode,
        profile_id=None,
        feature_columns=feature_columns,
    )
    prediction = predict_with_uncertainty(model_dir, feature_frame)
    final_predictions = [float(value) for value in prediction["prediction"]]
    split_std = _optional_float_list(prediction.get("split_prediction_std"))
    split_p10 = _optional_float_list(prediction.get("split_prediction_p10"))
    split_p90 = _optional_float_list(prediction.get("split_prediction_p90"))

    rows: list[CandidateSuggestion] = []
    for index, ((candidate, plan), pred) in enumerate(zip(planned, final_predictions, strict=False)):
        nearest_ref, nearest_jaccard = _nearest_known(records, candidate.groups)
        already_tested = candidate.groups in tested_sets
        rows.append(
            CandidateSuggestion(
                rank=0,
                groups=sorted(candidate.groups),
                prediction=pred,
                split_prediction_std=_at(split_std, index),
                split_prediction_p10=_at(split_p10, index),
                split_prediction_p90=_at(split_p90, index),
                group_count=len(candidate.groups),
                added_groups=sorted(candidate.groups - candidate.anchor.groups),
                removed_groups=sorted(candidate.anchor.groups - candidate.groups),
                anchor_ref=candidate.anchor.ref,
                anchor_score=candidate.anchor.score,
                parent_ref=plan.parent_ref,
                nearest_ref=nearest_ref,
                nearest_jaccard=nearest_jaccard,
                already_tested=already_tested,
                branch_add_commands=plan.commands,
                generation_reason=candidate.reason,
            )
        )
    ranked = sorted(
        rows,
        key=lambda item: (
            -item.prediction,
            item.split_prediction_std if item.split_prediction_std is not None else 999.0,
            item.nearest_jaccard,
            item.group_count,
        ),
    )
    suggestions = [
        CandidateSuggestion(**{**item.to_dict(), "rank": rank})
        for rank, item in enumerate(ranked[: config.top], start=1)
    ]
    output_dir = output_dir or run_dir / "candidate_searches" / f"{datetime.now().strftime('%Y%m%dT%H%M%S')}-{config.target}"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "candidate_search.json"
    csv_path = output_dir / "candidates.csv"
    markdown_path = output_dir / "summary.md"
    result = CandidateSearchResult(
        output_dir=output_dir,
        target=config.target,
        config=config,
        generated_count=len(raw_candidates),
        scored_count=len(planned),
        suggestions=suggestions,
        json_path=json_path,
        csv_path=csv_path,
        markdown_path=markdown_path,
        low_confidence=False,
    )
    _write_artifacts(result)
    return result


def _validate_config(config: CandidateSearchConfig) -> None:
    if config.top <= 0:
        raise ValueError("top must be positive.")
    if config.candidates <= 0:
        raise ValueError("candidates must be positive.")
    if config.beam <= 0:
        raise ValueError("beam must be positive.")
    if config.depth <= 0:
        raise ValueError("depth must be positive.")
    if not 0.0 <= config.exploration <= 1.0:
        raise ValueError("exploration must be between 0 and 1.")
    if config.max_groups is not None and config.max_groups <= 0:
        raise ValueError("max_groups must be positive.")


def _generate_raw_candidates(
    *,
    anchors: list[_Anchor],
    universe: set[str],
    priors: dict[str, dict[str, float]],
    tested_sets: set[frozenset[str]],
    max_groups: int,
    config: CandidateSearchConfig,
) -> list[_RawCandidate]:
    candidates: dict[frozenset[str], _RawCandidate] = {}
    deterministic_target = max(1, int(config.candidates * max(0.0, min(1.0, 1.0 - config.exploration))))
    frontier = anchors[: max(1, config.beam)]
    positive = _ordered_positive_groups(universe, priors)
    negative = _ordered_negative_groups(universe, priors)

    for depth_index in range(max(1, config.depth)):
        expanded: list[_RawCandidate] = []
        for anchor in frontier:
            for candidate in _mutations(anchor, universe, positive, negative, priors=priors, max_groups=max_groups):
                if _accept_candidate(candidate.groups, tested_sets, include_tested=config.include_tested):
                    _remember(candidates, candidate)
                    expanded.append(candidate)
                if len(candidates) >= deterministic_target:
                    break
            if len(candidates) >= deterministic_target:
                break
        if len(candidates) >= deterministic_target:
            break
        frontier = [
            _Anchor(candidate.groups, candidate.anchor.ref, candidate.anchor.score, candidate.reason)
            for candidate in sorted(expanded, key=lambda item: item.heuristic, reverse=True)[: config.beam]
        ]
        if not frontier:
            break

    rng = random.Random(config.seed)
    exploration_target = max(0, config.candidates - len(candidates))
    attempts = 0
    while len(candidates) < config.candidates and attempts < max(100, exploration_target * 25):
        attempts += 1
        anchor = _weighted_anchor(anchors, rng)
        groups = set(anchor.groups)
        for _ in range(rng.randint(1, max(1, config.depth))):
            groups = _random_mutation(groups, universe, positive, negative, rng, max_groups=max_groups)
        frozen = frozenset(groups)
        if not _accept_candidate(frozen, tested_sets, include_tested=config.include_tested):
            continue
        candidate = _RawCandidate(
            groups=frozen,
            anchor=anchor,
            reason="stochastic_exploration",
            heuristic=_heuristic(frozen, priors),
        )
        _remember(candidates, candidate)

    return sorted(candidates.values(), key=lambda item: item.heuristic, reverse=True)


def _mutations(
    anchor: _Anchor,
    universe: set[str],
    positive: list[str],
    negative: list[str],
    *,
    priors: dict[str, dict[str, float]],
    max_groups: int,
) -> list[_RawCandidate]:
    rows: list[_RawCandidate] = []
    current = set(anchor.groups)
    absent_positive = [group for group in positive if group not in current][:20]
    present_negative = [group for group in negative if group in current][:10]
    for group in absent_positive:
        groups = frozenset([*current, group])
        if len(groups) <= max_groups:
            rows.append(_RawCandidate(groups, anchor, f"add:{group}", _heuristic(groups, priors)))
    if len(current) > 1:
        for group in present_negative:
            groups = frozenset(current - {group})
            rows.append(_RawCandidate(groups, anchor, f"remove:{group}", _heuristic(groups, priors)))
    for remove_group in present_negative:
        for add_group in absent_positive[:10]:
            groups = frozenset((current - {remove_group}) | {add_group})
            if groups and len(groups) <= max_groups and groups <= universe:
                rows.append(_RawCandidate(groups, anchor, f"swap:{remove_group}->{add_group}", _heuristic(groups, priors)))
    return rows


def _remember(candidates: dict[frozenset[str], _RawCandidate], candidate: _RawCandidate) -> None:
    existing = candidates.get(candidate.groups)
    if existing is None or candidate.heuristic > existing.heuristic:
        candidates[candidate.groups] = candidate


def _accept_candidate(groups: frozenset[str], tested_sets: set[frozenset[str]], *, include_tested: bool) -> bool:
    if not groups:
        return False
    if not include_tested and groups in tested_sets:
        return False
    return True


def _historical_priors(records: list[MetaRecord], *, target: str) -> dict[str, dict[str, float]]:
    scored = [record for record in records if _score(record, target) is not None]
    all_groups = sorted({group for record in records for group in _record_groups(record)})
    priors: dict[str, dict[str, float]] = {}
    for group in all_groups:
        active_scores = [_score(record, target) for record in scored if group in _record_groups(record)]
        inactive_scores = [_score(record, target) for record in scored if group not in _record_groups(record)]
        active_values = [float(value) for value in active_scores if value is not None]
        inactive_values = [float(value) for value in inactive_scores if value is not None]
        active_mean = sum(active_values) / len(active_values) if active_values else 0.0
        inactive_mean = sum(inactive_values) / len(inactive_values) if inactive_values else active_mean
        priors[group] = {
            "active_n": float(len(active_values)),
            "active_mean": active_mean,
            "active_mean_diff": active_mean - inactive_mean,
            "add_n": 0.0,
            "mean_delta": 0.0,
            "median_delta": 0.0,
            "win_rate": 0.0,
        }

    entity_groups = {record.entity_key: _record_groups(record) for record in records}
    entity_scores = {record.entity_key: _score(record, target) for record in records if _score(record, target) is not None}
    deltas: dict[str, list[float]] = {}
    for record in records:
        score = _score(record, target)
        if score is None or record.parent_key is None:
            continue
        parent_score = entity_scores.get(record.parent_key)
        if parent_score is None:
            continue
        added = _record_groups(record) - entity_groups.get(record.parent_key, frozenset())
        for group in added:
            deltas.setdefault(group, []).append(float(score - parent_score))
    for group, values in deltas.items():
        ordered = sorted(values)
        mid = len(ordered) // 2
        median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
        current = priors.setdefault(group, {})
        current.update(
            {
                "add_n": float(len(values)),
                "mean_delta": float(sum(values) / len(values)),
                "median_delta": float(median),
                "win_rate": float(sum(1 for value in values if value > 0) / len(values)),
            }
        )
    return priors


def _anchors(records: list[MetaRecord], *, target: str) -> list[_Anchor]:
    scored = [record for record in records if _score(record, target) is not None and _record_groups(record)]
    anchors: dict[frozenset[str], _Anchor] = {}
    by_score = sorted(scored, key=lambda record: float(_score(record, target) or 0.0), reverse=True)
    for record in by_score[:25]:
        _add_anchor(anchors, record, target, "top_score")
    entity_scores = {record.entity_key: _score(record, target) for record in records if _score(record, target) is not None}
    deltas = [
        (record, float(_score(record, target) or 0.0) - float(entity_scores.get(record.parent_key, 0.0)))
        for record in scored
        if record.parent_key in entity_scores
    ]
    for record, _ in sorted(deltas, key=lambda item: item[1], reverse=True)[:25]:
        _add_anchor(anchors, record, target, "top_delta")
    singles = [record for record in scored if len(_record_groups(record)) == 1]
    for record in sorted(singles, key=lambda item: float(_score(item, target) or 0.0), reverse=True)[:10]:
        _add_anchor(anchors, record, target, "single_root")
    return sorted(anchors.values(), key=lambda item: item.score or 0.0, reverse=True)


def _add_anchor(anchors: dict[frozenset[str], _Anchor], record: MetaRecord, target: str, reason: str) -> None:
    groups = _record_groups(record)
    if not groups:
        return
    score = _score(record, target)
    ref = record.branch_id or record.hypothesis_id or record.node_id
    existing = anchors.get(groups)
    if existing is None or (score or 0.0) > (existing.score or 0.0):
        anchors[groups] = _Anchor(groups=groups, ref=str(ref), score=score, reason=reason)


def _parent_plan(
    records: list[MetaRecord],
    groups: frozenset[str],
    *,
    priors: dict[str, dict[str, float]],
    mode: str,
) -> _ParentPlan:
    candidates: list[tuple[int, float, str, frozenset[str]]] = []
    for record in records:
        record_groups = _record_groups(record)
        if not record_groups or not record_groups <= groups:
            continue
        ref = record.branch_id or record.hypothesis_id
        if not ref:
            continue
        score = record.cv_score if record.cv_score is not None else -1.0
        candidates.append((len(record_groups), float(score), str(ref), record_groups))
    if not candidates:
        return _ParentPlan(None, frozenset(), sorted(groups), [])
    _, _, parent_ref, parent_groups = sorted(candidates, key=lambda item: (item[0], item[1], item[2]), reverse=True)[0]
    missing = sorted(groups - parent_groups, key=lambda group: _prior_value(group, priors), reverse=True)
    commands = []
    current_parent = parent_ref
    for group in missing:
        commands.append(f"uv run tml branch add parent={current_parent} source={group} mode={mode}")
        current_parent = "<BRANCH_FROM_PREVIOUS_STEP>"
    return _ParentPlan(parent_ref, parent_groups, missing, commands)


def _nearest_known(records: list[MetaRecord], groups: frozenset[str]) -> tuple[str | None, float]:
    best_ref = None
    best_score = -1.0
    for record in records:
        other = _record_groups(record)
        if not other:
            continue
        union = groups | other
        score = len(groups & other) / len(union) if union else 0.0
        if score > best_score:
            best_ref = record.branch_id or record.hypothesis_id or record.node_id
            best_score = score
    return (str(best_ref) if best_ref else None, max(0.0, best_score))


def _hypothesis_universe(project_dir: Path, records: list[MetaRecord], *, mode: str) -> set[str]:
    universe = {group for record in records for group in _record_groups(record)}
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT hypothesis_id
            FROM materializations
            WHERE mode=? AND active=1 AND status='active'
            """,
            (mode,),
        ).fetchall()
    universe.update(str(row["hypothesis_id"]).zfill(6) for row in rows)
    universe.discard("000000")
    return universe


def _record_groups(record: MetaRecord) -> frozenset[str]:
    return frozenset(
        component.source_id.zfill(6)
        for component in record.components
        if component.source_type == "hypothesis" and component.source_id
    )


def _score(record: MetaRecord, target: str) -> float | None:
    if target == "cv_score":
        return record.cv_score
    if target == "public_score":
        return record.public_score
    if target == "public_gap":
        return record.public_score - record.cv_score if record.public_score is not None and record.cv_score is not None else None
    return None


def _ordered_positive_groups(universe: set[str], priors: dict[str, dict[str, float]]) -> list[str]:
    return sorted(universe, key=lambda group: _prior_value(group, priors), reverse=True)


def _ordered_negative_groups(universe: set[str], priors: dict[str, dict[str, float]]) -> list[str]:
    return sorted(universe, key=lambda group: _prior_value(group, priors))


def _prior_value(group: str, priors: dict[str, dict[str, float]]) -> float:
    item = priors.get(group, {})
    add_n = float(item.get("add_n") or 0.0)
    shrink = add_n / (add_n + 3.0) if add_n else 0.0
    return float(item.get("mean_delta") or 0.0) * shrink + float(item.get("active_mean_diff") or 0.0) * 0.25


def _heuristic(groups: frozenset[str], priors: dict[str, dict[str, float]]) -> float:
    return sum(_prior_value(group, priors) for group in groups) - len(groups) * 0.000001


def _weighted_anchor(anchors: list[_Anchor], rng: random.Random) -> _Anchor:
    if not anchors:
        raise ValueError("No anchors available.")
    scores = [max(0.0, (anchor.score or 0.0) - 0.963) + 0.0001 for anchor in anchors]
    total = sum(scores)
    pick = rng.random() * total
    running = 0.0
    for anchor, score in zip(anchors, scores, strict=False):
        running += score
        if running >= pick:
            return anchor
    return anchors[-1]


def _random_mutation(
    groups: set[str],
    universe: set[str],
    positive: list[str],
    negative: list[str],
    rng: random.Random,
    *,
    max_groups: int,
) -> set[str]:
    choices = ["add", "remove" if len(groups) > 1 else "add", "swap" if groups and len(groups) < max_groups else "add"]
    op = rng.choice(choices)
    result = set(groups)
    absent = [group for group in positive if group not in result]
    present_negative = [group for group in negative if group in result]
    if op == "add" and absent and len(result) < max_groups:
        result.add(rng.choice(absent[: max(1, min(20, len(absent)))]))
    elif op == "remove" and present_negative and len(result) > 1:
        result.remove(rng.choice(present_negative[: max(1, min(10, len(present_negative)))]))
    elif op == "swap" and absent and present_negative:
        result.remove(rng.choice(present_negative[: max(1, min(10, len(present_negative)))]))
        result.add(rng.choice(absent[: max(1, min(20, len(absent)))]))
    return result & universe


def _optional_float_list(value: object) -> list[float | None] | None:
    if not isinstance(value, list):
        return None
    return [float(item) if isinstance(item, int | float) else None for item in value]


def _at(values: list[float | None] | None, index: int) -> float | None:
    if values is None or index >= len(values):
        return None
    return values[index]


def _latest_mode(records: list[MetaRecord]) -> str:
    for record in reversed(records):
        if record.mode:
            return record.mode
    return "autogluon"


def _write_artifacts(result: CandidateSearchResult) -> None:
    pd = _require_pandas()
    payload = result.to_dict()
    result.json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = []
    for item in result.suggestions:
        rows.append(
            {
                **item.to_dict(),
                "groups": ",".join(item.groups),
                "added_groups": ",".join(item.added_groups),
                "removed_groups": ",".join(item.removed_groups),
                "branch_add_commands": "\n".join(item.branch_add_commands),
            }
        )
    pd.DataFrame(rows).to_csv(result.csv_path, index=False)
    lines = [
        "# Meta-Model Candidate Search",
        "",
        f"- Target: `{result.target}`",
        f"- Generated: {result.generated_count}",
        f"- Scored: {result.scored_count}",
        f"- Suggestions: {len(result.suggestions)}",
        f"- Build a candidate: `uv run tml meta build search={result.output_dir} rank=<rank>`",
        f"- Build and run a candidate: `uv run tml meta build search={result.output_dir} rank=<rank> run=true`",
        "",
        "## Top Candidates",
        "",
    ]
    for item in result.suggestions:
        lines.extend(
            [
                f"### {item.rank}. predicted {item.prediction:.6f}",
                "",
                f"- Groups: `{','.join(item.groups)}`",
                f"- Split std: {_format_optional(item.split_prediction_std)}",
                f"- Nearest known: `{item.nearest_ref}` Jaccard={item.nearest_jaccard:.3f}",
                f"- Anchor: `{item.anchor_ref}`",
                f"- Added: `{','.join(item.added_groups) or 'none'}`",
                f"- Removed: `{','.join(item.removed_groups) or 'none'}`",
                f"- Build: `uv run tml meta build search={result.output_dir} rank={item.rank}`",
                f"- Build and run: `uv run tml meta build search={result.output_dir} rank={item.rank} run=true`",
                "- Low-level branch-add steps:",
                "",
            ]
        )
        if item.branch_add_commands:
            lines.append("```bash")
            lines.extend(item.branch_add_commands)
            lines.append("```")
        else:
            lines.append("No branch-add commands; candidate already exists or has no missing groups.")
        lines.append("")
    result.markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Meta-model candidate search requires pandas. Install project requirements with uv pip.") from exc
    return pd
