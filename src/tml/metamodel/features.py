from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from tml.db.connect import connect
from tml.db.state import ensure_project_db
from tml.metamodel.importer import dataset_fingerprint
from tml.metamodel.records import MetaComponent, MetaRecord


TARGET_COLUMNS = {"cv_score", "public_score", "public_gap", "private_score"}
IDENTIFIER_COLUMNS = {
    "node_id",
    "kind",
    "run_id",
    "step",
    "hypothesis_id",
    "branch_id",
    "parent_kind",
    "parent_id",
    "grandparent_kind",
    "grandparent_id",
    "root_id",
    "status",
    "created_at",
    "finished_at",
    "run_seconds",
    "materialization_file",
    "code_hash",
    "metric_name",
    "submit_status",
    "active_groups",
    "active_group_versions",
    "active_group_revisions",
    "components_json",
}


@dataclass(frozen=True)
class FeatureBuildResult:
    feature_csv: Path
    feature_spec_json: Path
    row_count: int
    feature_count: int
    feature_columns: list[str]
    leakage_policy: str


def build_feature_frame(records: list[MetaRecord]):
    pd = _require_pandas()
    rows = _build_feature_rows(records, candidate_records=[], include_record_rows=True)
    return pd.DataFrame(rows)


def build_candidate_frames(
    project_dir: Path,
    records: list[MetaRecord],
    *,
    candidates: list[tuple[list[str], str | None]],
    mode: str,
    profile_id: str | None,
    feature_columns: list[str],
):
    pd = _require_pandas()
    candidate_records = [
        _candidate_record(
            project_dir,
            records,
            groups=groups,
            parent=parent,
            mode=mode,
            profile_id=profile_id,
            node_id=f"candidate-{index:06d}",
        )
        for index, (groups, parent) in enumerate(candidates)
    ]
    rows = _build_feature_rows(records, candidate_records=candidate_records, include_record_rows=False)
    frame = pd.DataFrame(rows)
    for column in feature_columns:
        if column not in frame.columns:
            frame[column] = None
    return pd.DataFrame(frame[feature_columns])


def _build_feature_rows(
    records: list[MetaRecord],
    *,
    candidate_records: list[MetaRecord],
    include_record_rows: bool,
) -> list[dict[str, Any]]:
    ordered = sorted(records, key=_record_sort_key)
    all_records = [*ordered, *candidate_records]
    all_group_keys = sorted({component.logical_key for record in all_records for component in record.components})
    rows: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    prior_scores_by_group: dict[str, list[float]] = {}
    prior_effects_by_group: dict[str, list[float]] = {}
    best_score_by_entity: dict[str, float] = {}
    group_to_column = {key: f"group__{_safe_feature_name(key)}" for key in all_group_keys}
    revision_to_column = {key: f"revision__{_safe_feature_name(key)}" for key in all_group_keys}
    components_by_entity = _components_by_entity(ordered)

    for record in ordered:
        row = _feature_row(
            record,
            all_group_keys=all_group_keys,
            group_to_column=group_to_column,
            revision_to_column=revision_to_column,
            components_by_entity=components_by_entity,
            seen_groups=seen_groups,
            prior_scores_by_group=prior_scores_by_group,
            prior_effects_by_group=prior_effects_by_group,
            best_score_by_entity=best_score_by_entity,
        )
        if include_record_rows:
            rows.append(row)
        _update_feature_state(record, row, seen_groups, prior_scores_by_group, prior_effects_by_group, best_score_by_entity)

    for record in candidate_records:
        rows.append(
            _feature_row(
                record,
                all_group_keys=all_group_keys,
                group_to_column=group_to_column,
                revision_to_column=revision_to_column,
                components_by_entity=components_by_entity,
                seen_groups=seen_groups,
                prior_scores_by_group=prior_scores_by_group,
                prior_effects_by_group=prior_effects_by_group,
                best_score_by_entity=best_score_by_entity,
            )
        )

    return rows


def _feature_row(
    record: MetaRecord,
    *,
    all_group_keys: list[str],
    group_to_column: dict[str, str],
    revision_to_column: dict[str, str],
    components_by_entity: dict[str, list[MetaComponent]],
    seen_groups: set[str],
    prior_scores_by_group: dict[str, list[float]],
    prior_effects_by_group: dict[str, list[float]],
    best_score_by_entity: dict[str, float],
) -> dict[str, Any]:
    group_keys = [component.logical_key for component in record.components]
    version_keys = [component.version_key for component in record.components]
    group_set = set(group_keys)
    unseen = sorted(group_set - seen_groups)
    parent_score = best_score_by_entity.get(record.parent_key or "")
    grandparent_score = best_score_by_entity.get(record.grandparent_key or "")
    parent_delta = (
        parent_score - grandparent_score
        if parent_score is not None and grandparent_score is not None
        else None
    )
    parent_components = components_by_entity.get(record.parent_key or "", [])
    parent_logic = {component.logical_key: component for component in parent_components}
    current_logic = {component.logical_key: component for component in record.components}
    new_groups = [key for key in current_logic if key not in parent_logic]
    changed_groups = [
        key
        for key, component in current_logic.items()
        if key in parent_logic and component.version_key != parent_logic[key].version_key
    ]
    removed_groups = [key for key in parent_logic if key not in current_logic]
    historical_scores = [score for key in group_set for score in prior_scores_by_group.get(key, [])]
    historical_effects = [effect for key in group_set for effect in prior_effects_by_group.get(key, [])]
    historical_counts = [len(prior_scores_by_group.get(key, [])) for key in group_set]

    row: dict[str, Any] = {
        "node_id": record.node_id,
        "kind": record.kind,
        "run_id": record.run_id,
        "step": record.step,
        "mode": record.mode,
        "profile_id": record.profile_id,
        "profile_hash": record.profile_hash,
        "status": record.status,
        "created_at": record.created_at,
        "finished_at": record.finished_at,
        "run_seconds": record.run_seconds,
        "hypothesis_id": record.hypothesis_id,
        "branch_id": record.branch_id,
        "parent_kind": record.parent_kind,
        "parent_id": record.parent_id,
        "grandparent_kind": record.grandparent_kind,
        "grandparent_id": record.grandparent_id,
        "root_id": record.root_id,
        "graph_depth": record.graph_depth,
        "materialization_file": record.materialization_file,
        "code_hash": record.code_hash,
        "cv_score": record.cv_score,
        "public_score": record.public_score,
        "public_gap": record.public_score - record.cv_score if record.public_score is not None and record.cv_score is not None else None,
        "metric_name": record.metric_name,
        "submit_status": record.submit_status,
        "dataset_fingerprint": record.dataset_fingerprint,
        "profile_seed": record.profile_seed,
        "profile_time_limit": record.profile_time_limit,
        "profile_presets": record.profile_presets,
        "profile_validation_strategy": record.profile_validation_strategy,
        "profile_class_balance": record.profile_class_balance,
        "profile_use_gpu": record.profile_use_gpu,
        "profile_fold_config": record.profile_fold_config,
        "n_active_groups": len(group_set),
        "n_component_versions": len(set(version_keys)),
        "n_new_groups_vs_parent": len(new_groups) if parent_components else None,
        "n_changed_groups_vs_parent": len(changed_groups) if parent_components else None,
        "n_removed_groups_vs_parent": len(removed_groups) if parent_components else None,
        "parent_cv_score": parent_score,
        "parent_delta_vs_grandparent": parent_delta,
        "has_unseen_group": int(bool(unseen)),
        "n_unseen_groups": len(unseen),
        "known_group_coverage": 1.0 - (len(unseen) / len(group_set)) if group_set else None,
        "hist_group_mean_cv": mean(historical_scores) if historical_scores else None,
        "hist_group_max_cv": max(historical_scores) if historical_scores else None,
        "hist_group_use_count_sum": sum(historical_counts) if historical_counts else 0,
        "hist_group_use_count_mean": mean(historical_counts) if historical_counts else 0,
        "hist_group_effect_mean": mean(historical_effects) if historical_effects else None,
        "component_families": ";".join(sorted({component.family or "" for component in record.components if component.family})),
        "component_group_names": ";".join(sorted({component.group_name or "" for component in record.components if component.group_name})),
        "active_groups": ";".join(group_keys),
        "active_group_versions": ";".join(version_keys),
    }
    revision_by_group = {component.logical_key: component.revision for component in record.components}
    for group_key in all_group_keys:
        row[group_to_column[group_key]] = int(group_key in group_set)
        row[revision_to_column[group_key]] = revision_by_group.get(group_key) or 0
    return row


def _update_feature_state(
    record: MetaRecord,
    row: dict[str, Any],
    seen_groups: set[str],
    prior_scores_by_group: dict[str, list[float]],
    prior_effects_by_group: dict[str, list[float]],
    best_score_by_entity: dict[str, float],
) -> None:
    group_set = {component.logical_key for component in record.components}
    parent_score = row.get("parent_cv_score")
    if record.cv_score is not None:
        for group_key in group_set:
            prior_scores_by_group.setdefault(group_key, []).append(record.cv_score)
        if isinstance(parent_score, int | float):
            effect = record.cv_score - float(parent_score)
            for group_key in group_set:
                prior_effects_by_group.setdefault(group_key, []).append(effect)
        previous = best_score_by_entity.get(record.entity_key)
        if previous is None or record.cv_score > previous:
            best_score_by_entity[record.entity_key] = record.cv_score
    seen_groups.update(group_set)


def write_feature_artifacts(records: list[MetaRecord], output_dir: Path) -> tuple[Any, FeatureBuildResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = build_feature_frame(records)
    feature_csv = output_dir / "meta_features.csv"
    frame.to_csv(feature_csv, index=False)
    feature_columns = selectable_feature_columns(frame)
    spec = {
        "schema_version": 1,
        "kind": "tml_meta_model_features",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "feature_columns": feature_columns,
        "target_columns": sorted(TARGET_COLUMNS),
        "identifier_columns": sorted(IDENTIFIER_COLUMNS),
        "leakage_policy": (
            "Targets are excluded from feature_columns. Parent and group-history features "
            "are computed only from earlier records sorted by created_at, step, and node_id."
        ),
    }
    feature_spec_json = output_dir / "feature_spec.json"
    feature_spec_json.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return frame, FeatureBuildResult(
        feature_csv=feature_csv,
        feature_spec_json=feature_spec_json,
        row_count=len(frame),
        feature_count=len(feature_columns),
        feature_columns=feature_columns,
        leakage_policy=str(spec["leakage_policy"]),
    )


def selectable_feature_columns(frame) -> list[str]:
    excluded = TARGET_COLUMNS | IDENTIFIER_COLUMNS | {"private_score"}
    return [column for column in frame.columns if column not in excluded]


def build_candidate_frame(
    project_dir: Path,
    records: list[MetaRecord],
    *,
    groups: list[str],
    parent: str | None,
    mode: str,
    profile_id: str | None,
    feature_columns: list[str],
):
    pd = _require_pandas()
    candidate = _candidate_record(
        project_dir,
        records,
        groups=groups,
        parent=parent,
        mode=mode,
        profile_id=profile_id,
        node_id="candidate",
    )
    full = build_feature_frame([*records, candidate])
    row = full[full["node_id"] == "candidate"].copy()
    for column in feature_columns:
        if column not in row.columns:
            row[column] = None
    return pd.DataFrame(row[feature_columns])


def _candidate_record(
    project_dir: Path,
    records: list[MetaRecord],
    *,
    groups: list[str],
    parent: str | None,
    mode: str,
    profile_id: str | None,
    node_id: str,
) -> MetaRecord:
    components = _candidate_components(project_dir, groups, mode=mode)
    parent_kind, parent_id = _candidate_parent(parent)
    root_id, graph_depth, grandparent_kind, grandparent_id = _candidate_graph_context(records, parent_kind, parent_id)
    profile_id = profile_id or _latest_profile_id(records, mode)
    return MetaRecord(
        node_id=node_id,
        kind="candidate",
        run_id="",
        step=None,
        mode=mode,
        profile_id=profile_id or "",
        profile_hash=None,
        status="candidate",
        created_at=datetime.now().isoformat(timespec="seconds"),
        finished_at=None,
        run_seconds=None,
        hypothesis_id=None,
        branch_id=None,
        parent_kind=parent_kind,
        parent_id=parent_id,
        grandparent_kind=grandparent_kind,
        grandparent_id=grandparent_id,
        root_id=root_id,
        graph_depth=graph_depth,
        materialization_file=None,
        code_hash=None,
        cv_score=None,
        public_score=None,
        private_score=None,
        metric_name=None,
        submit_status=None,
        dataset_fingerprint=dataset_fingerprint(project_dir),
        components=components,
    )


def _candidate_components(project_dir: Path, groups: list[str], *, mode: str) -> list[MetaComponent]:
    db_path = ensure_project_db(project_dir)
    components: list[MetaComponent] = []
    with connect(db_path) as conn:
        for token in groups:
            parsed = _parse_group_token(token)
            hypothesis_id = parsed["source_id"]
            revision = parsed["revision"]
            file_name = parsed["file"]
            if file_name:
                row = conn.execute(
                    """
                    SELECT hypothesis_id, mode, file, code_hash, hypothesis_revision
                    FROM materializations
                    WHERE hypothesis_id=? AND mode=? AND file=?
                    ORDER BY active DESC, file DESC
                    LIMIT 1
                    """,
                    (hypothesis_id, mode, file_name),
                ).fetchone()
            elif revision is not None:
                row = conn.execute(
                    """
                    SELECT hypothesis_id, mode, file, code_hash, hypothesis_revision
                    FROM materializations
                    WHERE hypothesis_id=? AND mode=? AND hypothesis_revision=?
                    ORDER BY active DESC, file DESC
                    LIMIT 1
                    """,
                    (hypothesis_id, mode, revision),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT hypothesis_id, mode, file, code_hash, hypothesis_revision
                    FROM materializations
                    WHERE hypothesis_id=? AND mode=?
                    ORDER BY active DESC, file DESC
                    LIMIT 1
                    """,
                    (hypothesis_id, mode),
                ).fetchone()
            if row is None:
                components.append(
                    MetaComponent(
                        role="candidate",
                        source_type="hypothesis",
                        source_id=hypothesis_id,
                        mode=mode,
                        file=file_name or "",
                        code_hash="",
                        path="",
                        revision=revision,
                    )
                )
                continue
            components.append(
                MetaComponent(
                    role="candidate",
                    source_type="hypothesis",
                    source_id=str(row["hypothesis_id"]),
                    mode=str(row["mode"]),
                    file=str(row["file"]),
                    code_hash=str(row["code_hash"]),
                    path=f"hypotheses/{row['hypothesis_id']}/materializations/{row['file']}",
                    revision=int(row["hypothesis_revision"]) if row["hypothesis_revision"] else revision,
                )
            )
    return components


def _parse_group_token(token: str) -> dict[str, Any]:
    text = token.strip()
    file_name = None
    revision = None
    if ":" in text:
        parts = text.split(":")
        text = parts[0]
        if len(parts) >= 2 and parts[1]:
            revision = int(parts[1])
        if len(parts) >= 3 and parts[2]:
            file_name = parts[2]
    elif "@" in text:
        text, revision_text = text.split("@", 1)
        revision = int(revision_text)
    source_id = text.zfill(6) if text.isdigit() else text
    return {"source_id": source_id, "revision": revision, "file": file_name}


def _candidate_parent(parent: str | None) -> tuple[str | None, str | None]:
    if not parent:
        return None, None
    text = parent.strip()
    if text.upper().startswith("B"):
        return "branch", f"B{int(text[1:]):06d}" if text[1:].isdigit() else text
    return "hypothesis", text.zfill(6) if text.isdigit() else text


def _candidate_graph_context(
    records: list[MetaRecord],
    parent_kind: str | None,
    parent_id: str | None,
) -> tuple[str | None, int | None, str | None, str | None]:
    if not parent_kind or not parent_id:
        return None, 0, None, None
    if parent_kind != "branch":
        return parent_id, 1, None, None
    parent_record = None
    parent_key = f"branch:{parent_id}"
    for record in records:
        if record.entity_key == parent_key:
            parent_record = record
    if parent_record is None:
        return None, None, None, None
    depth = parent_record.graph_depth + 1 if parent_record.graph_depth is not None else None
    return parent_record.root_id, depth, parent_record.parent_kind, parent_record.parent_id


def _latest_profile_id(records: list[MetaRecord], mode: str) -> str | None:
    matching = [record.profile_id for record in records if record.mode == mode and record.profile_id]
    return matching[-1] if matching else None


def _components_by_entity(records: list[MetaRecord]) -> dict[str, list[MetaComponent]]:
    by_entity: dict[str, list[MetaComponent]] = {}
    for record in records:
        by_entity[record.entity_key] = record.components
    return by_entity


def _record_sort_key(record: MetaRecord) -> tuple[str, int, str]:
    return (record.created_at or "", record.step or 0, record.node_id)


def _safe_feature_name(value: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_]+", "_", value)
    return safe.strip("_").lower() or "unknown"


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Meta-model feature building requires pandas. Install project requirements with uv pip.") from exc
    return pd
