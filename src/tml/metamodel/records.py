from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any


@dataclass(frozen=True)
class MetaComponent:
    role: str
    source_type: str
    source_id: str
    mode: str
    file: str
    code_hash: str
    path: str
    revision: int | None = None
    group_name: str | None = None
    family: str | None = None
    title: str | None = None

    @property
    def logical_key(self) -> str:
        return f"{self.source_type}:{self.source_id}"

    @property
    def version_key(self) -> str:
        parts = (self.source_type, self.source_id, self.mode, self.file, self.code_hash)
        return ":".join(str(part or "") for part in parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "mode": self.mode,
            "file": self.file,
            "code_hash": self.code_hash,
            "path": self.path,
            "revision": self.revision,
            "group_name": self.group_name,
            "family": self.family,
            "title": self.title,
            "logical_key": self.logical_key,
            "version_key": self.version_key,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MetaComponent":
        revision = payload.get("revision")
        return cls(
            role=str(payload.get("role") or ""),
            source_type=str(payload.get("source_type") or ""),
            source_id=str(payload.get("source_id") or ""),
            mode=str(payload.get("mode") or ""),
            file=str(payload.get("file") or ""),
            code_hash=str(payload.get("code_hash") or ""),
            path=str(payload.get("path") or ""),
            revision=int(revision) if isinstance(revision, int | float) and revision else None,
            group_name=_optional_str(payload.get("group_name")),
            family=_optional_str(payload.get("family")),
            title=_optional_str(payload.get("title")),
        )


@dataclass
class MetaRecord:
    node_id: str
    kind: str
    run_id: str
    step: int | None
    mode: str
    profile_id: str
    profile_hash: str | None
    status: str
    created_at: str
    finished_at: str | None
    run_seconds: int | None
    hypothesis_id: str | None
    branch_id: str | None
    parent_kind: str | None
    parent_id: str | None
    grandparent_kind: str | None
    grandparent_id: str | None
    root_id: str | None
    graph_depth: int | None
    materialization_file: str | None
    code_hash: str | None
    cv_score: float | None
    public_score: float | None
    private_score: float | None
    metric_name: str | None
    submit_status: str | None
    dataset_fingerprint: str | None
    profile_seed: int | None = None
    profile_time_limit: int | None = None
    profile_presets: str | None = None
    profile_validation_strategy: str | None = None
    profile_class_balance: str | None = None
    profile_use_gpu: bool | None = None
    profile_fold_config: str | None = None
    components: list[MetaComponent] = field(default_factory=list)

    @property
    def entity_kind(self) -> str:
        if self.kind == "branch" or self.branch_id:
            return "branch"
        if self.kind == "rerun" and self.branch_id:
            return "branch"
        return "root"

    @property
    def entity_id(self) -> str:
        return self.branch_id or self.hypothesis_id or self.node_id

    @property
    def entity_key(self) -> str:
        return f"{self.entity_kind}:{self.entity_id}"

    @property
    def parent_key(self) -> str | None:
        if not self.parent_kind or not self.parent_id:
            return None
        kind = "branch" if self.parent_kind == "branch" or self.parent_id.upper().startswith("B") else "root"
        return f"{kind}:{self.parent_id}"

    @property
    def grandparent_key(self) -> str | None:
        if not self.grandparent_kind or not self.grandparent_id:
            return None
        kind = "branch" if self.grandparent_kind == "branch" or self.grandparent_id.upper().startswith("B") else "root"
        return f"{kind}:{self.grandparent_id}"

    def to_dataset_row(self) -> dict[str, Any]:
        components = [component.to_dict() for component in self.components]
        active_groups = [component.logical_key for component in self.components]
        active_versions = [component.version_key for component in self.components]
        active_revisions = [
            f"{component.logical_key}@{component.revision}"
            for component in self.components
            if component.revision is not None
        ]
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "run_id": self.run_id,
            "step": self.step,
            "mode": self.mode,
            "profile_id": self.profile_id,
            "profile_hash": self.profile_hash,
            "status": self.status,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "run_seconds": self.run_seconds,
            "hypothesis_id": self.hypothesis_id,
            "branch_id": self.branch_id,
            "parent_kind": self.parent_kind,
            "parent_id": self.parent_id,
            "grandparent_kind": self.grandparent_kind,
            "grandparent_id": self.grandparent_id,
            "root_id": self.root_id,
            "graph_depth": self.graph_depth,
            "materialization_file": self.materialization_file,
            "code_hash": self.code_hash,
            "cv_score": self.cv_score,
            "public_score": self.public_score,
            "public_gap": self.public_score - self.cv_score if self.public_score is not None and self.cv_score is not None else None,
            "private_score": self.private_score,
            "metric_name": self.metric_name,
            "submit_status": self.submit_status,
            "dataset_fingerprint": self.dataset_fingerprint,
            "profile_seed": self.profile_seed,
            "profile_time_limit": self.profile_time_limit,
            "profile_presets": self.profile_presets,
            "profile_validation_strategy": self.profile_validation_strategy,
            "profile_class_balance": self.profile_class_balance,
            "profile_use_gpu": self.profile_use_gpu,
            "profile_fold_config": self.profile_fold_config,
            "n_active_groups": len(active_groups),
            "active_groups": ";".join(active_groups),
            "active_group_versions": ";".join(active_versions),
            "active_group_revisions": ";".join(active_revisions),
            "components_json": json.dumps(components, sort_keys=True, separators=(",", ":")),
        }


def records_from_rows(rows: list[dict[str, Any]]) -> list[MetaRecord]:
    records: list[MetaRecord] = []
    for row in rows:
        components_payload = row.get("components_json")
        components: list[MetaComponent] = []
        if isinstance(components_payload, str) and components_payload:
            try:
                parsed = json.loads(components_payload)
            except json.JSONDecodeError:
                parsed = []
            if isinstance(parsed, list):
                components = [MetaComponent.from_dict(item) for item in parsed if isinstance(item, dict)]
        records.append(
            MetaRecord(
                node_id=str(row.get("node_id") or ""),
                kind=str(row.get("kind") or ""),
                run_id=str(row.get("run_id") or ""),
                step=_optional_int(row.get("step")),
                mode=str(row.get("mode") or ""),
                profile_id=str(row.get("profile_id") or ""),
                profile_hash=_optional_str(row.get("profile_hash")),
                status=str(row.get("status") or ""),
                created_at=str(row.get("created_at") or ""),
                finished_at=_optional_str(row.get("finished_at")),
                run_seconds=_optional_int(row.get("run_seconds")),
                hypothesis_id=_optional_str(row.get("hypothesis_id")),
                branch_id=_optional_str(row.get("branch_id")),
                parent_kind=_optional_str(row.get("parent_kind")),
                parent_id=_optional_str(row.get("parent_id")),
                grandparent_kind=_optional_str(row.get("grandparent_kind")),
                grandparent_id=_optional_str(row.get("grandparent_id")),
                root_id=_optional_str(row.get("root_id")),
                graph_depth=_optional_int(row.get("graph_depth")),
                materialization_file=_optional_str(row.get("materialization_file")),
                code_hash=_optional_str(row.get("code_hash")),
                cv_score=_optional_float(row.get("cv_score")),
                public_score=_optional_float(row.get("public_score")),
                private_score=_optional_float(row.get("private_score")),
                metric_name=_optional_str(row.get("metric_name")),
                submit_status=_optional_str(row.get("submit_status")),
                dataset_fingerprint=_optional_str(row.get("dataset_fingerprint")),
                profile_seed=_optional_int(row.get("profile_seed")),
                profile_time_limit=_optional_int(row.get("profile_time_limit")),
                profile_presets=_optional_str(row.get("profile_presets")),
                profile_validation_strategy=_optional_str(row.get("profile_validation_strategy")),
                profile_class_balance=_optional_str(row.get("profile_class_balance")),
                profile_use_gpu=_optional_bool(row.get("profile_use_gpu")),
                profile_fold_config=_optional_str(row.get("profile_fold_config")),
                components=components,
            )
        )
    return records


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None
