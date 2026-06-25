from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from tml.core.profiles import load_profile
from tml.db.connect import connect
from tml.db.state import ensure_project_db
from tml.metamodel.records import MetaComponent, MetaRecord, records_from_rows
from tml.utils.hashing import sha256_file
from tml.utils.yaml_io import read_yaml


@dataclass(frozen=True)
class DatasetBuildResult:
    project_dir: Path
    output_dir: Path
    dataset_csv: Path
    dataset_parquet: Path | None
    metadata_json: Path
    record_count: int
    cv_score_count: int
    public_score_count: int
    missing_fields: dict[str, int]
    dataset_fingerprint: str


def default_dataset_dir(project_dir: Path) -> Path:
    return project_dir / "meta_models" / "datasets"


def build_meta_dataset(project_dir: Path, output_dir: Path | None = None) -> tuple[list[MetaRecord], DatasetBuildResult]:
    output_dir = output_dir or default_dataset_dir(project_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_meta_records(project_dir)
    rows = [record.to_dataset_row() for record in records]

    pd = _require_pandas()
    frame = pd.DataFrame(rows)
    dataset_csv = output_dir / "meta_dataset.csv"
    frame.to_csv(dataset_csv, index=False)
    dataset_parquet = _try_write_parquet(frame, output_dir / "meta_dataset.parquet")
    missing_fields = _missing_field_counts(rows)
    metadata = {
        "schema_version": 1,
        "kind": "tml_meta_model_dataset",
        "project_dir": str(project_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_csv": str(dataset_csv),
        "dataset_parquet": str(dataset_parquet) if dataset_parquet else None,
        "record_count": len(records),
        "cv_score_count": sum(1 for record in records if record.cv_score is not None),
        "public_score_count": sum(1 for record in records if record.public_score is not None),
        "missing_fields": missing_fields,
        "dataset_fingerprint": records[0].dataset_fingerprint if records else dataset_fingerprint(project_dir),
        "source_structures": {
            "primary": "project SQLite tml.db",
            "tables": [
                "nodes",
                "evaluations",
                "submissions",
                "branches",
                "branch_components",
                "branch_edges",
                "hypothesis_revisions",
                "materializations",
                "profiles",
            ],
            "yaml_fallbacks": ["branches/*/branch.yaml", "hypotheses/*/manifest.yaml"],
        },
    }
    metadata_json = output_dir / "meta_dataset.json"
    metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = DatasetBuildResult(
        project_dir=project_dir,
        output_dir=output_dir,
        dataset_csv=dataset_csv,
        dataset_parquet=dataset_parquet,
        metadata_json=metadata_json,
        record_count=len(records),
        cv_score_count=int(metadata["cv_score_count"]),
        public_score_count=int(metadata["public_score_count"]),
        missing_fields=missing_fields,
        dataset_fingerprint=str(metadata["dataset_fingerprint"]),
    )
    return records, result


def load_records_from_csv(path: Path) -> list[MetaRecord]:
    pd = _require_pandas()
    frame = pd.read_csv(path, dtype=str)
    rows = frame.where(pd.notnull(frame), None).to_dict(orient="records")
    return records_from_rows(rows)


def load_meta_records(project_dir: Path) -> list[MetaRecord]:
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        node_rows = [dict(row) for row in conn.execute(_NODE_SQL).fetchall()]
        branches = {str(row["branch_id"]): dict(row) for row in conn.execute("SELECT * FROM branches").fetchall()}
        branch_edges = _branch_edges(conn)
        branch_components = _branch_components(conn)
        materializations = _materializations(conn)
        revision_payloads = _revision_payloads(conn, project_dir)
        profile_hashes = {(str(row["mode"]), str(row["profile_id"])): str(row["profile_hash"]) for row in conn.execute("SELECT * FROM profiles").fetchall()}

    profile_payloads = _profile_payloads(project_dir, node_rows)
    fingerprint = dataset_fingerprint(project_dir)
    root_by_branch: dict[str, str | None] = {}
    depth_by_branch: dict[str, int | None] = {}
    records: list[MetaRecord] = []

    for row in node_rows:
        kind = str(row.get("kind") or "root")
        branch_id = _optional_str(row.get("branch_id"))
        hypothesis_id = _optional_str(row.get("hypothesis_id"))
        mode = str(row.get("mode") or "")
        profile_id = str(row.get("profile_id") or "")
        parent_kind, parent_id = _parent_for_row(row, branches, branch_edges)
        grandparent_kind, grandparent_id = _grandparent_for(parent_kind, parent_id, branches, branch_edges)
        root_id = _root_for_record(kind, branch_id, hypothesis_id, parent_kind, parent_id, branches, branch_edges, root_by_branch)
        depth = _depth_for_record(kind, branch_id, branches, branch_edges, depth_by_branch)
        profile_payload = profile_payloads.get((mode, profile_id), {})
        components = _components_for_row(
            row,
            branch_components=branch_components,
            materializations=materializations,
            revision_payloads=revision_payloads,
            project_dir=project_dir,
        )
        profile_fold_config = _profile_fold_config(profile_payload)
        record = MetaRecord(
            node_id=str(row.get("node_id") or ""),
            kind=kind,
            run_id=str(row.get("run_id") or ""),
            step=_optional_int(row.get("step")),
            mode=mode,
            profile_id=profile_id,
            profile_hash=profile_hashes.get((mode, profile_id)),
            status=str(row.get("node_status") or row.get("evaluation_status") or ""),
            created_at=str(row.get("created_at") or ""),
            finished_at=_optional_str(row.get("finished_at")),
            run_seconds=_optional_int(row.get("run_seconds")),
            hypothesis_id=hypothesis_id,
            branch_id=branch_id,
            parent_kind=parent_kind,
            parent_id=parent_id,
            grandparent_kind=grandparent_kind,
            grandparent_id=grandparent_id,
            root_id=root_id,
            graph_depth=depth,
            materialization_file=_optional_str(row.get("materialization_file")),
            code_hash=_optional_str(row.get("code_hash")),
            cv_score=_optional_float(row.get("cv_score")),
            public_score=_optional_float(row.get("public_score")),
            private_score=_optional_float(row.get("private_score")),
            metric_name=_optional_str(row.get("metric_name")),
            submit_status=_optional_str(row.get("submit_status")),
            dataset_fingerprint=fingerprint,
            profile_seed=_optional_int(profile_payload.get("seed")),
            profile_time_limit=_optional_int(profile_payload.get("time_limit") or profile_payload.get("time")),
            profile_presets=_optional_str(profile_payload.get("presets") or profile_payload.get("preset")),
            profile_validation_strategy=_optional_str(profile_payload.get("validation_strategy")),
            profile_class_balance=_optional_str(profile_payload.get("class_balance")),
            profile_use_gpu=_optional_bool(profile_payload.get("use_gpu")),
            profile_fold_config=profile_fold_config,
            components=components,
        )
        records.append(record)

    return sorted(records, key=_record_sort_key)


def dataset_fingerprint(project_dir: Path) -> str:
    payload: dict[str, Any] = {
        "project_yaml": _safe_file_hash(project_dir / "project.yaml"),
        "data_files": [],
    }
    data_dir = project_dir / "data"
    if data_dir.exists():
        for path in sorted(data_dir.iterdir()):
            if not path.is_file():
                continue
            stat = path.stat()
            payload["data_files"].append(
                {
                    "name": path.name,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


_NODE_SQL = """
WITH sub AS (
  SELECT
    node_id,
    MAX(public_score) AS public_score,
    MAX(private_score) AS private_score,
    MAX(local_score) AS local_score,
    MAX(public_rank) AS public_rank,
    MAX(metric) AS metric_name,
    MAX(submit_status) AS submit_status
  FROM submissions
  GROUP BY node_id
)
SELECT
  n.node_id,
  n.run_id,
  n.step,
  n.kind,
  n.hypothesis_id,
  n.hypothesis_revision,
  n.materialization_file,
  n.branch_id,
  n.mode,
  n.profile_id,
  n.status AS node_status,
  n.created_at,
  n.finished_at,
  n.run_seconds,
  e.code_hash,
  COALESCE(e.metric, sub.local_score) AS cv_score,
  e.status AS evaluation_status,
  sub.public_score,
  sub.private_score,
  sub.metric_name,
  sub.submit_status
FROM nodes n
LEFT JOIN evaluations e ON e.node_id = n.node_id
LEFT JOIN sub ON sub.node_id = n.node_id
ORDER BY n.created_at, n.step, n.node_id
"""


def _branch_edges(conn) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT branch_id, parent_kind, parent_id, child_kind, child_id, edge_kind
        FROM branch_edges
        ORDER BY branch_id
        """
    ).fetchall()
    return {str(row["branch_id"]): dict(row) for row in rows}


def _branch_components(conn) -> dict[str, list[dict[str, Any]]]:
    rows = conn.execute(
        """
        SELECT branch_id, role, source_type, source_id, mode, file, code_hash, path
        FROM branch_components
        ORDER BY branch_id, source_type, source_id, mode, file, code_hash
        """
    ).fetchall()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["branch_id"]), []).append(dict(row))
    return grouped


def _materializations(conn) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT hypothesis_id, mode, file, code_hash, hypothesis_revision, status, active
        FROM materializations
        """
    ).fetchall()
    result: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["hypothesis_id"]), str(row["mode"]), str(row["file"]), str(row["code_hash"]))
        result[key] = dict(row)
    return result


def _revision_payloads(conn, project_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = conn.execute("SELECT hypothesis_id, revision, path FROM hypothesis_revisions").fetchall()
    payloads: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        hypothesis_id = str(row["hypothesis_id"])
        revision = int(row["revision"])
        path = project_dir / str(row["path"])
        payload = read_yaml(path) if path.exists() else {}
        payloads[(hypothesis_id, revision)] = payload if isinstance(payload, dict) else {}
    return payloads


def _profile_payloads(project_dir: Path, rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    payloads: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        mode = str(row.get("mode") or "")
        profile_id = str(row.get("profile_id") or "")
        if not mode or not profile_id or (mode, profile_id) in payloads:
            continue
        try:
            payload = load_profile(project_dir, mode, profile_id)
        except Exception:
            payload = {}
        payloads[(mode, profile_id)] = payload if isinstance(payload, dict) else {}
    return payloads


def _components_for_row(
    row: dict[str, Any],
    *,
    branch_components: dict[str, list[dict[str, Any]]],
    materializations: dict[tuple[str, str, str, str], dict[str, Any]],
    revision_payloads: dict[tuple[str, int], dict[str, Any]],
    project_dir: Path,
) -> list[MetaComponent]:
    branch_id = _optional_str(row.get("branch_id"))
    if branch_id:
        payloads = branch_components.get(branch_id) or _branch_yaml_components(project_dir, branch_id)
    else:
        payloads = [_root_component_payload(row)]

    components: list[MetaComponent] = []
    seen_versions: set[str] = set()
    for payload in payloads:
        component = _component_from_payload(payload, materializations=materializations, revision_payloads=revision_payloads)
        if not component.source_id:
            continue
        if component.version_key in seen_versions:
            continue
        seen_versions.add(component.version_key)
        components.append(component)
    return sorted(components, key=lambda item: (item.source_type, item.source_id, item.mode, item.file, item.code_hash))


def _root_component_payload(row: dict[str, Any]) -> dict[str, Any]:
    mode = str(row.get("mode") or "")
    hypothesis_id = str(row.get("hypothesis_id") or "")
    file_name = str(row.get("materialization_file") or "")
    code_hash = str(row.get("code_hash") or "")
    return {
        "role": "root",
        "source_type": "hypothesis",
        "source_id": hypothesis_id,
        "mode": mode,
        "file": file_name,
        "code_hash": code_hash,
        "path": f"hypotheses/{hypothesis_id}/materializations/{file_name}" if hypothesis_id and file_name else "",
    }


def _branch_yaml_components(project_dir: Path, branch_id: str) -> list[dict[str, Any]]:
    branch_yaml = project_dir / "branches" / branch_id / "branch.yaml"
    payload = read_yaml(branch_yaml) if branch_yaml.exists() else {}
    components = payload.get("components") if isinstance(payload, dict) else None
    return [dict(item) for item in components if isinstance(item, dict)] if isinstance(components, list) else []


def _component_from_payload(
    payload: dict[str, Any],
    *,
    materializations: dict[tuple[str, str, str, str], dict[str, Any]],
    revision_payloads: dict[tuple[str, int], dict[str, Any]],
) -> MetaComponent:
    source_id = str(payload.get("source_id") or "")
    mode = str(payload.get("mode") or "")
    file_name = str(payload.get("file") or "")
    code_hash = str(payload.get("code_hash") or "")
    mat = materializations.get((source_id, mode, file_name, code_hash), {})
    revision = _optional_int(payload.get("revision") or mat.get("hypothesis_revision"))
    revision_payload = revision_payloads.get((source_id, revision or 0), {})
    return MetaComponent(
        role=str(payload.get("role") or ""),
        source_type=str(payload.get("source_type") or ""),
        source_id=source_id,
        mode=mode,
        file=file_name,
        code_hash=code_hash,
        path=str(payload.get("path") or ""),
        revision=revision,
        group_name=_optional_str(revision_payload.get("group_name")),
        family=_optional_str(revision_payload.get("family")),
        title=_optional_str(revision_payload.get("title")),
    )


def _parent_for_row(
    row: dict[str, Any],
    branches: dict[str, dict[str, Any]],
    branch_edges: dict[str, dict[str, Any]],
) -> tuple[str | None, str | None]:
    branch_id = _optional_str(row.get("branch_id"))
    if not branch_id:
        return None, None
    edge = branch_edges.get(branch_id)
    if edge:
        return str(edge.get("parent_kind") or ""), _normalize_parent_id(edge.get("parent_id"))
    branch = branches.get(branch_id, {})
    parent_ref = _normalize_parent_id(branch.get("parent_ref"))
    if not parent_ref:
        return None, None
    return ("branch", parent_ref) if parent_ref.upper().startswith("B") else ("hypothesis", parent_ref)


def _grandparent_for(
    parent_kind: str | None,
    parent_id: str | None,
    branches: dict[str, dict[str, Any]],
    branch_edges: dict[str, dict[str, Any]],
) -> tuple[str | None, str | None]:
    if parent_kind != "branch" or not parent_id:
        return None, None
    fake_row = {"branch_id": parent_id}
    return _parent_for_row(fake_row, branches, branch_edges)


def _root_for_record(
    kind: str,
    branch_id: str | None,
    hypothesis_id: str | None,
    parent_kind: str | None,
    parent_id: str | None,
    branches: dict[str, dict[str, Any]],
    branch_edges: dict[str, dict[str, Any]],
    cache: dict[str, str | None],
) -> str | None:
    if kind == "root" and hypothesis_id:
        return hypothesis_id
    if parent_kind and parent_kind != "branch" and parent_id:
        return parent_id
    if branch_id:
        return _root_for_branch(branch_id, branches, branch_edges, cache, set())
    return hypothesis_id


def _root_for_branch(
    branch_id: str,
    branches: dict[str, dict[str, Any]],
    branch_edges: dict[str, dict[str, Any]],
    cache: dict[str, str | None],
    visiting: set[str],
) -> str | None:
    if branch_id in cache:
        return cache[branch_id]
    if branch_id in visiting:
        return None
    parent_kind, parent_id = _parent_for_row({"branch_id": branch_id}, branches, branch_edges)
    if not parent_id:
        cache[branch_id] = None
    elif parent_kind == "branch" or parent_id.upper().startswith("B"):
        cache[branch_id] = _root_for_branch(parent_id, branches, branch_edges, cache, {*visiting, branch_id})
    else:
        cache[branch_id] = parent_id
    return cache[branch_id]


def _depth_for_record(
    kind: str,
    branch_id: str | None,
    branches: dict[str, dict[str, Any]],
    branch_edges: dict[str, dict[str, Any]],
    cache: dict[str, int | None],
) -> int | None:
    if kind == "root" or not branch_id:
        return 0
    return _depth_for_branch(branch_id, branches, branch_edges, cache, set())


def _depth_for_branch(
    branch_id: str,
    branches: dict[str, dict[str, Any]],
    branch_edges: dict[str, dict[str, Any]],
    cache: dict[str, int | None],
    visiting: set[str],
) -> int | None:
    if branch_id in cache:
        return cache[branch_id]
    if branch_id in visiting:
        return None
    parent_kind, parent_id = _parent_for_row({"branch_id": branch_id}, branches, branch_edges)
    if not parent_id or parent_kind != "branch":
        cache[branch_id] = 1
        return 1
    parent_depth = _depth_for_branch(parent_id, branches, branch_edges, cache, {*visiting, branch_id})
    cache[branch_id] = None if parent_depth is None else parent_depth + 1
    return cache[branch_id]


def _profile_fold_config(profile: dict[str, Any]) -> str | None:
    keys = ("validation_strategy", "num_bag_folds", "num_stack_levels", "auto_stack")
    payload = {key: profile.get(key) for key in keys if key in profile}
    fit_args = profile.get("fit_args")
    if isinstance(fit_args, dict):
        for key in ("num_bag_folds", "num_stack_levels", "auto_stack"):
            if key in fit_args:
                payload[f"fit_args.{key}"] = fit_args[key]
    return json.dumps(payload, sort_keys=True) if payload else None


def _missing_field_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    important = [
        "branch_id",
        "parent_id",
        "root_id",
        "profile_hash",
        "profile_seed",
        "profile_fold_config",
        "cv_score",
        "public_score",
    ]
    return {field: sum(1 for row in rows if row.get(field) in (None, "")) for field in important}


def _safe_file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return sha256_file(path)
    except OSError:
        return None


def _try_write_parquet(frame, path: Path) -> Path | None:
    try:
        frame.to_parquet(path, index=False)
    except Exception:
        return None
    return path


def _record_sort_key(record: MetaRecord) -> tuple[str, int, str]:
    return (record.created_at or "", record.step or 0, record.node_id)


def _normalize_parent_id(value: object) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.upper().startswith("B") and text[1:].isdigit():
        return f"B{int(text[1:]):06d}"
    if text.isdigit():
        return text.zfill(6)
    return text


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


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Meta-model dataset building requires pandas. Install project requirements with uv pip.") from exc
    return pd
