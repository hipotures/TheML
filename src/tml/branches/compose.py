from __future__ import annotations

import importlib.util
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tml.core.config import active_mode, active_profile_id, load_project_config
from tml.core.errors import TmlError
from tml.db.connect import connect
from tml.db.state import (
    branch_by_composition,
    branch_by_id,
    branch_component_rows,
    ensure_project_db,
    branch_node_count,
    branch_node_paths,
    delete_branch_records,
    next_branch_id,
    upsert_branch,
    ensure_project_db,
)
from tml.features.validation import validate_group_code_source
from tml.hypotheses.revisions import migrate_hypothesis_dir, revision_records
from tml.utils.hashing import sha256_file, sha256_text
from tml.utils.yaml_io import read_yaml, write_yaml


@dataclass(frozen=True)
class BranchSource:
    ref: str
    source_type: str
    source_id: str
    mode: str
    file: str
    code_path: Path
    code_hash: str
    path: str
    parent_kind: str
    parent_id: str


@dataclass(frozen=True)
class CreatedBranch:
    branch_id: str
    branch_path: Path
    materialization_path: Path
    parent: BranchSource
    source: BranchSource
    composition_hash: str


@dataclass(frozen=True)
class BranchCompositionPlan:
    parent: BranchSource
    source: BranchSource
    components: list[dict[str, Any]]
    composition_hash: str
    existing_branch: dict[str, Any] | None


@dataclass(frozen=True)
class BranchDeletePlan:
    branch_id: str
    path: Path
    mode: str
    parent_ref: str
    source_ref: str
    node_count: int
    force: bool


@dataclass(frozen=True)
class BranchRebaseTarget:
    branch_id: str
    node_id: str | None
    step: int | None
    status: str | None
    metric: float | None


@dataclass(frozen=True)
class BranchRebaseComponentChange:
    role: str
    source_type: str
    source_id: str
    old_file: str
    new_file: str
    old_code_hash: str
    new_code_hash: str
    old_score: float | None
    new_score: float | None


@dataclass(frozen=True)
class BranchRebasePlan:
    source_branch_id: str
    mode: str
    source_score: float | None
    source_node_id: str | None
    source_step: int | None
    composition_hash: str
    existing_branch_id: str | None
    changed_components: list[BranchRebaseComponentChange]
    total_components: int


@dataclass(frozen=True)
class RebasedBranch:
    branch_id: str
    source_branch_id: str
    mode: str
    branch_path: Path
    materialization_path: Path
    composition_hash: str
    existing: bool
    changed_components: int
    total_components: int


def add_branch(project_dir: Path, *, parent_ref: str, source_ref: str, mode: str | None = None) -> CreatedBranch:
    plan = plan_branch_composition(project_dir, parent_ref=parent_ref, source_ref=source_ref, mode=mode)
    mode = plan.parent.mode
    parent = plan.parent
    source = plan.source
    components = plan.components
    composition_hash = plan.composition_hash
    if plan.existing_branch is not None:
        branch_id = str(plan.existing_branch["branch_id"])
        materialization_path = project_dir / str(plan.existing_branch["path"]).rsplit("/", 1)[0] / "materializations" / str(plan.existing_branch["materialization_file"])
        return CreatedBranch(
            branch_id=branch_id,
            branch_path=materialization_path.parents[1],
            materialization_path=materialization_path,
            parent=parent,
            source=source,
            composition_hash=composition_hash,
        )

    branch_id = next_branch_id(project_dir)
    branch_dir = project_dir / "branches" / branch_id
    mat_dir = branch_dir / "materializations"
    mat_dir.mkdir(parents=True, exist_ok=True)
    materialization_path = mat_dir / f"{mode}-001.py"
    code = build_branch_materialization_source(project_dir, components)
    validate_group_code_source(code)
    materialization_path.write_text(code, encoding="utf-8")
    code_hash = sha256_file(materialization_path)
    created_at = datetime.now().isoformat(timespec="seconds")
    summary = f"Add {source.source_id} groups to parent {parent.source_id}."
    stored_parent_ref = parent.ref
    stored_source_ref = source.ref
    write_yaml(
        branch_dir / "branch.yaml",
        {
            "schema_version": 1,
            "branch_id": branch_id,
            "operation": "add_existing_groups",
            "parent_ref": stored_parent_ref,
            "source_ref": stored_source_ref,
            "parent": _source_manifest(parent),
            "source": _source_manifest(source),
            "parent_edge": {"kind": parent.parent_kind, "id": parent.parent_id},
            "mode": mode,
            "status": "materialized",
            "materialization_file": materialization_path.name,
            "code_hash": code_hash,
            "composition_hash": composition_hash,
            "summary": summary,
            "created_at": created_at,
            "components": components,
        },
    )
    upsert_branch(
        project_dir,
        branch_dir,
        parent_ref=stored_parent_ref,
        source_ref=stored_source_ref,
        parent_kind=parent.parent_kind,
        parent_id=parent.parent_id,
        mode=mode,
        materialization_file=materialization_path.name,
        code_hash=code_hash,
        composition_hash=composition_hash,
        summary=summary,
        components=components,
    )
    return CreatedBranch(
        branch_id=branch_id,
        branch_path=branch_dir,
        materialization_path=materialization_path,
        parent=parent,
        source=source,
        composition_hash=composition_hash,
    )


def plan_branch_composition(project_dir: Path, *, parent_ref: str, source_ref: str, mode: str | None = None) -> BranchCompositionPlan:
    config = load_project_config(project_dir)
    mode = mode or active_mode(config)
    parent = resolve_branch_source(project_dir, parent_ref, mode=mode)
    source = resolve_branch_source(project_dir, source_ref, mode=mode)
    if parent.code_hash == source.code_hash:
        raise TmlError("Branch add source resolves to the same code as parent.")

    components = _component_records(project_dir, parent=parent, source=source)
    composition_hash = _composition_hash(components)
    existing = branch_by_composition(project_dir, mode=mode, composition_hash=composition_hash)
    return BranchCompositionPlan(
        parent=parent,
        source=source,
        components=components,
        composition_hash=composition_hash,
        existing_branch=existing,
    )


def branch_rebase_targets(
    project_dir: Path,
    *,
    branch_id: str | None = None,
    node_id: str | None = None,
    step: int | None = None,
) -> list[BranchRebaseTarget]:
    selectors = [branch_id is not None, node_id is not None, step is not None]
    if sum(selectors) != 1:
        raise TmlError("Specify exactly one of id=<branch>, node=<node_id>, or step=<n>.")
    if branch_id is not None:
        normalized = _normalize_branch_id(branch_id)
        row = branch_by_id(project_dir, normalized)
        if row is None:
            return []
        return [BranchRebaseTarget(branch_id=normalized, node_id=None, step=None, status=str(row.get("status") or ""), metric=None)]

    db_path = ensure_project_db(project_dir)
    params: tuple[Any, ...]
    where: str
    if node_id is not None:
        where = "n.node_id=?"
        params = (node_id,)
    else:
        where = "n.step=?"
        params = (step,)
    with connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT n.branch_id, n.node_id, n.step, n.status, e.metric
            FROM nodes n
            LEFT JOIN evaluations e ON e.node_id=n.node_id
            WHERE n.kind='branch'
              AND n.branch_id IS NOT NULL
              AND {where}
            ORDER BY n.step, n.node_id
            """,
            params,
        ).fetchall()
    return [
        BranchRebaseTarget(
            branch_id=str(row["branch_id"]),
            node_id=str(row["node_id"]),
            step=int(row["step"]) if row["step"] is not None else None,
            status=str(row["status"] or ""),
            metric=float(row["metric"]) if isinstance(row["metric"], int | float) else None,
        )
        for row in rows
    ]


def rebase_branch(project_dir: Path, *, branch_id: str) -> RebasedBranch:
    normalized, source_row, original_components, components, plan = _branch_rebase_parts(project_dir, branch_id=branch_id)
    mode = plan.mode
    composition_hash = plan.composition_hash
    existing = branch_by_composition(project_dir, mode=mode, composition_hash=composition_hash)
    changed_count = len(plan.changed_components)
    if existing is not None:
        existing_branch_id = str(existing["branch_id"])
        materialization_path = project_dir / str(existing["path"]).rsplit("/", 1)[0] / "materializations" / str(
            existing["materialization_file"]
        )
        return RebasedBranch(
            branch_id=existing_branch_id,
            source_branch_id=normalized,
            mode=mode,
            branch_path=materialization_path.parents[1],
            materialization_path=materialization_path,
            composition_hash=composition_hash,
            existing=True,
            changed_components=changed_count,
            total_components=len(components),
        )

    branch_id_value = next_branch_id(project_dir)
    branch_dir = project_dir / "branches" / branch_id_value
    mat_dir = branch_dir / "materializations"
    mat_dir.mkdir(parents=True, exist_ok=True)
    materialization_path = mat_dir / f"{mode}-001.py"
    code = build_branch_materialization_source(project_dir, components)
    validate_group_code_source(code)
    materialization_path.write_text(code, encoding="utf-8")
    code_hash = sha256_file(materialization_path)
    created_at = datetime.now().isoformat(timespec="seconds")
    summary = f"Rebase {normalized} with active hypothesis materializations."
    write_yaml(
        branch_dir / "branch.yaml",
        {
            "schema_version": 1,
            "branch_id": branch_id_value,
            "operation": "rebase_active_materializations",
            "source_branch_id": normalized,
            "parent_ref": normalized,
            "source_ref": normalized,
            "parent": {
                "ref": normalized,
                "source_type": "branch",
                "source_id": normalized,
                "mode": mode,
                "file": str(source_row.get("materialization_file") or ""),
                "code_hash": str(source_row.get("code_hash") or ""),
                "path": str(source_row.get("path") or ""),
            },
            "parent_edge": {"kind": "branch", "id": normalized},
            "mode": mode,
            "status": "materialized",
            "materialization_file": materialization_path.name,
            "code_hash": code_hash,
            "composition_hash": composition_hash,
            "summary": summary,
            "created_at": created_at,
            "changed_components": changed_count,
            "components": components,
        },
    )
    upsert_branch(
        project_dir,
        branch_dir,
        parent_ref=normalized,
        source_ref=normalized,
        parent_kind="branch",
        parent_id=normalized,
        mode=mode,
        materialization_file=materialization_path.name,
        code_hash=code_hash,
        composition_hash=composition_hash,
        summary=summary,
        components=components,
    )
    return RebasedBranch(
        branch_id=branch_id_value,
        source_branch_id=normalized,
        mode=mode,
        branch_path=branch_dir,
        materialization_path=materialization_path,
        composition_hash=composition_hash,
        existing=False,
        changed_components=changed_count,
        total_components=len(components),
    )


def branch_rebase_plan(project_dir: Path, *, branch_id: str, target: BranchRebaseTarget | None = None) -> BranchRebasePlan:
    _, _, _, _, plan = _branch_rebase_parts(project_dir, branch_id=branch_id, target=target)
    return plan


def branch_delete_plan(project_dir: Path, *, branch_id: str, force: bool = False) -> BranchDeletePlan:
    normalized = _normalize_branch_id(branch_id)
    row = branch_by_id(project_dir, normalized)
    if row is None:
        raise TmlError(f"Branch does not exist: {normalized}")
    branch_path = project_dir / str(row["path"]).rsplit("/", 1)[0]
    return BranchDeletePlan(
        branch_id=normalized,
        path=branch_path,
        mode=str(row.get("mode") or ""),
        parent_ref=str(row.get("parent_ref") or ""),
        source_ref=str(row.get("source_ref") or ""),
        node_count=branch_node_count(project_dir, normalized),
        force=force,
    )


def delete_branch(project_dir: Path, *, branch_id: str, force: bool = False) -> BranchDeletePlan:
    plan = branch_delete_plan(project_dir, branch_id=branch_id, force=force)
    if plan.node_count and not force:
        raise TmlError(f"Branch {plan.branch_id} has {plan.node_count} run node(s); use force=true.")
    node_paths = branch_node_paths(project_dir, plan.branch_id) if force else []
    delete_branch_records(project_dir, plan.branch_id, force=force)
    for node_path in node_paths:
        shutil.rmtree(project_dir / node_path, ignore_errors=True)
    shutil.rmtree(plan.path, ignore_errors=True)
    return plan


def resolve_branch_source(project_dir: Path, ref: str, *, mode: str) -> BranchSource:
    text = str(ref).strip()
    if not text:
        raise TmlError("Missing branch source reference.")
    if text.upper().startswith("B"):
        return _resolve_branch(project_dir, text, mode=mode)
    if text.isdigit():
        return _resolve_hypothesis(project_dir, text.zfill(6), mode=mode)
    return _resolve_node(project_dir, text, mode=mode)


def build_branch_materialization_source(project_dir: Path, components: list[dict[str, Any]]) -> str:
    loaded_components = []
    for index, component in enumerate(sorted(components, key=_component_sort_key), start=1):
        code_path = project_dir / str(component["path"])
        source = code_path.read_text(encoding="utf-8")
        validate_group_code_source(source)
        groups = _load_feature_group_specs(source, module_name=f"_tml_branch_probe_{index}")
        source_key = _source_key(component)
        loaded_components.append(
            {
                **component,
                "source_key": source_key,
                "source_literal": repr(source),
                "groups": groups,
            }
        )
    lines = [
        '"""Deterministic branch composition of existing feature groups."""',
        "",
        "_MODULE_CACHE = {}",
        "",
    ]
    for component in loaded_components:
        constant = f"{component['source_key'].upper()}_SOURCE"
        lines.append(f"{constant} = {component['source_literal']}")
        lines.append("")
    lines.extend(
        [
            "def _load_branch_module(key, source):",
            "    import types",
            "",
            "    if key in _MODULE_CACHE:",
            "        return _MODULE_CACHE[key]",
            "    module = types.ModuleType(key)",
            "    exec(source, module.__dict__)",
            "    _MODULE_CACHE[key] = module",
            "    return module",
            "",
        ]
    )
    feature_group_entries: list[str] = []
    for component in loaded_components:
        constant = f"{component['source_key'].upper()}_SOURCE"
        module_key = component["source_key"]
        name_map = {str(group["name"]): _branch_group_name(component, str(group["name"])) for group in component["groups"]}
        for group_index, group in enumerate(component["groups"], start=1):
            original_name = str(group["name"])
            branch_name = name_map[original_name]
            wrapper_name = f"_branch_group_{module_key}_{group_index:03d}"
            original_dep_names = [str(dep) for dep in group.get("depends_on", []) or []]
            branch_dep_names = [name_map[dep] for dep in original_dep_names if dep in name_map]
            missing = [dep for dep in original_dep_names if dep not in name_map]
            if missing:
                raise TmlError(f"Cannot compose group {original_name!r}: missing dependencies {missing}")
            lines.extend(
                [
                    f"def {wrapper_name}(raw, deps, aux):",
                    f"    module = _load_branch_module({module_key!r}, {constant})",
                    f"    original_deps = {{{', '.join(f'{dep!r}: deps[{name_map[dep]!r}]' for dep in original_dep_names)}}}",
                    f"    return getattr(module, {str(group['fn_name'])!r})(raw, original_deps, aux)",
                    "",
                ]
            )
            description = str(group.get("description") or f"Imported from {component['source_id']}/{original_name}.")
            feature_group_entries.append(
                "{"
                f"'name': {branch_name!r}, "
                f"'fn': {wrapper_name}, "
                f"'depends_on': {branch_dep_names!r}, "
                f"'description': {description!r}"
                "}"
            )
    lines.append("FEATURE_GROUPS = [")
    for entry in feature_group_entries:
        lines.append(f"    {entry},")
    lines.append("]")
    lines.append("")
    return "\n".join(lines)


def _resolve_hypothesis(project_dir: Path, hypothesis_id: str, *, mode: str) -> BranchSource:
    hdir = project_dir / "hypotheses" / hypothesis_id
    migrate_hypothesis_dir(project_dir, hdir)
    if not revision_records(hdir):
        raise TmlError(f"Hypothesis does not exist: {hypothesis_id}")
    manifest = read_yaml(hdir / "manifest.yaml")
    mats = manifest.get("materializations") if isinstance(manifest.get("materializations"), dict) else {}
    mat = mats.get(mode) if isinstance(mats.get(mode), dict) else {}
    file_name = str(mat.get("active") or f"{mode}-001.py")
    code_path = hdir / "materializations" / file_name
    if not code_path.exists():
        raise TmlError(f"Hypothesis {hypothesis_id} has no {mode} materialization: {file_name}")
    return BranchSource(
        ref=hypothesis_id,
        source_type="hypothesis",
        source_id=hypothesis_id,
        mode=mode,
        file=file_name,
        code_path=code_path,
        code_hash=sha256_file(code_path),
        path=code_path.relative_to(project_dir).as_posix(),
        parent_kind="hypothesis",
        parent_id=hypothesis_id,
    )


def _resolve_branch(project_dir: Path, branch_id: str, *, mode: str) -> BranchSource:
    normalized = _normalize_branch_id(branch_id)
    branch_dir = project_dir / "branches" / normalized
    payload = read_yaml(branch_dir / "branch.yaml")
    if not payload:
        raise TmlError(f"Branch does not exist: {normalized}")
    branch_mode = str(payload.get("mode") or mode)
    if branch_mode != mode:
        raise TmlError(f"Branch {normalized} is mode={branch_mode}, not {mode}.")
    file_name = str(payload.get("materialization_file") or f"{mode}-001.py")
    code_path = branch_dir / "materializations" / file_name
    if not code_path.exists():
        raise TmlError(f"Branch {normalized} has no materialization file: {file_name}")
    return BranchSource(
        ref=normalized,
        source_type="branch",
        source_id=normalized,
        mode=mode,
        file=file_name,
        code_path=code_path,
        code_hash=sha256_file(code_path),
        path=code_path.relative_to(project_dir).as_posix(),
        parent_kind="branch",
        parent_id=normalized,
    )


def _resolve_node(project_dir: Path, node_ref: str, *, mode: str) -> BranchSource:
    from tml.db.connect import connect
    from tml.db.state import ensure_project_db

    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT n.kind, n.hypothesis_id, n.branch_id, n.mode, e.code_hash
            FROM nodes n
            LEFT JOIN evaluations e ON e.node_id=n.node_id
            WHERE n.node_id=?
            """,
            (node_ref,),
        ).fetchone()
    if row is None:
        raise TmlError(f"Node does not exist: {node_ref}")
    node_mode = str(row["mode"] or mode)
    if node_mode != mode:
        raise TmlError(f"Node {node_ref} is mode={node_mode}, not {mode}.")
    if str(row["kind"] or "root") == "branch":
        return _resolve_branch(project_dir, str(row["branch_id"]), mode=mode)
    hid = str(row["hypothesis_id"] or "")
    source = _resolve_hypothesis(project_dir, hid, mode=mode)
    code_hash = str(row["code_hash"] or "")
    if code_hash and source.code_hash != code_hash:
        raise TmlError(f"Node {node_ref} does not match the active materialization for hypothesis {hid}.")
    return BranchSource(
        **{**source.__dict__, "ref": node_ref, "parent_kind": "node", "parent_id": node_ref}
    )


def _component_records(project_dir: Path, *, parent: BranchSource, source: BranchSource) -> list[dict[str, Any]]:
    if parent.source_type == "branch":
        components = branch_component_rows(project_dir, parent.source_id)
        parent_components = [{**component, "role": "parent"} for component in components]
    else:
        parent_components = [_component_record(parent, role="parent")]
    if source.source_type == "branch":
        components = branch_component_rows(project_dir, source.source_id)
        source_components = [{**component, "role": "source"} for component in components]
    else:
        source_components = [_component_record(source, role="source")]
    by_logical_component: dict[tuple[str, str, str], dict[str, Any]] = {}
    unique: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for component in [*parent_components, *source_components]:
        logical_key = _component_logical_key(component)
        previous = by_logical_component.get(logical_key)
        if previous is not None and _component_version_key(previous) != _component_version_key(component):
            continue
        by_logical_component[logical_key] = component
        key = (
            str(component["source_type"]),
            str(component["source_id"]),
            str(component["mode"]),
            str(component["file"]),
            str(component["code_hash"]),
        )
        unique[key] = component
    return sorted(unique.values(), key=_component_sort_key)


def _branch_rebase_parts(
    project_dir: Path,
    *,
    branch_id: str,
    target: BranchRebaseTarget | None = None,
) -> tuple[str, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], BranchRebasePlan]:
    normalized = _normalize_branch_id(branch_id)
    source_row = branch_by_id(project_dir, normalized)
    if source_row is None:
        raise TmlError(f"Branch does not exist: {normalized}")
    mode = str(source_row.get("mode") or "")
    if not mode:
        raise TmlError(f"Branch {normalized} has no mode.")
    config = load_project_config(project_dir)
    profile_id = active_profile_id(config, mode=mode)
    original_components = branch_component_rows(project_dir, normalized)
    if not original_components:
        raise TmlError(f"Branch {normalized} has no components.")
    components = _active_component_records(project_dir, original_components, mode=mode)
    composition_hash = _composition_hash(components)
    existing = branch_by_composition(project_dir, mode=mode, composition_hash=composition_hash)
    source_score, source_node_id, source_step = _branch_best_score(project_dir, normalized, mode=mode)
    if target is not None:
        source_score = target.metric if target.metric is not None else source_score
        source_node_id = target.node_id or source_node_id
        source_step = target.step if target.step is not None else source_step
    plan = BranchRebasePlan(
        source_branch_id=normalized,
        mode=mode,
        source_score=source_score,
        source_node_id=source_node_id,
        source_step=source_step,
        composition_hash=composition_hash,
        existing_branch_id=str(existing["branch_id"]) if existing is not None else None,
        changed_components=_component_changes(
            original_components,
            components,
            project_dir=project_dir,
            mode=mode,
            profile_id=profile_id,
        ),
        total_components=len(components),
    )
    return normalized, source_row, original_components, components, plan


def _branch_best_score(project_dir: Path, branch_id: str, *, mode: str) -> tuple[float | None, str | None, int | None]:
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT e.metric, n.node_id, n.step
            FROM evaluations e
            LEFT JOIN nodes n ON n.node_id=e.node_id
            WHERE e.kind='branch'
              AND e.branch_id=?
              AND e.mode=?
              AND e.status='complete'
              AND e.metric IS NOT NULL
            ORDER BY e.metric DESC, e.node_id DESC
            LIMIT 1
            """,
            (branch_id, mode),
        ).fetchone()
    if row is None:
        return None, None, None
    return (
        float(row["metric"]) if isinstance(row["metric"], int | float) else None,
        str(row["node_id"] or "") or None,
        int(row["step"]) if row["step"] is not None else None,
    )


def _active_component_records(project_dir: Path, components: list[dict[str, Any]], *, mode: str) -> list[dict[str, Any]]:
    refreshed: list[dict[str, Any]] = []
    for component in components:
        if str(component.get("source_type") or "") != "hypothesis":
            refreshed.append(dict(component))
            continue
        source = _resolve_hypothesis(project_dir, str(component.get("source_id") or "").zfill(6), mode=mode)
        refreshed.append(_component_record(source, role=str(component.get("role") or "")))
    return _dedupe_component_records(refreshed)


def _dedupe_component_records(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_logical_component: dict[tuple[str, str, str], dict[str, Any]] = {}
    unique: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for component in components:
        logical_key = _component_logical_key(component)
        previous = by_logical_component.get(logical_key)
        if previous is not None and _component_version_key(previous) != _component_version_key(component):
            continue
        by_logical_component[logical_key] = component
        key = (
            str(component["source_type"]),
            str(component["source_id"]),
            str(component["mode"]),
            str(component["file"]),
            str(component["code_hash"]),
        )
        unique[key] = component
    return sorted(unique.values(), key=_component_sort_key)


def _component_changes(
    original_components: list[dict[str, Any]],
    refreshed_components: list[dict[str, Any]],
    *,
    project_dir: Path,
    mode: str,
    profile_id: str,
) -> list[BranchRebaseComponentChange]:
    refreshed_by_logical = {_component_logical_key(component): component for component in refreshed_components}
    changes: list[BranchRebaseComponentChange] = []
    for component in original_components:
        refreshed = refreshed_by_logical.get(_component_logical_key(component))
        if refreshed is None:
            continue
        if _component_version_key(component) != _component_version_key(refreshed):
            old_score = _component_score(project_dir, component, mode=mode, profile_id=profile_id)
            new_score = _component_score(project_dir, refreshed, mode=mode, profile_id=profile_id)
            changes.append(
                BranchRebaseComponentChange(
                    role=str(component.get("role") or ""),
                    source_type=str(component.get("source_type") or ""),
                    source_id=str(component.get("source_id") or ""),
                    old_file=str(component.get("file") or ""),
                    new_file=str(refreshed.get("file") or ""),
                    old_code_hash=str(component.get("code_hash") or ""),
                    new_code_hash=str(refreshed.get("code_hash") or ""),
                    old_score=old_score,
                    new_score=new_score,
                )
            )
    return changes


def _component_score(project_dir: Path, component: dict[str, Any], *, mode: str, profile_id: str) -> float | None:
    if str(component.get("source_type") or "") != "hypothesis":
        return None
    hypothesis_id = str(component.get("source_id") or "")
    file_name = str(component.get("file") or "")
    code_hash = str(component.get("code_hash") or "")
    if not hypothesis_id or not file_name or not code_hash:
        return None
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT e.metric
            FROM evaluations e
            WHERE e.kind='root'
              AND e.mode=?
              AND e.profile_id=?
              AND e.hypothesis_id=?
              AND e.materialization_file=?
              AND e.code_hash=?
              AND e.status='complete'
              AND e.metric IS NOT NULL
            ORDER BY e.metric DESC, e.node_id DESC
            LIMIT 1
            """,
            (mode, profile_id, hypothesis_id, file_name, code_hash),
        ).fetchone()
    if row is None:
        return None
    metric = row["metric"]
    return float(metric) if isinstance(metric, int | float) else None


def _component_record(source: BranchSource, *, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "source_type": source.source_type,
        "source_id": source.source_id,
        "mode": source.mode,
        "file": source.file,
        "code_hash": source.code_hash,
        "path": source.path,
    }


def _component_logical_key(component: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(component.get("source_type") or ""),
        str(component.get("source_id") or ""),
        str(component.get("mode") or ""),
    )


def _component_version_key(component: dict[str, Any]) -> tuple[str, str]:
    return (
        str(component.get("file") or ""),
        str(component.get("code_hash") or ""),
    )


def _source_manifest(source: BranchSource) -> dict[str, str]:
    return {
        "ref": source.ref,
        "source_type": source.source_type,
        "source_id": source.source_id,
        "mode": source.mode,
        "file": source.file,
        "code_hash": source.code_hash,
        "path": source.path,
    }


def _composition_hash(components: list[dict[str, Any]]) -> str:
    payload = [
        {
            "source_type": component["source_type"],
            "source_id": component["source_id"],
            "mode": component["mode"],
            "file": component["file"],
            "code_hash": component["code_hash"],
        }
        for component in sorted(components, key=_component_sort_key)
    ]
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _load_feature_group_specs(source: str, *, module_name: str) -> list[dict[str, Any]]:
    module = _load_source_module(source, module_name=module_name)
    groups = getattr(module, "FEATURE_GROUPS", None)
    if not isinstance(groups, list) or not groups:
        raise TmlError("Materialization must define non-empty FEATURE_GROUPS.")
    specs: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, dict):
            raise TmlError("Every FEATURE_GROUPS entry must be a dict.")
        fn = group.get("fn")
        if not callable(fn):
            raise TmlError(f"Feature group {group.get('name')!r} has no callable fn.")
        specs.append(
            {
                "name": str(group.get("name") or ""),
                "fn_name": str(getattr(fn, "__name__", "")),
                "depends_on": [str(dep) for dep in group.get("depends_on", []) or []],
                "description": str(group.get("description") or ""),
            }
        )
    return specs


def _load_source_module(source: str, *, module_name: str):
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    if spec is None:
        raise TmlError(f"Cannot create module spec for {module_name}.")
    module = importlib.util.module_from_spec(spec)
    exec(source, module.__dict__)
    return module


def _source_key(component: dict[str, Any]) -> str:
    raw = f"{component['source_type']}_{component['source_id']}_{component['file']}_{str(component['code_hash'])[:10]}"
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in raw).strip("_")


def _branch_group_name(component: dict[str, Any], group_name: str) -> str:
    prefix = _source_key(component)
    safe_group = "".join(ch.lower() if ch.isalnum() else "_" for ch in group_name).strip("_")
    return f"{prefix}__{safe_group}"


def _component_sort_key(component: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(component.get("source_type") or ""),
        str(component.get("source_id") or ""),
        str(component.get("mode") or ""),
        str(component.get("file") or ""),
        str(component.get("code_hash") or ""),
    )


def _normalize_branch_id(value: str) -> str:
    text = str(value).strip()
    if text.upper().startswith("B") and text[1:].isdigit():
        return f"B{int(text[1:]):06d}"
    if text.isdigit():
        return f"B{int(text):06d}"
    return text
