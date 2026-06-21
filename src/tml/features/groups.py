from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any


LOCAL_PREFIX_RE = re.compile(r"^G\d+_")


def has_feature_groups(module: object) -> bool:
    groups = getattr(module, "FEATURE_GROUPS", None)
    return isinstance(groups, list) and bool(groups)


def run_feature_groups(
    raw: Any,
    feature_groups: list[dict[str, Any]],
    *,
    aux: Any | None = None,
    log_path: Path | None = None,
    ctx: dict[str, Any] | None = None,
) -> Any:
    import numpy as np
    import pandas as pd

    globals()["np"] = np
    globals()["pd"] = pd
    raw = raw.copy().reset_index(drop=True)
    _validate_unique_columns(raw, "raw/base frame")
    aux = pd.DataFrame() if aux is None else aux.copy()
    ordered = _topological_sort(feature_groups)
    local_blocks: dict[str, pd.DataFrame] = {}
    final_blocks = [raw]
    runtime_ctx: dict[str, Any] = {
        "raw_columns": tuple(raw.columns),
        "feature_group_manifest": [],
        **(ctx or {}),
    }

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")

    for index, group in enumerate(ordered, start=1):
        name = str(group["name"])
        depends_on = [str(dep) for dep in group.get("depends_on", []) or []]
        deps = {dep: local_blocks[dep] for dep in depends_on}
        started_at = time.perf_counter()
        status = "ok"
        warnings: list[dict[str, Any]] = []
        try:
            block = group["fn"](raw.copy(), {k: v.copy() for k, v in deps.items()}, aux.copy(), runtime_ctx)
            block = _normalize_block(block, raw=raw, group_name=name)
            warnings = _column_warnings(block)
            local_blocks[name] = block.copy()
            prefix = f"G{index:03d}"
            final_block = block.copy()
            final_block.columns = [f"{prefix}_{column}" for column in final_block.columns]
            _validate_unique_columns(final_block, f"prefixed group {name!r}")
            final_blocks.append(final_block)
            runtime_ctx["feature_group_manifest"].append(
                {
                    "name": name,
                    "prefix": prefix,
                    "depends_on": depends_on,
                    "description": str(group.get("description", "")),
                    "created_columns_count": int(len(block.columns)),
                }
            )
        except Exception:
            status = "failed"
            raise
        finally:
            if log_path is not None:
                _append_group_log(
                    log_path,
                    {
                        "group": name,
                        "seconds": round(time.perf_counter() - started_at, 6),
                        "rows": int(len(raw)),
                        "depends_on": depends_on,
                        "created_columns_count": int(len(local_blocks.get(name, pd.DataFrame()).columns)),
                        "created_columns_sample": list(local_blocks.get(name, pd.DataFrame()).columns[:20]),
                        "warnings": warnings,
                        "status": status,
                    },
                )

    final = pd.concat(final_blocks, axis=1).copy()
    _validate_unique_columns(final, "final feature frame")
    return final


def _topological_sort(feature_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for group in feature_groups:
        if not isinstance(group, dict):
            raise TypeError("Every feature group must be a dict")
        name = str(group.get("name") or "")
        if not name:
            raise ValueError("Every feature group must define a non-empty name")
        if name in by_name:
            raise ValueError(f"Duplicate feature group name: {name}")
        if not callable(group.get("fn")):
            raise ValueError(f"Feature group {name!r} must define callable field 'fn'")
        by_name[name] = group

    ordered: list[dict[str, Any]] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str, stack: list[str]) -> None:
        if name in visited:
            return
        if name in visiting:
            raise ValueError(f"Cyclic FEATURE_GROUPS dependency detected: {' -> '.join(stack + [name])}")
        if name not in by_name:
            raise ValueError(f"Unknown FEATURE_GROUPS dependency: {name}")
        visiting.add(name)
        for dep in by_name[name].get("depends_on", []) or []:
            dep_name = str(dep)
            if dep_name not in by_name:
                raise ValueError(f"Feature group {name!r} depends on missing group {dep_name!r}")
            visit(dep_name, stack + [name])
        visiting.remove(name)
        visited.add(name)
        ordered.append(by_name[name])

    for group in feature_groups:
        visit(str(group["name"]), [])
    return ordered


def _normalize_block(block: object, *, raw: Any, group_name: str) -> Any:
    if not isinstance(block, pd.DataFrame):
        raise TypeError(f"Feature group {group_name!r} must return a pandas DataFrame")
    if len(block) != len(raw):
        raise ValueError(f"Feature group {group_name!r} changed row count: {len(block)} != {len(raw)}")
    if not block.index.equals(raw.index):
        raise ValueError(f"Feature group {group_name!r} must preserve the input index and row order")
    block = block.copy()
    _validate_local_columns(block, raw=raw, group_name=group_name)
    block = block.replace([np.inf, -np.inf], np.nan)
    _validate_scalar_cells(block, group_name)
    return block


def _validate_local_columns(block: Any, *, raw: Any, group_name: str) -> None:
    _validate_unique_columns(block, f"feature group {group_name!r}")
    raw_columns = {str(column) for column in raw.columns}
    for column in block.columns:
        text = str(column)
        if not text:
            raise ValueError(f"Feature group {group_name!r} returned an empty column name")
        if LOCAL_PREFIX_RE.match(text):
            raise ValueError(f"Feature group {group_name!r} must not create final-prefixed column {text!r}")
        if text in raw_columns:
            raise ValueError(f"Feature group {group_name!r} returned raw/input column {text!r}; return only new features")


def _validate_unique_columns(frame: Any, label: str) -> None:
    if frame.columns.duplicated().any():
        dupes = frame.columns[frame.columns.duplicated()].astype(str).tolist()
        raise ValueError(f"Duplicate columns in {label}: {dupes[:20]}")


def _validate_scalar_cells(block: Any, group_name: str) -> None:
    object_columns = block.select_dtypes(include=["object"]).columns
    for column in object_columns:
        sample = block[column].dropna()
        if sample.empty:
            continue
        nested = sample.map(lambda value: isinstance(value, (dict, list, tuple, set))).any()
        if bool(nested):
            raise ValueError(f"Feature group {group_name!r} returned nested object values in column {column!r}")


def _column_warnings(block: Any) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    all_null = [str(column) for column in block.columns if bool(block[column].isna().all())]
    if all_null:
        warnings.append({"kind": "all_null_columns", "columns_sample": all_null[:20], "count": len(all_null)})
    constant = [
        str(column)
        for column in block.columns
        if not block[column].isna().all() and int(block[column].nunique(dropna=False)) <= 1
    ]
    if constant:
        warnings.append({"kind": "constant_columns", "columns_sample": constant[:20], "count": len(constant)})
    return warnings


def _append_group_log(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
