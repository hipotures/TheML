from __future__ import annotations

import ast
import re
from typing import Any


BANNED_HYPOTHESIS_FIELDS = {
    "baseline_model_panel",
    "model_panel_rationale",
    "validation_strategy",
    "materialization_hint",
    "autogluon_config",
    "training_strategy",
}

BANNED_CALLS = {
    "pd.read_csv",
    "pandas.read_csv",
    "open",
}

BANNED_NAMES = {
    "TabularPredictor",
}

BANNED_METHODS = {
    "predict",
    "predict_proba",
}


def validate_root_hypothesis(payload: dict[str, Any]) -> None:
    missing = [
        field
        for field in (
            "title",
            "group_name",
            "family",
            "summary",
            "depends_on",
            "strategy",
            "expected_signal",
            "risk",
        )
        if field not in payload
    ]
    if missing:
        raise ValueError(f"ROOT hypothesis is missing required fields: {missing}")
    banned = sorted(BANNED_HYPOTHESIS_FIELDS.intersection(payload))
    if banned:
        raise ValueError(f"ROOT hypothesis contains banned model/training fields: {banned}")
    if payload.get("depends_on") not in ([], None):
        raise ValueError("ROOT hypotheses must use depends_on=[] in the first group-contract version")
    group_name = str(payload.get("group_name") or "")
    if not re.match(r"^[a-z][a-z0-9_]*$", group_name):
        raise ValueError(f"Invalid group_name {group_name!r}; expected snake_case")


def validate_group_code_source(source: str) -> None:
    tree = ast.parse(source)
    has_feature_groups = False
    function_names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            pass
        elif isinstance(node, ast.Assign):
            if _contains_call(node.value):
                raise ValueError("Group materialization must not call functions in top-level assignments")
        elif isinstance(node, ast.AnnAssign):
            if node.value is not None and _contains_call(node.value):
                raise ValueError("Group materialization must not call functions in top-level assignments")
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            pass
        else:
            raise ValueError("Group materialization must not execute top-level code outside imports, definitions, and registry assignments")
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            raise ValueError("Group materialization must not define main()")
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "FEATURE_GROUPS":
                    has_feature_groups = True
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "FEATURE_GROUPS":
            has_feature_groups = True

    if not has_feature_groups:
        raise ValueError("Group materialization must define FEATURE_GROUPS")

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            raise ValueError(f"Group materialization must not use {node.id}")
        if isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            if call_name in BANNED_CALLS:
                raise ValueError(f"Group materialization must not call {call_name}")
            if isinstance(node.func, ast.Attribute) and node.func.attr in BANNED_METHODS:
                raise ValueError(f"Group materialization must not call .{node.func.attr}()")

    for name in function_names:
        if name.startswith("add_"):
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
            args = [arg.arg for arg in function.args.args]
            if args[:4] != ["raw", "deps", "aux", "ctx"]:
                raise ValueError(f"Feature function {name} must use signature (raw, deps, aux, ctx)")


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _contains_call(node: ast.AST) -> bool:
    return any(isinstance(child, ast.Call) for child in ast.walk(node))
