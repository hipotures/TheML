from __future__ import annotations

import importlib.util
from pathlib import Path


COMPONENTS = (
    ("000001", "autogluon-001.py"),
    ("000021", "autogluon-001.py"),
    ("000022", "autogluon-001.py"),
    ("000023", "autogluon-001.py"),
    ("000024", "autogluon-001.py"),
    ("000025", "autogluon-001.py"),
    ("000026", "autogluon-001.py"),
    ("000027", "autogluon-001.py"),
)


def _project_dir() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_feature_groups(hypothesis_id: str, filename: str) -> list[dict]:
    source_path = _project_dir() / "hypotheses" / hypothesis_id / "materializations" / filename
    module_name = f"_tml_aide_branch_{hypothesis_id}_{filename.replace('-', '_').replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load materialization {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.FEATURE_GROUPS)


FEATURE_GROUPS = []
for _hypothesis_id, _filename in COMPONENTS:
    FEATURE_GROUPS.extend(_load_feature_groups(_hypothesis_id, _filename))
