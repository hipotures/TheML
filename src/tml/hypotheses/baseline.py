from __future__ import annotations

from datetime import datetime
from pathlib import Path

from tml.db.state import upsert_hypothesis, upsert_materialization
from tml.utils.hashing import sha256_file
from tml.utils.yaml_io import read_yaml, write_yaml


BASELINE_HYPOTHESIS_ID = "000000"
BASELINE_MODE = "autogluon"
BASELINE_MATERIALIZATION_FILE = "autogluon-001.py"

BASELINE_CODE = '''"""Raw AutoGluon baseline.

This materialization intentionally defines no feature groups. The fixed
AutoGluon wrapper will train on the raw project columns, with profile-level
ignored columns still applied.
"""

FEATURE_GROUPS = []
'''


def ensure_root_baseline(project_dir: Path) -> Path:
    hdir = project_dir / "hypotheses" / BASELINE_HYPOTHESIS_ID
    mat_dir = hdir / "materializations"
    mat_dir.mkdir(parents=True, exist_ok=True)
    _write_baseline_hypothesis(hdir)
    code_path = mat_dir / BASELINE_MATERIALIZATION_FILE
    if not code_path.exists() or code_path.read_text(encoding="utf-8") != BASELINE_CODE:
        code_path.write_text(BASELINE_CODE, encoding="utf-8")
    _write_baseline_manifest(hdir, code_path)
    upsert_hypothesis(project_dir, hdir)
    upsert_materialization(project_dir, hdir, BASELINE_MODE, code_path, status="active", active=True)
    return hdir


def _write_baseline_hypothesis(hdir: Path) -> None:
    existing = read_yaml(hdir / "hypothesis.yaml")
    created_at = existing.get("created_at") if isinstance(existing.get("created_at"), str) else None
    payload = {
        "schema_version": 1,
        "hypothesis_id": BASELINE_HYPOTHESIS_ID,
        "title": "Raw AutoGluon baseline",
        "group_name": "raw_autogluon_baseline",
        "family": "baseline",
        "summary": "Train AutoGluon on the raw project columns without generated feature groups.",
        "depends_on": [],
        "strategy": "Use the project train and test tables as-is. Do not create additional preprocessing feature columns.",
        "expected_signal": "Provides the reference score for AutoGluon before any generated feature engineering is added.",
        "risk": "This baseline may underperform generated features, but it should expose the raw AutoGluon reference point.",
        "enabled": True,
        "created_at": created_at or datetime.now().isoformat(timespec="seconds"),
    }
    write_yaml(hdir / "hypothesis.yaml", payload)


def _write_baseline_manifest(hdir: Path, code_path: Path) -> None:
    manifest_path = hdir / "manifest.yaml"
    existing = read_yaml(manifest_path)
    materializations = existing.get("materializations") if isinstance(existing.get("materializations"), dict) else {}
    autogluon = materializations.get(BASELINE_MODE) if isinstance(materializations.get(BASELINE_MODE), dict) else {}
    created_at = autogluon.get("created_at") if isinstance(autogluon.get("created_at"), str) else None
    manifest = {
        "materializations": {
            BASELINE_MODE: {
                "active": code_path.name,
                "sha256": sha256_file(code_path),
                "created_at": created_at or datetime.now().isoformat(timespec="seconds"),
            }
        },
        "feature_group": {
            "logical_name": "raw_autogluon_baseline",
            "version_id": f"raw_autogluon_baseline@{BASELINE_HYPOTHESIS_ID}",
            "source_hypothesis_id": BASELINE_HYPOTHESIS_ID,
            "operation": "raw_autogluon_baseline",
            "depends_on": [],
            "code_artifact": code_path.name,
        },
    }
    write_yaml(manifest_path, manifest)
