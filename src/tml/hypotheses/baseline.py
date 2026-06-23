from __future__ import annotations

from datetime import datetime
from pathlib import Path

from tml.db.state import upsert_hypothesis, upsert_materialization
from tml.utils.hashing import sha256_file
from tml.utils.yaml_io import read_yaml, write_yaml
from .revisions import append_materialization, repair_manifest


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
    legacy = hdir / "hypothesis.yaml"
    revision_path = hdir / "01-hypothesis.yaml"
    if legacy.exists() and not revision_path.exists():
        legacy.replace(revision_path)
    elif legacy.exists():
        legacy.unlink()
    existing = read_yaml(revision_path)
    created_at = existing.get("created_at") if isinstance(existing.get("created_at"), str) else None
    payload = {
        "schema_version": 1,
        "hypothesis_id": BASELINE_HYPOTHESIS_ID,
        "revision": 1,
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
    write_yaml(revision_path, payload)


def _write_baseline_manifest(hdir: Path, code_path: Path) -> None:
    repair_manifest(hdir.parents[1], hdir)
    append_materialization(hdir, BASELINE_MODE, code_path, revision=1, active=True)
