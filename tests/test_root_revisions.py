from __future__ import annotations

import json
from pathlib import Path

from tml.ai.client import AiResponse
from tml.db.connect import connect
from tml.db.reindex import reindex_project
from tml.db.state import materialization_rows, root_run_rows
from tml.execution.result import ExecutionResult
from tml.hypotheses.materialize import materialize_missing
from tml.hypotheses.revisions import migrate_root_revisions
from tml.hypotheses.revise import revise_root_hypothesis
from tml.hypotheses.run import run_missing
from tml.utils.yaml_io import read_yaml, write_yaml


def test_backfill_moves_hypothesis_and_repairs_manifest(tmp_path: Path) -> None:
    project_dir = _project(tmp_path)
    hdir = _legacy_hypothesis(project_dir, "000001")
    _write_group_code(hdir / "materializations" / "autogluon-001.py")

    migrate_root_revisions(project_dir)
    migrate_root_revisions(project_dir)

    assert not (hdir / "hypothesis.yaml").exists()
    revision = read_yaml(hdir / "01-hypothesis.yaml")
    assert revision["revision"] == 1
    manifest = read_yaml(hdir / "manifest.yaml")
    assert manifest["hypothesis"] == {"id": "000001", "latest_revision": 1}
    assert manifest["revisions"][1]["file"] == "01-hypothesis.yaml"
    assert manifest["materializations"]["autogluon"]["files"][0]["revision"] == 1
    assert "feature_group" not in manifest


def test_revise_creates_numbered_revision_and_no_change_skips(monkeypatch, tmp_path: Path) -> None:
    project_dir = _project(tmp_path)
    hdir = _legacy_hypothesis(project_dir, "000001")
    migrate_root_revisions(project_dir)
    responses = [
        {
            "operation": "revise_existing_root_hypothesis",
            "decision": "revise",
            "title": "Revised",
            "group_name": "example_group",
            "family": "example",
            "summary": "Revised summary.",
            "depends_on": [],
            "strategy": "Use safer bins.",
            "expected_signal": "May help.",
            "risk": "May not help.",
            "change_summary": "Changed bins.",
            "semantic_goal_preserved": True,
        },
        {
            "operation": "revise_existing_root_hypothesis",
            "decision": "no_change",
            "reason": "No safe change.",
        },
    ]

    def fake_invocation(*args, **kwargs):
        payload = responses.pop(0)
        artifact_dir = kwargs["artifact_dir"]
        prefix = kwargs["response_prefix"]
        (artifact_dir / f"{prefix}.request.md").write_text("request", encoding="utf-8")
        (artifact_dir / f"{prefix}.response.md").write_text(json.dumps(payload), encoding="utf-8")
        return AiResponse(text=json.dumps(payload), metadata={})

    monkeypatch.setattr("tml.hypotheses.revise.run_model_invocation", fake_invocation)

    created = revise_root_hypothesis(project_dir, hypothesis_id="1", count=2)

    assert created == [hdir / "02-hypothesis.yaml"]
    assert not (hdir / "03-hypothesis.yaml").exists()
    assert not (hdir / "hypothesis.yaml").exists()
    assert read_yaml(hdir / "02-hypothesis.yaml")["revision"] == 2
    manifest = read_yaml(hdir / "manifest.yaml")
    assert manifest["hypothesis"]["latest_revision"] == 2
    assert manifest["revisions"][2]["prefix"] == "02-hypothesis"


def test_materialize_revision_mapping(monkeypatch, tmp_path: Path) -> None:
    project_dir = _project(tmp_path)
    hdir = _legacy_hypothesis(project_dir, "000001")
    migrate_root_revisions(project_dir)
    _write_revision(hdir, 2)

    def fake_invocation(*args, **kwargs):
        return AiResponse(text=json.dumps({"code": _group_code()}), metadata={})

    monkeypatch.setattr("tml.hypotheses.materialize.run_model_invocation", fake_invocation)

    created = materialize_missing(project_dir, mode="autogluon", hypothesis_id="000001", revision=2)

    assert created == 1
    manifest = read_yaml(hdir / "manifest.yaml")
    files = manifest["materializations"]["autogluon"]["files"]
    assert files[-1]["file"] == "autogluon-001.py"
    assert files[-1]["revision"] == 2
    assert "source" not in files[-1]
    assert "feature_group" not in manifest


def test_run_records_revision_metadata_and_promotes_better(monkeypatch, tmp_path: Path) -> None:
    project_dir = _project(tmp_path)
    hdir = _legacy_hypothesis(project_dir, "000001")
    migrate_root_revisions(project_dir)
    _write_revision(hdir, 2)
    _write_group_code(hdir / "materializations" / "autogluon-001.py")
    manifest = read_yaml(hdir / "manifest.yaml")
    manifest["materializations"]["autogluon"] = {
        "active": "autogluon-001.py",
        "files": [
            {
                "file": "autogluon-001.py",
                "revision": 2,
                "sha256": "placeholder",
                "created_at": "2026-01-01T00:00:00",
            }
        ],
    }
    write_yaml(hdir / "manifest.yaml", manifest)
    reindex_project(project_dir, project_dir / "tml.db")

    def fake_run(*args, **kwargs):
        return ExecutionResult(
            status="ok",
            returncode=0,
            stdout="",
            stderr="",
            metric=0.9,
            maximize=True,
            payload={"run_stats": {"eval_metric": "accuracy"}},
        )

    monkeypatch.setattr("tml.hypotheses.run.run_python_script", fake_run)

    ran = run_missing(project_dir, mode="autogluon", hypothesis_id="000001", revision=2)

    assert ran == ["000001:2"]
    rows = root_run_rows(project_dir, mode="autogluon", profile_id="autogluon-root-start-v1", hypothesis_id="000001")
    assert rows[0]["hypothesis_revision"] == 2
    assert rows[0]["materialization_file"] == "autogluon-001.py"
    manifest = read_yaml(hdir / "manifest.yaml")
    assert manifest["materializations"]["autogluon"]["active"] == "autogluon-001.py"


def test_reindex_rebuilds_revision_tables(tmp_path: Path) -> None:
    project_dir = _project(tmp_path)
    hdir = _legacy_hypothesis(project_dir, "000001")
    migrate_root_revisions(project_dir)
    _write_revision(hdir, 2)
    _write_group_code(hdir / "materializations" / "autogluon-001.py")
    manifest = read_yaml(hdir / "manifest.yaml")
    manifest["materializations"]["autogluon"] = {
        "active": "autogluon-001.py",
        "files": [
            {
                "file": "autogluon-001.py",
                "revision": 2,
                "sha256": "placeholder",
                "created_at": "2026-01-01T00:00:00",
            }
        ],
    }
    write_yaml(hdir / "manifest.yaml", manifest)

    reindex_project(project_dir, project_dir / "tml.db")

    rows = materialization_rows(project_dir, mode="autogluon", hypothesis_id="000001")
    assert rows[0]["hypothesis_revision"] == 2
    with connect(project_dir / "tml.db") as conn:
        revisions = conn.execute(
            "SELECT hypothesis_id, revision, path FROM hypothesis_revisions ORDER BY revision"
        ).fetchall()
    assert [(row["hypothesis_id"], row["revision"]) for row in revisions] == [("000000", 1), ("000001", 1), ("000001", 2)]


def _project(tmp_path: Path) -> Path:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    write_yaml(
        project_dir / "project.yaml",
        {
            "schema_version": 1,
            "project_id": "test-project",
            "kind": "kaggle",
            "target": {"maximize": True},
            "root": {"active_mode": "autogluon"},
        },
    )
    (project_dir / "hypotheses").mkdir()
    (project_dir / "runs").mkdir()
    (project_dir / "prompts" / "root").mkdir(parents=True)
    (project_dir / "prompts" / "blocks").mkdir(parents=True)
    (project_dir / "prompts" / "root" / "revise-hypothesis.md.j2").write_text("revise {{ hypothesis_id }}", encoding="utf-8")
    (project_dir / "prompts" / "root" / "materialize-autogluon.md.j2").write_text("materialize", encoding="utf-8")
    (project_dir / "task.md").write_text("# Task\n", encoding="utf-8")
    return project_dir


def _legacy_hypothesis(project_dir: Path, hypothesis_id: str) -> Path:
    hdir = project_dir / "hypotheses" / hypothesis_id
    (hdir / "materializations").mkdir(parents=True)
    write_yaml(hdir / "hypothesis.yaml", _revision_payload(hypothesis_id, 1))
    write_yaml(
        hdir / "manifest.yaml",
        {
            "materializations": {
                "autogluon": {
                    "active": "autogluon-001.py",
                    "sha256": "legacy",
                    "created_at": "2026-01-01T00:00:00",
                }
            },
            "feature_group": {"logical_name": "example_group"},
        },
    )
    return hdir


def _write_revision(hdir: Path, revision: int) -> None:
    write_yaml(hdir / f"{revision:02d}-hypothesis.yaml", _revision_payload(hdir.name, revision))
    manifest = read_yaml(hdir / "manifest.yaml")
    manifest["hypothesis"]["latest_revision"] = max(int(manifest["hypothesis"]["latest_revision"]), revision)
    manifest["revisions"][revision] = {"file": f"{revision:02d}-hypothesis.yaml", "prefix": f"{revision:02d}-hypothesis"}
    write_yaml(hdir / "manifest.yaml", manifest)


def _revision_payload(hypothesis_id: str, revision: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "hypothesis_id": hypothesis_id,
        "revision": revision,
        "enabled": True,
        "created_at": "2026-01-01T00:00:00",
        "title": f"Example {revision}",
        "group_name": "example_group",
        "family": "example",
        "summary": f"Summary {revision}.",
        "depends_on": [],
        "strategy": f"Strategy {revision}.",
        "expected_signal": "Signal.",
        "risk": "Risk.",
    }


def _write_group_code(path: Path) -> None:
    path.write_text(_group_code(), encoding="utf-8")


def _group_code() -> str:
    return (
        "from __future__ import annotations\n\n"
        "import pandas as pd\n\n"
        "def add_example(raw, deps, aux):\n"
        "    _ = (raw, deps, aux)\n"
        "    return pd.DataFrame(index=raw.index)\n\n"
        "FEATURE_GROUPS = [{'name': 'example', 'fn': add_example, 'depends_on': []}]\n"
    )
