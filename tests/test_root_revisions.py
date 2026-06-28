from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from tml.ai.client import AiResponse
from tml.cli.main import app
from tml.db.connect import connect
from tml.db.reindex import reindex_project
from tml.db.state import materialization_rows, revision_status_rows, root_run_rows
from tml.execution.result import ExecutionResult
from tml.hypotheses.materialize import materialize_missing
from tml.hypotheses.revisions import migrate_root_revisions
from tml.hypotheses.revise import revise_root_hypothesis, root_revise_batch_plan
from tml.hypotheses.run import run_missing
from tml.utils.hashing import sha256_file
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


def test_root_revise_batch_plan_selects_enabled_below_max_in_fair_order(tmp_path: Path) -> None:
    project_dir = _project(tmp_path)
    for hypothesis_id in ("000000", "000001", "000002", "000003", "000004", "000005", "000006"):
        _legacy_hypothesis(project_dir, hypothesis_id)
    migrate_root_revisions(project_dir)
    _write_revision(project_dir / "hypotheses" / "000001", 2)
    _write_revision(project_dir / "hypotheses" / "000004", 2)
    _set_enabled(project_dir / "hypotheses" / "000002", False)

    plan = root_revise_batch_plan(project_dir, count=3, max_revision=2)

    assert [(item.hypothesis_id, item.latest_revision, item.next_revision) for item in plan.items] == [
        ("000003", 1, 2),
        ("000005", 1, 2),
        ("000006", 1, 2),
    ]
    assert plan.requested_count == 3
    assert plan.planned_count == 3
    assert plan.max_revision == 2


def test_root_revise_batch_plan_replans_after_candidates_reach_max(tmp_path: Path) -> None:
    project_dir = _project(tmp_path)
    for hypothesis_id in ("000001", "000002", "000003", "000004"):
        _legacy_hypothesis(project_dir, hypothesis_id)
    migrate_root_revisions(project_dir)
    _write_revision(project_dir / "hypotheses" / "000001", 2)
    _write_revision(project_dir / "hypotheses" / "000002", 2)

    plan = root_revise_batch_plan(project_dir, count=3, max_revision=2)

    assert [item.hypothesis_id for item in plan.items] == ["000003", "000004"]
    assert plan.planned_count == 2


def test_root_revise_batch_plan_without_max_orders_by_latest_revision(tmp_path: Path) -> None:
    project_dir = _project(tmp_path)
    for hypothesis_id in ("000001", "000002", "000003", "000004"):
        _legacy_hypothesis(project_dir, hypothesis_id)
    migrate_root_revisions(project_dir)
    _write_revision(project_dir / "hypotheses" / "000001", 2)
    _write_revision(project_dir / "hypotheses" / "000003", 2)
    _write_revision(project_dir / "hypotheses" / "000003", 3)

    plan = root_revise_batch_plan(project_dir, count=3, max_revision=None)

    assert [(item.hypothesis_id, item.latest_revision, item.next_revision) for item in plan.items] == [
        ("000002", 1, 2),
        ("000004", 1, 2),
        ("000001", 2, 3),
    ]


def test_revise_single_hypothesis_stops_at_max_revision(monkeypatch, tmp_path: Path) -> None:
    project_dir = _project(tmp_path)
    hdir = _legacy_hypothesis(project_dir, "000005")
    migrate_root_revisions(project_dir)
    responses = [_revision_response("Two"), _revision_response("Three"), _revision_response("Four")]

    def fake_invocation(*args, **kwargs):
        payload = responses.pop(0)
        artifact_dir = kwargs["artifact_dir"]
        prefix = kwargs["response_prefix"]
        (artifact_dir / f"{prefix}.request.md").write_text("request", encoding="utf-8")
        (artifact_dir / f"{prefix}.response.md").write_text(json.dumps(payload), encoding="utf-8")
        return AiResponse(text=json.dumps(payload), metadata={})

    monkeypatch.setattr("tml.hypotheses.revise.run_model_invocation", fake_invocation)

    created = revise_root_hypothesis(project_dir, hypothesis_id="000005", count=3, max_revision=4)

    assert created == [
        hdir / "02-hypothesis.yaml",
        hdir / "03-hypothesis.yaml",
        hdir / "04-hypothesis.yaml",
    ]
    assert not (hdir / "05-hypothesis.yaml").exists()
    assert responses == []


def test_revise_single_hypothesis_at_max_creates_nothing(monkeypatch, tmp_path: Path) -> None:
    project_dir = _project(tmp_path)
    hdir = _legacy_hypothesis(project_dir, "000005")
    migrate_root_revisions(project_dir)
    _write_revision(hdir, 2)

    def fake_invocation(*args, **kwargs):
        raise AssertionError("revise should not invoke the model when max is already reached")

    monkeypatch.setattr("tml.hypotheses.revise.run_model_invocation", fake_invocation)

    created = revise_root_hypothesis(project_dir, hypothesis_id="000005", count=3, max_revision=2)

    assert created == []
    assert not (hdir / "03-hypothesis.yaml").exists()


def test_root_revise_cli_prints_planned_count_before_confirmation(monkeypatch, tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    project_dir = _workspace_project(workspace)
    for hypothesis_id in ("000001", "000002", "000003"):
        _legacy_hypothesis(project_dir, hypothesis_id)
    migrate_root_revisions(project_dir)
    _write_revision(project_dir / "hypotheses" / "000001", 2)
    monkeypatch.setenv("TML_CWD", str(workspace))
    runner = CliRunner()

    result = runner.invoke(app, ["root", "revise", "count=10", "max=2"], input="n\n")

    assert result.exit_code == 0
    assert "Planned revisions" in result.output
    assert "2" in result.output
    assert "000002:2" in result.output
    assert "000003:2" in result.output
    assert "Create 2 ROOT revisions?" in result.output
    assert not (project_dir / "hypotheses" / "000002" / "02-hypothesis.yaml").exists()
    assert not (project_dir / "hypotheses" / "000003" / "02-hypothesis.yaml").exists()


def test_root_revise_cli_does_not_prompt_when_no_candidates(monkeypatch, tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    project_dir = _workspace_project(workspace)
    for hypothesis_id in ("000001", "000002"):
        hdir = _legacy_hypothesis(project_dir, hypothesis_id)
        migrate_root_revisions(project_dir)
        _write_revision(hdir, 2)
    monkeypatch.setenv("TML_CWD", str(workspace))
    runner = CliRunner()

    result = runner.invoke(app, ["root", "revise", "count=10", "max=2"])

    assert result.exit_code == 0
    assert "Planned revisions" in result.output
    assert "0" in result.output
    assert "No ROOT revisions to create" in result.output
    assert "Create " not in result.output


def test_root_revise_help_lists_add_parameters_without_delete_details() -> None:
    result = CliRunner().invoke(app, ["root", "revise", "--help"])

    assert result.exit_code == 0
    assert "max=<M>" in result.output
    assert "delete" not in result.output
    assert "rev=<N>" not in result.output
    assert "revision=<N>" not in result.output


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
    _write_group_code(hdir / "materializations" / "autogluon-002.py", suffix="# second materialization")
    component_hash = sha256_file(hdir / "materializations" / "autogluon-002.py")
    manifest["materializations"]["autogluon"]["files"].append(
        {
            "file": "autogluon-002.py",
            "revision": 2,
            "sha256": component_hash,
            "created_at": "2026-01-01T00:00:00",
        }
    )
    write_yaml(hdir / "manifest.yaml", manifest)
    code_hash = sha256_file(hdir / "materializations" / "autogluon-001.py")
    run_dir = project_dir / "runs" / "20260101T000000-test"
    node_dir = run_dir / "artifacts" / "20260101T000001-node"
    branch_node_dir = run_dir / "artifacts" / "20260101T000002-branch"
    failed_root_dir = run_dir / "artifacts" / "20260101T000003-root-failed"
    node_dir.mkdir(parents=True)
    branch_node_dir.mkdir(parents=True)
    failed_root_dir.mkdir(parents=True)
    write_yaml(run_dir / "run.yaml", {"run_id": run_dir.name})
    write_yaml(
        node_dir / "node.start.yaml",
        {
            "run_id": run_dir.name,
            "step": 1,
            "kind": "root",
            "hypothesis_id": "000001",
            "mode": "autogluon",
            "profile_id": "autogluon-root-start-v1",
            "created_at": "2026-01-01T00:00:00",
        },
    )
    write_yaml(node_dir / "node.done.yaml", {"created_at": "2026-01-01T00:00:10"})
    write_yaml(
        node_dir / "artifact-manifest.yaml",
        {
            "schema_version": 1,
            "node_id": node_dir.name,
            "hypothesis_id": "000001",
            "mode": "autogluon",
            "profile_id": "autogluon-root-start-v1",
            "code_hash": code_hash,
            "metric": 0.8,
        },
    )
    write_yaml(
        branch_node_dir / "node.start.yaml",
        {
            "run_id": run_dir.name,
            "step": 2,
            "kind": "branch",
            "branch_id": "B000001",
            "mode": "autogluon",
            "profile_id": "autogluon-root-start-v1",
            "created_at": "2026-01-01T00:00:20",
        },
    )
    write_yaml(branch_node_dir / "failed.yaml", {"created_at": "2026-01-01T00:00:25", "status": "failed"})
    write_yaml(
        branch_node_dir / "01-branch.yaml",
        {
            "schema_version": 1,
            "branch_id": "B000001",
            "mode": "autogluon",
            "components": [
                {
                    "role": "source",
                    "source_type": "hypothesis",
                    "source_id": "000001",
                    "mode": "autogluon",
                    "file": "autogluon-002.py",
                    "code_hash": component_hash,
                    "path": "hypotheses/000001/materializations/autogluon-002.py",
                }
            ],
        },
    )
    write_yaml(
        failed_root_dir / "node.start.yaml",
        {
            "run_id": run_dir.name,
            "step": 3,
            "kind": "root",
            "hypothesis_id": "000001",
            "mode": "autogluon",
            "profile_id": "autogluon-root-start-v1",
            "created_at": "2026-01-01T00:00:30",
        },
    )
    write_yaml(
        failed_root_dir / "failed.yaml",
        {
            "created_at": "2026-01-01T00:00:35",
            "hypothesis_id": "000001",
            "mode": "autogluon",
            "profile_id": "autogluon-root-start-v1",
            "code_hash": component_hash,
            "status": "failed",
        },
    )

    reindex_project(project_dir, project_dir / "tml.db")

    rows = materialization_rows(project_dir, mode="autogluon", hypothesis_id="000001")
    assert rows[0]["hypothesis_revision"] == 2
    with connect(project_dir / "tml.db") as conn:
        evaluations = conn.execute(
            "SELECT hypothesis_revision, materialization_file, metric FROM evaluations WHERE hypothesis_id='000001'"
        ).fetchall()
        revisions = conn.execute(
            "SELECT hypothesis_id, revision, path FROM hypothesis_revisions ORDER BY revision"
        ).fetchall()
        run_components = conn.execute(
            "SELECT node_id, source_id, file, code_hash FROM run_components WHERE source_id='000001'"
        ).fetchall()
    status_rows = revision_status_rows(
        project_dir,
        hypothesis_id="000001",
        mode="autogluon",
        profile_id="autogluon-root-start-v1",
    )
    assert [(row["hypothesis_revision"], row["materialization_file"], row["metric"]) for row in evaluations] == [
        (2, "autogluon-001.py", 0.8),
        (2, "autogluon-002.py", None),
    ]
    assert [(row["node_id"], row["source_id"], row["file"], row["code_hash"]) for row in run_components] == [
        ("20260101T000002-branch", "000001", "autogluon-002.py", component_hash)
    ]
    component_rows = [row for row in status_rows if row["materialization_file"] == "autogluon-002.py"]
    assert component_rows[0]["component_status"] == "failed"
    assert component_rows[0]["component_statuses"] == "failed"
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
            "root": {"active_mode": "autogluon", "active_profiles": {"autogluon": "autogluon-root-start-v1"}},
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


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_yaml(
        workspace / "tml.yaml",
        {
            "schema_version": 1,
            "active_project": {"kind": "kaggle", "slug": "demo"},
            "models": {"hypothesis": "mock"},
            "providers": {"mock": {"kind": "mock"}},
        },
    )
    return workspace


def _workspace_project(workspace: Path) -> Path:
    project_dir = workspace / "projects" / "kaggle" / "demo"
    project_dir.mkdir(parents=True)
    write_yaml(
        project_dir / "project.yaml",
        {
            "schema_version": 1,
            "project_id": "demo",
            "kind": "kaggle",
            "target": {"maximize": True},
            "root": {"active_mode": "autogluon", "active_profiles": {"autogluon": "autogluon-root-start-v1"}},
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


def _revision_response(title: str) -> dict[str, object]:
    return {
        "operation": "revise_existing_root_hypothesis",
        "decision": "revise",
        "title": title,
        "group_name": "example_group",
        "family": "example",
        "summary": f"{title} summary.",
        "depends_on": [],
        "strategy": f"{title} strategy.",
        "expected_signal": "May help.",
        "risk": "May not help.",
        "change_summary": f"{title} change.",
        "semantic_goal_preserved": True,
    }


def _set_enabled(hdir: Path, enabled: bool) -> None:
    revision = read_yaml(hdir / "01-hypothesis.yaml")
    revision["enabled"] = enabled
    write_yaml(hdir / "01-hypothesis.yaml", revision)


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


def _write_group_code(path: Path, *, suffix: str = "") -> None:
    text = _group_code()
    if suffix:
        text += f"\n{suffix}\n"
    path.write_text(text, encoding="utf-8")


def _group_code() -> str:
    return (
        "from __future__ import annotations\n\n"
        "import pandas as pd\n\n"
        "def add_example(raw, deps, aux):\n"
        "    _ = (raw, deps, aux)\n"
        "    return pd.DataFrame(index=raw.index)\n\n"
        "FEATURE_GROUPS = [{'name': 'example', 'fn': add_example, 'depends_on': []}]\n"
    )
