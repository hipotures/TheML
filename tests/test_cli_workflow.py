from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import yaml
from typer.testing import CliRunner

from tml.cli.main import app


runner = CliRunner()


def invoke(root: Path, *args: str, env: dict[str, str] | None = None):
    command_env = {"TML_CWD": str(root)}
    if env:
        command_env.update(env)
    return runner.invoke(app, list(args), env=command_env)


def fake_kaggle_cli(tmp_path: Path, *, sample_submission: str | None = None) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "kaggle"
    script.write_text(
        """#!{python}
from pathlib import Path
import sys

Path(sys.argv[0]).with_name("kaggle.args").write_text("\\n".join(sys.argv[1:]))
out = Path(sys.argv[sys.argv.index("-p") + 1])
out.mkdir(parents=True, exist_ok=True)
sample = {sample_submission!r}
if sample is not None:
    (out / "sample_submission.csv").write_text(sample, encoding="utf-8")
""".format(python=sys.executable, sample_submission=sample_submission),
        encoding="utf-8",
    )
    script.chmod(0o755)
    return {"PATH": str(bin_dir)}


def test_init_project_defaults_to_kaggle_and_writes_root_config(tmp_path: Path):
    env = fake_kaggle_cli(tmp_path)

    result = invoke(tmp_path, "init", "project", "playground-series-s6e6", env=env)
    assert result.exit_code == 0, result.output

    project_dir = tmp_path / "projects" / "kaggle" / "playground-series-s6e6"
    assert not (tmp_path / "projects" / "local").exists()
    assert (tmp_path / ".gitignore").exists()
    assert (tmp_path / "tml.yaml").exists()
    root_config = (tmp_path / "tml.yaml").read_text(encoding="utf-8")
    assert "project_kind: kaggle" in root_config
    assert "download_data: true" in root_config
    assert "active_project: null" in root_config
    kaggle_args = (tmp_path / "bin" / "kaggle.args").read_text(encoding="utf-8")
    assert "competitions\ndownload\n-c\nplayground-series-s6e6" in kaggle_args
    assert (project_dir / "project.yaml").exists()
    project_config = (project_dir / "project.yaml").read_text(encoding="utf-8")
    assert "kind: kaggle" in project_config
    assert "kaggle_slug: playground-series-s6e6" in project_config
    assert "target_column: null" in project_config
    assert "metric: null" in project_config
    assert (project_dir / "task.md").exists()
    assert (project_dir / "profiles" / "root" / "autogluon-root-start-v1.yaml").exists()
    assert not (project_dir / "code").exists()

    result = invoke(tmp_path, "project", "use", "playground-series-s6e6")
    assert result.exit_code == 0, result.output
    used_config = (tmp_path / "tml.yaml").read_text(encoding="utf-8")
    assert "kind: kaggle" in used_config
    assert "slug: playground-series-s6e6" in used_config

    result = invoke(tmp_path, "root", "status")
    assert result.exit_code == 0, result.output
    assert "Active project: playground-series-s6e6" in result.output
    assert "Hypotheses: 0" in result.output


def test_init_project_reads_root_tml_yaml_defaults(tmp_path: Path):
    (tmp_path / "tml.yaml").write_text(
        """
schema_version: 1
defaults:
  project_kind: kaggle
  download_data: false
  root_mode: legacy
  prompt_output: tmp
  probe_output: prompt-lab
active_project: null
active_run: null
models:
  hypothesis: mock
  code: codex-test
  review: mock
  bugfix: mock
""",
        encoding="utf-8",
    )

    result = invoke(tmp_path, "init", "project", "playground-series-s6e6")
    assert result.exit_code == 0, result.output

    project_yaml = tmp_path / "projects" / "kaggle" / "playground-series-s6e6" / "project.yaml"
    project = yaml.safe_load(project_yaml.read_text(encoding="utf-8"))
    assert project["root"]["active_mode"] == "legacy"
    assert project["models"]["code"] == "codex-test"


def test_init_project_infers_submission_columns_when_data_exists(tmp_path: Path):
    data_dir = tmp_path / "projects" / "kaggle" / "playground-series-s6e6" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "sample_submission.csv").write_text("id,class\n1,A\n", encoding="utf-8")

    (tmp_path / "tml.yaml").write_text(
        """
schema_version: 1
defaults:
  project_kind: kaggle
  download_data: false
active_project: null
active_run: null
models:
  hypothesis: mock
  code: mock
  review: mock
  bugfix: mock
""",
        encoding="utf-8",
    )

    result = invoke(tmp_path, "init", "project", "playground-series-s6e6")
    assert result.exit_code == 0, result.output

    project_yaml = tmp_path / "projects" / "kaggle" / "playground-series-s6e6" / "project.yaml"
    project = yaml.safe_load(project_yaml.read_text(encoding="utf-8"))
    assert project["target"]["id_column"] == "id"
    assert project["target"]["target_column"] == "class"
    assert project["target"]["submission_kind"] == "labels"


def test_root_mock_workflow_reindexes_and_records_node_artifacts(tmp_path: Path):
    env = fake_kaggle_cli(tmp_path)
    assert invoke(tmp_path, "init", "project", "playground-series-s6e6", env=env).exit_code == 0
    assert invoke(tmp_path, "project", "use", "playground-series-s6e6").exit_code == 0

    result = invoke(tmp_path, "root", "generate", "count=1")
    assert result.exit_code == 0, result.output
    hypothesis = (
        tmp_path
        / "projects"
        / "kaggle"
        / "playground-series-s6e6"
        / "hypotheses"
        / "000001"
        / "hypothesis.yaml"
    )
    assert "hypothesis_id: '000001'" in hypothesis.read_text(encoding="utf-8")

    result = invoke(tmp_path, "root", "materialize", "mode=legacy")
    assert result.exit_code == 0, result.output
    materialization = hypothesis.parent / "materializations" / "legacy-001.py"
    assert "TML_RESULT_JSON" in materialization.read_text(encoding="utf-8")

    result = invoke(tmp_path, "root", "run", "mode=legacy")
    assert result.exit_code == 0, result.output
    project_dir = tmp_path / "projects" / "kaggle" / "playground-series-s6e6"
    node_dirs = list((project_dir / "runs").glob("*/artifacts/*"))
    assert len(node_dirs) == 1
    node_dir = node_dirs[0]
    assert (node_dir / "node.start.yaml").exists()
    assert (node_dir / "01-hypothesis.yaml").exists()
    assert (node_dir / "02-code.py").exists()
    assert (node_dir / "artifact-manifest.yaml").exists()
    assert (node_dir / "node.done.yaml").exists()

    result = invoke(tmp_path, "reindex")
    assert result.exit_code == 0, result.output
    db_path = project_dir / "tml.db"
    assert db_path.exists()
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("select count(*) from hypotheses").fetchone()[0] == 1
        assert conn.execute("select count(*) from materializations").fetchone()[0] == 1
        assert conn.execute("select status from nodes").fetchone()[0] == "complete"

    result = invoke(tmp_path, "root", "status")
    assert result.exit_code == 0, result.output
    assert "Hypotheses: 1" in result.output
    assert "Evaluated: 1" in result.output

    db_path.unlink()
    result = invoke(tmp_path, "reindex")
    assert result.exit_code == 0, result.output
    assert db_path.exists()


def test_prompt_render_probe_and_diff_do_not_create_nodes(tmp_path: Path):
    env = fake_kaggle_cli(tmp_path)
    assert invoke(tmp_path, "init", "project", "playground-series-s6e6", env=env).exit_code == 0
    assert invoke(tmp_path, "project", "use", "playground-series-s6e6").exit_code == 0
    assert invoke(tmp_path, "root", "generate", "count=1").exit_code == 0
    assert invoke(tmp_path, "root", "materialize", "mode=autogluon").exit_code == 0

    render = invoke(tmp_path, "prompt", "render", "tmp=true")
    assert render.exit_code == 0, render.output
    rendered_path = Path(render.output.strip().splitlines()[-1])
    assert rendered_path.exists()
    assert "ROOT hypothesis" in rendered_path.read_text(encoding="utf-8")

    code_render = invoke(tmp_path, "prompt", "render", "1", "code", "tmp=true")
    assert code_render.exit_code == 0, code_render.output
    code_rendered_path = Path(code_render.output.strip().splitlines()[-1])
    assert "preprocess(df)" in code_rendered_path.read_text(encoding="utf-8")

    probe = invoke(tmp_path, "prompt", "probe", "tmp=true")
    assert probe.exit_code == 0, probe.output
    probe_path = Path(probe.output.strip().splitlines()[-1])
    assert (probe_path / "request.md").exists()
    assert (probe_path / "response.json").exists()

    assert not list((tmp_path / "projects" / "kaggle" / "playground-series-s6e6" / "runs").glob("*"))

    assert invoke(tmp_path, "root", "run").exit_code == 0
    diff = invoke(tmp_path, "prompt", "diff", "1", "code")
    assert diff.exit_code == 0, diff.output
    assert "saved request differs" in diff.output or "saved request matches" in diff.output


def test_root_ensure_is_idempotent_for_failed_autogluon_attempt(tmp_path: Path):
    env = fake_kaggle_cli(tmp_path)
    assert invoke(tmp_path, "init", "project", "playground-series-s6e6", env=env).exit_code == 0
    assert invoke(tmp_path, "project", "use", "playground-series-s6e6").exit_code == 0

    first = invoke(tmp_path, "root", "ensure", "count=1")
    assert first.exit_code == 0, first.output
    project_dir = tmp_path / "projects" / "kaggle" / "playground-series-s6e6"
    node_dirs = list((project_dir / "runs").glob("*/artifacts/*"))
    assert len(node_dirs) == 1
    assert (node_dirs[0] / "failed.yaml").exists()

    second = invoke(tmp_path, "root", "ensure", "count=1")
    assert second.exit_code == 0, second.output
    node_dirs_after_second = list((project_dir / "runs").glob("*/artifacts/*"))
    assert len(node_dirs_after_second) == 1
