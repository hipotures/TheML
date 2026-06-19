from __future__ import annotations

import sqlite3
import shutil
import sys
from pathlib import Path

import yaml
from typer.testing import CliRunner

from tml.cli.main import app


runner = CliRunner()


def invoke(root: Path, *args: str, env: dict[str, str] | None = None):
    ensure_root_resources(root)
    command_env = {"TML_CWD": str(root)}
    if env:
        command_env.update(env)
    return runner.invoke(app, list(args), env=command_env)


def ensure_root_resources(root: Path) -> None:
    source_root = Path(__file__).resolve().parents[1]
    for name in ("profiles", "prompts"):
        target = root / name
        if not target.exists():
            shutil.copytree(source_root / name, target)


def fake_kaggle_cli(tmp_path: Path, *, sample_submission: str | None = None) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "kaggle"
    script.write_text(
        """#!{python}
from pathlib import Path
import sys

args_path = Path(sys.argv[0]).with_name("kaggle.args")
previous = args_path.read_text() + "\\n---\\n" if args_path.exists() else ""
args_path.write_text(previous + "\\n".join(sys.argv[1:]))
if "pages" in sys.argv:
    import csv
    page = sys.argv[sys.argv.index("--page-name") + 1]
    pages = {{
        "abstract": "**Your Goal:** Predict the stellar class.",
        "data-description": "**train.csv** - the training set, with `class` as target\\n**test.csv** - predict `class`",
        "Evaluation": "Submissions are evaluated on balanced accuracy between the predicted class and observed target. Predict a class label (GALAXY, STAR, QSO).",
    }}
    writer = csv.DictWriter(sys.stdout, fieldnames=["name", "content"])
    writer.writeheader()
    writer.writerow({{"name": page, "content": pages.get(page, "")}})
    raise SystemExit(0)
out = Path(sys.argv[sys.argv.index("-p") + 1])
out.mkdir(parents=True, exist_ok=True)
sample = {sample_submission!r}
if sample is not None:
    import zipfile
    with zipfile.ZipFile(out / "playground-series-s6e6.zip", "w") as archive:
        archive.writestr("sample_submission.csv", sample)
""".format(python=sys.executable, sample_submission=sample_submission),
        encoding="utf-8",
    )
    script.chmod(0o755)
    return {"PATH": str(bin_dir)}


def test_init_project_reports_paths_and_downloaded_data(tmp_path: Path):
    env = fake_kaggle_cli(
        tmp_path,
        sample_submission="id,target\n1,0\n",
    )

    result = invoke(tmp_path, "init", "project", "playground-series-s6e6", env=env)

    assert result.exit_code == 0, result.output
    project_dir = "projects/kaggle/playground-series-s6e6"
    assert "Project initialized" in result.output
    assert "Initialized project" in result.output
    assert project_dir in result.output
    assert "Project config" in result.output
    assert f"{project_dir}/project.yaml" in result.output
    assert "Task file" in result.output
    assert f"{project_dir}/task.md" in result.output
    assert "Data dir" in result.output
    assert f"{project_dir}/data" in result.output
    assert "Kaggle data" in result.output
    assert "downloaded" in result.output
    assert "sample_submission.csv" in result.output
    assert "Next: uv run tml project use playground-series-s6e6" in result.output
    assert str(tmp_path) not in result.output
    project_yaml = tmp_path / "projects" / "kaggle" / "playground-series-s6e6" / "project.yaml"
    project = yaml.safe_load(project_yaml.read_text(encoding="utf-8"))
    assert project["target"]["target_column"] == "target"
    assert not (project_yaml.parent / "data" / "sample_submission.csv").exists()
    assert (project_yaml.parent / "data" / "sample_submission.csv.gz").exists()


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
    assert "target_column: class" in project_config
    assert "autogluon_metric: balanced_accuracy" in project_config
    assert (project_dir / "task.md").exists()
    assert not (project_dir / "profiles").exists()
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
    assert "models" not in project


def test_init_project_uses_root_autogluon_profiles_without_copying_defaults(tmp_path: Path):
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

    project_dir = tmp_path / "projects" / "kaggle" / "playground-series-s6e6"
    project = yaml.safe_load((project_dir / "project.yaml").read_text(encoding="utf-8"))
    assert project["root"]["active_profiles"]["autogluon"] == "ag-medium-10m-v1"

    assert not list((project_dir / "profiles" / "autogluon").glob("*.yaml"))
    assert not list((project_dir / "profiles" / "legacy").glob("*.yaml"))

    root_profile_dir = Path("profiles/autogluon")
    profile_files = {path.name for path in root_profile_dir.glob("*.yaml")}
    assert "ag-medium-10m-v1.yaml" in profile_files
    assert "ag-fast-boost-v1.yaml" in profile_files
    assert "ag-s6e6-boost-gpu-ens-cv3-v1.yaml" in profile_files
    assert "ag-full-best-30m-gpu-v1.yaml" in profile_files
    assert Path("profiles/legacy/legacy-root-start-v1.yaml").exists()

    default_profile = yaml.safe_load((root_profile_dir / "ag-medium-10m-v1.yaml").read_text(encoding="utf-8"))
    assert default_profile["profile_id"] == "ag-medium-10m-v1"
    assert default_profile["mode"] == "autogluon"
    assert default_profile["presets"] == "medium_quality"
    assert default_profile["time_limit"] == 600
    assert default_profile["included_model_types"] == ["XGB", "GBM", "CAT"]
    assert default_profile["use_gpu"] is False
    assert "profiles" not in default_profile

    gpu_cv_profile = yaml.safe_load(
        (root_profile_dir / "ag-s6e6-boost-gpu-ens-cv3-v1.yaml").read_text(encoding="utf-8")
    )
    assert gpu_cv_profile["time_limit"] == 900
    assert gpu_cv_profile["fit_args"]["num_bag_folds"] == 3
    assert gpu_cv_profile["hyperparameters"]["XGB"][0]["device"] == "cuda"


def test_autogluon_profiles_are_root_yaml_not_project_code():
    project_source = Path("src/tml/core/project.py").read_text(encoding="utf-8")
    assert "AUTOGLUON_PROFILES" not in project_source
    assert Path("profiles/autogluon/ag-medium-10m-v1.yaml").exists()
    assert Path("profiles/autogluon/ag-s6e6-boost-gpu-ens-cv3-v1.yaml").exists()


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


def test_init_project_uses_kaggle_pages_prompt_to_fill_task_and_config(tmp_path: Path):
    env = fake_kaggle_cli(
        tmp_path,
        sample_submission="id,class\n577347,STAR\n",
    )

    result = invoke(tmp_path, "init", "project", "playground-series-s6e6", env=env)
    assert result.exit_code == 0, result.output

    project_dir = tmp_path / "projects" / "kaggle" / "playground-series-s6e6"
    task_text = (project_dir / "task.md").read_text(encoding="utf-8")
    assert "## Goal" in task_text
    assert "Predict the stellar class" in task_text
    assert "## Evaluation" in task_text
    assert "balanced accuracy" in task_text
    assert "## Data description" in task_text

    project = yaml.safe_load((project_dir / "project.yaml").read_text(encoding="utf-8"))
    assert project["target"]["id_column"] == "id"
    assert project["target"]["target_column"] == "class"
    assert project["target"]["problem_type"] == "multiclass"
    assert project["target"]["autogluon_metric"] == "balanced_accuracy"
    assert project["target"]["sklearn_metric"] == "sklearn.metrics.balanced_accuracy_score"
    assert project["target"]["maximize"] is True
    assert project["target"]["submission_kind"] == "labels"
    assert project["target"]["metric_description"]

    metadata_log = project_dir / "logs" / "project-metadata"
    assert (metadata_log / "request.md").exists()
    assert (metadata_log / "request.json").exists()
    assert (metadata_log / "response.md").exists()
    assert (metadata_log / "response.json").exists()
    response = yaml.safe_load((metadata_log / "response.md").read_text(encoding="utf-8"))
    assert response["target"]["autogluon_metric"] == "balanced_accuracy"
    assert response["target"]["sklearn_metric"] == "sklearn.metrics.balanced_accuracy_score"


def test_prompt_render_project_metadata_uses_metadata_template(tmp_path: Path):
    env = fake_kaggle_cli(
        tmp_path,
        sample_submission="id,class\n577347,STAR\n",
    )
    assert invoke(tmp_path, "init", "project", "playground-series-s6e6", env=env).exit_code == 0
    assert invoke(tmp_path, "project", "use", "playground-series-s6e6").exit_code == 0

    result = invoke(tmp_path, "prompt", "render", "project", "metadata", env=env)
    assert result.exit_code == 0, result.output

    request_path = Path(result.output.strip())
    assert request_path.exists()
    request_text = request_path.read_text(encoding="utf-8")
    assert "You are configuring a Kaggle machine learning project." in request_text
    assert "Competition slug: playground-series-s6e6" in request_text
    assert "balanced accuracy" in request_text
    assert "Sample submission header: ['id', 'class']" in request_text
    assert "Generate ROOT hypotheses" not in request_text

    request_json = yaml.safe_load((request_path.parent / "request.json").read_text(encoding="utf-8"))
    assert request_json["template_id"] == "project.metadata"


def test_prompt_probe_project_metadata_uses_metadata_role(tmp_path: Path):
    env = fake_kaggle_cli(
        tmp_path,
        sample_submission="id,class\n577347,STAR\n",
    )
    assert invoke(tmp_path, "init", "project", "playground-series-s6e6", env=env).exit_code == 0
    assert invoke(tmp_path, "project", "use", "playground-series-s6e6").exit_code == 0

    result = invoke(tmp_path, "prompt", "probe", "project", "metadata", "tmp=true", env=env)
    assert result.exit_code == 0, result.output

    out_dir = Path(result.output.strip())
    request_json = yaml.safe_load((out_dir / "request.json").read_text(encoding="utf-8"))
    response = yaml.safe_load((out_dir / "response.md").read_text(encoding="utf-8"))
    assert request_json["template_id"] == "project.metadata"
    assert response["target"]["autogluon_metric"] == "balanced_accuracy"
    assert response["target"]["sklearn_metric"] == "sklearn.metrics.balanced_accuracy_score"


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
        stored_paths = [
            row[0]
            for table in ("projects", "profiles", "hypotheses", "runs", "nodes", "artifacts")
            for row in conn.execute(f"select path from {table}").fetchall()
        ]
        assert stored_paths
        assert all(not Path(path).is_absolute() for path in stored_paths)

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

    render = invoke(tmp_path, "prompt", "render", "root", "hypothesis")
    assert render.exit_code == 0, render.output
    rendered_path = Path(render.output.strip().splitlines()[-1])
    assert rendered_path.exists()
    assert "ROOT hypothesis" in rendered_path.read_text(encoding="utf-8")

    code_render = invoke(tmp_path, "prompt", "render", "1", "code", "tmp=true")
    assert code_render.exit_code == 0, code_render.output
    code_rendered_path = Path(code_render.output.strip().splitlines()[-1])
    assert "preprocess(df)" in code_rendered_path.read_text(encoding="utf-8")

    probe = invoke(tmp_path, "prompt", "probe", "root", "hypothesis")
    assert probe.exit_code == 0, probe.output
    probe_path = Path(probe.output.strip().splitlines()[-1])
    assert (probe_path / "request.md").exists()
    assert (probe_path / "response.json").exists()
    request_json = yaml.safe_load((probe_path / "request.json").read_text(encoding="utf-8"))
    assert request_json["project_dir"] == "."
    assert not Path(request_json["template_path"]).is_absolute()
    assert str(tmp_path) not in (probe_path / "request.md").read_text(encoding="utf-8")

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
    failed_text = (node_dirs[0] / "failed.yaml").read_text(encoding="utf-8")
    assert "data/train.csv" in failed_text
    assert str(tmp_path) not in failed_text

    second = invoke(tmp_path, "root", "ensure", "count=1")
    assert second.exit_code == 0, second.output
    node_dirs_after_second = list((project_dir / "runs").glob("*/artifacts/*"))
    assert len(node_dirs_after_second) == 1
