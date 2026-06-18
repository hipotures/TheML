from __future__ import annotations

from datetime import datetime
from pathlib import Path

from tml.ai import AiRequest, client_for_model
from tml.core.config import active_mode, load_project_config
from tml.core.ids import timestamp_id
from tml.core.metadata import render_project_metadata_prompt
from tml.prompts.context import project_prompt_context
from tml.prompts.renderer import render_template
from tml.utils.atomic import atomic_write_json, atomic_write_text


def render_prompt(
    project_dir: Path,
    *,
    target: str | None = None,
    stage: str | None = None,
    tmp_root: Path | None = None,
) -> Path:
    out_dir = _tmp_dir(project_dir, tmp_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = _render_for_target(project_dir, target=target, stage=stage)
    path = out_dir / "request.md"
    atomic_write_text(path, rendered["rendered"])
    atomic_write_json(out_dir / "request.json", _request_json(project_dir, rendered, "render"))
    return path


def probe_prompt(
    project_dir: Path,
    *,
    tmp: bool,
    target: str | None = None,
    stage: str | None = None,
    model_override: str | None = None,
    tmp_root: Path | None = None,
) -> Path:
    out_dir = _tmp_dir(project_dir, tmp_root) if tmp else project_dir / "prompt-lab" / timestamp_id()
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_project_config(project_dir)
    role = _role_for_target(target, stage)
    models = config.get("models", {}) if isinstance(config.get("models"), dict) else {}
    model = model_override or str(models.get(role) or models.get("hypothesis") or "mock")
    rendered = _render_for_target(project_dir, target=target, stage=stage)
    atomic_write_text(out_dir / "request.md", rendered["rendered"])
    atomic_write_json(out_dir / "request.json", _request_json(project_dir, rendered, "probe", model=model))
    response = client_for_model(model).call(AiRequest(role=role, model=model, prompt=rendered["rendered"]))
    atomic_write_text(out_dir / "response.md", response.text)
    atomic_write_json(out_dir / "response.json", {"text": response.text, **response.metadata})
    return out_dir


def _render_for_target(project_dir: Path, *, target: str | None, stage: str | None) -> dict[str, str]:
    config = load_project_config(project_dir)
    if target == "project" and stage == "metadata":
        return render_project_metadata_prompt(
            project_dir,
            slug=str(config.get("kaggle_slug") or config.get("project_id") or project_dir.name),
        )
    if stage == "code":
        mode = active_mode(config)
        hypothesis = _hypothesis_for_target(project_dir, target)
        return render_template(
            project_dir,
            f"root.materialize-{mode}",
            project_prompt_context(project_dir, hypothesis=hypothesis),
        )
    return render_template(project_dir, "root.hypothesis", project_prompt_context(project_dir, count=1))


def _role_for_target(target: str | None, stage: str | None) -> str:
    if target == "project" and stage == "metadata":
        return "metadata"
    if stage == "code":
        return "code"
    return "hypothesis"


def _hypothesis_for_target(project_dir: Path, target: str | None) -> dict[str, object]:
    if target and target.isdigit():
        path = project_dir / "hypotheses" / target.zfill(6) / "hypothesis.yaml"
        if path.exists():
            from tml.utils.yaml_io import read_yaml

            return read_yaml(path)
    return {"hypothesis_id": target or "next"}


def _request_json(project_dir: Path, rendered: dict[str, str], kind: str, model: str = "mock") -> dict[str, object]:
    return {
        "kind": kind,
        "model": model,
        "provider": model,
        "messages": [{"role": "user", "content": rendered["rendered"]}],
        "template_id": rendered["template_id"],
        "template_path": rendered["template_path"],
        "template_hash": rendered["template_hash"],
        "context_hash": rendered["rendered_hash"],
        "rendered_prompt_hash": rendered["rendered_hash"],
        "project_dir": ".",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def _tmp_dir(project_dir: Path, tmp_root: Path | None) -> Path:
    root = tmp_root or Path("/tmp")
    return root / "tml" / project_dir.name / timestamp_id()
