from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from tml.ai import AiRequest, client_for_model
from tml.core.config import load_project_config
from tml.prompts.context import project_prompt_context
from tml.prompts.renderer import render_template
from tml.utils.atomic import atomic_write_json, atomic_write_text
from tml.utils.hashing import sha256_file
from tml.utils.yaml_io import read_yaml, write_yaml

from .model import hypothesis_dirs


def materialize_missing(project_dir: Path, mode: str) -> int:
    config = load_project_config(project_dir)
    model = str(config.get("models", {}).get("code", "mock"))
    client = client_for_model(model)
    created = 0
    for hdir in hypothesis_dirs(project_dir):
        mat_dir = hdir / "materializations"
        mat_dir.mkdir(parents=True, exist_ok=True)
        target = mat_dir / f"{mode}-001.py"
        if target.exists():
            continue
        hypothesis = read_yaml(hdir / "hypothesis.yaml")
        template_id = f"root.materialize-{mode}"
        rendered = render_template(
            project_dir,
            template_id,
            project_prompt_context(project_dir, hypothesis=hypothesis),
        )
        request = {
            "role": "code",
            "model": model,
            "provider": model,
            "messages": [{"role": "user", "content": rendered["rendered"]}],
            "template_id": rendered["template_id"],
            "template_path": rendered["template_path"],
            "template_hash": rendered["template_hash"],
            "rendered_prompt_hash": rendered["rendered_hash"],
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        atomic_write_json(mat_dir / f"{mode}-001.request.json", request)
        atomic_write_text(mat_dir / f"{mode}-001.request.md", rendered["rendered"])
        response = client.call(AiRequest(role="code", model=model, prompt=f"{mode}\n\n{rendered['rendered']}"))
        atomic_write_text(mat_dir / f"{mode}-001.response.md", response.text)
        atomic_write_json(mat_dir / f"{mode}-001.response.json", {"text": response.text, **response.metadata})
        code = _parse_code(response.text)
        atomic_write_text(target, code)
        _update_manifest(hdir, mode, target)
        created += 1
    return created


def _parse_code(text: str) -> str:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(parsed, dict) and isinstance(parsed.get("code"), str):
        return parsed["code"]
    return text


def _update_manifest(hdir: Path, mode: str, path: Path) -> None:
    manifest_path = hdir / "manifest.yaml"
    manifest = read_yaml(manifest_path)
    versions = manifest.setdefault("materializations", {})
    if not isinstance(versions, dict):
        versions = {}
        manifest["materializations"] = versions
    versions[mode] = {
        "active": path.name,
        "sha256": sha256_file(path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_yaml(manifest_path, manifest)
