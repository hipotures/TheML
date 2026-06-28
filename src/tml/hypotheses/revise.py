from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tml.ai import ModelInvocation, run_model_invocation
from tml.ai.models import resolve_model_spec, resolve_role_model
from tml.core.config import repo_models_for_project, repo_providers_for_project, repo_root_for_project
from tml.db.state import upsert_hypothesis, upsert_hypothesis_revision, upsert_project
from tml.features.validation import validate_root_hypothesis
from tml.prompts.context import project_prompt_context
from tml.prompts.renderer import render_template

from .revisions import (
    migrate_hypothesis_dir,
    next_revision_number,
    normalize_hypothesis_id,
    revision_records,
    write_revision,
)


@dataclass(frozen=True)
class RootRevisePlan:
    hypothesis_id: str
    count: int
    next_revision: int
    role: str
    model: str
    provider: str
    provider_kind: object
    resolved_model: str | None
    reasoning_effort: str | None
    timeout_seconds: object
    sandbox: str


@dataclass(frozen=True)
class RootReviseBatchItem:
    hypothesis_id: str
    latest_revision: int
    next_revision: int


@dataclass(frozen=True)
class RootReviseBatchPlan:
    requested_count: int
    max_revision: int | None
    items: list[RootReviseBatchItem]

    @property
    def planned_count(self) -> int:
        return len(self.items)


def root_revise_plan(
    project_dir: Path,
    *,
    hypothesis_id: str,
    count: int,
    max_revision: int | None = None,
) -> RootRevisePlan:
    upsert_project(project_dir)
    hid = normalize_hypothesis_id(hypothesis_id)
    hdir = project_dir / "hypotheses" / hid
    migrate_hypothesis_dir(project_dir, hdir)
    models = repo_models_for_project(project_dir)
    model, role_options = resolve_role_model(models, "hypothesis")
    providers = repo_providers_for_project(project_dir)
    spec = resolve_model_spec(model, providers)
    provider_config = {**(spec.provider_config or {}), **role_options}
    return RootRevisePlan(
        hypothesis_id=hid,
        count=_effective_revision_count(hdir, count=count, max_revision=max_revision),
        next_revision=next_revision_number(hdir),
        role="hypothesis",
        model=model,
        provider=spec.provider,
        provider_kind=provider_config.get("kind"),
        resolved_model=spec.model,
        reasoning_effort=spec.reasoning_effort,
        timeout_seconds=provider_config.get("timeout_seconds"),
        sandbox="read_only",
    )


def root_revise_batch_plan(project_dir: Path, *, count: int, max_revision: int | None = None) -> RootReviseBatchPlan:
    upsert_project(project_dir)
    candidates: list[RootReviseBatchItem] = []
    for hdir in sorted((project_dir / "hypotheses").glob("*")):
        if not hdir.is_dir() or hdir.name == "000000":
            continue
        migrate_hypothesis_dir(project_dir, hdir)
        records = revision_records(hdir)
        if not records:
            continue
        latest = records[-1]
        if not latest.payload.get("enabled", True):
            continue
        if max_revision is not None and latest.revision >= max_revision:
            continue
        candidates.append(
            RootReviseBatchItem(
                hypothesis_id=hdir.name,
                latest_revision=latest.revision,
                next_revision=latest.revision + 1,
            )
        )
    candidates.sort(key=lambda item: (item.latest_revision, item.hypothesis_id))
    return RootReviseBatchPlan(
        requested_count=count,
        max_revision=max_revision,
        items=candidates[:count],
    )


def revise_root_hypothesis(
    project_dir: Path,
    *,
    hypothesis_id: str,
    count: int,
    max_revision: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[Path]:
    upsert_project(project_dir)
    hid = normalize_hypothesis_id(hypothesis_id)
    hdir = project_dir / "hypotheses" / hid
    migrate_hypothesis_dir(project_dir, hdir)
    models = repo_models_for_project(project_dir)
    model, role_options = resolve_role_model(models, "hypothesis")
    providers = repo_providers_for_project(project_dir)
    spec = resolve_model_spec(model, providers)
    provider_config = {**(spec.provider_config or {}), **role_options}
    web_search_enabled = _web_search_enabled(provider_config.get("web_search"))
    created: list[Path] = []
    for _ in range(_effective_revision_count(hdir, count=count, max_revision=max_revision)):
        previous = revision_records(hdir)
        if not previous:
            raise FileNotFoundError(f"Missing ROOT hypothesis {hid}")
        next_revision = next_revision_number(hdir)
        latest = previous[-1].payload
        if progress is not None:
            progress(f"Revising ROOT hypothesis {hid} rev {next_revision} with {model}...")
        rendered = render_template(
            project_dir,
            "root.revise-hypothesis",
            project_prompt_context(
                project_dir,
                hypothesis=latest,
                hypothesis_id=hid,
                previous_revisions=[record.payload for record in previous],
                web_search_enabled=web_search_enabled,
            ),
        )
        response = run_model_invocation(
            ModelInvocation(
                role="hypothesis",
                model=model,
                prompt=rendered["rendered"],
                template_id=rendered["template_id"],
                template_path=rendered["template_path"],
                template_hash=rendered["template_hash"],
                rendered_prompt_hash=rendered["rendered_hash"],
                cwd=repo_root_for_project(project_dir),
                sandbox="read_only",
                metadata={"hypothesis_id": hid, "revision": next_revision},
            ),
            artifact_dir=hdir,
            providers=providers,
            role_options=role_options,
            response_prefix=f"{next_revision:02d}-hypothesis",
        )
        payload = _parse_revision_response(response.text)
        if str(payload.get("decision") or "").strip().lower() == "no_change":
            break
        revision_payload = _revision_payload_from_response(payload, latest)
        validate_root_hypothesis(revision_payload)
        _validate_revision_compatibility(previous[-1].payload, revision_payload)
        path = write_revision(hdir, next_revision, revision_payload)
        upsert_hypothesis_revision(project_dir, hdir, next_revision)
        upsert_hypothesis(project_dir, hdir)
        created.append(path)
    return created


def _effective_revision_count(hdir: Path, *, count: int, max_revision: int | None) -> int:
    if max_revision is None:
        return count
    records = revision_records(hdir)
    if not records:
        return count
    remaining = max_revision - records[-1].revision
    return max(0, min(count, remaining))


def revision_status_rows(project_dir: Path, *, hypothesis_id: str, mode: str, profile_id: str) -> list[dict[str, Any]]:
    from tml.db.state import revision_status_rows as db_revision_status_rows

    hid = normalize_hypothesis_id(hypothesis_id)
    migrate_hypothesis_dir(project_dir, project_dir / "hypotheses" / hid)
    return db_revision_status_rows(project_dir, hypothesis_id=hid, mode=mode, profile_id=profile_id)


def _parse_revision_response(text: str) -> dict[str, Any]:
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Revise response must be a JSON object")
    return parsed


def _web_search_enabled(value: object) -> bool:
    if str(value or "").strip().lower() in {"live", "cached"}:
        return True
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _revision_payload_from_response(payload: dict[str, Any], latest: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "title",
        "group_name",
        "family",
        "summary",
        "depends_on",
        "strategy",
        "expected_signal",
        "risk",
    )
    revision_payload = {key: payload[key] for key in allowed if key in payload}
    revision_payload.setdefault("group_name", latest.get("group_name"))
    revision_payload.setdefault("family", latest.get("family"))
    revision_payload.setdefault("depends_on", latest.get("depends_on") or [])
    return revision_payload


def _validate_revision_compatibility(previous: dict[str, Any], current: dict[str, Any]) -> None:
    if current.get("group_name") != previous.get("group_name"):
        raise ValueError("ROOT revision must keep the same group_name")
    if list(current.get("depends_on") or []) != list(previous.get("depends_on") or []):
        raise ValueError("ROOT revision must keep depends_on compatible with the previous revision")
