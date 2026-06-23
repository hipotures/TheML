from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tml.utils.hashing import sha256_file
from tml.utils.yaml_io import read_yaml, write_yaml


@dataclass(frozen=True)
class HypothesisRevision:
    hypothesis_id: str
    revision: int
    path: Path
    prefix: str
    payload: dict[str, Any]


def normalize_hypothesis_id(value: object) -> str:
    text = str(value or "").strip()
    return text.zfill(6) if text.isdigit() else text


def revision_prefix(revision: int) -> str:
    return f"{int(revision):02d}-hypothesis"


def revision_file_name(revision: int) -> str:
    return f"{revision_prefix(revision)}.yaml"


def revision_from_file(path: Path) -> int | None:
    stem = path.stem
    if len(stem) < 13 or not stem[:2].isdigit() or stem[2:] != "-hypothesis":
        return None
    return int(stem[:2])


def migrate_root_revisions(project_dir: Path) -> None:
    for hdir in sorted((project_dir / "hypotheses").glob("*")):
        if hdir.is_dir():
            migrate_hypothesis_dir(project_dir, hdir)


def migrate_hypothesis_dir(project_dir: Path, hdir: Path) -> None:
    legacy_path = hdir / "hypothesis.yaml"
    first_path = hdir / revision_file_name(1)
    if legacy_path.exists() and not first_path.exists():
        legacy_path.replace(first_path)
    elif legacy_path.exists():
        legacy_path.unlink()

    for path in sorted(hdir.glob("??-hypothesis.yaml")):
        revision = revision_from_file(path)
        if revision is None:
            continue
        payload = read_yaml(path)
        changed = False
        defaults = {
            "schema_version": 1,
            "hypothesis_id": hdir.name,
            "revision": revision,
            "enabled": True,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        for key, value in defaults.items():
            if key not in payload:
                payload[key] = value
                changed = True
        if str(payload.get("hypothesis_id") or "") != hdir.name:
            payload["hypothesis_id"] = hdir.name
            changed = True
        if int(payload.get("revision") or 0) != revision:
            payload["revision"] = revision
            changed = True
        if changed:
            write_yaml(path, payload)
    repair_manifest(project_dir, hdir)


def revision_records(hdir: Path) -> list[HypothesisRevision]:
    records: list[HypothesisRevision] = []
    for path in sorted(hdir.glob("??-hypothesis.yaml")):
        revision = revision_from_file(path)
        if revision is None:
            continue
        records.append(
            HypothesisRevision(
                hypothesis_id=hdir.name,
                revision=revision,
                path=path,
                prefix=revision_prefix(revision),
                payload=read_yaml(path),
            )
        )
    return records


def latest_revision_record(hdir: Path) -> HypothesisRevision:
    records = revision_records(hdir)
    if not records:
        raise FileNotFoundError(f"No canonical ROOT revisions in {hdir}")
    return records[-1]


def load_revision(hdir: Path, revision: int | None = None) -> HypothesisRevision:
    if revision is None:
        return latest_revision_record(hdir)
    path = hdir / revision_file_name(revision)
    if not path.exists():
        raise FileNotFoundError(f"Missing hypothesis revision: {path}")
    return HypothesisRevision(
        hypothesis_id=hdir.name,
        revision=revision,
        path=path,
        prefix=revision_prefix(revision),
        payload=read_yaml(path),
    )


def next_revision_number(hdir: Path) -> int:
    records = revision_records(hdir)
    return (records[-1].revision if records else 0) + 1


def write_revision(hdir: Path, revision: int, payload: dict[str, Any]) -> Path:
    out = hdir / revision_file_name(revision)
    payload = {
        **payload,
        "schema_version": 1,
        "hypothesis_id": hdir.name,
        "revision": revision,
        "enabled": bool(payload.get("enabled", True)),
        "created_at": str(payload.get("created_at") or datetime.now().isoformat(timespec="seconds")),
    }
    write_yaml(out, payload)
    repair_manifest(hdir.parents[1], hdir)
    return out


def repair_manifest(project_dir: Path, hdir: Path) -> None:
    records = revision_records(hdir)
    latest = records[-1].revision if records else 0
    old = read_yaml(hdir / "manifest.yaml")
    old_mats = old.get("materializations") if isinstance(old.get("materializations"), dict) else {}
    manifest: dict[str, Any] = {
        "hypothesis": {"id": hdir.name, "latest_revision": latest},
        "revisions": {
            record.revision: {"file": record.path.name, "prefix": record.prefix}
            for record in records
        },
        "materializations": {},
    }
    mat_dir = hdir / "materializations"
    if mat_dir.exists():
        modes = sorted({path.name.split("-", 1)[0] for path in mat_dir.glob("*.py")})
        modes.extend(mode for mode in old_mats if isinstance(mode, str) and mode not in modes)
        for mode in modes:
            old_mode = old_mats.get(mode) if isinstance(old_mats.get(mode), dict) else {}
            active = old_mode.get("active") if isinstance(old_mode.get("active"), str) else None
            existing_entries = _existing_file_entries(old_mode)
            files: list[dict[str, Any]] = []
            for code in sorted(mat_dir.glob(f"{mode}-*.py")):
                previous = existing_entries.get(code.name, {})
                files.append(
                    {
                        "file": code.name,
                        "revision": _int_or_default(previous.get("revision"), 1),
                        "sha256": sha256_file(code),
                        "created_at": str(previous.get("created_at") or old_mode.get("created_at") or datetime.now().isoformat(timespec="seconds")),
                    }
                )
            if files or active:
                if not active and files:
                    active = files[-1]["file"]
                manifest["materializations"][mode] = {
                    "active": active,
                    "files": files,
                }
    write_yaml(hdir / "manifest.yaml", manifest)


def materialization_entries(hdir: Path, mode: str) -> list[dict[str, Any]]:
    manifest = read_yaml(hdir / "manifest.yaml")
    mats = manifest.get("materializations") if isinstance(manifest.get("materializations"), dict) else {}
    mode_entry = mats.get(mode) if isinstance(mats.get(mode), dict) else {}
    files = mode_entry.get("files") if isinstance(mode_entry.get("files"), list) else []
    return [dict(item) for item in files if isinstance(item, dict)]


def materialization_revision(hdir: Path, mode: str, file_name: str) -> int:
    for entry in materialization_entries(hdir, mode):
        if entry.get("file") == file_name:
            return _int_or_default(entry.get("revision"), 1)
    return 1


def active_materialization_file(hdir: Path, mode: str) -> str | None:
    manifest = read_yaml(hdir / "manifest.yaml")
    mats = manifest.get("materializations") if isinstance(manifest.get("materializations"), dict) else {}
    mode_entry = mats.get(mode) if isinstance(mats.get(mode), dict) else {}
    active = mode_entry.get("active")
    return active if isinstance(active, str) else None


def append_materialization(hdir: Path, mode: str, code_path: Path, *, revision: int, active: bool = False) -> None:
    manifest = read_yaml(hdir / "manifest.yaml")
    mats = manifest.setdefault("materializations", {})
    if not isinstance(mats, dict):
        mats = {}
        manifest["materializations"] = mats
    mode_entry = mats.setdefault(mode, {})
    if not isinstance(mode_entry, dict):
        mode_entry = {}
        mats[mode] = mode_entry
    files = mode_entry.setdefault("files", [])
    if not isinstance(files, list):
        files = []
        mode_entry["files"] = files
    created_at = datetime.now().isoformat(timespec="seconds")
    replacement = {
        "file": code_path.name,
        "revision": int(revision),
        "sha256": sha256_file(code_path),
        "created_at": created_at,
    }
    files[:] = [item for item in files if not isinstance(item, dict) or item.get("file") != code_path.name]
    files.append(replacement)
    if active or not mode_entry.get("active"):
        mode_entry["active"] = code_path.name
    write_yaml(hdir / "manifest.yaml", manifest)


def set_active_materialization(hdir: Path, mode: str, file_name: str) -> None:
    manifest = read_yaml(hdir / "manifest.yaml")
    mats = manifest.setdefault("materializations", {})
    if not isinstance(mats, dict):
        mats = {}
        manifest["materializations"] = mats
    mode_entry = mats.setdefault(mode, {})
    if not isinstance(mode_entry, dict):
        mode_entry = {}
        mats[mode] = mode_entry
    mode_entry["active"] = file_name
    write_yaml(hdir / "manifest.yaml", manifest)


def _existing_file_entries(mode_entry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    files = mode_entry.get("files")
    if not isinstance(files, list):
        return {}
    entries: dict[str, dict[str, Any]] = {}
    for item in files:
        if isinstance(item, dict) and isinstance(item.get("file"), str):
            entries[str(item["file"])] = item
    return entries


def _int_or_default(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
