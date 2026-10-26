#!/usr/bin/env python3
"""Cross-platform manifest path resolution and V6 compatibility manifest.

Legacy model-decay manifests retain their original absolute paths as immutable
provenance.  Validation uses repository-relative paths as the primary key and
consults absolute/translated paths only as explicit legacy fallbacks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
MANIFEST_VERSION = "manifest-portability-v6.0.0"
OUTPUT_PATH = (
    REPO_ROOT
    / "research_data"
    / "shared"
    / "manifests"
    / "manifest_portability_v6.json"
)
REPORT_PATH = REPO_ROOT / "reports" / "data_execution" / "MANIFEST_PORTABILITY_V6.md"


class ManifestResolutionError(ValueError):
    """Raised when no safe, existing artifact path can be resolved."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_relative(value: str | os.PathLike[str]) -> Path:
    text = str(value).replace("\\", "/").strip()
    pure = PurePosixPath(text)
    if pure.is_absolute() or ".." in pure.parts:
        raise ManifestResolutionError(f"unsafe repository-relative path: {value}")
    return Path(*pure.parts)


def legacy_relative_path(
    original_path: str | os.PathLike[str],
    repository_root: Path,
) -> Path:
    """Extract a repository-relative key from Windows or WSL legacy metadata."""

    text = str(original_path).replace("\\", "/")
    root_name = repository_root.name
    marker = f"/{root_name}/"
    lowered = text.lower()
    marker_index = lowered.find(marker.lower())
    if marker_index >= 0:
        return _clean_relative(text[marker_index + len(marker) :])
    for stable_marker in ("research_data/", "reports/", "scripts/", "tests/"):
        marker_index = lowered.find(stable_marker.lower())
        if marker_index >= 0:
            return _clean_relative(text[marker_index:])
    path = Path(str(original_path))
    if not path.is_absolute():
        return _clean_relative(original_path)
    raise ManifestResolutionError(
        f"cannot derive a repository-relative key from legacy path: {original_path}"
    )


def translate_explicit_path(
    original_path: str | os.PathLike[str],
) -> Path | None:
    """Translate only recognized WSL/Windows drive forms."""

    text = str(original_path)
    wsl = re.match(r"^/mnt/([A-Za-z])/(.+)$", text.replace("\\", "/"))
    if wsl and os.name == "nt":
        return Path(f"{wsl.group(1).upper()}:\\{wsl.group(2).replace('/', os.sep)}")
    windows = re.match(r"^([A-Za-z]):[\\/](.+)$", text)
    if windows and os.name != "nt":
        return Path(
            f"/mnt/{windows.group(1).lower()}/"
            + windows.group(2).replace("\\", "/")
        )
    return None


def resolve_manifest_path(
    *,
    artifact_relative_path: str | os.PathLike[str] | None,
    repository_root: Path | None,
    repository_root_hint: str | os.PathLike[str] | None = None,
    original_absolute_path: str | os.PathLike[str] | None = None,
    allow_translation: bool = True,
    require_exists: bool = True,
) -> Path:
    """Resolve with the preregistered V6 precedence.

    1. repository-relative key under the configured repository root;
    2. repository-relative key under an explicit root hint;
    3. the original absolute path when it still exists;
    4. an explicitly recognized Windows/WSL translation.
    """

    relative = (
        _clean_relative(artifact_relative_path)
        if artifact_relative_path not in (None, "")
        else None
    )
    candidates: list[Path] = []
    if relative is not None and repository_root is not None:
        candidates.append(repository_root.resolve() / relative)
    if relative is not None and repository_root_hint not in (None, "", "."):
        hint = Path(str(repository_root_hint))
        if hint.is_absolute():
            candidates.append(hint.resolve() / relative)
    if original_absolute_path not in (None, ""):
        original = Path(str(original_absolute_path))
        if original.is_absolute():
            candidates.append(original)
        if allow_translation:
            translated = translate_explicit_path(original_absolute_path)
            if translated is not None:
                candidates.append(translated)
        if repository_root is not None:
            try:
                candidates.append(
                    repository_root.resolve()
                    / legacy_relative_path(original_absolute_path, repository_root)
                )
            except ManifestResolutionError:
                pass

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).casefold()
        if key in seen:
            continue
        seen.add(key)
        if not require_exists or candidate.is_file():
            return candidate
    if relative is not None and repository_root is None:
        raise ManifestResolutionError(
            "repository_root is required for a repository-relative validation key"
        )
    raise ManifestResolutionError(
        "manifest artifact did not resolve to an existing file: "
        + ", ".join(str(path) for path in candidates)
    )


def validate_portable_record(
    record: Mapping[str, Any],
    repository_root: Path | None,
) -> Path:
    path = resolve_manifest_path(
        artifact_relative_path=record.get("artifact_relative_path"),
        repository_root=repository_root,
        repository_root_hint=record.get("repository_root_hint"),
        original_absolute_path=record.get("original_absolute_path"),
    )
    expected_size = int(record["size_bytes"])
    if path.stat().st_size != expected_size:
        raise ManifestResolutionError(
            f"size mismatch for {path}: {path.stat().st_size} != {expected_size}"
        )
    actual_checksum = sha256_file(path)
    if actual_checksum != record["checksum"]:
        raise ManifestResolutionError(
            f"checksum mismatch for {path}: {actual_checksum} != {record['checksum']}"
        )
    return path


def build_compatibility_manifest(repository_root: Path = REPO_ROOT) -> dict[str, Any]:
    manifest_dir = repository_root / "research_data" / "shared" / "manifests"
    records: list[dict[str, Any]] = []
    for legacy_path in sorted(manifest_dir.glob("model_decay_*.manifest.json")):
        legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
        artifact_relative = legacy_relative_path(legacy["path"], repository_root)
        schema_relative = legacy_relative_path(legacy["schema_path"], repository_root)
        artifact = repository_root / artifact_relative
        schema = repository_root / schema_relative
        if not artifact.is_file() or not schema.is_file():
            raise ManifestResolutionError(
                f"legacy manifest target missing after relative resolution: {legacy_path}"
            )
        if sha256_file(artifact) != legacy["file_checksum_sha256"]:
            raise ManifestResolutionError(
                f"legacy artifact checksum mismatch: {legacy_path}"
            )
        if sha256_file(schema) != legacy["schema_checksum_sha256"]:
            raise ManifestResolutionError(
                f"legacy schema checksum mismatch: {legacy_path}"
            )
        records.append(
            {
                "legacy_manifest_relative_path": legacy_path.relative_to(
                    repository_root
                ).as_posix(),
                "artifact_relative_path": artifact_relative.as_posix(),
                "repository_root_hint": ".",
                "original_absolute_path": legacy["path"],
                "checksum": legacy["file_checksum_sha256"],
                "size_bytes": artifact.stat().st_size,
                "manifest_version": MANIFEST_VERSION,
                "schema_relative_path": schema_relative.as_posix(),
                "original_schema_absolute_path": legacy["schema_path"],
                "schema_checksum": legacy["schema_checksum_sha256"],
                "schema_size_bytes": schema.stat().st_size,
                "row_count": legacy["row_count"],
            }
        )
    payload = {
        "manifest_version": MANIFEST_VERSION,
        "validation_key": "artifact_relative_path",
        "repository_root_hint": ".",
        "legacy_manifests_modified": False,
        "record_count": len(records),
        "records": records,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def write_report(payload: Mapping[str, Any]) -> None:
    lines = [
        "# Manifest Portability V6",
        "",
        "The legacy model-decay manifests retain their original `/mnt/k/...` "
        "paths as provenance metadata. They were not rewritten.",
        "",
        "Validation now uses this precedence:",
        "",
        "1. `artifact_relative_path` under the configured repository root;",
        "2. the same relative key under an explicit repository-root hint;",
        "3. an existing original absolute path as legacy metadata;",
        "4. explicit Windows/WSL drive translation.",
        "",
        f"- Compatibility manifest version: `{payload['manifest_version']}`.",
        f"- Compatibility records: {payload['record_count']}.",
        "- Validation key: `artifact_relative_path`.",
        "- Legacy manifests modified: no.",
        "- Locked SET50 V4 binaries modified: no.",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    payload = build_compatibility_manifest()
    for record in payload["records"]:
        validate_portable_record(record, REPO_ROOT)
    write_report(payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    return argparse.Namespace()


def main(argv: Sequence[str] | None = None) -> int:
    parse_args(argv)
    payload = run()
    print(
        f"manifest_portability_v6: PASS ({payload['record_count']} records, "
        "legacy manifests unchanged)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
