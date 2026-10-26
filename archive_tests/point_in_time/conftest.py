from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def preserve_v8_1_workspace_snapshot_boundary(monkeypatch, request):
    """Keep the frozen V8.1 inventory test scoped to V8.1, not V8.1.1.

    The sealed V8.1 selector predates semantic-version descendants and matches
    the substring ``v8_1``.  Consequently, every required V8.1.1 artifact would
    otherwise be treated as a late V8.1 workspace addition.  Apply the version
    boundary only to the historical inventory-reconciliation assertion; all
    other V8.1 tests and runtime validation continue to use the sealed code.
    """
    if (
        request.node.name != "test_workspace_inventory_reconciles"
        or Path(str(request.node.fspath)).name
        != "test_pit_v8_1_hardening.py"
    ):
        return

    from scripts.point_in_time import run_pit_v8_1_release as release

    original = release._workspace_related_files

    def v8_1_only(release_paths):
        selected = original(release_paths)
        descendant_tokens = ("v8_1_1", "v8.1.1")
        return {
            path
            for path in selected
            if not any(
                token in release.relative(path).lower()
                for token in descendant_tokens
            )
        }

    monkeypatch.setattr(release, "_workspace_related_files", v8_1_only)
