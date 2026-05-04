"""Tests for ``scripts/sync_wshobson_cursor_skills.py``."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_sync_module():
    path = _REPO_ROOT / "scripts" / "sync_wshobson_cursor_skills.py"
    name = "_sync_wshobson_cursor_skills"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_sync_skills_copies_skill_md(tmp_path: Path) -> None:
    m = _load_sync_module()
    source = tmp_path / "plugins" / "demo-plugin" / "skills" / "demo-skill"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Demo.\n---\n\n# Demo\n",
        encoding="utf-8",
    )
    dest = tmp_path / "out" / "skills"
    manifest = tmp_path / ".cursor" / m._MANIFEST

    copied, skipped = m.sync_skills(
        source_plugins=tmp_path / "plugins",
        dest_skills=dest,
        repo_root=tmp_path,
        dry_run=False,
        purge_stale=True,
    )

    assert skipped == 0
    assert copied == 1
    out_dir = dest / "demo-skill"
    assert (out_dir / "SKILL.md").is_file()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["managed_skill_ids"] == ["demo-skill"]


def test_sync_skills_skips_without_skill_md(tmp_path: Path) -> None:
    m = _load_sync_module()
    source = tmp_path / "plugins" / "p" / "skills" / "empty-skill"
    source.mkdir(parents=True)
    dest = tmp_path / "out2" / "skills"

    copied, skipped = m.sync_skills(
        source_plugins=tmp_path / "plugins",
        dest_skills=dest,
        repo_root=tmp_path,
        dry_run=False,
        purge_stale=True,
    )

    assert copied == 0
    assert skipped == 1


def test_sync_skills_raises_when_plugins_missing(tmp_path: Path) -> None:
    m = _load_sync_module()
    with pytest.raises(FileNotFoundError):
        m.sync_skills(
            source_plugins=tmp_path / "nope",
            dest_skills=tmp_path / "d",
            repo_root=tmp_path,
            dry_run=False,
            purge_stale=False,
        )


def test_collect_skill_dirs_rejects_duplicate_ids(tmp_path: Path) -> None:
    m = _load_sync_module()
    for plugin in ("a", "b"):
        d = tmp_path / "plugins" / plugin / "skills" / "dup-skill"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            "---\nname: dup-skill\ndescription: x\n---\n",
            encoding="utf-8",
        )
    with pytest.raises(ValueError, match="Duplicate skill id"):
        m._collect_skill_dirs(tmp_path / "plugins")


def test_sync_removes_stale_managed_skill(tmp_path: Path) -> None:
    m = _load_sync_module()
    root = tmp_path
    dest = root / "skills"
    dest.mkdir(parents=True)
    stale = dest / "gone-skill"
    stale.mkdir()
    (stale / "SKILL.md").write_text(
        "---\nname: gone-skill\ndescription: x\n---\n", encoding="utf-8"
    )
    manifest = root / ".cursor" / m._MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"managed_skill_ids": ["gone-skill", "keep-skill"]}),
        encoding="utf-8",
    )

    keep = root / "plugins" / "p" / "skills" / "keep-skill"
    keep.mkdir(parents=True)
    (keep / "SKILL.md").write_text("---\nname: keep-skill\ndescription: x\n---\n", encoding="utf-8")

    m.sync_skills(
        source_plugins=root / "plugins",
        dest_skills=dest,
        repo_root=root,
        dry_run=False,
        purge_stale=True,
    )

    assert not stale.exists()
    assert (dest / "keep-skill" / "SKILL.md").is_file()
