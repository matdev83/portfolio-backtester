#!/usr/bin/env python3
"""Copy wshobson/agents plugin skills into project-local Cursor skills.

Reads ``third_party/wshobson-agents/plugins/<plugin>/skills/<skill>/`` and writes
each skill to ``.cursor/skills/<skill>/`` (same layout as upstream: one folder
per skill with ``SKILL.md``). That matches Cursor's project-skill convention
(see ``.cursor/skills/<skill-name>/SKILL.md``).

Skill directory names are unique across the upstream repo; the sync script
raises if that invariant breaks. A manifest under ``.cursor/`` records which
skill ids were installed so a later sync can remove skills dropped upstream.

Re-run after ``git submodule update --remote third_party/wshobson-agents``.

Usage (from repo root)::

    ./.venv/Scripts/python.exe scripts/sync_wshobson_cursor_skills.py
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

SKILL_FILENAME = "SKILL.md"
# Legacy destination prefix from an earlier sync layout.
_LEGACY_PREFIX = "wshobson__"
_MANIFEST = ".wshobson-agents-sync-manifest.json"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_source(root: Path) -> Path:
    return root / "third_party" / "wshobson-agents" / "plugins"


def _default_dest(root: Path) -> Path:
    return root / ".cursor" / "skills"


def _manifest_path(root: Path) -> Path:
    return root / ".cursor" / _MANIFEST


def _collect_skill_dirs(source_plugins: Path) -> list[tuple[str, Path]]:
    """Return (skill_id, skill_dir) sorted by skill_id. Raises on duplicate ids."""
    seen: dict[str, Path] = {}
    for plugin_dir in sorted(p for p in source_plugins.iterdir() if p.is_dir()):
        skills_root = plugin_dir / "skills"
        if not skills_root.is_dir():
            continue
        for skill_dir in sorted(s for s in skills_root.iterdir() if s.is_dir()):
            skill_id = skill_dir.name
            if skill_id in seen:
                msg = (
                    f"Duplicate skill id {skill_id!r} under plugins "
                    f"{seen[skill_id].parent.parent.name!r} and "
                    f"{plugin_dir.name!r}; fix upstream or extend this script."
                )
                raise ValueError(msg)
            seen[skill_id] = skill_dir
    return sorted(seen.items(), key=lambda x: x[0])


def _load_managed_ids(manifest: Path) -> set[str]:
    if not manifest.is_file():
        return set()
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Ignoring manifest %s: %s", manifest, e)
        return set()
    raw = data.get("managed_skill_ids")
    if not isinstance(raw, list):
        return set()
    return {str(x) for x in raw}


def _write_managed_ids(manifest: Path, skill_ids: list[str], *, dry_run: bool) -> None:
    payload = {"managed_skill_ids": sorted(skill_ids)}
    if dry_run:
        logger.info("Would write manifest %s (%d ids)", manifest, len(skill_ids))
        return
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sync_skills(
    *,
    source_plugins: Path,
    dest_skills: Path,
    repo_root: Path,
    dry_run: bool,
    purge_stale: bool,
) -> tuple[int, int]:
    """Copy skill directories. Returns (skills_copied, skills_skipped)."""
    if not source_plugins.is_dir():
        raise FileNotFoundError(
            f"Missing submodule checkout: {source_plugins}. "
            "Run: git submodule update --init third_party/wshobson-agents"
        )

    dest_skills.mkdir(parents=True, exist_ok=True)
    manifest = _manifest_path(repo_root)
    pairs = _collect_skill_dirs(source_plugins)
    eligible = [(skill_id, d) for skill_id, d in pairs if (d / SKILL_FILENAME).is_file()]
    skipped = len(pairs) - len(eligible)
    for _, skill_dir in pairs:
        if (skill_dir / SKILL_FILENAME).is_file():
            continue
        logger.warning("Skip %s (no %s)", skill_dir, SKILL_FILENAME)
    new_ids = [skill_id for skill_id, _ in eligible]

    if purge_stale:
        old_ids = _load_managed_ids(manifest)
        for stale_id in sorted(old_ids.difference(new_ids)):
            stale_dir = dest_skills / stale_id
            if not stale_dir.is_dir():
                continue
            if dry_run:
                logger.info("Would remove stale skill dir %s", stale_dir)
            else:
                shutil.rmtree(stale_dir)
                logger.info("Removed stale skill dir %s", stale_dir.name)

        for child in sorted(dest_skills.iterdir(), key=lambda p: p.name):
            if not child.is_dir() or not child.name.startswith(_LEGACY_PREFIX):
                continue
            if dry_run:
                logger.info("Would remove legacy prefixed dir %s", child)
            else:
                shutil.rmtree(child)
                logger.info("Removed legacy dir %s", child.name)

    copied = 0

    for skill_id, skill_dir in eligible:
        dest_dir = dest_skills / skill_id
        if dry_run:
            logger.info("Would sync %s -> %s", skill_dir, dest_dir)
            copied += 1
            continue
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(skill_dir, dest_dir)
        logger.info("Synced %s", dest_dir.name)
        copied += 1

    _write_managed_ids(manifest, new_ids, dry_run=dry_run)

    return copied, skipped


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=_default_source(root),
        help="Path to wshobson agents ``plugins`` directory",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=_default_dest(root),
        help="Cursor project skills directory (default: .cursor/skills)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing",
    )
    parser.add_argument(
        "--no-purge",
        action="store_true",
        help="Do not remove stale managed skills or legacy wshobson__* dirs",
    )
    args = parser.parse_args()

    try:
        copied, skipped = sync_skills(
            source_plugins=args.source.resolve(),
            dest_skills=args.dest.resolve(),
            repo_root=root,
            dry_run=args.dry_run,
            purge_stale=not args.no_purge,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error("%s", e)
        return 1

    logger.info("Done: %d skills synced, %d skipped", copied, skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
