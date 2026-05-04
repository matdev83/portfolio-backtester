"""Repository hygiene checks (anti-patterns, silent failures)."""

from __future__ import annotations

import re
from pathlib import Path


def test_no_silent_except_exception_pass_in_package_src() -> None:
    """Ban ``except Exception:`` immediately followed by ``pass`` (silent swallow)."""
    repo_root = Path(__file__).resolve().parents[2]
    root = repo_root / "src" / "portfolio_backtester"
    pattern = re.compile(r"except\s+Exception\s*:\s*\n\s*pass\b")
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if pattern.search(text):
            offenders.append(str(path.relative_to(repo_root)))
    assert not offenders, "Silent except Exception: pass found in:\n" + "\n".join(offenders)
