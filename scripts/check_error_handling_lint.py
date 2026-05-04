#!/usr/bin/env python3
"""Optional gate: ruff BLE001 on boundary packages (see docs/error_handling.md)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    targets = [
        root / "src" / "portfolio_backtester" / "interfaces",
        root / "src" / "portfolio_backtester" / "research",
    ]
    cmd = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        *[str(p) for p in targets],
        "--select",
        "BLE001",
    ]
    return subprocess.call(cmd, cwd=str(root))


if __name__ == "__main__":
    raise SystemExit(main())
