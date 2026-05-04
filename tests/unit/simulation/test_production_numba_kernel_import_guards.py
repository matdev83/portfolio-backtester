"""Guard: legacy Numba kernels must not gain new production call sites."""

from __future__ import annotations

import ast
from pathlib import Path

_FORBIDDEN = frozenset(
    {
        "drifting_weights_returns_kernel",
        "detailed_commission_slippage_kernel",
        "trade_tracking_kernel",
    }
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_PKG = _REPO_ROOT / "src" / "portfolio_backtester"


def _names_from_import(node: ast.AST) -> frozenset[str]:
    names: set[str] = set()
    if isinstance(node, ast.Import):
        for alias in node.names:
            base = alias.name.split(".")[-1]
            if alias.asname:
                names.add(alias.asname)
            else:
                names.add(base)
    elif isinstance(node, ast.ImportFrom):
        if node.module is None:
            return frozenset()
        for alias in node.names:
            if alias.name == "*":
                continue
            if alias.asname:
                names.add(alias.asname)
            else:
                names.add(alias.name)
    return frozenset(names)


def test_production_py_files_do_not_import_legacy_kernels_except_allowlist() -> None:
    allow_trade_tracking_only = frozenset({"trading/numba_trade_tracker.py"})
    violations: list[str] = []

    for path in sorted(_SRC_PKG.rglob("*.py")):
        rel = path.relative_to(_SRC_PKG).as_posix()
        if rel == "numba_kernels.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported: set[str] = set()
        for stmt in ast.walk(tree):
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                imported |= _names_from_import(stmt)
        banned_here = imported & _FORBIDDEN
        if not banned_here:
            continue
        if rel in allow_trade_tracking_only:
            if banned_here <= {"trade_tracking_kernel"}:
                continue
        violations.append(f"{rel}: {sorted(banned_here)}")

    assert not violations, "Unexpected legacy kernel imports:\n" + "\n".join(violations)
