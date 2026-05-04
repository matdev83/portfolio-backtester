"""Concrete signal/portfolio ``generate_target_weights`` must not return None (StrategyLogic API)."""

from __future__ import annotations

import ast
import pathlib

import pytest

_PKG_ROOT = (
    pathlib.Path(__file__).resolve().parents[3] / "src" / "portfolio_backtester" / "strategies"
)
_SCAN_SUBDIRS = ("builtins/signal", "builtins/portfolio", "user/signal", "user/portfolio", "signal")


def _iter_strategy_py_files() -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for rel in _SCAN_SUBDIRS:
        d = _PKG_ROOT / pathlib.Path(rel)
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*.py")):
            if not p.is_file():
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except OSError:
                continue
            if "def generate_target_weights" not in text:
                continue
            out.append(p)
    return out


def _stmt_returns_none(stmt: ast.stmt) -> bool:
    if isinstance(stmt, ast.Return):
        if stmt.value is None:
            return True
        return isinstance(stmt.value, ast.Constant) and stmt.value.value is None
    if isinstance(stmt, ast.If):
        for s in stmt.body + stmt.orelse:
            if _stmt_returns_none(s):
                return True
    if isinstance(stmt, (ast.For, ast.While)):
        for s in stmt.body + stmt.orelse:
            if _stmt_returns_none(s):
                return True
    elif isinstance(stmt, ast.With):
        for s in stmt.body:
            if _stmt_returns_none(s):
                return True
    if isinstance(stmt, ast.Try):
        for s in stmt.body + stmt.orelse + stmt.finalbody:
            if _stmt_returns_none(s):
                return True
        for handler in stmt.handlers:
            for s in handler.body:
                if _stmt_returns_none(s):
                    return True
    return False


def _function_def_returns_none(body: list[ast.stmt]) -> bool:
    for st in body:
        if _stmt_returns_none(st):
            return True
    return False


@pytest.mark.parametrize(
    "path", _iter_strategy_py_files(), ids=lambda p: str(p.relative_to(_PKG_ROOT))
)
def test_generate_target_weights_has_no_none_return(path: pathlib.Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if not isinstance(item, ast.FunctionDef) or item.name != "generate_target_weights":
                    continue
                assert not _function_def_returns_none(
                    item.body
                ), f"{path}:{item.lineno} generate_target_weights must not return None"
