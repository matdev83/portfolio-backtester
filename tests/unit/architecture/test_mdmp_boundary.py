"""Guardrails: OHLCV and MDMP coupling stay within documented boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PKG_ROOT = _REPO_ROOT / "src" / "portfolio_backtester"

_MDMP_MODULES_REL = frozenset(
    {
        Path("data_sources") / "mdmp_facade.py",
    },
)

_MARKET_FETCH_VENDOR_MODULES = frozenset(
    {
        "alpaca_trade_api",
        "finnhub",
        "pandas_datareader",
        "polygon",
        "polygon_api_client",
        "tiingo",
        "yfinance",
    },
)


def _rel_module_path(py_path: Path) -> Path:
    return py_path.relative_to(_PKG_ROOT)


def _is_mdmp_symbol(name: str) -> bool:
    return name == "market_data_multi_provider" or name.startswith("market_data_multi_provider.")


def _imports_market_vendor(node: ast.AST, top_name: str) -> bool:
    if isinstance(node, ast.Import):
        for alias in node.names:
            base = alias.name.split(".", 1)[0]
            if base == top_name:
                return True
    elif isinstance(node, ast.ImportFrom) and node.module:
        base = node.module.split(".", 1)[0]
        if base == top_name:
            return True
    return False


def _mdmp_refs_in_tree(tree: ast.AST) -> list[tuple[int, str]]:
    refs: list[tuple[int, str]] = []

    class V(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                if _is_mdmp_symbol(alias.name):
                    refs.append((node.lineno, f"import {alias.name}"))
            self.generic_visit(node)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            mod = node.module or ""
            if _is_mdmp_symbol(mod):
                refs.append((node.lineno, f"from {mod}"))
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Attribute) and node.func.attr == "import_module":
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "importlib":
                    arg0 = node.args[0] if node.args else None
                    mod: str | None = None
                    if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                        mod = arg0.value
                    if mod and _is_mdmp_symbol(mod):
                        refs.append((node.lineno, f"importlib.import_module({mod!r})"))
            self.generic_visit(node)

    V().visit(tree)
    return refs


def test_mdmp_public_imports_are_allowlisted_only() -> None:
    offenders: dict[str, list[str]] = {}
    for py in sorted(_PKG_ROOT.rglob("*.py")):
        rel = _rel_module_path(py)
        if rel.parts[:1] == ("__pycache__",):
            continue
        raw = py.read_bytes()
        try:
            tree = ast.parse(raw, filename=str(py))
        except SyntaxError as e:
            raise AssertionError(f"Syntax error in {rel}: {e}") from e

        refs = _mdmp_refs_in_tree(tree)
        if not refs:
            continue

        if rel not in _MDMP_MODULES_REL:
            offenders[str(rel)] = [f"line {ln}: {msg}" for ln, msg in refs]

    assert not offenders, (
        "market_data_multi_provider must only be referenced under src/portfolio_backtester "
        f"in: data_sources/mdmp_facade.py\n"
        f"Violations:\n{offenders}"
    )


def test_no_vendor_market_fetch_imports_under_package() -> None:
    violations: dict[str, list[str]] = {}
    for py in sorted(_PKG_ROOT.rglob("*.py")):
        rel_path = py.read_text(encoding="utf-8")
        rel = _rel_module_path(py)
        try:
            tree = ast.parse(rel_path, filename=str(py))
        except SyntaxError as e:
            raise AssertionError(f"Syntax error in {rel}: {e}") from e

        bad: list[str] = []
        for node in ast.walk(tree):
            for vendor in _MARKET_FETCH_VENDOR_MODULES:
                if _imports_market_vendor(node, vendor):
                    lineno = getattr(node, "lineno", 0)
                    bad.append(f"line {lineno}: forbidden import {vendor!r}")

        if bad:
            violations[str(rel)] = bad

    assert not violations, (
        "Direct vendor client imports are not allowed under portfolio_backtester; "
        "use MDMP via MarketDataMultiProviderDataSource or MDMP constellation APIs.\n"
        f"{violations}"
    )


def test_no_parquet_serialization_under_package_src() -> None:
    parquet_hits: list[str] = []
    for py in sorted(_PKG_ROOT.rglob("*.py")):
        text = py.read_text(encoding="utf-8")
        if ".to_parquet(" in text or ".to_parquet (" in text:
            parquet_hits.append(str(_rel_module_path(py)))

    assert not parquet_hits, (
        "portfolio_backtester must not write canonical OHLCV/holdings via pandas to_parquet; "
        f"those belong to MDMP cache ownership.\nHits: {parquet_hits}"
    )


def test_no_parquet_deserialization_under_package_src() -> None:
    parquet_reads: list[str] = []
    for py in sorted(_PKG_ROOT.rglob("*.py")):
        text = py.read_text(encoding="utf-8")
        if ".read_parquet(" in text or ".read_parquet (" in text:
            parquet_reads.append(str(_rel_module_path(py)))

    assert not parquet_reads, (
        "portfolio_backtester must not load canonical OHLCV/holdings via pandas read_parquet; "
        f"use MDMP APIs instead.\nHits: {parquet_reads}"
    )
