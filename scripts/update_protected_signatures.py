"""Utility to refresh api_stable signature registry.

Run with:
    python scripts/update_protected_signatures.py
It will import the entire `src.portfolio_backtester` package tree so that
all `@api_stable` decorated callables register themselves, then overwrite
`src/portfolio_backtester/api_stability/api_stable_signatures.json` with
the current signature meta-data.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

# Root dotted package to scan
PACKAGE_ROOT = "portfolio_backtester"
SIGNATURE_PATH = (
    Path(__file__).parent.parent
    / "src"
    / "portfolio_backtester"
    / "api_stability"
    / "api_stable_signatures.json"
)


def _import_all_package_files(package_root: str) -> None:
    """Import every *.py file inside *package_root* recursively.

    This guarantees that any module-level side-effects (notably
    `@api_stable` decorators) are executed exactly once, without having
    to maintain a hand-written import list.
    """
    pkg = importlib.import_module(package_root)
    base_paths = list(pkg.__path__)

    for base in base_paths:
        base_path = Path(base).resolve()
        for file_path in base_path.rglob("*.py"):
            rel_mod = file_path.with_suffix("").relative_to(base_path.parent)
            module_name = ".".join(rel_mod.parts)
            if module_name.endswith(".__init__"):
                module_name = module_name[:-9]
            try:
                import contextlib
                import io
                import logging

                # Suppress any noisy stdout / stderr and lower loglevel during import
                _stdout, _stderr = io.StringIO(), io.StringIO()
                prev_level = logging.getLogger().level
                logging.getLogger().setLevel(logging.CRITICAL)
                with contextlib.redirect_stdout(_stdout), contextlib.redirect_stderr(_stderr):
                    importlib.import_module(module_name)
                logging.getLogger().setLevel(prev_level)
            except Exception as exc:
                print(f"[WARN] Skipped {module_name}: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate api_stable signature snapshot.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    if not args.quiet:
        print("[update_protected_signatures] Regenerating API-stable signatures.")
        print(
            "Any RuntimeWarnings or '[WARN] Skipped …' messages you see below are expected – they merely\n"
            "indicate helper packages or CLI entry-points that don’t need to be imported for signature\n"
            "collection.",
            flush=True,
        )

    # Prepend 'src' directory to sys.path so top-level package 'portfolio_backtester' is importable
    repo_root = Path(__file__).parent.parent
    src_path = repo_root / "src"
    # Ensure both repo root (for 'src.*' namespace packages) and the
    # nested 'src' path (for top-level 'portfolio_backtester') are on PYTHONPATH.
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(src_path))

    # Eager import the full package tree
    _import_all_package_files(PACKAGE_ROOT)

    # Import key sub-packages that sit outside the main tree but still inside portfolio_backtester
    importlib.import_module("portfolio_backtester.core")
    importlib.import_module("portfolio_backtester.roro_signals")

    from portfolio_backtester.api_stability import registry  # noqa: E402

    refs = {}
    for k, meta in registry.get_registered_methods().items():
        d = meta.as_dict()
        d["type_hints"] = {name: str(t) for name, t in d["type_hints"].items()}
        refs[k] = d
    SIGNATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SIGNATURE_PATH.write_text(json.dumps(refs, indent=2, sort_keys=True))

    if not args.quiet:
        print(f"Signature file written: {SIGNATURE_PATH}\nEntries: {len(refs)}")


if __name__ == "__main__":  # pragma: no cover
    main()
