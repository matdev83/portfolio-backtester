"""
Inspect the SOLID strategy registry and print details for a given strategy.

Usage:
  .\.venv\Scripts\python.exe scripts/inspect_registry.py --name SimpleMomentumPortfolioStrategy
"""

from __future__ import annotations

import argparse
import inspect
from typing import Optional, Type

from portfolio_backtester.strategies._core.registry import get_strategy_registry
from portfolio_backtester.strategies._core.base.base.base_strategy import BaseStrategy


def _print_strategy_info(name: str) -> None:
    registry = get_strategy_registry()
    all_strategies = registry.get_all_strategies()
    count = len(all_strategies)
    print(f"Registry count: {count}")

    strategy_class: Optional[Type[BaseStrategy]] = registry.get_strategy_class(name)

    print(f"Lookup name: {name}")
    print(f"Present: {strategy_class is not None}")
    print(f"is_strategy_registered: {registry.is_strategy_registered(name)}")
    if strategy_class is None:
        return

    print(f"Resolved: {strategy_class}")
    print(f"Module: {getattr(strategy_class, '__module__', '<unknown>')}")
    print(f"isclass: {inspect.isclass(strategy_class)}")
    try:
        print(f"issubclass(BaseStrategy): {issubclass(strategy_class, BaseStrategy)}")
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"issubclass check raised: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect strategy registry entry")
    parser.add_argument(
        "--name",
        default="SimpleMomentumPortfolioStrategy",
        help="Strategy class name to look up (default: SimpleMomentumPortfolioStrategy)",
    )
    args = parser.parse_args()
    _print_strategy_info(args.name)


if __name__ == "__main__":
    main()
