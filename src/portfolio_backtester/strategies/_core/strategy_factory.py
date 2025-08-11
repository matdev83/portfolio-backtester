"""Core location for StrategyFactory after strategy layout refactor.

User guidance
-------------
- Built-in strategies now live under ``portfolio_backtester.strategies.builtins``
  with categories ``portfolio``, ``signal``, and ``meta``. Example:
  ``portfolio_backtester.strategies.builtins.portfolio.simple_momentum_portfolio_strategy``.
- Framework internals (base classes, registry) live under
  ``portfolio_backtester.strategies._core``.
- Prefer strategy discovery + factory usage over importing concrete classes directly:

    from portfolio_backtester.strategies._core.strategy_factory import StrategyFactory
    strategy = StrategyFactory.create_strategy({"type": "SimpleMomentumPortfolioStrategy", ...})

This module re-exports the concrete ``StrategyFactory`` API from the new
implementation module in ``_core``.
"""

from .strategy_factory_impl import StrategyFactory  # noqa: F401

__all__ = ["StrategyFactory"]
