from ..builtins.portfolio.calmar_momentum_portfolio_strategy import (  # noqa: F401
    CalmarMomentumPortfolioStrategy,
)

__all__ = ["CalmarMomentumPortfolioStrategy"]

# Provide legacy attribute used by tests for patching and proxy it into builtins module
from ...features.calmar_ratio import CalmarRatio  # noqa: F401
from ..builtins.portfolio import calmar_momentum_portfolio_strategy as _builtins_calmar
from typing import Any as _Any


def _CalmarRatio_proxy(*args: _Any, **kwargs: _Any):  # noqa: N802
    return CalmarRatio(*args, **kwargs)


setattr(_builtins_calmar, "CalmarRatio", _CalmarRatio_proxy)
