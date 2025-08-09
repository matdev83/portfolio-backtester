# Direct import of optimized function - no fallback needed

from ..builtins.portfolio.filtered_lagged_momentum_portfolio_strategy import (  # noqa: F401
    FilteredLaggedMomentumPortfolioStrategy,
)

__all__ = ["FilteredLaggedMomentumPortfolioStrategy"]
