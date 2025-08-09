from ..builtins.portfolio.base_momentum_portfolio_strategy import (  # noqa: F401
    BaseMomentumPortfolioStrategy,
)

__all__ = ["BaseMomentumPortfolioStrategy"]

# Provide legacy module attributes used in tests
import logging as _logging

logger = _logging.getLogger(__name__)
