from abc import ABC
from .base_strategy import BaseStrategy


# Custom exception for naming convention validation
class NamingConventionError(ValueError):
    """Raised when a class name doesn't follow naming conventions."""

    def __init__(self, message):
        super().__init__(message)


class PortfolioStrategy(BaseStrategy, ABC):
    """
    Abstract base class for portfolio-based strategies.

    === FOR CODING AGENTS: AUTO-DISCOVERY REQUIREMENTS ===
    🚨 IMPORTANT: Strategies are AUTOMATICALLY DISCOVERED! 🚨
    DO NOT manually register strategies - the system finds them automatically.

    TO CREATE A PORTFOLIO STRATEGY:
    1. Class name MUST end with "PortfolioStrategy"
       ✅ Good: class MyAwesomePortfolioStrategy(PortfolioStrategy)
       ❌ Bad:  class MyAwesome(PortfolioStrategy)

    2. File name MUST end with "_portfolio_strategy.py"
       ✅ Good: my_awesome_portfolio_strategy.py
       ❌ Bad:  my_awesome.py

    3. File MUST be in: src/portfolio_backtester/strategies/portfolio/
       ✅ Good: src/portfolio_backtester/strategies/portfolio/my_awesome_portfolio_strategy.py
       ❌ Bad:  any other directory

    4. Class must be CONCRETE (implement all abstract methods)
       ✅ Good: Implement all required methods from PortfolioStrategy
       ❌ Bad:  Leave abstract methods unimplemented

    5. Import and inherit correctly:
       ✅ Good: from ..base.portfolio_strategy import PortfolioStrategy
                class MyAwesomePortfolioStrategy(PortfolioStrategy):

    The system will AUTOMATICALLY find and register your strategy!
    NEVER call register_strategy() or hardcode class names!
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__name__ == "PortfolioStrategy":
            return

        # Allow test classes and classes in test modules to bypass naming convention
        if (
            cls.__name__.startswith(("Test", "Mock", "Incomplete", "Illegal"))
            and not cls.__name__.startswith("IncorrectlyNamed")
            or cls.__module__
            and ("test" in cls.__module__ or "tests" in cls.__module__)
        ):
            return

        if not cls.__name__.endswith("PortfolioStrategy"):
            raise NamingConventionError(
                f"Portfolio-based strategy class names must end with PortfolioStrategy: {cls.__name__}"
            )
