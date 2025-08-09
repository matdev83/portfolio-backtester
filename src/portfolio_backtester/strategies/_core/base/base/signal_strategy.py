from abc import ABC
from .base_strategy import BaseStrategy


# Custom exception for naming convention validation
class NamingConventionError(ValueError):
    """Raised when a class name doesn't follow naming conventions."""

    def __init__(self, message):
        super().__init__(message)


class SignalStrategy(BaseStrategy, ABC):
    """
    Abstract base class for signal-based strategies.

    === FOR CODING AGENTS: AUTO-DISCOVERY REQUIREMENTS ===
    🚨 IMPORTANT: Strategies are AUTOMATICALLY DISCOVERED! 🚨
    DO NOT manually register strategies - the system finds them automatically.

    TO CREATE A SIGNAL STRATEGY:
    1. Class name MUST end with "SignalStrategy"
       ✅ Good: class MyAwesomeSignalStrategy(SignalStrategy)
       ❌ Bad:  class MyAwesome(SignalStrategy)

    2. File name MUST end with "_signal_strategy.py"
       ✅ Good: my_awesome_signal_strategy.py
       ❌ Bad:  my_awesome.py

    3. File MUST be in: src/portfolio_backtester/strategies/signal/
       ✅ Good: src/portfolio_backtester/strategies/signal/my_awesome_signal_strategy.py
       ❌ Bad:  any other directory

    4. Class must be CONCRETE (implement all abstract methods)
       ✅ Good: Implement all required methods from SignalStrategy
       ❌ Bad:  Leave abstract methods unimplemented

    5. Import and inherit correctly:
       ✅ Good: from ..base.signal_strategy import SignalStrategy
                class MyAwesomeSignalStrategy(SignalStrategy):

    The system will AUTOMATICALLY find and register your strategy!
    NEVER call register_strategy() or hardcode class names!
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__name__ == "SignalStrategy":
            return

        # Allow test classes and classes in test modules to bypass naming convention
        if (
            cls.__name__.startswith(("Test", "Mock", "Incomplete", "Illegal"))
            and not cls.__name__.startswith("IncorrectlyNamed")
            or cls.__module__
            and ("test" in cls.__module__ or "tests" in cls.__module__)
        ):
            return

        if not cls.__name__.endswith("SignalStrategy"):
            raise NamingConventionError(
                f"Signal-based strategy class names must end with SignalStrategy: {cls.__name__}"
            )
