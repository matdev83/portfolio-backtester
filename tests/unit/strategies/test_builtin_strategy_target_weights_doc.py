"""Built-in/user strategies advertise ``generate_target_weights`` as the full-period authoring API."""

from __future__ import annotations

import inspect

import pytest

from portfolio_backtester.strategies.builtins.portfolio.fixed_weight_portfolio_strategy import (
    FixedWeightPortfolioStrategy,
)
from portfolio_backtester.strategies.builtins.signal.donchian_asri_signal_strategy import (
    AsriThresholdSignalStrategy,
    DonchianAsriSignalStrategy,
)
from portfolio_backtester.strategies.builtins.signal.ema_crossover_signal_strategy import (
    EmaCrossoverSignalStrategy,
)
from portfolio_backtester.strategies.builtins.signal.ema_roro_signal_strategy import (
    EmaRoroSignalStrategy,
)
from portfolio_backtester.strategies.builtins.signal.mmm_qs_swing_nasdaq_signal_strategy import (
    MmmQsSwingNasdaqSignalStrategy,
)
from portfolio_backtester.strategies.builtins.signal.seasonal_signal_strategy import (
    SeasonalSignalStrategy,
)
from portfolio_backtester.strategies.builtins.signal.uvxy_rsi_signal_strategy import (
    UvxyRsiSignalStrategy,
)
from portfolio_backtester.strategies.user.signal.hello_world_signal_strategy import (
    HelloWorldSignalStrategy,
)


_PUBLISHING_STRATEGIES = pytest.mark.parametrize(
    "cls",
    [
        FixedWeightPortfolioStrategy,
        EmaCrossoverSignalStrategy,
        EmaRoroSignalStrategy,
        MmmQsSwingNasdaqSignalStrategy,
        DonchianAsriSignalStrategy,
        AsriThresholdSignalStrategy,
        UvxyRsiSignalStrategy,
        HelloWorldSignalStrategy,
        SeasonalSignalStrategy,
    ],
)


@_PUBLISHING_STRATEGIES
def test_strategy_defines_generate_target_weights(cls: type) -> None:
    assert hasattr(cls, "generate_target_weights")
    assert callable(getattr(cls, "generate_target_weights"))


@_PUBLISHING_STRATEGIES
def test_class_docstring_documents_generate_target_weights(cls: type) -> None:
    doc = inspect.getdoc(cls) or ""
    assert (
        "generate_target_weights" in doc
    ), f"{cls.__name__} missing generate_target_weights doc cue"
