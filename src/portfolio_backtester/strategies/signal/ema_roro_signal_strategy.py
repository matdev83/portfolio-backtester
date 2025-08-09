"""
EMA Crossover Strategy with RoRo Signal Integration

This strategy extends the basic EMA crossover strategy by integrating Risk-on/Risk-off (RoRo) signals.
During risk-off periods, the leverage is reduced by half to manage downside risk.

- Long signal: Fast EMA crosses above Slow EMA
- Exit signal: Fast EMA crosses below Slow EMA
- Risk management: During RoRo risk-off periods, leverage is reduced by 50%
"""



from ..builtins.signal.ema_roro_signal_strategy import EmaRoroSignalStrategy  # noqa: F401

__all__ = ["EmaRoroSignalStrategy"]
