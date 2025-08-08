import warnings

# Backward compatibility imports with deprecation warnings
with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "The 'roro_signals' module is deprecated and will be removed in a future version. "
        "Please use 'risk_off_signals' instead. "
        "Migration guide: BaseRoRoSignal -> IRiskOffSignalGenerator, "
        "DummyRoRoSignal -> DummyRiskOffSignalGenerator. "
        "Note: Signal semantics have changed - True now means risk-off (was risk-on).",
        DeprecationWarning,
        stacklevel=2
    )

from .roro_signals import BaseRoRoSignal, DummyRoRoSignal

__all__ = [
    "BaseRoRoSignal",
    "DummyRoRoSignal",
]
