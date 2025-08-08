"""
Signal Strategy Components

This module contains signal generation components that implement the ISignalGenerator interface.
Importing this module ensures all signal generators are registered with the factory.
"""

# Import signal generators to trigger factory registration
from ...utils.signal_processing.signal_generator import UvxySignalGenerator
from .ema_signal_generator import EmaCrossoverSignalGenerator

# Import the factory for external use
from ...interfaces.signal_generator_interface import signal_generator_factory

__all__ = ["UvxySignalGenerator", "EmaCrossoverSignalGenerator", "signal_generator_factory"]
