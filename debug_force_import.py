#!/usr/bin/env python3
"""Force fresh import of all strategy modules."""

import sys

# Clear any cached strategy modules
modules_to_clear = [m for m in sys.modules.keys() if "strategy" in m.lower()]
print(f"Clearing {len(modules_to_clear)} cached strategy modules...")
for module_name in modules_to_clear:
    if module_name in sys.modules:
        del sys.modules[module_name]

# Force reimport of the testing strategies
print("Force importing testing strategies...")
try:

    print("✅ Testing strategies imported successfully")
except Exception as e:
    print(f"❌ Error importing testing strategies: {e}")

# Force reimport of meta strategies
print("Force importing meta strategies...")
try:

    print("✅ Meta strategies imported successfully")
except Exception as e:
    print(f"❌ Error importing meta strategies: {e}")

# Now test the enumerator again
print("\nTesting strategy enumerator after fresh imports...")
from portfolio_backtester.interfaces.strategy_enumerator_interface import create_strategy_enumerator

enumerator = create_strategy_enumerator()
strategies = enumerator.enumerate_strategies_with_params()
available = list(strategies.keys())
print(f"Available strategies ({len(available)}): {available}")

# Check our specific strategies
print(f'simple_meta found: {"simple_meta" in strategies}')
print(f'stop_loss_tester found: {"stop_loss_tester" in strategies}')

# Check what base strategy sees
from portfolio_backtester.strategies.base.base_strategy import BaseStrategy

subclasses = [cls.__name__ for cls in BaseStrategy.__subclasses__()]
print(f"BaseStrategy.__subclasses__(): {subclasses}")
