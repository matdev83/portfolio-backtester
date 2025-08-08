#!/usr/bin/env python3
"""Debug strategy resolution issues."""

# Let's test the deeper validation layers directly
from portfolio_backtester.interfaces.strategy_resolver_interface import create_strategy_resolver
from portfolio_backtester.interfaces.strategy_enumerator_interface import create_strategy_enumerator

print("Testing strategy resolver...")
resolver = create_strategy_resolver()
result = resolver.resolve_strategy("simple_meta")
print(f"simple_meta resolved: {result}")

result2 = resolver.resolve_strategy("stop_loss_tester")
print(f"stop_loss_tester resolved: {result2}")

print("\nTesting strategy enumerator...")
enumerator = create_strategy_enumerator()
strategies = enumerator.enumerate_strategies_with_params()
print(f'simple_meta in strategies: {"simple_meta" in strategies}')
print(f'stop_loss_tester in strategies: {"stop_loss_tester" in strategies}')

# Show available strategies
available = list(strategies.keys())
print(f"Available strategies ({len(available)}): {available}")

# Check specific strategies we need
for needed in ["simple_meta", "stop_loss_tester"]:
    if needed in strategies:
        strategy_class = strategies[needed]
        print(f"\n{needed} -> {strategy_class}")
        try:
            params = strategy_class.tunable_parameters()
            print(
                f"  Tunable params: {list(params.keys()) if isinstance(params, dict) else params}"
            )
        except Exception as e:
            print(f"  Error getting params: {e}")
    else:
        print(f"\n{needed} -> NOT FOUND")
