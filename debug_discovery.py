import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

try:
    from portfolio_backtester.strategies._core.registry import get_strategy_registry
    reg = get_strategy_registry()
    strategies = reg.get_all_strategies()
    print(f"Total strategies: {len(strategies)}")
    print("Strategies found:")
    for name in sorted(strategies.keys()):
        print(f"  - {name}")
        
    target = "DriftRegimeConditionalFactorPortfolioStrategy"
    if target in strategies:
        print(f"\nSUCCESS: {target} found!")
    else:
        print(f"\nFAILURE: {target} NOT found!")
        
except Exception as e:
    print(f"Error during discovery: {e}")
    import traceback
    traceback.print_exc()
