"""
Examples demonstrating the use of deprecation decorators for API stability.

This file shows practical examples of how to use @deprecated and @deprecated_signature
decorators when making breaking changes to method signatures.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.portfolio_backtester.api_stability.protection import deprecated, deprecated_signature


class ExampleStrategy:
    """Example strategy class showing deprecation patterns."""
    
    @deprecated(
        reason="This method has been replaced with a more efficient implementation",
        version="2.0",
        removal_version="3.0",
        migration_guide="Use calculate_signals_v2() instead, which provides better performance and more accurate results"
    )
    def calculate_signals(self, data, lookback_period=20):
        """
        Old signal calculation method.
        
        This method is deprecated and will be removed in version 3.0.
        """
        # Legacy implementation
        return {"signals": "old_implementation", "lookback": lookback_period}
    
    def calculate_signals_v2(self, data, lookback_period=20, use_cache=True):
        """
        New and improved signal calculation method.
        
        This is the recommended method for signal calculation.
        """
        return {
            "signals": "new_implementation", 
            "lookback": lookback_period,
            "cached": use_cache
        }
    
    @deprecated_signature(
        old_signature="set_parameters(period, threshold, format='json')",
        new_signature="set_parameters(period, threshold, output_format='json')",
        version="2.1",
        removal_version="3.0",
        parameter_mapping={"format": "output_format"}
    )
    def set_parameters(self, period, threshold, output_format='json'):
        """
        Set strategy parameters with improved parameter naming.
        
        The 'format' parameter has been renamed to 'output_format' for clarity.
        """
        return {
            "period": period,
            "threshold": threshold,
            "output_format": output_format
        }
    
    @deprecated_signature(
        old_signature="backtest(data, start_date, end_date, verbose=False, format='dict')",
        new_signature="backtest(data, start_date, end_date, show_progress=False, output_format='dict')",
        version="2.2",
        removal_version="3.0",
        parameter_mapping={"verbose": "show_progress", "format": "output_format"}
    )
    def backtest(self, data, start_date, end_date, show_progress=False, output_format='dict'):
        """
        Run backtest with improved parameter naming.
        
        Parameter changes:
        - 'verbose' renamed to 'show_progress' for clarity
        - 'format' renamed to 'output_format' for consistency
        """
        return {
            "data": "processed",
            "start": start_date,
            "end": end_date,
            "progress": show_progress,
            "format": output_format
        }


class ExamplePortfolioManager:
    """Example portfolio manager showing deprecation for complex signature changes."""
    
    @deprecated(
        reason="Method signature will change to support new risk management features",
        version="2.3",
        removal_version="3.0",
        migration_guide="This method will be replaced with rebalance_portfolio_v2() which includes risk constraints"
    )
    def rebalance_portfolio(self, weights, cash_available=10000):
        """
        Old portfolio rebalancing method.
        
        Will be replaced with a version that includes risk management.
        """
        return {"weights": weights, "cash": cash_available, "version": "old"}
    
    def rebalance_portfolio_v2(self, weights, cash_available=10000, risk_limit=0.05, max_position_size=0.1):
        """
        New portfolio rebalancing method with risk management.
        
        This is the recommended method for portfolio rebalancing.
        """
        return {
            "weights": weights, 
            "cash": cash_available,
            "risk_limit": risk_limit,
            "max_position": max_position_size,
            "version": "new"
        }


def demonstrate_deprecation_warnings():
    """
    Demonstrate how deprecation warnings work in practice.
    
    Run this function to see deprecation warnings in action.
    """
    print("=== Deprecation Examples ===\n")
    
    strategy = ExampleStrategy()
    portfolio_manager = ExamplePortfolioManager()
    
    print("1. Using deprecated method (will show warning):")
    result1 = strategy.calculate_signals("sample_data")
    print(f"Result: {result1}\n")
    
    print("2. Using new method (no warning):")
    result2 = strategy.calculate_signals_v2("sample_data")
    print(f"Result: {result2}\n")
    
    print("3. Using deprecated signature with old parameter names (will show warning):")
    result3 = strategy.set_parameters(10, 0.5, format='xml')
    print(f"Result: {result3}\n")
    
    print("4. Using new signature with new parameter names (no warning):")
    result4 = strategy.set_parameters(10, 0.5, output_format='xml')
    print(f"Result: {result4}\n")
    
    print("5. Using deprecated method with complex signature changes:")
    result5 = strategy.backtest("data", "2020-01-01", "2021-01-01", verbose=True, format='json')
    print(f"Result: {result5}\n")
    
    print("6. Using deprecated portfolio method:")
    result6 = portfolio_manager.rebalance_portfolio({"AAPL": 0.5, "GOOGL": 0.5})
    print(f"Result: {result6}\n")


if __name__ == "__main__":
    # Run with warnings enabled to see deprecation messages
    import warnings
    warnings.simplefilter("always", DeprecationWarning)
    
    demonstrate_deprecation_warnings()