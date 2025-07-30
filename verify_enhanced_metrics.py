#!/usr/bin/env python3
"""
Verification script for enhanced performance metrics integration.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def verify_imports():
    """Verify all necessary imports work correctly."""
    print("🔍 Verifying imports...")
    
    try:
        from portfolio_backtester.trading.trade_tracker import TradeTracker, Trade
        from portfolio_backtester.reporting.performance_metrics import calculate_metrics
        from portfolio_backtester.backtester_logic.portfolio_logic import calculate_portfolio_returns
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def verify_new_metrics():
    """Verify that new metrics are properly integrated."""
    print("\\n🧪 Verifying new metrics integration...")
    
    try:
        import pandas as pd
        import numpy as np
        from portfolio_backtester.reporting.performance_metrics import calculate_metrics
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)
        benchmark_returns = pd.Series(np.random.normal(0.0008, 0.015, 100), index=dates)
        
        # Test without trade stats
        metrics_basic = calculate_metrics(returns, benchmark_returns, 'SPY')
        print(f"✅ Basic metrics calculated: {len(metrics_basic)} metrics")
        
        # Test with trade stats
        trade_stats = {
            'num_trades': 10,
            'win_rate_pct': 60.0,
            'total_commissions_paid': 50.0,
            'avg_mfe': 100.0,
            'avg_mae': -50.0,
            'information_score': 0.8,
            'min_trade_duration_days': 1,
            'max_trade_duration_days': 30,
            'mean_trade_duration_days': 10.0,
            'max_margin_load': 0.9,
            'mean_margin_load': 0.7,
            'trades_per_month': 2.0
        }
        
        metrics_enhanced = calculate_metrics(returns, benchmark_returns, 'SPY', trade_stats=trade_stats)
        print(f"✅ Enhanced metrics calculated: {len(metrics_enhanced)} metrics")
        
        # Verify new metrics are present
        expected_new_metrics = [
            'Number of Trades',
            'Win Rate (%)',
            'Commissions Paid',
            'Avg MFE',
            'Avg MAE',
            'Information Score',
            'Min Trade Duration (days)',
            'Max Trade Duration (days)',
            'Mean Trade Duration (days)',
            'Max Margin Load',
            'Mean Margin Load',
            'Trades per Month',
            'Max DD Recovery Time (days)',
            'Max Flat Period (days)'
        ]
        
        missing = [m for m in expected_new_metrics if m not in metrics_enhanced.index]
        if missing:
            print(f"❌ Missing metrics: {missing}")
            return False
        
        print("✅ All expected new metrics present")
        
        # Verify values are correctly passed through
        assert metrics_enhanced['Number of Trades'] == 10
        assert metrics_enhanced['Win Rate (%)'] == 60.0
        assert metrics_enhanced['Avg MFE'] == 100.0
        print("✅ Metric values correctly integrated")
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_trade_tracker():
    """Verify trade tracker functionality."""
    print("\\n📊 Verifying trade tracker...")
    
    try:
        import pandas as pd
        import numpy as np
        from portfolio_backtester.trading.trade_tracker import TradeTracker
        
        # Initialize tracker
        tracker = TradeTracker(portfolio_value=100000)
        
        # Simulate some trades
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        
        for i, date in enumerate(dates):
            weights = pd.Series({
                'AAPL': 0.5 if i < 3 else 0.0,
                'MSFT': 0.3 if i < 4 else 0.0
            })
            
            prices = pd.Series({
                'AAPL': 150 + i,
                'MSFT': 300 + i * 2
            })
            
            tracker.update_positions(date, weights, prices, 5.0)
            tracker.update_mfe_mae(date, prices)
        
        # Close positions
        final_prices = pd.Series({'AAPL': 160, 'MSFT': 320})
        tracker.close_all_positions(dates[-1], final_prices)
        
        # Get statistics
        stats = tracker.get_trade_statistics()
        
        print(f"✅ Trade tracker generated {stats['num_trades']} trades")
        print(f"✅ Win rate: {stats['win_rate_pct']:.1f}%")
        
        # Verify trade history
        history = tracker.get_trade_history_dataframe()
        print(f"✅ Trade history: {len(history)} records")
        
        return True
        
    except Exception as e:
        print(f"❌ Trade tracker verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_portfolio_logic():
    """Verify portfolio logic integration."""
    print("\\n💼 Verifying portfolio logic integration...")
    
    try:
        import pandas as pd
        import numpy as np
        from portfolio_backtester.backtester_logic.portfolio_logic import calculate_portfolio_returns
        
        # Create minimal test data
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        tickers = ['AAPL', 'MSFT']
        
        # Create signals (weights)
        signals = pd.DataFrame(
            np.random.uniform(0, 0.5, (len(dates), len(tickers))),
            index=dates,
            columns=tickers
        )
        
        # Create price data
        price_data = pd.DataFrame(
            np.random.uniform(100, 200, (len(dates), len(tickers))),
            index=dates,
            columns=tickers
        )
        
        # Create returns data
        returns_data = price_data.pct_change().fillna(0)
        
        # Test configuration
        scenario_config = {
            'rebalance_frequency': 'D',
            'transaction_costs_bps': 10
        }
        
        global_config = {
            'portfolio_value': 100000,
            'slippage_bps': 2.5,
            'commission_min_per_order': 1.0,
            'commission_per_share': 0.005,
            'commission_max_percent_of_trade': 0.005
        }
        
        # Test without trade tracking
        result_no_tracking = calculate_portfolio_returns(
            signals, scenario_config, price_data, returns_data, 
            tickers, global_config, track_trades=False
        )
        
        if isinstance(result_no_tracking, tuple):
            portfolio_returns, trade_tracker = result_no_tracking
            assert trade_tracker is None, "Should not have trade tracker when disabled"
        else:
            portfolio_returns = result_no_tracking
        
        print("✅ Portfolio logic without trade tracking works")
        
        # Test with trade tracking
        result_with_tracking = calculate_portfolio_returns(
            signals, scenario_config, price_data, returns_data,
            tickers, global_config, track_trades=True
        )
        
        if isinstance(result_with_tracking, tuple):
            portfolio_returns_tracked, trade_tracker = result_with_tracking
            assert trade_tracker is not None, "Should have trade tracker when enabled"
            stats = trade_tracker.get_trade_statistics()
            print(f"✅ Portfolio logic with trade tracking works: {stats['num_trades']} trades")
        else:
            print("❌ Expected tuple return when trade tracking enabled")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Portfolio logic verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification tests."""
    print("=" * 80)
    print("ENHANCED PERFORMANCE METRICS VERIFICATION")
    print("=" * 80)
    
    tests = [
        verify_imports,
        verify_new_metrics,
        verify_trade_tracker,
        verify_portfolio_logic
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test.__name__:<30} {status}")
    
    print(f"\\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\\n🎉 ALL VERIFICATIONS PASSED!")
        print("Enhanced performance metrics are ready for use.")
        return 0
    else:
        print(f"\\n❌ {total - passed} verification(s) failed.")
        print("Please check the implementation before using enhanced metrics.")
        return 1

if __name__ == "__main__":
    sys.exit(main())