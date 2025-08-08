"""
Integration test for daily evaluation system.

Tests the complete workflow from intramonth strategy through daily evaluation
to accurate trade duration calculation.
"""

import pandas as pd
import numpy as np
from unittest.mock import Mock
from portfolio_backtester.optimization.wfo_window import WFOWindow
from portfolio_backtester.backtesting.window_evaluator import WindowEvaluator
from portfolio_backtester.backtesting.position_tracker import PositionTracker
from portfolio_backtester.optimization.evaluator import BacktestEvaluator


class MockIntramonthStrategy:
    """Mock intramonth strategy for testing daily evaluation."""
    
    def __init__(self, entry_day=1, hold_days=5):
        self.entry_day = entry_day
        self.hold_days = hold_days
        self.current_positions = {}
        self.exit_dates = {}
    
    def generate_signals(self, all_historical_data, benchmark_historical_data, 
                        non_universe_historical_data, current_date, **kwargs):
        """Generate signals for the mock strategy."""
        
        # Simple logic: enter on first business day of month, exit after hold_days
        if current_date.day <= 3:  # Approximate first few business days
            # Check if we should enter (not already in position)
            if 'TLT' not in self.current_positions:
                # Enter position
                self.current_positions['TLT'] = current_date
                self.exit_dates['TLT'] = current_date + pd.Timedelta(days=self.hold_days)
                return pd.DataFrame({'TLT': [1.0], 'SPY': [0.0]}, index=[current_date])
        
        # Check if we should exit
        if 'TLT' in self.current_positions:
            if current_date >= self.exit_dates['TLT']:
                # Exit position
                del self.current_positions['TLT']
                del self.exit_dates['TLT']
                return pd.DataFrame({'TLT': [0.0], 'SPY': [0.0]}, index=[current_date])
            else:
                # Hold position
                return pd.DataFrame({'TLT': [1.0], 'SPY': [0.0]}, index=[current_date])
        
        # No position
        return pd.DataFrame({'TLT': [0.0], 'SPY': [0.0]}, index=[current_date])


class TestDailyEvaluationIntegration:
    """Integration tests for the complete daily evaluation system."""
    
    def test_complete_daily_evaluation_workflow(self):
        """Test the complete workflow from strategy to trade duration calculation."""
        
        # Create test window (short for testing)
        window = WFOWindow(
            train_start=pd.Timestamp('2024-01-01'),
            train_end=pd.Timestamp('2024-12-31'),
            test_start=pd.Timestamp('2025-01-01'),
            test_end=pd.Timestamp('2025-01-15'),  # 15-day test window
            evaluation_frequency='D',
            strategy_name='test_intramonth'
        )
        
        # Create test data
        dates = pd.bdate_range('2024-01-01', '2025-01-20')
        daily_data = pd.DataFrame({
            'TLT': np.random.randn(len(dates)) * 0.01 + 1.0001,  # Small random returns
            'SPY': np.random.randn(len(dates)) * 0.01 + 1.0001
        }, index=dates)
        daily_data = daily_data.cumprod() * 100  # Convert to price series
        
        benchmark_data = daily_data[['SPY']].copy()
        
        # Create mock strategy
        strategy = MockIntramonthStrategy(entry_day=1, hold_days=5)
        
        # Create window evaluator
        evaluator = WindowEvaluator()
        
        # Evaluate the window
        result = evaluator.evaluate_window(
            window=window,
            strategy=strategy,
            daily_data=daily_data,
            benchmark_data=benchmark_data,
            universe_tickers=['TLT', 'SPY'],
            benchmark_ticker='SPY'
        )
        
        # Verify results
        assert result is not None
        assert hasattr(result, 'trades')
        assert hasattr(result, 'final_weights')
        assert hasattr(result, 'window_returns')
        
        # Should have some trades (strategy enters and exits positions)
        if result.trades:
            # Check trade duration is reasonable (around 5 business days)
            for trade in result.trades:
                assert hasattr(trade, 'duration_days')
                assert 3 <= trade.duration_days <= 7  # Allow some flexibility
                assert trade.ticker == 'TLT'
    
    def test_position_tracker_with_realistic_signals(self):
        """Test position tracker with realistic intramonth strategy signals."""
        
        tracker = PositionTracker()
        
        # Create test dates (first 10 business days of January)
        dates = pd.bdate_range('2025-01-01', '2025-01-15')
        
        # Simulate intramonth strategy: enter on day 1, exit on day 6
        for i, date in enumerate(dates):
            if i == 0:
                # Enter position on first day
                signals = pd.DataFrame({'TLT': [1.0], 'SPY': [0.0]}, index=[date])
            elif i == 5:
                # Exit position on 6th day (5 business days later)
                signals = pd.DataFrame({'TLT': [0.0], 'SPY': [0.0]}, index=[date])
            else:
                # Hold position
                if i < 5:
                    signals = pd.DataFrame({'TLT': [1.0], 'SPY': [0.0]}, index=[date])
                else:
                    signals = pd.DataFrame({'TLT': [0.0], 'SPY': [0.0]}, index=[date])
            
            tracker.update_positions(signals, date)
        
        # Check completed trades
        trades = tracker.get_completed_trades()
        assert len(trades) == 1
        
        trade = trades[0]
        assert trade.ticker == 'TLT'
        assert trade.entry_date == dates[0]
        assert trade.exit_date == dates[5]
        assert trade.duration_days == 5  # Exactly 5 business days
    
    def test_window_evaluator_caching(self):
        """Test that window evaluator caching works correctly."""
        
        evaluator = WindowEvaluator()
        
        # Create test data
        dates = pd.date_range('2024-01-01', '2025-01-10', freq='D')
        data = pd.DataFrame({
            'TLT': np.ones(len(dates)) * 100,
            'SPY': np.ones(len(dates)) * 400
        }, index=dates)
        
        train_start = pd.Timestamp('2024-01-01')
        current_date = pd.Timestamp('2024-06-30')
        
        # First call should populate cache
        result1 = evaluator._get_historical_data(data, current_date, train_start)
        assert len(evaluator.data_cache) == 1
        
        # Second call should use cache
        result2 = evaluator._get_historical_data(data, current_date, train_start)
        assert result1.equals(result2)
        assert len(evaluator.data_cache) == 1  # No new cache entries
        
        # Different date should create new cache entry
        new_date = pd.Timestamp('2024-07-31')
        result3 = evaluator._get_historical_data(data, new_date, train_start)
        assert len(evaluator.data_cache) == 2
        assert not result1.equals(result3)  # Different data
    
    def test_enhanced_evaluator_integration(self):
        """Test integration of enhanced evaluator with daily evaluation."""
        
        # Create enhanced evaluator
        evaluator = BacktestEvaluator(['sharpe_ratio'], False)
        
        # Test frequency determination
        intramonth_config = {
            'strategy_class': 'SeasonalSignalStrategy',
            'name': 'test_strategy'
        }
        
        freq = evaluator._determine_evaluation_frequency(intramonth_config)
        assert freq == 'D'
        
        # Test universe ticker extraction
        mock_strategy = Mock()
        mock_strategy.get_universe.return_value = ['TLT', 'GLD']
        
        tickers = evaluator._get_universe_tickers(mock_strategy)
        assert tickers == ['TLT', 'GLD']
        
        # Test metrics calculation with trades
        returns = pd.Series([0.01, 0.02, -0.005, 0.015, 0.01])
        
        mock_trades = []
        for i in range(3):
            trade = Mock()
            trade.duration_days = 4 + i  # 4, 5, 6 days
            mock_trades.append(trade)
        
        metrics = evaluator._calculate_metrics(returns, mock_trades)
        
        # Check that trade metrics are included
        assert 'num_trades' in metrics
        assert 'avg_trade_duration' in metrics
        assert 'max_trade_duration' in metrics
        assert 'min_trade_duration' in metrics
        
        assert metrics['num_trades'] == 3
        assert metrics['avg_trade_duration'] == 5.0
        assert metrics['max_trade_duration'] == 6
        assert metrics['min_trade_duration'] == 4
    
    def test_daily_vs_monthly_evaluation_comparison(self):
        """Test that daily evaluation produces different results than monthly."""
        
        # This test demonstrates why daily evaluation is necessary
        # for intramonth strategies
        
        # Create position tracker for daily evaluation
        daily_tracker = PositionTracker()
        
        # Create dates spanning multiple months
        dates = pd.bdate_range('2025-01-01', '2025-02-28')
        
        # Simulate intramonth strategy with monthly evaluation
        # (only evaluated on first day of each month)
        monthly_eval_dates = [dates[0], dates[22]]  # Approx first day of Jan and Feb
        
        # With monthly evaluation, position would be held for entire month
        for date in monthly_eval_dates:
            if date == dates[0]:
                # Enter position in January
                signals = pd.DataFrame({'TLT': [1.0]}, index=[date])
            else:
                # Exit position in February
                signals = pd.DataFrame({'TLT': [0.0]}, index=[date])
            
            daily_tracker.update_positions(signals, date)
        
        monthly_trades = daily_tracker.get_completed_trades()
        
        # Reset for daily evaluation
        daily_tracker.reset()
        
        # With daily evaluation, position can be closed mid-month
        for i, date in enumerate(dates[:10]):  # First 10 days
            if i == 0:
                # Enter position
                signals = pd.DataFrame({'TLT': [1.0]}, index=[date])
            elif i == 5:
                # Exit position after 5 days
                signals = pd.DataFrame({'TLT': [0.0]}, index=[date])
            else:
                # Hold or stay out
                weight = 1.0 if i < 5 else 0.0
                signals = pd.DataFrame({'TLT': [weight]}, index=[date])
            
            daily_tracker.update_positions(signals, date)
        
        daily_trades = daily_tracker.get_completed_trades()
        
        # Compare results
        if monthly_trades and daily_trades:
            monthly_duration = monthly_trades[0].duration_days
            daily_duration = daily_trades[0].duration_days
            
            # Daily evaluation should produce much shorter trade duration
            assert daily_duration < monthly_duration
            assert daily_duration == 5  # Exactly as intended
            assert monthly_duration > 15  # Much longer due to monthly evaluation
    
    def test_wfo_window_evaluation_dates_integration(self):
        """Test integration between WFO window and evaluation date generation."""
        
        # Create window with daily evaluation
        window = WFOWindow(
            train_start=pd.Timestamp('2024-01-01'),
            train_end=pd.Timestamp('2024-12-31'),
            test_start=pd.Timestamp('2025-01-01'),
            test_end=pd.Timestamp('2025-01-31'),
            evaluation_frequency='D'
        )
        
        # Create available dates (business days)
        available_dates = pd.bdate_range('2025-01-01', '2025-01-31')
        
        # Get evaluation dates
        eval_dates = window.get_evaluation_dates(available_dates)
        
        # Should get all business days in January
        assert len(eval_dates) == len(available_dates)
        assert all(eval_dates == available_dates)
        
        # Test with monthly evaluation
        monthly_window = WFOWindow(
            train_start=pd.Timestamp('2024-01-01'),
            train_end=pd.Timestamp('2024-12-31'),
            test_start=pd.Timestamp('2025-01-01'),
            test_end=pd.Timestamp('2025-01-31'),
            evaluation_frequency='M'
        )
        
        monthly_eval_dates = monthly_window.get_evaluation_dates(available_dates)
        
        # Should get only first date
        assert len(monthly_eval_dates) == 1
        assert monthly_eval_dates[0] == pd.Timestamp('2025-01-01')
        
        # Daily should have many more evaluation dates than monthly
        assert len(eval_dates) > len(monthly_eval_dates)
    
    def test_end_to_end_trade_duration_accuracy(self):
        """End-to-end test verifying accurate trade duration calculation."""
        
        # This test simulates the complete workflow and verifies that
        # trade durations match the strategy's intended hold period
        
        # Create a simple strategy that holds for exactly 7 business days
        class TestStrategy:
            def __init__(self):
                self.positions = {}
                self.exit_dates = {}
            
            def generate_signals(self, all_historical_data, benchmark_historical_data,
                               non_universe_historical_data, current_date, **kwargs):
                
                # Enter position on first call
                if not self.positions and current_date.day == 1:
                    self.positions['TLT'] = current_date
                    # Calculate exit date (7 business days later)
                    exit_date = current_date
                    for _ in range(7):
                        exit_date = exit_date + pd.Timedelta(days=1)
                        while exit_date.weekday() >= 5:  # Skip weekends
                            exit_date = exit_date + pd.Timedelta(days=1)
                    self.exit_dates['TLT'] = exit_date
                    return pd.DataFrame({'TLT': [1.0]}, index=[current_date])
                
                # Exit position on calculated date
                elif 'TLT' in self.positions and current_date >= self.exit_dates['TLT']:
                    del self.positions['TLT']
                    del self.exit_dates['TLT']
                    return pd.DataFrame({'TLT': [0.0]}, index=[current_date])
                
                # Hold position
                elif 'TLT' in self.positions:
                    return pd.DataFrame({'TLT': [1.0]}, index=[current_date])
                
                # No position
                else:
                    return pd.DataFrame({'TLT': [0.0]}, index=[current_date])
        
        # Create test window
        window = WFOWindow(
            train_start=pd.Timestamp('2024-01-01'),
            train_end=pd.Timestamp('2024-12-31'),
            test_start=pd.Timestamp('2025-01-01'),
            test_end=pd.Timestamp('2025-01-20'),
            evaluation_frequency='D'
        )
        
        # Create test data
        dates = pd.bdate_range('2024-01-01', '2025-01-25')
        daily_data = pd.DataFrame({
            'TLT': np.ones(len(dates)) * 100,  # Constant price for simplicity
            'SPY': np.ones(len(dates)) * 400
        }, index=dates)
        
        # Create strategy and evaluator
        strategy = TestStrategy()
        evaluator = WindowEvaluator()
        
        # Evaluate window
        result = evaluator.evaluate_window(
            window=window,
            strategy=strategy,
            daily_data=daily_data,
            benchmark_data=daily_data[['SPY']],
            universe_tickers=['TLT', 'SPY'],
            benchmark_ticker='SPY'
        )
        
        # Verify trade duration accuracy
        assert result.trades is not None
        if result.trades:
            trade = result.trades[0]
            assert trade.ticker == 'TLT'
            # Should be exactly 7 business days (the strategy's intended hold period)
            assert trade.duration_days == 7
            
            # Verify entry and exit dates are business days apart
            actual_duration = len(pd.bdate_range(trade.entry_date, trade.exit_date)) - 1
            assert actual_duration == trade.duration_days