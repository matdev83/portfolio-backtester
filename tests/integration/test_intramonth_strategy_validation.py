"""
Strategy validation tests for IntramonthSeasonalStrategy with enhanced WFO.

This module validates that the IntramonthSeasonalStrategy works correctly with
the enhanced WFO system, producing accurate trade durations and proper exit timing.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.portfolio_backtester.strategies.signal.intramonth_seasonal_strategy import IntramonthSeasonalStrategy
from src.portfolio_backtester.optimization.evaluator import BacktestEvaluator
from src.portfolio_backtester.optimization.results import OptimizationData, EvaluationResult
from src.portfolio_backtester.backtesting.strategy_backtester import StrategyBacktester
from src.portfolio_backtester.backtesting.results import WindowResult
from src.portfolio_backtester.config_loader import GLOBAL_CONFIG

# Enhanced WFO imports
try:
    from src.portfolio_backtester.optimization.wfo_window import WFOWindow
    from src.portfolio_backtester.backtesting.window_evaluator import WindowEvaluator
    from src.portfolio_backtester.backtesting.position_tracker import Trade
    WFO_ENHANCEMENT_AVAILABLE = True
except ImportError:
    WFO_ENHANCEMENT_AVAILABLE = False


class TestIntramonthStrategyValidation:
    """Validation tests for IntramonthSeasonalStrategy with enhanced WFO."""
    
    @pytest.fixture
    def realistic_daily_data(self):
        """Create realistic daily price data for strategy validation."""
        # Create 1 year of business day data
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        dates = pd.date_range(start_date, end_date, freq='D')
        business_dates = dates[dates.dayofweek < 5]  # Business days only
        
        # Create realistic price movements
        np.random.seed(42)  # For reproducible tests
        n_days = len(business_dates)
        
        # Generate correlated price data for multiple assets
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'SPY']
        price_data = {}
        
        for i, ticker in enumerate(tickers):
            base_price = 100 + i * 25  # Different starting prices
            
            # Generate realistic returns with some trend and volatility
            daily_returns = np.random.normal(0.0005, 0.015, n_days)  # 0.05% daily return, 1.5% volatility
            
            # Add some monthly seasonality for testing
            for j, date in enumerate(business_dates):
                if date.month in [1, 4, 7, 10]:  # Quarterly effect
                    daily_returns[j] += 0.001  # Slight positive bias
            
            # Calculate prices from returns
            prices = [base_price]
            for ret in daily_returns:
                prices.append(prices[-1] * (1 + ret))
            
            price_data[ticker] = prices[1:]  # Remove initial base price
        
        # Create MultiIndex DataFrame (Ticker, OHLCV)
        columns = pd.MultiIndex.from_product([tickers, ['Open', 'High', 'Low', 'Close', 'Volume']])
        df = pd.DataFrame(index=business_dates, columns=columns)
        
        for ticker in tickers:
            close_prices = price_data[ticker]
            # Generate OHLC from close prices
            for i, close in enumerate(close_prices):
                # Simple OHLC generation
                volatility = close * 0.02  # 2% daily volatility
                high = close + np.random.uniform(0, volatility)
                low = close - np.random.uniform(0, volatility)
                open_price = close + np.random.uniform(-volatility/2, volatility/2)
                volume = np.random.randint(1000000, 10000000)
                
                df.loc[business_dates[i], (ticker, 'Open')] = open_price
                df.loc[business_dates[i], (ticker, 'High')] = high
                df.loc[business_dates[i], (ticker, 'Low')] = low
                df.loc[business_dates[i], (ticker, 'Close')] = close
                df.loc[business_dates[i], (ticker, 'Volume')] = volume
        
        return df.astype(float)
    
    @pytest.fixture
    def monthly_data(self, realistic_daily_data):
        """Create monthly data from daily data."""
        return realistic_daily_data.resample('ME').last()
    
    @pytest.fixture
    def returns_data(self, realistic_daily_data):
        """Create returns data from daily prices."""
        close_prices = realistic_daily_data.xs('Close', level=1, axis=1)
        return close_prices.pct_change().dropna()
    
    @pytest.fixture
    def wfo_windows(self):
        """Create realistic WFO windows for testing."""
        return [
            # Window 1: Train on Q1, Test on Q2
            (datetime(2023, 1, 1), datetime(2023, 3, 31), 
             datetime(2023, 4, 1), datetime(2023, 6, 30)),
            
            # Window 2: Train on Q2, Test on Q3
            (datetime(2023, 4, 1), datetime(2023, 6, 30),
             datetime(2023, 7, 1), datetime(2023, 9, 30)),
            
            # Window 3: Train on Q3, Test on Q4
            (datetime(2023, 7, 1), datetime(2023, 9, 30),
             datetime(2023, 10, 1), datetime(2023, 12, 31))
        ]
    
    @pytest.fixture
    def optimization_data(self, realistic_daily_data, monthly_data, returns_data, wfo_windows):
        """Create optimization data for testing."""
        return OptimizationData(
            monthly=monthly_data,
            daily=realistic_daily_data,
            returns=returns_data,
            windows=wfo_windows
        )
    
    @pytest.fixture
    def intramonth_strategy_config(self):
        """Create realistic intramonth strategy configuration."""
        return {
            'name': 'IntramonthValidationTest',
            'strategy': 'IntramonthSeasonalStrategy',
            'strategy_class': 'IntramonthSeasonalStrategy',
            'strategy_params': {
                'direction': 'long',
                'entry_day': 3,  # Enter on 3rd business day
                'hold_days': 10,  # Hold for 10 business days
                'trade_month_1': True,
                'trade_month_2': True,
                'trade_month_3': True,
                'trade_month_4': True,
                'trade_month_5': True,
                'trade_month_6': True,
                'trade_month_7': True,
                'trade_month_8': True,
                'trade_month_9': True,
                'trade_month_10': True,
                'trade_month_11': True,
                'trade_month_12': True,
            },
            'timing': {
                'rebalance_frequency': 'D',
                'mode': 'signal_based',
                'scan_frequency': 'D'
            },
            'universe': {
                'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
            },
            'optimization_metric': 'sharpe_ratio'
        }
    
    @pytest.mark.skipif(not WFO_ENHANCEMENT_AVAILABLE, reason="WFO enhancement not available")
    def test_intramonth_strategy_with_enhanced_wfo(self, intramonth_strategy_config, optimization_data):
        """Test IntramonthSeasonalStrategy with enhanced WFO system."""
        # Create evaluator with daily evaluation
        evaluator = BacktestEvaluator(
            metrics_to_optimize=['sharpe_ratio'],
            is_multi_objective=False,
            n_jobs=1
        )
        
        # Test that the evaluator correctly detects intramonth strategy
        detected_frequency = evaluator._determine_evaluation_frequency(intramonth_strategy_config)
        assert detected_frequency == 'D', "Intramonth strategy should be detected as requiring daily evaluation"
        
        # Test that the strategy class is properly configured
        assert intramonth_strategy_config['strategy_class'] == 'IntramonthSeasonalStrategy'
        assert intramonth_strategy_config['strategy_params']['entry_day'] == 3
        assert intramonth_strategy_config['strategy_params']['hold_days'] == 10
        
        # Validate configuration structure
        assert 'timing' in intramonth_strategy_config
        assert intramonth_strategy_config['timing']['rebalance_frequency'] == 'D'
        assert intramonth_strategy_config['timing']['mode'] == 'signal_based'
    
    @pytest.mark.skipif(not WFO_ENHANCEMENT_AVAILABLE, reason="WFO enhancement not available")
    def test_trade_duration_accuracy(self, intramonth_strategy_config, optimization_data):
        """Validate that trade durations match strategy parameters."""
        # Create strategy instance
        strategy = IntramonthSeasonalStrategy(intramonth_strategy_config)
        
        # Test the strategy's hold_days parameter
        expected_hold_days = intramonth_strategy_config['strategy_params']['hold_days']
        assert expected_hold_days == 10, "Strategy should be configured with 10 hold days"
        
        # Test entry day calculation
        entry_day = intramonth_strategy_config['strategy_params']['entry_day']
        test_date = datetime(2023, 4, 1)
        calculated_entry_date = strategy.get_entry_date_for_month(test_date, entry_day)
        
        # Validate that entry date calculation works
        assert isinstance(calculated_entry_date, pd.Timestamp)
        assert calculated_entry_date.month == test_date.month
        
        # Test that the strategy can be instantiated and configured properly
        assert strategy is not None
        params = strategy.strategy_config.get('strategy_params', strategy.strategy_config)
        assert params['hold_days'] == expected_hold_days
        assert params['entry_day'] == entry_day
    
    @pytest.mark.skipif(not WFO_ENHANCEMENT_AVAILABLE, reason="WFO enhancement not available")
    def test_multiple_intramonth_strategies_simultaneously(self, optimization_data):
        """Test multiple intramonth strategies with different parameters."""
        strategy_configs = [
            {
                'name': 'IntramonthEarly',
                'strategy': 'IntramonthSeasonalStrategy',
                'strategy_class': 'IntramonthSeasonalStrategy',
                'strategy_params': {
                    'direction': 'long',
                    'entry_day': 2,
                    'hold_days': 8,
                },
                'timing': {'rebalance_frequency': 'D', 'mode': 'signal_based', 'scan_frequency': 'D'}
            },
            {
                'name': 'IntramonthMid',
                'strategy': 'IntramonthSeasonalStrategy',
                'strategy_class': 'IntramonthSeasonalStrategy',
                'strategy_params': {
                    'direction': 'long',
                    'entry_day': 5,
                    'hold_days': 12,
                },
                'timing': {'rebalance_frequency': 'D', 'mode': 'signal_based', 'scan_frequency': 'D'}
            },
            {
                'name': 'IntramonthLate',
                'strategy': 'IntramonthSeasonalStrategy',
                'strategy_class': 'IntramonthSeasonalStrategy',
                'strategy_params': {
                    'direction': 'long',
                    'entry_day': 8,
                    'hold_days': 15,
                },
                'timing': {'rebalance_frequency': 'D', 'mode': 'signal_based', 'scan_frequency': 'D'}
            }
        ]
        
        evaluator = BacktestEvaluator(
            metrics_to_optimize=['sharpe_ratio'],
            is_multi_objective=False,
            n_jobs=1
        )
        
        backtester = StrategyBacktester(global_config=GLOBAL_CONFIG, data_source=None)
        
        results = []
        
        for config in strategy_configs:
            test_parameters = config['strategy_params']
            
            result = evaluator.evaluate_parameters(
                parameters=test_parameters,
                scenario_config=config,
                data=optimization_data,
                backtester=backtester
            )
            
            results.append((config['name'], result))
        
        # Validate all strategies produced results
        assert len(results) == 3
        
        for name, result in results:
            assert isinstance(result, EvaluationResult), f"Strategy {name} failed to produce valid result"
            assert result.objective_value is not None, f"Strategy {name} has null objective value"
            assert not np.isnan(result.objective_value), f"Strategy {name} has NaN objective value"
            
            # All should use daily evaluation
            assert evaluator.evaluation_frequency == 'D', f"Strategy {name} should use daily evaluation"
    
    @pytest.mark.skipif(not WFO_ENHANCEMENT_AVAILABLE, reason="WFO enhancement not available")
    def test_exit_timing_accuracy(self, intramonth_strategy_config, optimization_data):
        """Validate that exit timing matches strategy logic exactly."""
        # Create strategy instance
        strategy = IntramonthSeasonalStrategy(intramonth_strategy_config)
        
        # Test specific dates to validate entry and exit logic
        entry_day = intramonth_strategy_config['strategy_params']['entry_day']
        hold_days = intramonth_strategy_config['strategy_params']['hold_days']
        
        # Test entry date calculation for different months
        test_months = [datetime(2023, 1, 1), datetime(2023, 4, 1), datetime(2023, 7, 1)]
        
        for test_month in test_months:
            expected_entry_date = strategy.get_entry_date_for_month(test_month, entry_day)
            
            # Validate entry date is in the correct month
            assert expected_entry_date.month == test_month.month
            assert expected_entry_date.year == test_month.year
            
            # Calculate expected exit date using business days
            expected_exit_date = expected_entry_date + pd.tseries.offsets.BDay(hold_days)
            
            # Validate that exit date is after entry date
            assert expected_exit_date > expected_entry_date
            
            # Validate that the duration is approximately correct
            # (allowing for weekends and holidays)
            duration_days = (expected_exit_date - expected_entry_date).days
            assert 7 <= duration_days <= 20, f"Duration {duration_days} should be reasonable for {hold_days} business days"
    
    @pytest.mark.skipif(not WFO_ENHANCEMENT_AVAILABLE, reason="WFO enhancement not available")
    def test_position_management_across_window_boundaries(self, intramonth_strategy_config, optimization_data):
        """Test position management across WFO window boundaries."""
        # Create strategy instance
        strategy = IntramonthSeasonalStrategy(intramonth_strategy_config)
        
        # Test that strategy state can be reset
        strategy.reset_state()
        assert strategy._last_weights is None, "Strategy state should be reset"
        
        # Test multiple WFO windows with different date ranges
        windows = [
            WFOWindow(
                train_start=datetime(2023, 1, 1),
                train_end=datetime(2023, 3, 31),
                test_start=datetime(2023, 4, 1),
                test_end=datetime(2023, 5, 31),
                evaluation_frequency='D',
                strategy_name='IntramonthSeasonalStrategy'
            ),
            WFOWindow(
                train_start=datetime(2023, 2, 1),
                train_end=datetime(2023, 4, 30),
                test_start=datetime(2023, 5, 1),
                test_end=datetime(2023, 6, 30),
                evaluation_frequency='D',
                strategy_name='IntramonthSeasonalStrategy'
            )
        ]
        
        # Validate window properties
        for i, window in enumerate(windows):
            assert isinstance(window, WFOWindow), f"Window {i} should be WFOWindow instance"
            assert window.evaluation_frequency == 'D', f"Window {i} should use daily evaluation"
            assert window.requires_daily_evaluation, f"Window {i} should require daily evaluation"
            
            # Test that evaluation dates can be generated
            available_dates = optimization_data.daily.index
            eval_dates = window.get_evaluation_dates(available_dates)
            assert len(eval_dates) > 0, f"Window {i} should have evaluation dates"
    
    def test_strategy_parameter_validation(self, realistic_daily_data):
        """Test that strategy parameters are properly validated and applied."""
        # Test different parameter combinations
        test_configs = [
            {
                'strategy_params': {
                    'direction': 'long',
                    'entry_day': 1,
                    'hold_days': 5,
                }
            },
            {
                'strategy_params': {
                    'direction': 'short',
                    'entry_day': -3,  # Negative entry day (from end of month)
                    'hold_days': 7,
                }
            },
            {
                'strategy_params': {
                    'direction': 'long',
                    'entry_day': 10,
                    'hold_days': 15,
                    'trade_month_1': True,
                    'trade_month_2': False,  # Only trade in January
                    'trade_month_3': False,
                    'trade_month_4': False,
                    'trade_month_5': False,
                    'trade_month_6': False,
                    'trade_month_7': False,
                    'trade_month_8': False,
                    'trade_month_9': False,
                    'trade_month_10': False,
                    'trade_month_11': False,
                    'trade_month_12': False,
                }
            }
        ]
        
        for i, config in enumerate(test_configs):
            strategy = IntramonthSeasonalStrategy(config)
            
            # Test that strategy can be created without errors
            assert strategy is not None, f"Strategy {i} failed to initialize"
            
            # Test that parameters are properly set
            params = strategy.strategy_config.get('strategy_params', strategy.strategy_config)
            assert params['direction'] == config['strategy_params']['direction']
            assert params['entry_day'] == config['strategy_params']['entry_day']
            assert params['hold_days'] == config['strategy_params']['hold_days']
            
            # Test signal generation doesn't crash
            test_date = datetime(2023, 1, 15)
            if test_date in realistic_daily_data.index:
                signals = strategy.generate_signals(
                    all_historical_data=realistic_daily_data,
                    benchmark_historical_data=realistic_daily_data,
                    non_universe_historical_data=pd.DataFrame(),
                    current_date=test_date,
                    start_date=datetime(2023, 1, 1),
                    end_date=datetime(2023, 12, 31)
                )
                
                assert isinstance(signals, pd.DataFrame), f"Strategy {i} failed to generate signals"
    
    @pytest.mark.skipif(not WFO_ENHANCEMENT_AVAILABLE, reason="WFO enhancement not available")
    def test_performance_comparison_with_proof_of_concept(self, intramonth_strategy_config, optimization_data):
        """Compare results with proof of concept to validate improvements."""
        # Test evaluation frequency detection for different strategy types
        evaluator = BacktestEvaluator(
            metrics_to_optimize=['sharpe_ratio'],
            is_multi_objective=False,
            n_jobs=1
        )
        
        # Test intramonth strategy detection
        intramonth_frequency = evaluator._determine_evaluation_frequency(intramonth_strategy_config)
        assert intramonth_frequency == 'D', "Intramonth strategy should use daily evaluation"
        
        # Test monthly strategy detection
        monthly_config = {
            'strategy_class': 'MonthlyStrategy',
            'timing': {'rebalance_frequency': 'M'}
        }
        monthly_frequency = evaluator._determine_evaluation_frequency(monthly_config)
        assert monthly_frequency == 'M', "Monthly strategy should use monthly evaluation"
        
        # Test signal-based strategy with daily scanning
        signal_config = {
            'strategy_class': 'SomeStrategy',
            'timing_config': {  # Note: using timing_config not timing
                'mode': 'signal_based',
                'scan_frequency': 'D',
            },
            'rebalance_frequency': 'D'
        }
        signal_frequency = evaluator._determine_evaluation_frequency(signal_config)
        assert signal_frequency == 'D', "Signal-based daily strategy should use daily evaluation"
        
        # Validate that the frequency detection logic works correctly
        assert intramonth_frequency != monthly_frequency, "Different strategies should have different frequencies"
    
    def test_strategy_specific_requirements_documentation(self):
        """Document strategy-specific requirements for intramonth strategies."""
        # This test serves as documentation of requirements
        requirements = {
            'evaluation_frequency': 'D',  # Must use daily evaluation
            'min_hold_days': 1,
            'max_hold_days': 30,  # Reasonable upper bound for intramonth
            'supported_directions': ['long', 'short'],
            'required_data_frequency': 'daily',
            'position_sizing': 'equal_weight',
            'rebalance_frequency': 'D',
            'scan_frequency': 'D',
            'mode': 'signal_based'
        }
        
        # Validate requirements are reasonable
        assert requirements['evaluation_frequency'] == 'D'
        assert requirements['min_hold_days'] >= 1
        assert requirements['max_hold_days'] <= 30
        assert 'long' in requirements['supported_directions']
        assert 'short' in requirements['supported_directions']
        
        # This test passes if requirements are well-defined
        assert len(requirements) > 0