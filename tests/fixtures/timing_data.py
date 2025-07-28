"""
Timing data fixtures for generating standardized timing framework test data.

This module provides the TimingDataFixture class with methods for generating
timing configurations, scenarios, and test data for timing framework tests.
"""

import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from functools import lru_cache
from unittest.mock import Mock


class TimingDataFixture:
    """
    Fixture class for generating standardized timing framework test data.
    
    Provides methods to create timing configurations, scenarios, and mock
    objects for comprehensive timing framework testing.
    """
    
    @staticmethod
    @lru_cache(maxsize=20)
    def time_based_config(
        frequency: str = 'M',
        offset: int = 0,
        start_date: str = '2020-01-01',
        end_date: str = '2023-12-31'
    ) -> Dict[str, Any]:
        """
        Create time-based timing configuration.
        
        Args:
            frequency: Rebalancing frequency ('M', 'Q', 'D', 'W')
            offset: Offset in days from the base frequency
            start_date: Start date for timing
            end_date: End date for timing
            
        Returns:
            Dictionary with time-based timing configuration
        """
        return {
            'timing_type': 'time_based',
            'rebalance_frequency': frequency,
            'rebalance_offset': offset,
            'start_date': start_date,
            'end_date': end_date
        }
    
    @staticmethod
    @lru_cache(maxsize=20)
    def signal_based_config(
        signal_threshold: float = 0.05,
        min_rebalance_interval: int = 30,
        max_rebalance_interval: int = 90,
        signal_type: str = 'momentum'
    ) -> Dict[str, Any]:
        """
        Create signal-based timing configuration.
        
        Args:
            signal_threshold: Threshold for triggering rebalancing
            min_rebalance_interval: Minimum days between rebalances
            max_rebalance_interval: Maximum days between rebalances
            signal_type: Type of signal to monitor
            
        Returns:
            Dictionary with signal-based timing configuration
        """
        return {
            'timing_type': 'signal_based',
            'signal_threshold': signal_threshold,
            'min_rebalance_interval': min_rebalance_interval,
            'max_rebalance_interval': max_rebalance_interval,
            'signal_type': signal_type
        }
    
    @staticmethod
    @lru_cache(maxsize=20)
    def hybrid_timing_config(
        primary_frequency: str = 'M',
        signal_threshold: float = 0.1,
        emergency_rebalance: bool = True
    ) -> Dict[str, Any]:
        """
        Create hybrid timing configuration combining time and signal-based approaches.
        
        Args:
            primary_frequency: Primary time-based frequency
            signal_threshold: Signal threshold for emergency rebalancing
            emergency_rebalance: Whether to allow emergency rebalancing
            
        Returns:
            Dictionary with hybrid timing configuration
        """
        return {
            'timing_type': 'hybrid',
            'primary_frequency': primary_frequency,
            'signal_threshold': signal_threshold,
            'emergency_rebalance': emergency_rebalance,
            'min_emergency_interval': 7  # Minimum days between emergency rebalances
        }
    
    @staticmethod
    def create_rebalance_dates(
        start_date: str = '2020-01-01',
        end_date: str = '2023-12-31',
        frequency: str = 'ME'
    ) -> pd.DatetimeIndex:
        """
        Create standardized rebalance dates for testing.
        
        Args:
            start_date: Start date string
            end_date: End date string
            frequency: Pandas frequency string
            
        Returns:
            DatetimeIndex with rebalance dates
        """
        return pd.date_range(start=start_date, end=end_date, freq=frequency)
    
    @staticmethod
    def create_available_dates(
        start_date: str = '2020-01-01',
        end_date: str = '2023-12-31',
        freq: str = 'B'
    ) -> pd.DatetimeIndex:
        """
        Create available trading dates for testing.
        
        Args:
            start_date: Start date string
            end_date: End date string
            freq: Frequency ('B' for business days, 'D' for daily)
            
        Returns:
            DatetimeIndex with available trading dates
        """
        return pd.date_range(start=start_date, end=end_date, freq=freq)
    
    @staticmethod
    def create_mock_strategy_context(
        strategy_name: str = 'TestStrategy',
        config: Optional[Dict[str, Any]] = None
    ) -> Mock:
        """
        Create mock strategy context for timing tests.
        
        Args:
            strategy_name: Name of the strategy
            config: Strategy configuration dictionary
            
        Returns:
            Mock object representing strategy context
        """
        mock_context = Mock()
        mock_context.strategy_name = strategy_name
        mock_context.config = config or {}
        mock_context.get_current_positions.return_value = {}
        mock_context.get_signal_history.return_value = pd.DataFrame()
        return mock_context
    
    @staticmethod
    def create_timing_scenarios() -> Dict[str, Dict[str, Any]]:
        """
        Create comprehensive timing test scenarios.
        
        Returns:
            Dictionary mapping scenario names to timing configurations
        """
        return {
            'monthly_standard': TimingDataFixture.time_based_config('M', 0),
            'monthly_offset': TimingDataFixture.time_based_config('M', -5),
            'quarterly_standard': TimingDataFixture.time_based_config('Q', 0),
            'daily_standard': TimingDataFixture.time_based_config('D', 0),
            'weekly_standard': TimingDataFixture.time_based_config('W', 0),
            'signal_conservative': TimingDataFixture.signal_based_config(0.1, 30, 90),
            'signal_aggressive': TimingDataFixture.signal_based_config(0.02, 7, 30),
            'signal_momentum': TimingDataFixture.signal_based_config(0.05, 14, 60, 'momentum'),
            'signal_volatility': TimingDataFixture.signal_based_config(0.08, 21, 45, 'volatility'),
            'hybrid_monthly': TimingDataFixture.hybrid_timing_config('M', 0.1, True),
            'hybrid_quarterly': TimingDataFixture.hybrid_timing_config('Q', 0.15, True),
            'hybrid_no_emergency': TimingDataFixture.hybrid_timing_config('M', 0.1, False)
        }
    
    @staticmethod
    def create_migration_scenarios() -> Dict[str, Dict[str, Any]]:
        """
        Create migration test scenarios for backward compatibility testing.
        
        Returns:
            Dictionary mapping migration scenario names to configurations
        """
        return {
            'legacy_monthly_momentum': {
                'old_config': {
                    'rebalance_frequency': 'monthly',
                    'strategy_type': 'momentum',
                    'lookback_months': 3
                },
                'new_config': {
                    'timing_type': 'time_based',
                    'rebalance_frequency': 'M',
                    'strategy_params': {
                        'lookback_months': 3
                    }
                },
                'expected_behavior': 'equivalent_rebalancing'
            },
            'legacy_daily_uvxy': {
                'old_config': {
                    'rebalance_frequency': 'daily',
                    'strategy_type': 'uvxy_rsi',
                    'rsi_period': 14
                },
                'new_config': {
                    'timing_type': 'time_based',
                    'rebalance_frequency': 'D',
                    'strategy_params': {
                        'rsi_period': 14
                    }
                },
                'expected_behavior': 'equivalent_rebalancing'
            },
            'legacy_signal_based': {
                'old_config': {
                    'rebalance_trigger': 'signal',
                    'signal_threshold': 0.05,
                    'strategy_type': 'momentum'
                },
                'new_config': {
                    'timing_type': 'signal_based',
                    'signal_threshold': 0.05,
                    'signal_type': 'momentum'
                },
                'expected_behavior': 'equivalent_signaling'
            }
        }
    
    @staticmethod
    def create_compatibility_test_data() -> Dict[str, Any]:
        """
        Create test data for strategy-timing compatibility testing.
        
        Returns:
            Dictionary with compatibility test data and expectations
        """
        return {
            'strategy_timing_combinations': [
                ('momentum', 'time_based_monthly'),
                ('momentum', 'signal_based'),
                ('calmar', 'time_based_quarterly'),
                ('sortino', 'time_based_monthly'),
                ('ema_roro', 'time_based_daily'),
                ('uvxy_rsi', 'time_based_daily'),
                ('uvxy_rsi', 'signal_based'),
                ('low_vol_factor', 'time_based_monthly')
            ],
            'incompatible_combinations': [
                ('momentum', 'time_based_daily'),  # Momentum needs monthly data
                ('ema_roro', 'time_based_monthly'),  # EMA RoRo needs daily rebalancing
            ],
            'expected_rebalance_counts': {
                ('momentum', 'time_based_monthly', '2020-01-01', '2020-12-31'): 12,
                ('momentum', 'time_based_quarterly', '2020-01-01', '2020-12-31'): 4,
                ('ema_roro', 'time_based_daily', '2020-01-01', '2020-01-31'): 23,  # Business days
            }
        }
    
    @staticmethod
    def create_timing_state_scenarios() -> Dict[str, Dict[str, Any]]:
        """
        Create timing state management test scenarios.
        
        Returns:
            Dictionary with timing state test scenarios
        """
        return {
            'initial_state': {
                'scheduled_dates': [],
                'last_rebalance': None,
                'next_rebalance': None,
                'rebalance_count': 0,
                'emergency_rebalances': 0
            },
            'active_state': {
                'scheduled_dates': [
                    '2020-01-31', '2020-02-29', '2020-03-31'
                ],
                'last_rebalance': '2020-01-31',
                'next_rebalance': '2020-02-29',
                'rebalance_count': 1,
                'emergency_rebalances': 0
            },
            'emergency_state': {
                'scheduled_dates': [
                    '2020-01-31', '2020-02-29', '2020-03-31'
                ],
                'last_rebalance': '2020-01-15',  # Emergency rebalance
                'next_rebalance': '2020-02-29',
                'rebalance_count': 2,
                'emergency_rebalances': 1
            }
        }
    
    @staticmethod
    def create_edge_case_scenarios() -> Dict[str, Dict[str, Any]]:
        """
        Create edge case scenarios for robust timing testing.
        
        Returns:
            Dictionary with edge case test scenarios
        """
        return {
            'weekend_rebalance': {
                'config': TimingDataFixture.time_based_config('M', 0),
                'test_dates': ['2020-02-29', '2020-03-31'],  # Some may fall on weekends
                'expected_behavior': 'adjust_to_business_day'
            },
            'holiday_rebalance': {
                'config': TimingDataFixture.time_based_config('M', 0),
                'test_dates': ['2020-01-01', '2020-07-04', '2020-12-25'],  # Holidays
                'expected_behavior': 'adjust_to_next_business_day'
            },
            'missing_data_rebalance': {
                'config': TimingDataFixture.time_based_config('M', 0),
                'missing_dates': ['2020-02-28', '2020-03-31'],
                'expected_behavior': 'skip_or_defer_rebalance'
            },
            'rapid_signal_changes': {
                'config': TimingDataFixture.signal_based_config(0.01, 1, 30),
                'signal_pattern': 'high_frequency_oscillation',
                'expected_behavior': 'respect_min_interval'
            },
            'no_signal_for_long_period': {
                'config': TimingDataFixture.signal_based_config(0.1, 30, 90),
                'signal_pattern': 'long_quiet_period',
                'expected_behavior': 'force_rebalance_at_max_interval'
            }
        }
    
    @staticmethod
    def create_performance_test_scenarios() -> Dict[str, Dict[str, Any]]:
        """
        Create performance testing scenarios for timing framework.
        
        Returns:
            Dictionary with performance test scenarios
        """
        return {
            'high_frequency_daily': {
                'config': TimingDataFixture.time_based_config('D', 0),
                'date_range': ('2020-01-01', '2023-12-31'),
                'expected_rebalances': 1000,  # Approximate business days
                'performance_target': 'sub_second_execution'
            },
            'large_universe_monthly': {
                'config': TimingDataFixture.time_based_config('M', 0),
                'universe_size': 500,  # Large number of assets
                'date_range': ('2020-01-01', '2023-12-31'),
                'performance_target': 'linear_scaling'
            },
            'complex_signal_based': {
                'config': TimingDataFixture.signal_based_config(0.05, 7, 30),
                'signal_complexity': 'multi_factor',
                'date_range': ('2020-01-01', '2023-12-31'),
                'performance_target': 'acceptable_latency'
            }
        }
    
    @staticmethod
    def get_expected_rebalance_dates(
        config: Dict[str, Any],
        start_date: str,
        end_date: str
    ) -> List[str]:
        """
        Get expected rebalance dates for a given timing configuration.
        
        Args:
            config: Timing configuration dictionary
            start_date: Start date string
            end_date: End date string
            
        Returns:
            List of expected rebalance date strings
        """
        if config.get('timing_type') == 'time_based':
            freq = config.get('rebalance_frequency', 'M')
            if freq == 'M':
                freq = 'ME'  # Use month-end frequency
            
            dates = pd.date_range(start=start_date, end=end_date, freq=freq)
            
            # Apply offset if specified
            offset = config.get('rebalance_offset', 0)
            if offset != 0:
                dates = dates + pd.Timedelta(days=offset)
            
            return [date.strftime('%Y-%m-%d') for date in dates]
        
        elif config.get('timing_type') == 'signal_based':
            # For signal-based timing, we can't predict exact dates
            # Return approximate expected behavior
            return ['signal_dependent']
        
        else:
            return []