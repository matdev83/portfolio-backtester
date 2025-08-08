"""
Unit tests for WFOWindow class.

Tests the enhanced WFO window structure with daily evaluation support.
"""

import pandas as pd
from portfolio_backtester.optimization.wfo_window import WFOWindow


class TestWFOWindow:
    """Test cases for WFOWindow class."""
    
    def test_wfo_window_initialization(self):
        """Test basic WFO window initialization."""
        window = WFOWindow(
            train_start=pd.Timestamp('2024-01-01'),
            train_end=pd.Timestamp('2024-12-31'),
            test_start=pd.Timestamp('2025-01-01'),
            test_end=pd.Timestamp('2025-01-31'),
            evaluation_frequency='D',
            strategy_name='test_strategy'
        )
        
        assert window.train_start == pd.Timestamp('2024-01-01')
        assert window.train_end == pd.Timestamp('2024-12-31')
        assert window.test_start == pd.Timestamp('2025-01-01')
        assert window.test_end == pd.Timestamp('2025-01-31')
        assert window.evaluation_frequency == 'D'
        assert window.strategy_name == 'test_strategy'
    
    def test_daily_evaluation_dates(self):
        """Test that WFO window generates correct daily evaluation dates."""
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
        
        # Should return all business days in January 2025
        expected_dates = pd.bdate_range('2025-01-01', '2025-01-31')
        assert len(eval_dates) == len(expected_dates)
        assert all(eval_dates == expected_dates)
    
    def test_weekly_evaluation_dates(self):
        """Test that WFO window generates correct weekly evaluation dates."""
        window = WFOWindow(
            train_start=pd.Timestamp('2024-01-01'),
            train_end=pd.Timestamp('2024-12-31'),
            test_start=pd.Timestamp('2025-01-01'),
            test_end=pd.Timestamp('2025-01-31'),
            evaluation_frequency='W'
        )
        
        # Create available dates (business days)
        available_dates = pd.bdate_range('2025-01-01', '2025-01-31')
        
        # Get evaluation dates
        eval_dates = window.get_evaluation_dates(available_dates)
        
        # Should return first business day of each week
        assert len(eval_dates) > 0
        assert len(eval_dates) <= 5  # At most 5 weeks in January
        
        # Check that dates are from different weeks
        weeks = set()
        for date in eval_dates:
            week_key = (date.year, date.isocalendar()[1])
            assert week_key not in weeks  # Each week should appear only once
            weeks.add(week_key)
    
    def test_monthly_evaluation_dates(self):
        """Test that WFO window generates correct monthly evaluation dates."""
        window = WFOWindow(
            train_start=pd.Timestamp('2024-01-01'),
            train_end=pd.Timestamp('2024-12-31'),
            test_start=pd.Timestamp('2025-01-01'),
            test_end=pd.Timestamp('2025-01-31'),
            evaluation_frequency='M'
        )
        
        # Create available dates (business days)
        available_dates = pd.bdate_range('2025-01-01', '2025-01-31')
        
        # Get evaluation dates
        eval_dates = window.get_evaluation_dates(available_dates)
        
        # Should return only the first date of the test window
        assert len(eval_dates) == 1
        assert eval_dates[0] == pd.Timestamp('2025-01-01')
    
    def test_requires_daily_evaluation_property(self):
        """Test the requires_daily_evaluation property."""
        # Daily evaluation window
        daily_window = WFOWindow(
            train_start=pd.Timestamp('2024-01-01'),
            train_end=pd.Timestamp('2024-12-31'),
            test_start=pd.Timestamp('2025-01-01'),
            test_end=pd.Timestamp('2025-01-31'),
            evaluation_frequency='D'
        )
        assert daily_window.requires_daily_evaluation is True
        
        # Weekly evaluation window
        weekly_window = WFOWindow(
            train_start=pd.Timestamp('2024-01-01'),
            train_end=pd.Timestamp('2024-12-31'),
            test_start=pd.Timestamp('2025-01-01'),
            test_end=pd.Timestamp('2025-01-31'),
            evaluation_frequency='W'
        )
        assert weekly_window.requires_daily_evaluation is False
        
        # Monthly evaluation window
        monthly_window = WFOWindow(
            train_start=pd.Timestamp('2024-01-01'),
            train_end=pd.Timestamp('2024-12-31'),
            test_start=pd.Timestamp('2025-01-01'),
            test_end=pd.Timestamp('2025-01-31'),
            evaluation_frequency='M'
        )
        assert monthly_window.requires_daily_evaluation is False
    
    def test_window_length_properties(self):
        """Test window length calculation properties."""
        window = WFOWindow(
            train_start=pd.Timestamp('2024-01-01'),
            train_end=pd.Timestamp('2024-12-31'),
            test_start=pd.Timestamp('2025-01-01'),
            test_end=pd.Timestamp('2025-01-31'),
            evaluation_frequency='D'
        )
        
        # Test window length (January has 31 days)
        assert window.window_length_days == 31
        
        # Train window length (2024 has 366 days)
        assert window.train_length_days == 366
    
    def test_empty_available_dates(self):
        """Test behavior when no available dates in test window."""
        window = WFOWindow(
            train_start=pd.Timestamp('2024-01-01'),
            train_end=pd.Timestamp('2024-12-31'),
            test_start=pd.Timestamp('2025-01-01'),
            test_end=pd.Timestamp('2025-01-31'),
            evaluation_frequency='D'
        )
        
        # Empty available dates
        available_dates = pd.DatetimeIndex([])
        
        # Get evaluation dates
        eval_dates = window.get_evaluation_dates(available_dates)
        
        # Should return empty index
        assert len(eval_dates) == 0
        assert isinstance(eval_dates, pd.DatetimeIndex)
    
    def test_no_overlap_available_dates(self):
        """Test behavior when available dates don't overlap with test window."""
        window = WFOWindow(
            train_start=pd.Timestamp('2024-01-01'),
            train_end=pd.Timestamp('2024-12-31'),
            test_start=pd.Timestamp('2025-01-01'),
            test_end=pd.Timestamp('2025-01-31'),
            evaluation_frequency='D'
        )
        
        # Available dates outside test window
        available_dates = pd.bdate_range('2025-02-01', '2025-02-28')
        
        # Get evaluation dates
        eval_dates = window.get_evaluation_dates(available_dates)
        
        # Should return empty index
        assert len(eval_dates) == 0
    
    def test_partial_overlap_available_dates(self):
        """Test behavior when available dates partially overlap with test window."""
        window = WFOWindow(
            train_start=pd.Timestamp('2024-01-01'),
            train_end=pd.Timestamp('2024-12-31'),
            test_start=pd.Timestamp('2025-01-15'),
            test_end=pd.Timestamp('2025-01-31'),
            evaluation_frequency='D'
        )
        
        # Available dates that partially overlap
        available_dates = pd.bdate_range('2025-01-01', '2025-01-20')
        
        # Get evaluation dates
        eval_dates = window.get_evaluation_dates(available_dates)
        
        # Should return only dates in the overlap (Jan 15-20)
        expected_dates = pd.bdate_range('2025-01-15', '2025-01-20')
        assert len(eval_dates) == len(expected_dates)
        assert all(eval_dates == expected_dates)
    
    def test_string_representation(self):
        """Test string representation methods."""
        window = WFOWindow(
            train_start=pd.Timestamp('2024-01-01'),
            train_end=pd.Timestamp('2024-12-31'),
            test_start=pd.Timestamp('2025-01-01'),
            test_end=pd.Timestamp('2025-01-31'),
            evaluation_frequency='D',
            strategy_name='test_strategy'
        )
        
        # Test __str__
        str_repr = str(window)
        assert 'test_strategy' in str_repr
        assert '2024-01-01' in str_repr
        assert '2025-01-31' in str_repr
        assert 'Freq=D' in str_repr
        
        # Test __repr__
        repr_str = repr(window)
        assert 'WFOWindow(' in repr_str
        assert 'evaluation_frequency=\'D\'' in repr_str
        assert 'strategy_name=\'test_strategy\'' in repr_str
    
    def test_default_values(self):
        """Test default values for optional parameters."""
        window = WFOWindow(
            train_start=pd.Timestamp('2024-01-01'),
            train_end=pd.Timestamp('2024-12-31'),
            test_start=pd.Timestamp('2025-01-01'),
            test_end=pd.Timestamp('2025-01-31')
        )
        
        # Default evaluation frequency should be monthly
        assert window.evaluation_frequency == 'M'
        assert window.strategy_name is None
        assert window.requires_daily_evaluation is False