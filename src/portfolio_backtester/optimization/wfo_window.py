"""
Enhanced Walk-Forward Optimization Window with Daily Evaluation Support.

This module provides the WFOWindow dataclass that extends the traditional WFO window
concept to support different evaluation frequencies (daily, weekly, monthly) for
strategies that require intramonth position management.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class WFOWindow:
    """Enhanced walk-forward optimization window with daily evaluation support.
    
    This class extends the traditional WFO window concept to support strategies
    that need to be evaluated more frequently than just at the beginning of each
    test window. This is particularly important for intramonth strategies that
    need to manage position exits during the test period.
    
    Attributes:
        train_start: Start date of the training period
        train_end: End date of the training period
        test_start: Start date of the test period
        test_end: End date of the test period
        evaluation_frequency: Frequency of strategy evaluation ('D', 'W', 'M')
        strategy_name: Optional name of the strategy for debugging
    """
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    evaluation_frequency: str = 'M'  # D, W, M
    strategy_name: Optional[str] = None
    
    def get_evaluation_dates(self, available_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Get all dates when strategy should be evaluated in test window.
        
        This method determines the specific dates when the strategy should be
        evaluated within the test window based on the evaluation frequency and
        available trading dates.
        
        Args:
            available_dates: DatetimeIndex of all available trading dates
            
        Returns:
            DatetimeIndex of dates when strategy should be evaluated
        """
        # Filter available dates to test window
        window_dates = available_dates[
            (available_dates >= self.test_start) & 
            (available_dates <= self.test_end)
        ]
        
        if len(window_dates) == 0:
            logger.warning(
                f"No available dates found in test window {self.test_start} to {self.test_end}"
            )
            return pd.DatetimeIndex([])
        
        if self.evaluation_frequency == 'D':
            # Daily evaluation - return all business days in the window
            return window_dates
            
        elif self.evaluation_frequency == 'W':
            # Weekly evaluation - first business day of each week
            weekly_dates = []
            current_week = None
            
            for date in window_dates:
                # Get the week number (year, week)
                week_key = (date.year, date.isocalendar()[1])
                
                if current_week != week_key:
                    weekly_dates.append(date)
                    current_week = week_key
            
            return pd.DatetimeIndex(weekly_dates)
            
        else:  # Monthly evaluation (default)
            # Monthly evaluation - only the first date of the test window
            return pd.DatetimeIndex([self.test_start])
    
    @property
    def requires_daily_evaluation(self) -> bool:
        """Check if this window requires daily strategy evaluation.
        
        Returns:
            True if strategy needs to be evaluated daily, False otherwise
        """
        return self.evaluation_frequency == 'D'
    
    @property
    def window_length_days(self) -> int:
        """Get the length of the test window in calendar days.
        
        Returns:
            Number of calendar days in the test window
        """
        return (self.test_end - self.test_start).days + 1
    
    @property
    def train_length_days(self) -> int:
        """Get the length of the training window in calendar days.
        
        Returns:
            Number of calendar days in the training window
        """
        return (self.train_end - self.train_start).days + 1
    
    def __str__(self) -> str:
        """String representation of the window."""
        strategy_info = f" ({self.strategy_name})" if self.strategy_name else ""
        return (
            f"WFOWindow{strategy_info}: "
            f"Train={self.train_start.date()} to {self.train_end.date()}, "
            f"Test={self.test_start.date()} to {self.test_end.date()}, "
            f"Freq={self.evaluation_frequency}"
        )
    
    def __repr__(self) -> str:
        """Detailed representation of the window."""
        return (
            f"WFOWindow("
            f"train_start={self.train_start}, "
            f"train_end={self.train_end}, "
            f"test_start={self.test_start}, "
            f"test_end={self.test_end}, "
            f"evaluation_frequency='{self.evaluation_frequency}', "
            f"strategy_name='{self.strategy_name}'"
            f")"
        )