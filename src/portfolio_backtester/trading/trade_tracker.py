"""
Trade tracking system for detailed performance analysis.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade with all relevant information."""
    ticker: str
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    quantity: float  # Positive for long, negative for short
    entry_value: float  # Absolute value of position
    commission_entry: float
    commission_exit: float
    mfe: float = 0.0  # Maximum Favorable Excursion
    mae: float = 0.0  # Maximum Adverse Excursion
    duration_days: Optional[int] = None
    pnl_gross: Optional[float] = None
    pnl_net: Optional[float] = None
    is_winner: Optional[bool] = None
    
    def finalize(self):
        """Calculate derived fields when trade is closed."""
        if self.exit_date is not None and self.exit_price is not None:
            self.duration_days = (self.exit_date - self.entry_date).days
            
            # Calculate P&L
            if self.quantity > 0:  # Long position
                self.pnl_gross = (self.exit_price - self.entry_price) * abs(self.quantity)
            else:  # Short position
                self.pnl_gross = (self.entry_price - self.exit_price) * abs(self.quantity)
            
            self.pnl_net = self.pnl_gross - self.commission_entry - self.commission_exit
            self.is_winner = self.pnl_net > 0


class TradeTracker:
    """Comprehensive trade tracking system."""
    
    def __init__(self, portfolio_value: float = 100000.0):
        self.portfolio_value = portfolio_value
        self.trades: List[Trade] = []
        self.open_positions: Dict[str, Trade] = {}
        self.daily_margin_usage: pd.Series = pd.Series(dtype=float)
        
    def update_positions(
        self,
        date: pd.Timestamp,
        new_weights: pd.Series,
        prices: pd.Series,
        transaction_costs: float
    ) -> None:
        """Update positions based on new target weights."""
        # Calculate target quantities
        target_quantities = {}
        for ticker, weight in new_weights.items():
            if ticker in prices and not pd.isna(prices[ticker]) and prices[ticker] > 0:
                target_value = weight * self.portfolio_value
                target_quantities[ticker] = target_value / prices[ticker]
            else:
                target_quantities[ticker] = 0.0
        
        # Process position changes
        current_tickers = set(self.open_positions.keys())
        target_tickers = set(ticker for ticker, qty in target_quantities.items() if abs(qty) > 1e-6)
        
        # Close positions not in target
        for ticker in current_tickers - target_tickers:
            if ticker in prices:
                self._close_position(date, ticker, prices[ticker], transaction_costs)
        
        # Open new positions
        for ticker in target_tickers - current_tickers:
            if ticker in prices:
                self._open_position(date, ticker, target_quantities[ticker], prices[ticker], transaction_costs)
        
        # Update margin usage
        total_position_value = sum(abs(qty) * prices.get(ticker, 0) 
                                 for ticker, qty in target_quantities.items() 
                                 if abs(qty) > 1e-6)
        self.daily_margin_usage[date] = total_position_value / self.portfolio_value
    
    def update_mfe_mae(self, date: pd.Timestamp, prices: pd.Series) -> None:
        """Update Maximum Favorable/Adverse Excursion for open positions."""
        for ticker, trade in self.open_positions.items():
            if ticker in prices and not pd.isna(prices[ticker]):
                current_price = prices[ticker]
                
                # Calculate current P&L per share
                if trade.quantity > 0:  # Long position
                    pnl_per_share = current_price - trade.entry_price
                else:  # Short position
                    pnl_per_share = trade.entry_price - current_price
                
                # Update MFE (most favorable)
                if pnl_per_share > trade.mfe:
                    trade.mfe = pnl_per_share
                
                # Update MAE (most adverse)
                if pnl_per_share < trade.mae:
                    trade.mae = pnl_per_share
    
    def _open_position(self, date: pd.Timestamp, ticker: str, quantity: float, price: float, commission: float) -> None:
        """Open a new position."""
        entry_value = abs(quantity) * price
        
        trade = Trade(
            ticker=ticker,
            entry_date=date,
            exit_date=None,
            entry_price=price,
            exit_price=None,
            quantity=quantity,
            entry_value=entry_value,
            commission_entry=commission,
            commission_exit=0.0
        )
        
        self.open_positions[ticker] = trade
    
    def _close_position(self, date: pd.Timestamp, ticker: str, price: float, commission: float) -> None:
        """Close an existing position."""
        if ticker not in self.open_positions:
            return
        
        trade = self.open_positions[ticker]
        trade.exit_date = date
        trade.exit_price = price
        trade.commission_exit = commission
        
        # Finalize MFE/MAE in dollar terms
        trade.mfe = trade.mfe * abs(trade.quantity)
        trade.mae = trade.mae * abs(trade.quantity)
        
        trade.finalize()
        
        # Move to completed trades
        self.trades.append(trade)
        del self.open_positions[ticker]
    
    def close_all_positions(self, date: pd.Timestamp, prices: pd.Series) -> None:
        """Close all open positions at the end of backtesting."""
        tickers_to_close = list(self.open_positions.keys())
        
        for ticker in tickers_to_close:
            if ticker in prices and not pd.isna(prices[ticker]):
                self._close_position(date, ticker, prices[ticker], 0.0)
    
    def get_trade_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive trade statistics split by direction (All/Long/Short)."""
        if not self.trades:
            return self._get_empty_trade_stats()
        
        completed_trades = [t for t in self.trades if t.exit_date is not None]
        
        if not completed_trades:
            return self._get_empty_trade_stats()
        
        # Split trades by direction
        all_trades = completed_trades
        long_trades = [t for t in completed_trades if t.quantity > 0]
        short_trades = [t for t in completed_trades if t.quantity < 0]
        
        # Calculate statistics for each direction
        all_stats = self._calculate_direction_stats(all_trades, "all")
        long_stats = self._calculate_direction_stats(long_trades, "long")
        short_stats = self._calculate_direction_stats(short_trades, "short")
        
        # Combine all statistics
        combined_stats = {}
        
        # Add directional statistics with prefixes
        for direction, stats in [("all", all_stats), ("long", long_stats), ("short", short_stats)]:
            for key, value in stats.items():
                combined_stats[f"{direction}_{key}"] = value
        
        # Add portfolio-level statistics (not direction-specific)
        max_margin_load = self.daily_margin_usage.max() if not self.daily_margin_usage.empty else 0.0
        mean_margin_load = self.daily_margin_usage.mean() if not self.daily_margin_usage.empty else 0.0
        
        combined_stats.update({
            'max_margin_load': max_margin_load,
            'mean_margin_load': mean_margin_load
        })
        
        return combined_stats
    
    def get_directional_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get a summary of key metrics by direction for easy comparison."""
        stats = self.get_trade_statistics()
        
        summary = {}
        for direction in ['all', 'long', 'short']:
            prefix = f"{direction}_"
            summary[direction.title()] = {
                'trades': stats.get(f'{prefix}num_trades', 0),
                'win_rate': stats.get(f'{prefix}win_rate_pct', 0.0),
                'total_pnl': stats.get(f'{prefix}total_pnl_net', 0.0),
                'avg_profit': stats.get(f'{prefix}mean_profit', 0.0),
                'avg_loss': stats.get(f'{prefix}mean_loss', 0.0),
                'reward_risk': stats.get(f'{prefix}reward_risk_ratio', 0.0),
                'largest_win': stats.get(f'{prefix}largest_profit', 0.0),
                'largest_loss': stats.get(f'{prefix}largest_loss', 0.0)
            }
        
        return summary
    
    def get_trade_statistics_table(self) -> pd.DataFrame:
        """Get trade statistics formatted as a table with All/Long/Short columns."""
        stats = self.get_trade_statistics()
        
        if not self.trades:
            return pd.DataFrame()
        
        # Define the metrics we want to display
        metrics_config = [
            ('num_trades', 'Number of Trades', 'int'),
            ('num_winners', 'Number of Winners', 'int'),
            ('num_losers', 'Number of Losers', 'int'),
            ('win_rate_pct', 'Win Rate (%)', 'pct'),
            ('total_pnl_net', 'Total P&L Net', 'currency'),
            ('largest_profit', 'Largest Single Profit', 'currency'),
            ('largest_loss', 'Largest Single Loss', 'currency'),
            ('mean_profit', 'Mean Profit', 'currency'),
            ('mean_loss', 'Mean Loss', 'currency'),
            ('mean_trade_pnl', 'Mean Trade P&L', 'currency'),
            ('reward_risk_ratio', 'Reward/Risk Ratio', 'ratio'),
            ('total_commissions_paid', 'Commissions Paid', 'currency'),
            ('avg_mfe', 'Avg MFE', 'currency'),
            ('avg_mae', 'Avg MAE', 'currency'),
            ('min_trade_duration_days', 'Min Duration (days)', 'int'),
            ('max_trade_duration_days', 'Max Duration (days)', 'int'),
            ('mean_trade_duration_days', 'Mean Duration (days)', 'float'),
            ('information_score', 'Information Score', 'ratio'),
            ('trades_per_month', 'Trades per Month', 'float')
        ]
        
        # Build the table data
        table_data = []
        
        for metric_key, metric_name, format_type in metrics_config:
            row = {'Metric': metric_name}
            
            for direction in ['all', 'long', 'short']:
                direction_title = direction.title()
                value = stats.get(f'{direction}_{metric_key}', 0)
                
                # Format the value based on type
                if format_type == 'int':
                    formatted_value = f"{int(value)}"
                elif format_type == 'pct':
                    formatted_value = f"{value:.2f}%"
                elif format_type == 'currency':
                    formatted_value = f"${value:,.2f}"
                elif format_type == 'ratio':
                    if value == np.inf:
                        formatted_value = "∞"
                    else:
                        formatted_value = f"{value:.3f}"
                elif format_type == 'float':
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)
                
                row[direction_title] = formatted_value
            
            table_data.append(row)
        
        # Add portfolio-level metrics
        portfolio_metrics = [
            ('max_margin_load', 'Max Margin Load', 'pct'),
            ('mean_margin_load', 'Mean Margin Load', 'pct')
        ]
        
        for metric_key, metric_name, format_type in portfolio_metrics:
            row = {'Metric': metric_name}
            value = stats.get(metric_key, 0)
            
            if format_type == 'pct':
                formatted_value = f"{value * 100:.2f}%"
            else:
                formatted_value = f"{value:.4f}"
            
            # Portfolio metrics apply to all directions
            for direction in ['All', 'Long', 'Short']:
                row[direction] = formatted_value
            
            table_data.append(row)
        
        return pd.DataFrame(table_data)
    
    def _calculate_direction_stats(self, trades: List[Trade], direction: str) -> Dict[str, Any]:
        """Calculate statistics for a specific trade direction."""
        if not trades:
            return self._get_empty_direction_stats()
        
        # Basic trade counts
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.is_winner]
        losing_trades = [t for t in trades if not t.is_winner]
        num_winners = len(winning_trades)
        num_losers = len(losing_trades)
        win_rate = (num_winners / total_trades * 100) if total_trades > 0 else 0.0
        
        # P&L statistics
        pnl_values = [t.pnl_net for t in trades if t.pnl_net is not None]
        total_pnl_net = sum(pnl_values)
        total_commissions = sum(t.commission_entry + t.commission_exit for t in trades)
        
        # New metrics: Largest single trade profit/loss
        winning_pnls = [t.pnl_net for t in winning_trades if t.pnl_net is not None]
        losing_pnls = [t.pnl_net for t in losing_trades if t.pnl_net is not None]
        
        largest_profit = max(winning_pnls) if winning_pnls else 0.0
        largest_loss = min(losing_pnls) if losing_pnls else 0.0  # Most negative
        
        # Mean single trade profit/loss
        mean_profit = np.mean(winning_pnls) if winning_pnls else 0.0
        mean_loss = np.mean(losing_pnls) if losing_pnls else 0.0
        mean_trade_pnl = np.mean(pnl_values) if pnl_values else 0.0
        
        # Reward/Risk ratio (average win / average loss)
        reward_risk_ratio = abs(mean_profit / mean_loss) if mean_loss != 0 else np.inf if mean_profit > 0 else 0.0
        
        # Duration statistics
        durations = [t.duration_days for t in trades if t.duration_days is not None]
        min_duration = min(durations) if durations else 0
        max_duration = max(durations) if durations else 0
        mean_duration = np.mean(durations) if durations else 0
        
        # MFE/MAE statistics
        mfe_values = [t.mfe for t in trades]
        mae_values = [t.mae for t in trades]
        avg_mfe = np.mean(mfe_values) if mfe_values else 0.0
        avg_mae = np.mean(mae_values) if mae_values else 0.0
        
        # Calculate trades per month
        if trades:
            start_date = min(t.entry_date for t in trades)
            end_date = max(t.exit_date for t in trades if t.exit_date)
            total_months = ((end_date - start_date).days / 30.44) if end_date else 1
            trades_per_month = total_trades / total_months if total_months > 0 else 0
        else:
            trades_per_month = 0
        
        # Information Score (simplified)
        returns = [t.pnl_net / t.entry_value for t in trades if t.entry_value > 0]
        information_score = (np.mean(returns) / np.std(returns)) if len(returns) > 1 and np.std(returns) > 0 else 0.0
        
        return {
            'num_trades': total_trades,
            'num_winners': num_winners,
            'num_losers': num_losers,
            'win_rate_pct': win_rate,
            'total_commissions_paid': total_commissions,
            'total_pnl_net': total_pnl_net,
            'largest_profit': largest_profit,
            'largest_loss': largest_loss,
            'mean_profit': mean_profit,
            'mean_loss': mean_loss,
            'mean_trade_pnl': mean_trade_pnl,
            'reward_risk_ratio': reward_risk_ratio,
            'avg_mfe': avg_mfe,
            'avg_mae': avg_mae,
            'min_trade_duration_days': min_duration,
            'max_trade_duration_days': max_duration,
            'mean_trade_duration_days': mean_duration,
            'information_score': information_score,
            'trades_per_month': trades_per_month
        }
    
    def _get_empty_direction_stats(self) -> Dict[str, Any]:
        """Return empty trade statistics for a specific direction."""
        return {
            'num_trades': 0,
            'num_winners': 0,
            'num_losers': 0,
            'win_rate_pct': 0.0,
            'total_commissions_paid': 0.0,
            'total_pnl_net': 0.0,
            'largest_profit': 0.0,
            'largest_loss': 0.0,
            'mean_profit': 0.0,
            'mean_loss': 0.0,
            'mean_trade_pnl': 0.0,
            'reward_risk_ratio': 0.0,
            'avg_mfe': 0.0,
            'avg_mae': 0.0,
            'min_trade_duration_days': 0,
            'max_trade_duration_days': 0,
            'mean_trade_duration_days': 0.0,
            'information_score': 0.0,
            'trades_per_month': 0.0
        }
    
    def _get_empty_trade_stats(self) -> Dict[str, Any]:
        """Return empty trade statistics for cases with no trades."""
        empty_direction = self._get_empty_direction_stats()
        
        # Create empty stats for all directions
        combined_stats = {}
        for direction in ["all", "long", "short"]:
            for key, value in empty_direction.items():
                combined_stats[f"{direction}_{key}"] = value
        
        # Add portfolio-level statistics
        combined_stats.update({
            'max_margin_load': 0.0,
            'mean_margin_load': 0.0
        })
        
        return combined_stats