"""
TimingState dataclass for managing timing-related state across rebalancing periods.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Any, List, Tuple
import pandas as pd
import json
from datetime import datetime
import logging


@dataclass
class PositionInfo:
    """Detailed information about a position."""
    entry_date: pd.Timestamp
    entry_price: float
    entry_weight: float
    current_weight: float
    consecutive_periods: int = 0
    max_weight: float = 0.0
    min_weight: float = 0.0
    total_return: float = 0.0
    unrealized_pnl: float = 0.0
    
    def update_weight(self, new_weight: float, current_price: Optional[float] = None):
        """Update position weight and related metrics."""
        self.current_weight = new_weight
        self.max_weight = max(self.max_weight, new_weight)
        if self.min_weight == 0.0:  # First update
            self.min_weight = new_weight
        else:
            self.min_weight = min(self.min_weight, new_weight)
        
        if current_price is not None and self.entry_price > 0:
            self.total_return = (current_price - self.entry_price) / self.entry_price
            self.unrealized_pnl = self.current_weight * self.total_return


@dataclass
class TimingState:
    """Enhanced timing state management with advanced position tracking."""
    
    # Basic timing state
    last_signal_date: Optional[pd.Timestamp] = None
    last_weights: Optional[pd.Series] = None
    scheduled_dates: Set[pd.Timestamp] = field(default_factory=set)
    
    # Enhanced position tracking
    positions: Dict[str, PositionInfo] = field(default_factory=dict)
    position_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Legacy compatibility (maintained for backward compatibility)
    position_entry_dates: Dict[str, pd.Timestamp] = field(default_factory=dict)
    position_entry_prices: Dict[str, float] = field(default_factory=dict)
    consecutive_periods: Dict[str, int] = field(default_factory=dict)
    
    # State management metadata
    state_version: str = "1.0"
    last_updated: Optional[pd.Timestamp] = None
    debug_enabled: bool = False
    debug_log: List[Dict[str, Any]] = field(default_factory=list)
    
    def reset(self):
        """Reset all state for new backtest run."""
        self.last_signal_date = None
        self.last_weights = None
        self.scheduled_dates.clear()
        
        # Enhanced state
        self.positions.clear()
        self.position_history.clear()
        self.last_updated = None
        self.debug_log.clear()
        
        # Legacy compatibility
        self.position_entry_dates.clear()
        self.position_entry_prices.clear()
        self.consecutive_periods.clear()
        
        self._log_debug("State reset", {"action": "reset", "timestamp": pd.Timestamp.now()})
    
    def update_signal(self, date: pd.Timestamp, weights: pd.Series):
        """Update state after signal generation."""
        self.last_signal_date = date
        self.last_weights = weights.copy() if weights is not None else None
        self.last_updated = date
        
        self._log_debug("Signal updated", {
            "date": date,
            "num_assets": len(weights) if weights is not None else 0,
            "total_weight": weights.sum() if weights is not None else 0.0
        })
    
    def update_positions(self, date: pd.Timestamp, new_weights: pd.Series, prices: pd.Series):
        """Enhanced position tracking with detailed state management."""
        self.last_updated = date
        
        # Track position changes for history
        position_changes = []
        
        # Process each asset in new weights
        for asset in new_weights.index:
            new_weight = new_weights.get(asset, 0.0)
            current_price = prices.get(asset) if asset in prices.index else None
            
            if abs(new_weight) > 1e-10:  # New or existing position
                if asset not in self.positions:
                    # New position entry
                    position_info = PositionInfo(
                        entry_date=date,
                        entry_price=current_price if current_price is not None else 0.0,
                        entry_weight=new_weight,
                        current_weight=new_weight,
                        consecutive_periods=1,
                        max_weight=new_weight,
                        min_weight=new_weight
                    )
                    self.positions[asset] = position_info
                    
                    # Legacy compatibility
                    self.position_entry_dates[asset] = date
                    if current_price is not None:
                        self.position_entry_prices[asset] = current_price
                    self.consecutive_periods[asset] = 1
                    
                    position_changes.append({
                        "asset": asset,
                        "action": "entry",
                        "weight": new_weight,
                        "price": current_price
                    })
                    
                else:
                    # Update existing position
                    position_info = self.positions[asset]
                    old_weight = position_info.current_weight
                    position_info.update_weight(new_weight, current_price)
                    position_info.consecutive_periods += 1
                    
                    # Update legacy compatibility
                    self.consecutive_periods[asset] = position_info.consecutive_periods
                    
                    if abs(old_weight - new_weight) > 1e-10:
                        position_changes.append({
                            "asset": asset,
                            "action": "weight_change",
                            "old_weight": old_weight,
                            "new_weight": new_weight,
                            "price": current_price
                        })
            
            else:
                # Position exit
                if asset in self.positions:
                    position_info = self.positions[asset]
                    holding_days = (date - position_info.entry_date).days
                    
                    # Record position history before removal
                    self.position_history.append({
                        "asset": asset,
                        "entry_date": position_info.entry_date,
                        "exit_date": date,
                        "holding_days": holding_days,
                        "entry_price": position_info.entry_price,
                        "exit_price": current_price,
                        "entry_weight": position_info.entry_weight,
                        "max_weight": position_info.max_weight,
                        "min_weight": position_info.min_weight,
                        "consecutive_periods": position_info.consecutive_periods,
                        "total_return": position_info.total_return,
                        "final_pnl": position_info.unrealized_pnl
                    })
                    
                    # Remove from active positions
                    del self.positions[asset]
                    
                    # Legacy compatibility
                    self.position_entry_dates.pop(asset, None)
                    self.position_entry_prices.pop(asset, None)
                    self.consecutive_periods.pop(asset, None)
                    
                    position_changes.append({
                        "asset": asset,
                        "action": "exit",
                        "holding_days": holding_days,
                        "price": current_price
                    })
        
        # Log position changes
        if position_changes:
            self._log_debug("Position changes", {
                "date": date,
                "changes": position_changes,
                "active_positions": len(self.positions)
            })
    
    def get_position_holding_days(self, asset: str, current_date: pd.Timestamp) -> Optional[int]:
        """Get the number of days an asset has been held."""
        if asset in self.positions:
            return (current_date - self.positions[asset].entry_date).days
        elif asset in self.position_entry_dates:  # Legacy compatibility
            return (current_date - self.position_entry_dates[asset]).days
        return None
    
    def is_position_held(self, asset: str) -> bool:
        """Check if a position is currently held."""
        return asset in self.positions or asset in self.position_entry_dates
    
    def get_held_assets(self) -> Set[str]:
        """Get set of currently held assets."""
        return set(self.positions.keys()) | set(self.position_entry_dates.keys())
    
    # Enhanced state management methods
    
    def get_position_info(self, asset: str) -> Optional[PositionInfo]:
        """Get detailed position information for an asset."""
        return self.positions.get(asset)
    
    def get_consecutive_periods(self, asset: str) -> int:
        """Get consecutive periods for an asset (enhanced version)."""
        if asset in self.positions:
            return self.positions[asset].consecutive_periods
        return self.consecutive_periods.get(asset, 0)
    
    def get_position_return(self, asset: str, current_price: Optional[float] = None) -> Optional[float]:
        """Calculate current return for a position."""
        if asset not in self.positions:
            return None
        
        position_info = self.positions[asset]
        if position_info.entry_price <= 0:
            return None
        
        if current_price is not None:
            return (current_price - position_info.entry_price) / position_info.entry_price
        
        return position_info.total_return
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get summary of current portfolio state."""
        total_positions = len(self.positions)
        total_weight = sum(pos.current_weight for pos in self.positions.values())
        avg_holding_days = 0
        
        if self.last_updated and total_positions > 0:
            total_days = sum((self.last_updated - pos.entry_date).days for pos in self.positions.values())
            avg_holding_days = total_days / total_positions
        
        return {
            "total_positions": total_positions,
            "total_weight": total_weight,
            "avg_holding_days": avg_holding_days,
            "last_updated": self.last_updated,
            "total_historical_positions": len(self.position_history),
            "assets": list(self.positions.keys())
        }
    
    def get_position_statistics(self) -> Dict[str, Any]:
        """Get statistics about position history."""
        if not self.position_history:
            return {"total_trades": 0}
        
        holding_days = [pos["holding_days"] for pos in self.position_history]
        returns = [pos["total_return"] for pos in self.position_history if pos["total_return"] is not None]
        
        stats = {
            "total_trades": len(self.position_history),
            "avg_holding_days": sum(holding_days) / len(holding_days) if holding_days else 0,
            "min_holding_days": min(holding_days) if holding_days else 0,
            "max_holding_days": max(holding_days) if holding_days else 0,
        }
        
        if returns:
            stats.update({
                "avg_return": sum(returns) / len(returns),
                "min_return": min(returns),
                "max_return": max(returns),
                "positive_trades": sum(1 for r in returns if r > 0),
                "negative_trades": sum(1 for r in returns if r < 0),
                "win_rate": sum(1 for r in returns if r > 0) / len(returns)
            })
        
        return stats
    
    # State persistence methods
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary for persistence."""
        def serialize_timestamp(ts):
            return ts.isoformat() if ts is not None else None
        
        def serialize_series(series):
            return series.to_dict() if series is not None else None
        
        return {
            "state_version": self.state_version,
            "last_signal_date": serialize_timestamp(self.last_signal_date),
            "last_updated": serialize_timestamp(self.last_updated),
            "last_weights": serialize_series(self.last_weights),
            "scheduled_dates": [ts.isoformat() for ts in self.scheduled_dates],
            "positions": {
                asset: {
                    "entry_date": pos.entry_date.isoformat(),
                    "entry_price": pos.entry_price,
                    "entry_weight": pos.entry_weight,
                    "current_weight": pos.current_weight,
                    "consecutive_periods": pos.consecutive_periods,
                    "max_weight": pos.max_weight,
                    "min_weight": pos.min_weight,
                    "total_return": pos.total_return,
                    "unrealized_pnl": pos.unrealized_pnl
                }
                for asset, pos in self.positions.items()
            },
            "position_history": self.position_history.copy(),
            "debug_enabled": self.debug_enabled,
            "debug_log": self.debug_log.copy() if self.debug_enabled else []
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimingState':
        """Deserialize state from dictionary."""
        def parse_timestamp(ts_str):
            return pd.Timestamp(ts_str) if ts_str is not None else None
        
        def parse_series(series_dict):
            return pd.Series(series_dict) if series_dict is not None else None
        
        state = cls()
        state.state_version = data.get("state_version", "1.0")
        state.last_signal_date = parse_timestamp(data.get("last_signal_date"))
        state.last_updated = parse_timestamp(data.get("last_updated"))
        state.last_weights = parse_series(data.get("last_weights"))
        state.scheduled_dates = {pd.Timestamp(ts) for ts in data.get("scheduled_dates", [])}
        state.debug_enabled = data.get("debug_enabled", False)
        state.debug_log = data.get("debug_log", [])
        state.position_history = data.get("position_history", [])
        
        # Restore positions
        for asset, pos_data in data.get("positions", {}).items():
            position_info = PositionInfo(
                entry_date=pd.Timestamp(pos_data["entry_date"]),
                entry_price=pos_data["entry_price"],
                entry_weight=pos_data["entry_weight"],
                current_weight=pos_data["current_weight"],
                consecutive_periods=pos_data["consecutive_periods"],
                max_weight=pos_data["max_weight"],
                min_weight=pos_data["min_weight"],
                total_return=pos_data["total_return"],
                unrealized_pnl=pos_data["unrealized_pnl"]
            )
            state.positions[asset] = position_info
            
            # Maintain legacy compatibility
            state.position_entry_dates[asset] = position_info.entry_date
            state.position_entry_prices[asset] = position_info.entry_price
            state.consecutive_periods[asset] = position_info.consecutive_periods
        
        return state
    
    def save_to_file(self, filepath: str):
        """Save state to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'TimingState':
        """Load state from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    # Debugging utilities
    
    def enable_debug(self, enabled: bool = True):
        """Enable or disable debug logging."""
        self.debug_enabled = enabled
        if enabled:
            self._log_debug("Debug enabled", {"timestamp": pd.Timestamp.now()})
    
    def _log_debug(self, message: str, data: Dict[str, Any]):
        """Internal debug logging method."""
        if self.debug_enabled:
            log_entry = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "message": message,
                "data": data
            }
            self.debug_log.append(log_entry)
            
            # Keep debug log size manageable
            if len(self.debug_log) > 1000:
                self.debug_log = self.debug_log[-500:]  # Keep last 500 entries
    
    def get_debug_log(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get debug log entries."""
        if last_n is not None:
            return self.debug_log[-last_n:]
        return self.debug_log.copy()
    
    def clear_debug_log(self):
        """Clear debug log."""
        self.debug_log.clear()
        self._log_debug("Debug log cleared", {"timestamp": pd.Timestamp.now()})
    
    def print_state_summary(self):
        """Print a human-readable summary of the current state."""
        summary = self.get_portfolio_summary()
        stats = self.get_position_statistics()
        
        print("=" * 50)
        print("TIMING STATE SUMMARY")
        print("=" * 50)
        print(f"Last Updated: {summary['last_updated']}")
        print(f"Active Positions: {summary['total_positions']}")
        print(f"Total Weight: {summary['total_weight']:.4f}")
        print(f"Average Holding Days: {summary['avg_holding_days']:.1f}")
        
        if summary['assets']:
            print(f"Assets: {', '.join(summary['assets'])}")
        
        print(f"\nHistorical Statistics:")
        print(f"Total Trades: {stats['total_trades']}")
        if stats['total_trades'] > 0:
            print(f"Average Holding Days: {stats.get('avg_holding_days', 0):.1f}")
            if 'win_rate' in stats:
                print(f"Win Rate: {stats['win_rate']:.2%}")
                print(f"Average Return: {stats['avg_return']:.2%}")
        
        if self.debug_enabled:
            print(f"\nDebug Log Entries: {len(self.debug_log)}")
        
        print("=" * 50)