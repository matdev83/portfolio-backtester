"""
Asset Replacement Logic for Monte-Carlo Optimization

This module handles the random selection and replacement of assets with synthetic data
during the optimization process. It ensures that:
1. Random selection is consistent within each optimization run
2. Different assets are selected for each WFO window
3. Replacement percentage is configurable
4. Original data is preserved for training phases
"""

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .synthetic_data_generator import SyntheticDataGenerator

# Import Numba optimization with fallback
try:
    from ..numba_optimized import generate_ohlc_from_prices_fast, returns_to_prices_fast
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class ReplacementInfo:
    """Information about asset replacement for a specific run."""
    selected_assets: Set[str]
    replacement_percentage: float
    random_seed: Optional[int]
    total_assets: int


class AssetReplacementManager:
    """
    Manages the random selection and replacement of assets with synthetic data
    during Monte-Carlo optimization runs.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the Asset Replacement Manager.
        
        Args:
            config: Monte Carlo configuration dictionary
        """
        self.config = config
        self.replacement_history: List[ReplacementInfo] = []
        self._asset_stats_cache: Dict[str, Any] = {}
        self._full_data_cache: Dict[str, pd.DataFrame] = {}  # Cache for full historical data
        
        # Initialize synthetic data generator with improved version
        self.synthetic_generator = SyntheticDataGenerator(config)
        
        # Performance optimization for Stage 1 MC (during optimization)
        # Reduce validation and use faster generation methods
        if config.get('stage1_optimization', False):
            # Override config for faster generation during optimization
            optimized_config = config.copy()
            optimized_config['validation_config'] = {
                'enable_validation': False  # Skip validation for speed
            }
            optimized_config['generation_config'] = {
                'max_attempts': 1,  # Single attempt for speed
                'validation_tolerance': 1.0  # Very lenient
            }
            self.synthetic_generator = SyntheticDataGenerator(optimized_config)
            logger.debug("Initialized AssetReplacementManager for Stage 1 MC (optimization mode)")
        else:
            self.synthetic_generator = SyntheticDataGenerator(config)
            logger.debug("Initialized AssetReplacementManager for Stage 2 MC (stress testing mode)")
        
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"AssetReplacementManager initialized with {config.get('replacement_percentage', 0.1):.1%} replacement rate")
    
    def set_full_data_source(self, data_source, global_config: Dict):
        """
        Set the data source for accessing full historical data.
        
        Args:
            data_source: Data source instance (YFinanceDataSource, StooqDataSource, etc.)
            global_config: Global configuration containing universe and date ranges
        """
        self.data_source = data_source
        self.global_config = global_config
        logger.debug("Full data source configured for comprehensive statistical analysis")
    
    def _load_full_historical_data(self, asset: str) -> Optional[pd.DataFrame]:
        """
        Load complete historical data for an asset to perform proper statistical analysis.
        This bypasses the windowed data limitation and uses the full available history.
        
        Args:
            asset: Asset symbol to load data for
            
        Returns:
            Full historical OHLC data for the asset, or None if unavailable
        """
        if asset in self._full_data_cache:
            return self._full_data_cache[asset]
        
        if not hasattr(self, 'data_source') or not hasattr(self, 'global_config'):
            logger.warning("Full data source not configured. Cannot load comprehensive historical data.")
            return None
        
        try:
            # Load the full historical data for this asset
            # Use a much earlier start date to get comprehensive statistics
            extended_start = pd.Timestamp('2000-01-01')  # Go back further for better statistics
            current_end = pd.Timestamp.now().normalize()
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Loading full historical data for {asset} from {extended_start} to {current_end}")
            
            # Get full data from the data source
            full_data = self.data_source.get_data(
                tickers=[asset],
                start_date=extended_start.strftime('%Y-%m-%d'),
                end_date=current_end.strftime('%Y-%m-%d')
            )
            
            if full_data is None or full_data.empty:
                logger.warning(f"No full historical data available for {asset}")
                return None
            
            # Convert to OHLC format if needed
            if isinstance(full_data.columns, pd.MultiIndex):
                # Handle MultiIndex columns
                try:
                    asset_data = full_data.xs(asset, level='Ticker', axis=1, drop_level=True)
                except KeyError:
                    logger.warning(f"Asset {asset} not found in MultiIndex data")
                    return None
            else:
                # Handle simple column structure - create OHLC from Close prices
                if asset in full_data.columns:
                    close_prices = full_data[asset].dropna()
                    asset_data = pd.DataFrame({
                        'Open': close_prices,
                        'High': close_prices,
                        'Low': close_prices,
                        'Close': close_prices
                    })
                else:
                    logger.warning(f"Asset {asset} not found in data columns")
                    return None
            
            # Cache the result
            self._full_data_cache[asset] = asset_data
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Loaded {len(asset_data)} days of full historical data for {asset}")
            
            return asset_data
            
        except Exception as e:
            logger.error(f"Failed to load full historical data for {asset}: {e}")
            return None
    
    def select_assets_for_replacement(
        self, 
        universe: List[str],
        random_seed: Optional[int] = None
    ) -> Set[str]:
        """
        Randomly select assets for replacement with synthetic data.
        
        Args:
            universe: List of all assets in the universe
            random_seed: Optional random seed for reproducible selection
            
        Returns:
            Set of asset symbols to replace
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Calculate number of assets to replace
        num_to_replace = max(1, int(len(universe) * self.config.get('replacement_percentage', 0.1)))
        
        # Ensure we don't replace more assets than available
        num_to_replace = min(num_to_replace, len(universe))
        
        # Randomly select assets
        selected_assets = set(random.sample(universe, num_to_replace))
        
        # Log the selection
        if logger.isEnabledFor(logging.INFO):

            logger.info(f"Selected {len(selected_assets)} assets for replacement: {selected_assets}")
        
        # Store replacement info
        replacement_info = ReplacementInfo(
            selected_assets=selected_assets,
            replacement_percentage=self.config.get('replacement_percentage', 0.1),
            random_seed=random_seed,
            total_assets=len(universe)
        )
        self.replacement_history.append(replacement_info)
        
        return selected_assets
    
    def replace_asset_data(
        self,
        original_data: Dict[str, pd.DataFrame],
        assets_to_replace: Set[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        phase: str = "test"
    ) -> Dict[str, pd.DataFrame]:
        """
        Replace selected assets' data with synthetic data for the specified period.
        
        Args:
            original_data: Dictionary of asset symbol -> OHLC DataFrame
            assets_to_replace: Set of assets to replace
            start_date: Start date for replacement period
            end_date: End date for replacement period
            phase: Phase of optimization ("train" or "test")
            
        Returns:
            Dictionary with synthetic data replacing selected assets
        """
        if phase == "train":
            logger.warning("Asset replacement should NOT be used in training phase!")
            return original_data
        
        # Create a deep copy of the original dataset to avoid modifying it in place
        modified_data = {k: v.copy(deep=True) for k, v in original_data.items()}
        
        # Replace data for selected assets
        for asset in assets_to_replace:
            if asset not in original_data:
                if logger.isEnabledFor(logging.WARNING):

                    logger.warning(f"Asset {asset} not found in data. Skipping replacement.")
                continue
            
            try:
                # Get the original data for this asset
                asset_data = original_data[asset]
                
                # Find the data period that needs replacement
                mask = (asset_data.index >= start_date) & (asset_data.index <= end_date)
                period_data = asset_data.loc[mask]
                
                if len(period_data) == 0:
                    if logger.isEnabledFor(logging.WARNING):

                        logger.warning(f"No data found for {asset} in period {start_date} to {end_date}")
                    continue
                
                # Get historical data for parameter estimation (use data before the replacement period)
                # Use full historical data for comprehensive statistical analysis (no lookahead bias)
                full_historical_data = self._load_full_historical_data(asset)
                
                if full_historical_data is not None:
                    # Use full historical data for better statistical analysis
                    # Only use data up to start_date to avoid lookahead bias
                    historical_mask = full_historical_data.index < start_date
                    historical_data = full_historical_data.loc[historical_mask]
                    
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Using {len(historical_data)} days of full historical data for {asset} statistics")
                else:
                    # Fallback to windowed data if full data unavailable
                    historical_mask = asset_data.index < start_date
                    historical_data = asset_data.loc[historical_mask]
                    
                    if len(historical_data) < self.config.get('min_historical_observations', 252):
                        # Use more lenient requirement during optimization for performance
                        min_obs_required = self.config.get('min_historical_observations', 252)
                        if self.config.get('stage1_optimization', False):
                            min_obs_required = min(50, min_obs_required)  # Much lower requirement for stage 1
                        
                        if logger.isEnabledFor(logging.WARNING):
                            logger.warning(f"Insufficient historical data for {asset} ({len(historical_data)} < {min_obs_required}). Using all available data.")
                        # Get all data except the period we want to replace
                        non_period_mask = ~mask
                        historical_data = asset_data.loc[non_period_mask]
                        
                        # If still no data, skip this asset
                        if len(historical_data) == 0:
                            if logger.isEnabledFor(logging.WARNING):
                                logger.warning(f"No historical data available for {asset}. Skipping synthetic replacement.")
                            continue
                
                # Generate synthetic data
                synthetic_data = self._generate_synthetic_data_for_period(
                    historical_data=historical_data,
                    target_length=len(period_data),
                    target_dates=period_data.index,
                    asset_name=asset
                )
                
                # Replace data for the selected time period with synthetic data
                asset_mask = (modified_data[asset].index >= start_date) & (modified_data[asset].index <= end_date)
                if asset_mask.any():
                    # Handle each column individually to avoid dtype conflicts
                    for col in synthetic_data.columns:
                        if col in modified_data[asset].columns:
                            try:
                                # Get original column dtype for this specific asset
                                original_dtype = modified_data[asset][col].dtype
                                synthetic_col_data = synthetic_data[col]
                                
                                # Convert synthetic data to compatible dtype
                                if pd.api.types.is_integer_dtype(original_dtype):
                                    # For integer columns (like volume), round and convert
                                    synthetic_col_converted = synthetic_col_data.round().astype(original_dtype)
                                elif pd.api.types.is_float_dtype(original_dtype):
                                    # For float columns, direct conversion
                                    synthetic_col_converted = synthetic_col_data.astype(original_dtype)
                                else:
                                    # For other types, try direct conversion
                                    synthetic_col_converted = synthetic_col_data.astype(original_dtype)
                                
                                # Replace the data for this specific asset
                                modified_data[asset].loc[asset_mask, col] = synthetic_col_converted
                                
                            except Exception as col_error:
                                if logger.isEnabledFor(logging.WARNING):

                                    logger.warning(f"Failed to replace column {col} for {asset}: {col_error}")
                                # Skip this column if conversion fails
                                continue
                
                if logger.isEnabledFor(logging.INFO):

                
                    logger.info(f"Replaced {len(period_data)} observations for {asset} with synthetic data")
                
            except Exception as e:
                logger.error(f"Failed to replace data for {asset}: {e}")
                # Keep original data if replacement fails
                continue
        
        return modified_data
    
    def create_monte_carlo_dataset(
        self,
        original_data: Dict[str, pd.DataFrame],
        universe: List[str],
        test_start: pd.Timestamp,
        test_end: pd.Timestamp,
        run_id: Optional[str] = None,
        random_seed: Optional[int] = None
    ) -> Tuple[Dict[str, pd.DataFrame], ReplacementInfo]:
        """
        Create a Monte-Carlo dataset with randomly selected assets replaced by synthetic data.
        
        Args:
            original_data: Original asset data
            universe: List of assets in universe
            test_start: Start of test period
            test_end: End of test period
            run_id: Optional identifier for this run
            random_seed: Optional random seed
            
        Returns:
            Tuple of (modified_data, replacement_info)
        """
        # Skip replacement if disabled
        if not self.config.get('enable_synthetic_data', True):
            dummy_info = ReplacementInfo(
                selected_assets=set(),
                replacement_percentage=0.0,
                random_seed=random_seed,
                total_assets=len(universe)
            )
            return original_data, dummy_info
        
        # Select assets for replacement
        assets_to_replace = self.select_assets_for_replacement(universe, random_seed)
        
        # Replace asset data in test period only
        modified_data = self.replace_asset_data(
            original_data=original_data,
            assets_to_replace=assets_to_replace,
            start_date=test_start,
            end_date=test_end,
            phase="test"
        )
        
        # Get replacement info
        replacement_info = self.replacement_history[-1]
        
        if run_id:
            if logger.isEnabledFor(logging.INFO):

                logger.info(f"Run {run_id}: Created Monte-Carlo dataset with {len(assets_to_replace)} synthetic assets")
        
        return modified_data, replacement_info
    
    def _generate_synthetic_data_for_period(
        self,
        historical_data: pd.DataFrame,
        target_length: int,
        target_dates: pd.DatetimeIndex,
        asset_name: str
    ) -> pd.DataFrame:
        """
        Generate synthetic OHLC data for a specific time period.
        
        Args:
            historical_data: Historical OHLC data for parameter estimation
            target_length: Length of synthetic series needed
            target_dates: Target date index
            asset_name: Name of the asset
            
        Returns:
            DataFrame with synthetic OHLC data
        """
        # Check cache first
        cache_key = f"{asset_name}_{len(historical_data)}_{target_length}"
        if cache_key in self._asset_stats_cache:
            asset_stats = self._asset_stats_cache[cache_key]
        else:
            # Analyze historical data to get statistical properties
            asset_stats = self.synthetic_generator.analyze_asset_statistics(historical_data)
            self._asset_stats_cache[cache_key] = asset_stats
        
        # Generate synthetic returns
        synthetic_returns = self.synthetic_generator.generate_synthetic_returns(
            asset_stats=asset_stats,
            length=target_length,
            asset_name=asset_name
        )
        
        # Convert to prices starting from last historical price
        initial_price = historical_data['Close'].iloc[-1]
        synthetic_prices = self._returns_to_prices(synthetic_returns, initial_price)
        
        # Generate OHLC data
        synthetic_ohlc = self._generate_ohlc_from_prices(synthetic_prices)
        
        # Create DataFrame with target dates
        synthetic_df = pd.DataFrame(
            synthetic_ohlc,
            columns=['Open', 'High', 'Low', 'Close'],
            index=target_dates[:len(synthetic_ohlc)]  # Ensure length matches
        )
        
        # Add any additional columns that might be in the original data
        for col in historical_data.columns:
            if col not in synthetic_df.columns:
                if col.lower() in ['volume', 'vol']:
                    # Generate synthetic volume based on historical patterns
                    historical_volume = historical_data[col].dropna()
                    if len(historical_volume) > 0:
                        avg_volume = historical_volume.mean()
                        volume_std = historical_volume.std()
                        synthetic_volume = np.random.normal(avg_volume, volume_std, len(synthetic_df))
                        synthetic_volume = np.maximum(synthetic_volume, avg_volume * 0.1)  # Minimum volume
                        synthetic_df[col] = synthetic_volume
                    else:
                        synthetic_df[col] = 1000000  # Default volume
                else:
                    # For other columns, use forward fill from last historical value
                    if len(historical_data) > 0:
                        synthetic_df[col] = historical_data[col].iloc[-1]
        
        return synthetic_df
    
    def _returns_to_prices(self, returns: np.ndarray, initial_price: float) -> np.ndarray:
        """Convert returns to price levels."""
        if NUMBA_AVAILABLE:
            # Use Numba-jitted fast version for significant speedup
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Using Numba-jitted returns-to-prices conversion for {len(returns)} periods")
            
            return returns_to_prices_fast(returns, initial_price)
        
        # Fallback to original Python implementation
        price_relatives = 1 + returns
        prices = initial_price * np.cumprod(price_relatives)
        return prices
    
    def _generate_ohlc_from_prices(self, prices: np.ndarray) -> np.ndarray:
        """
        Generate OHLC data from price series (simplified but more realistic approach).
        Uses Numba-jitted fast version when available for 15-30x speedup.
        """
        if NUMBA_AVAILABLE:
            # Use Numba-jitted fast version for significant speedup
            random_seed = np.random.randint(0, 2**31 - 1)
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Using Numba-jitted OHLC generation for {len(prices)} periods")
            
            return generate_ohlc_from_prices_fast(prices, random_seed)
        
        # Fallback to original Python implementation
        ohlc = np.zeros((len(prices), 4))
        
        for i in range(len(prices)):
            close = prices[i]
            
            if i == 0:
                # First day: open at close price
                open_price = close
            else:
                # Subsequent days: open near previous close with small gap
                prev_close = prices[i-1]
                gap_factor = np.random.normal(1.0, 0.005)  # Small random gap
                open_price = prev_close * gap_factor
            
            # Generate realistic intraday range
            daily_volatility = abs(close - open_price) / open_price if open_price != 0 else 0.01
            daily_volatility = max(daily_volatility, 0.005)  # Minimum volatility
            
            # High and low based on realistic intraday movements
            intraday_range = open_price * daily_volatility * np.random.uniform(1.5, 4.0)
            
            # Determine high and low
            if close >= open_price:
                # Up day
                high = max(open_price, close) + intraday_range * np.random.uniform(0.2, 0.8)
                low = min(open_price, close) - intraday_range * np.random.uniform(0.1, 0.5)
            else:
                # Down day
                high = max(open_price, close) + intraday_range * np.random.uniform(0.1, 0.5)
                low = min(open_price, close) - intraday_range * np.random.uniform(0.2, 0.8)
            
            # Ensure low > 0 and logical ordering
            low = max(low, close * 0.5, open_price * 0.5)
            high = max(high, open_price, close)
            
            ohlc[i] = [open_price, high, low, close]
        
        return ohlc
    
    def get_replacement_statistics(self) -> Dict:
        """
        Get statistics about asset replacements performed.
        
        Returns:
            Dictionary with replacement statistics
        """
        if not self.replacement_history:
            return {"total_runs": 0}
        
        total_runs = len(self.replacement_history)
        total_assets_replaced = sum(len(info.selected_assets) for info in self.replacement_history)
        avg_replacement_percentage = np.mean([info.replacement_percentage for info in self.replacement_history])
        
        # Count how often each asset was replaced
        asset_replacement_counts = {}
        for info in self.replacement_history:
            for asset in info.selected_assets:
                asset_replacement_counts[asset] = asset_replacement_counts.get(asset, 0) + 1
        
        return {
            "total_runs": total_runs,
            "total_assets_replaced": total_assets_replaced,
            "avg_replacement_percentage": avg_replacement_percentage,
            "avg_assets_per_run": total_assets_replaced / total_runs if total_runs > 0 else 0,
            "asset_replacement_counts": asset_replacement_counts,
            "most_replaced_assets": sorted(asset_replacement_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def clear_cache(self):
        """Clear the asset statistics cache."""
        self._asset_stats_cache.clear()
        logger.info("Cleared asset statistics cache")
    
    def reset_history(self):
        """Reset the replacement history."""
        self.replacement_history.clear()
        logger.info("Reset replacement history")
    
    def get_replacement_info(self):
        """
        Get information about the most recent asset replacement.
        
        Returns:
            Object with selected_assets attribute containing list of replaced assets
        """
        class ReplacementInfoResult:
            def __init__(self, selected_assets):
                self.selected_assets = selected_assets
        
        # Get the most recent replacement from history
        if self.replacement_history:
            latest_replacement = self.replacement_history[-1]
            selected_assets = latest_replacement.selected_assets
        else:
            selected_assets = set()
        
        return ReplacementInfoResult(selected_assets) 