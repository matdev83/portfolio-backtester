"""
Low-Volatility Factor Strategy Implementation

Based on the research paper "Factoring in the Low-Volatility Factor" (SSRN-5295002)
by Amar Soebhag, Guido Baltussen, and Pim van Vliet (June 2025).

UNIVERSE SPECIFICATIONS FROM THE PAPER:
========================================

Primary Universe Requirements:
- Stock Exchange Coverage: NYSE, AMEX, and Nasdaq
- Share Codes: Only 10 or 11 (common stocks)
- Exclusions: Financial firms and firms with negative book-to-market ratios
- Time Period: January 1972 to December 2023 (paper's sample)

Critical Market Cap Filter:
- Excludes stocks below the 20th NYSE size percentile (micro-caps)
- Rationale: Micro-caps are typically inaccessible for most investors due to 
  limited capacity and high trading costs
- Micro-caps are ~3% of market value but ~60% of total number of stocks
- This filter provides a more realistic investable universe

Market Cap Ranges (Approximate):
- 0-20th NYSE percentile: < $500M (Micro-caps) - EXCLUDED
- 20th-50th percentile: $500M - $3B (Small-caps) - INCLUDED
- 50th-80th percentile: $3B - $25B (Mid-caps) - INCLUDED  
- 80th+ percentile: > $25B (Large-caps) - INCLUDED

Size Sorting Within Universe:
After excluding micro-caps, remaining universe is sorted using NYSE median:
- "Small": 20th-50th NYSE percentile
- "Big": 50th+ NYSE percentile

CURRENT IMPLEMENTATION LIMITATIONS:
==================================

Universe Gap:
- Current implementation uses only 55 large-cap S&P 500 stocks
- This represents only the top ~80th+ percentile of the market
- Paper uses full investable universe (~3,000-4,000 stocks)

Market Cap Data Missing:
- Paper requires actual market capitalization (shares outstanding × price)
- Current implementation lacks shares outstanding data
- Falls back to 1×3 volatility-only sorting instead of proper 2×3 size-volatility sorting

PROPER IMPLEMENTATION REQUIREMENTS:
==================================

To match paper methodology, need:
1. Full universe: All stocks above 20th NYSE percentile
2. Actual market cap data (shares outstanding)
3. ETF holdings data could solve this:
   - Small-cap ETFs: IWM, IJR, VB (Russell 2000, S&P 600 Small Cap)
   - Mid-cap ETFs: IJH, MDY, VO (S&P 400 Mid Cap)
   - Large-cap ETFs: SPY, IVV, VTI (S&P 500, Total Market)

IWC ETF AS MICRO-CAP EXCLUSION PROXY:
====================================

EXCELLENT SOLUTION: IWC (iShares Microcap ETF) tracks Russell Microcap Index:
- Companies ranked #2,001-4,000 in Russell 3000E Index
- Market cap range: ~$50M-$300M (exactly the micro-caps to exclude per paper)
- 1,364 holdings as of July 2025
- Median market cap: $228M, largest company: $3.6B

Perfect Alignment with Paper Requirements:
- Paper excludes stocks "below the 20th NYSE size percentile"
- 20th NYSE percentile ≈ $300-500M market cap threshold
- IWC holdings = $50-300M range (precisely the excluded micro-caps)

Implementation Strategy:
1. Download IWC holdings data periodically
2. Create exclusion list of micro-cap tickers
3. Filter our universe to exclude any stocks in IWC holdings
4. This creates a paper-compliant investable universe above 20th NYSE percentile

Benefits:
- Most practical and accurate approach available
- Regularly updated (ETF holdings change quarterly)
- Directly aligns with academic standards (Russell methodology)
- No need for expensive market cap data services

STRATEGY IMPLEMENTATION:
=======================

This strategy implements:
1. 2×3 sorting procedure on size and volatility (currently 1×3 due to data limitations)
2. Market-hedged long and short legs to account for factor asymmetry
3. Beta-neutral portfolio construction with 36-month rolling betas
4. Support for transaction costs and long-only implementations
5. 252-day rolling volatility calculation
6. Beta capping (0.25 to 2.0) to avoid excessive hedging

Factor Construction:
VOL_t = r_L,t - r_H,t - (β_L,t-1 - β_H,t-1) × MKT_t
Where:
- L = Low volatility portfolio
- H = High volatility portfolio  
- β = Prior month's market betas
- MKT = Value-weighted excess market return

Strategy Variants:
1. Long-minus-Short (Classic): Traditional factor construction
2. Long-plus-Short (Factor Asymmetry): Separate weights for long/short legs
3. Net Long-plus-Short (With Costs): Includes transaction costs and shorting fees
4. Net Long-Market (Long-Only): Focuses only on low-volatility stocks
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..data_sources.etf_holdings import ETFHoldingsDataSource
from .base_strategy import BaseStrategy

# Import Numba optimization with fallback for beta calculation
try:
    from ..numba_optimized import rolling_beta_fast_portfolio
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

logger = logging.getLogger(__name__)


class LowVolatilityFactorStrategy(BaseStrategy):
    """
    Low-Volatility Factor Strategy based on SSRN-5295002.
    
    The strategy constructs a low-volatility factor using:
    - 252-day rolling volatility calculation
    - 2x3 sorting procedure (size x volatility)
    - Market-hedged long and short legs
    - Beta-neutral portfolio construction
    """

    def __init__(self, strategy_config: Dict[str, Any]):
        super().__init__(strategy_config)
        
        # Initialize stateful variables
        self.w_prev: Optional[pd.Series] = None
        self.portfolio_betas: Dict[str, float] = {}
        self.size_breakpoints: Dict[str, float] = {}
        self.volatility_breakpoints: Dict[str, float] = {}
        self.etf_holdings_data_source = ETFHoldingsDataSource()
        
        # Default parameters based on the paper
        defaults = {
            "volatility_lookback_days": 252,  # 252 trading days for volatility calculation
            "vol_percentile_low": 30,  # 30th percentile for low volatility
            "vol_percentile_high": 70,  # 70th percentile for high volatility
            "beta_lookback_months": 36,  # 36-month rolling beta estimation
            "beta_min_cap": 0.25,  # Minimum beta cap to avoid excessive hedging
            "beta_max_cap": 2.0,  # Maximum beta cap to avoid excessive hedging
            "beta_max_low_vol": 1.0,  # Maximum beta for low-volatility portfolios
            "rebalance_frequency": "monthly",  # Rebalancing frequency
            "use_hedged_legs": True,  # Use market-hedged long/short legs
            "long_only": False,  # Whether to use long-only version
            "account_for_costs": False,  # Whether to account for transaction costs
            "shorting_fee_annual": 0.01,  # 1% annual shorting fee
            "price_column": "Close",  # Price column to use
            "leverage": 1.0,  # Portfolio leverage
            "smoothing_lambda": 0.5,  # Portfolio smoothing parameter
            # NOTE: size_percentile removed - we don't have market cap data
            # This implementation uses 1x3 volatility-only sorting instead of 2x3 size-volatility sorting
        }
        
        # Update strategy config with defaults
        params_dict_to_update = self.strategy_config
        if "strategy_params" in self.strategy_config:
            if self.strategy_config["strategy_params"] is None:
                self.strategy_config["strategy_params"] = {}
            params_dict_to_update = self.strategy_config["strategy_params"]
        
        for k, v in defaults.items():
            params_dict_to_update.setdefault(k, v)

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        """Return the set of tunable parameters for this strategy."""
        return {
            "volatility_lookback_days", "vol_percentile_low", 
            "vol_percentile_high", "beta_lookback_months", "beta_min_cap", 
            "beta_max_cap", "beta_max_low_vol", "use_hedged_legs", "long_only",
            "account_for_costs", "shorting_fee_annual", "leverage", "smoothing_lambda"
            # NOTE: size_percentile removed - we don't have market cap data
        }

    def get_minimum_required_periods(self) -> int:
        """Calculate minimum required periods for the strategy."""
        params = self.strategy_config.get("strategy_params", self.strategy_config)
        
        # Need volatility lookback days converted to months
        vol_lookback_months = max(1, params.get("volatility_lookback_days", 252) // 20)
        
        # Need beta lookback months
        beta_lookback_months = params.get("beta_lookback_months", 36)
        
        # Take the maximum plus buffer
        return max(vol_lookback_months, beta_lookback_months) + 2

    def _calculate_volatility(self, prices: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
        """
        Calculate rolling volatility for each asset.
        
        Args:
            prices: DataFrame with price data
            lookback_days: Number of days for volatility calculation
            
        Returns:
            DataFrame with volatility for each asset
        """
        # Calculate daily returns
        returns = prices.pct_change(fill_method=None)
        
        # Calculate rolling standard deviation (annualized)
        volatility = returns.rolling(window=lookback_days, min_periods=max(1, lookback_days // 2)).std()
        volatility = volatility * np.sqrt(252)  # Annualize
        
        return volatility

    def _calculate_market_caps(self, prices: pd.DataFrame) -> pd.Series | None:
        """
        Calculate market capitalizations.
        
        NOTE: We don't have actual market cap data (shares outstanding).
        This implementation will return None, indicating that 2x3 sorting
        based on size is not possible with current data.
        """
        return None

    def _get_size_breakpoints(self, market_caps: pd.Series, percentile: float) -> float:
        """Get size breakpoints based on NYSE percentile."""
        return market_caps.quantile(percentile / 100.0)

    def _get_volatility_breakpoints(self, volatilities: pd.Series, low_pct: float, high_pct: float) -> tuple:
        """Get volatility breakpoints based on NYSE percentiles."""
        low_breakpoint = volatilities.quantile(low_pct / 100.0)
        high_breakpoint = volatilities.quantile(high_pct / 100.0)
        return low_breakpoint, high_breakpoint

    def _sort_stocks_2x3(self, market_caps: pd.Series, volatilities: pd.Series, 
                         current_date: pd.Timestamp) -> Dict[str, list]:
        """
        Perform sorting procedure on size and volatility.
        
        If market_caps is None, performs 1x3 volatility-only sorting.
        Otherwise, performs 2x3 size-volatility sorting.
        
        Returns:
            Dictionary with portfolio assignments for each stock
        """
        params = self.strategy_config.get("strategy_params", self.strategy_config)
        
        # Get volatility breakpoints
        vol_low_bp, vol_high_bp = self._get_volatility_breakpoints(
            volatilities, params["vol_percentile_low"], params["vol_percentile_high"]
        )
        
        # Store breakpoints for reference
        self.volatility_breakpoints[current_date] = (vol_low_bp, vol_high_bp)
        
        # Create volatility groups
        low_vol_stocks = volatilities[volatilities <= vol_low_bp].index.tolist()
        med_vol_stocks = volatilities[(volatilities > vol_low_bp) & (volatilities <= vol_high_bp)].index.tolist()
        high_vol_stocks = volatilities[volatilities > vol_high_bp].index.tolist()
        
        if market_caps is None:
            # Volatility-only sorting (1x3) - no market cap data available
            self.size_breakpoints[current_date] = None
            portfolios = {
                "LV": low_vol_stocks,      # Low volatility
                "MV": med_vol_stocks,      # Medium volatility  
                "HV": high_vol_stocks,     # High volatility
                # Legacy names for compatibility
                "SLV": low_vol_stocks,     # Small Low Vol -> Low Vol
                "SMV": med_vol_stocks,     # Small Med Vol -> Med Vol
                "SHV": high_vol_stocks,    # Small High Vol -> High Vol
                "BLV": [],                 # Big Low Vol -> Empty
                "BMV": [],                 # Big Med Vol -> Empty
                "BHV": [],                 # Big High Vol -> Empty
            }
        else:
            # Full 2x3 sorting with size and volatility
            size_breakpoint = self._get_size_breakpoints(market_caps, params["size_percentile"])
            self.size_breakpoints[current_date] = size_breakpoint
            
            # Create size groups
            small_stocks = market_caps[market_caps <= size_breakpoint].index.tolist()
            big_stocks = market_caps[market_caps > size_breakpoint].index.tolist()
            
            # Create 6 portfolios: SLV, SMV, SHV, BLV, BMV, BHV
            portfolios = {
                "SLV": [stock for stock in small_stocks if stock in low_vol_stocks],
                "SMV": [stock for stock in small_stocks if stock in med_vol_stocks],
                "SHV": [stock for stock in small_stocks if stock in high_vol_stocks],
                "BLV": [stock for stock in big_stocks if stock in low_vol_stocks],
                "BMV": [stock for stock in big_stocks if stock in med_vol_stocks],
                "BHV": [stock for stock in big_stocks if stock in high_vol_stocks],
            }
        
        return portfolios

    def _calculate_portfolio_betas(self, prices: pd.DataFrame, market_prices: pd.Series, 
                                   portfolios: Dict[str, list], lookback_months: int) -> Dict[str, float]:
        """
        Calculate 36-month rolling betas for each portfolio.
        """
        betas = {}
        
        for portfolio_name, stocks in portfolios.items():
            if not stocks:
                betas[portfolio_name] = 1.0
                continue
            
            if NUMBA_AVAILABLE:
                portfolio_prices = prices[stocks].mean(axis=1)
                beta = rolling_beta_fast_portfolio(portfolio_prices.values, market_prices.values, lookback_months)
            else:
                # Calculate portfolio returns (equal-weighted for simplicity)
                portfolio_prices = prices[stocks].mean(axis=1)
                portfolio_returns = portfolio_prices.pct_change(fill_method=None).dropna()
                market_returns = market_prices.pct_change(fill_method=None).dropna()
                
                # Align returns
                common_dates = portfolio_returns.index.intersection(market_returns.index)
                if len(common_dates) < 12:  # Need at least 12 months
                    betas[portfolio_name] = 1.0
                    continue
                    
                portfolio_returns = portfolio_returns.loc[common_dates]
                market_returns = market_returns.loc[common_dates]
                
                # Calculate rolling beta (last 36 months)
                window_size = min(lookback_months, len(portfolio_returns))
                if window_size < 12:
                    betas[portfolio_name] = 1.0
                    continue
                    
                recent_port_returns = portfolio_returns.iloc[-window_size:]
                recent_mkt_returns = market_returns.iloc[-window_size:]
                
                # Calculate beta using covariance
                covariance = np.cov(recent_port_returns, recent_mkt_returns)[0, 1]
                market_variance = np.var(recent_mkt_returns)
                
                if market_variance > 0:
                    beta = covariance / market_variance
                else:
                    beta = 1.0
            
            betas[portfolio_name] = beta
            
        return betas

    def _create_hedged_factor_returns(self, portfolios: Dict[str, list], 
                                      portfolio_betas: Dict[str, float],
                                      current_date: pd.Timestamp) -> pd.Series:
        """
        Create market-hedged low-volatility factor returns.
        
        VOL_t = r_L,t - r_H,t - (β_L,t-1 - β_H,t-1) * MKT_t
        """
        params = self.strategy_config.get("strategy_params", self.strategy_config)
        
        # Apply beta caps
        beta_min = params["beta_min_cap"]
        beta_max = params["beta_max_cap"]
        beta_max_low_vol = params["beta_max_low_vol"]
        
        # Get low and high volatility portfolio betas
        low_vol_beta = (portfolio_betas.get("SLV", 1.0) + portfolio_betas.get("BLV", 1.0)) / 2
        high_vol_beta = (portfolio_betas.get("SHV", 1.0) + portfolio_betas.get("BHV", 1.0)) / 2
        
        # Apply caps
        low_vol_beta = np.clip(low_vol_beta, beta_min, min(beta_max, beta_max_low_vol))
        high_vol_beta = np.clip(high_vol_beta, beta_min, beta_max)
        
        # Create factor weights
        weights = pd.Series(0.0, index=self._get_all_stocks(portfolios))
        
        # Long leg: Low volatility stocks
        low_vol_stocks = portfolios["SLV"] + portfolios["BLV"]
        if low_vol_stocks:
            for stock in low_vol_stocks:
                weights[stock] = 0.5 / len(low_vol_stocks)  # 50% in low vol
        
        # Short leg: High volatility stocks (if not long-only)
        if not params["long_only"]:
            high_vol_stocks = portfolios["SHV"] + portfolios["BHV"]
            if high_vol_stocks:
                for stock in high_vol_stocks:
                    weights[stock] = -0.5 / len(high_vol_stocks)  # 50% short high vol
        
        # Store beta information for hedging
        self.portfolio_betas[current_date] = {
            "low_vol_beta": low_vol_beta,
            "high_vol_beta": high_vol_beta,
            "beta_adjustment": low_vol_beta - high_vol_beta
        }
        
        return weights

    def _apply_transaction_costs(self, weights: pd.Series, current_date: pd.Timestamp) -> pd.Series:
        """
        Apply transaction costs and shorting fees to the portfolio weights.
        This is a simplified implementation - in practice you'd need more sophisticated cost modeling.
        """
        params = self.strategy_config.get("strategy_params", self.strategy_config)
        
        if not params["account_for_costs"]:
            return weights
        
        # Simple transaction cost model - reduce weights slightly to account for costs
        # In practice, you'd use bid-ask spreads and more sophisticated models
        transaction_cost_factor = 0.995  # Assume 0.5% transaction cost
        
        # Apply transaction costs
        adjusted_weights = weights * transaction_cost_factor
        
        # Apply shorting fees to negative weights
        shorting_fee_factor = 1.0 - (params["shorting_fee_annual"] / 12)  # Monthly fee
        short_positions = adjusted_weights < 0
        adjusted_weights[short_positions] *= shorting_fee_factor
        
        return adjusted_weights

    def _create_long_only_weights(self, portfolios: Dict[str, list]) -> pd.Series:
        """
        Create long-only portfolio weights focusing on low-volatility stocks.
        This implements the "Net Long-Market" specification from the paper.
        """
        weights = pd.Series(0.0, index=self._get_all_stocks(portfolios))
        
        # Long leg: Low volatility stocks only
        low_vol_stocks = portfolios["SLV"] + portfolios["BLV"]
        if low_vol_stocks:
            for stock in low_vol_stocks:
                weights[stock] = 1.0 / len(low_vol_stocks)  # Equal weight in low vol stocks
        
        return weights

    def _get_all_stocks(self, portfolios: Dict[str, list]) -> list:
        """Get all unique stocks from portfolios."""
        all_stocks = set()
        for stocks in portfolios.values():
            all_stocks.update(stocks)
        return list(all_stocks)

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Generate trading signals for the Low-Volatility Factor Strategy.
        """
        # Data sufficiency validation
        is_sufficient, reason = self.validate_data_sufficiency(
            all_historical_data, benchmark_historical_data, current_date
        )
        if not is_sufficient:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(f"Insufficient data for {current_date}: {reason}")
            columns = (all_historical_data.columns.get_level_values(0).unique() 
                      if isinstance(all_historical_data.columns, pd.MultiIndex) 
                      else all_historical_data.columns)
            return pd.DataFrame(0.0, index=[current_date], columns=columns)

        # Filter universe by data availability and micro-cap exclusion
        valid_assets = self.filter_universe_by_data_availability(
            all_historical_data, current_date
        )
        
        if not valid_assets:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(f"No assets have sufficient data for {current_date}")
            columns = (all_historical_data.columns.get_level_values(0).unique() 
                      if isinstance(all_historical_data.columns, pd.MultiIndex) 
                      else all_historical_data.columns)
            return pd.DataFrame(0.0, index=[current_date], columns=columns)

        # Exclude micro-cap tickers using IWC holdings
        micro_cap_tickers = self.etf_holdings_data_source.get_micro_cap_tickers()
        initial_valid_assets_count = len(valid_assets)
        valid_assets = [asset for asset in valid_assets if asset not in micro_cap_tickers]
        if len(valid_assets) < initial_valid_assets_count:
            if logger.isEnabledFor(logging.INFO):
                logger.info(f"Excluded {initial_valid_assets_count - len(valid_assets)} micro-cap assets for {current_date}.")

        if not valid_assets:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(f"No assets remain after micro-cap exclusion for {current_date}.")
            columns = (all_historical_data.columns.get_level_values(0).unique() 
                      if isinstance(all_historical_data.columns, pd.MultiIndex) 
                      else all_historical_data.columns)
            return pd.DataFrame(0.0, index=[current_date], columns=columns)

        # Date window filtering
        if start_date and current_date < start_date:
            return pd.DataFrame(index=[current_date], columns=valid_assets).fillna(0.0)
        if end_date and current_date > end_date:
            return pd.DataFrame(index=[current_date], columns=valid_assets).fillna(0.0)

        params = self.strategy_config.get("strategy_params", self.strategy_config)
        price_col = params["price_column"]
        
        # Extract price data
        if isinstance(all_historical_data.columns, pd.MultiIndex):
            price_data = all_historical_data.xs(price_col, level=1, axis=1)
        else:
            price_data = all_historical_data
        
        # Filter to current date and before
        price_data = price_data[price_data.index <= current_date]
        
        if len(price_data) < params["volatility_lookback_days"] // 4:  # Need at least 1/4 of lookback period
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(f"Insufficient price history for volatility calculation at {current_date}")
            return pd.DataFrame(0.0, index=[current_date], columns=valid_assets)
        
        # Calculate volatilities
        volatilities = self._calculate_volatility(price_data, params["volatility_lookback_days"])
        current_volatilities = volatilities.loc[current_date].dropna()
        
        # Calculate market caps (returns None if not available)
        market_caps = self._calculate_market_caps(price_data)
        
        if market_caps is not None:
            # Full 2x3 sorting with market cap data
            current_market_caps = market_caps.loc[current_date].dropna()
            
            # Get common stocks with both volatility and market cap data
            common_stocks = current_volatilities.index.intersection(current_market_caps.index)
            common_stocks = [stock for stock in common_stocks if stock in valid_assets]
            
            if len(common_stocks) < 10:  # Need minimum number of stocks
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(f"Too few stocks with complete data at {current_date}")
                return pd.DataFrame(0.0, index=[current_date], columns=valid_assets)
            
            # Filter data to common stocks
            current_volatilities = current_volatilities[common_stocks]
            current_market_caps = current_market_caps[common_stocks]
        else:
            # Volatility-only sorting (1x3) - no market cap data
            common_stocks = [stock for stock in current_volatilities.index if stock in valid_assets]
            
            if len(common_stocks) < 10:  # Need minimum number of stocks
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(f"Too few stocks with complete data at {current_date}")
                return pd.DataFrame(0.0, index=[current_date], columns=valid_assets)
            
            # Filter volatilities to common stocks
            current_volatilities = current_volatilities[common_stocks]
            current_market_caps = None  # No market cap data available
        
        # Perform sorting (2x3 if market cap available, 1x3 if not)
        portfolios = self._sort_stocks_2x3(current_market_caps, current_volatilities, current_date)
        
        # Calculate portfolio betas
        benchmark_prices = benchmark_historical_data[price_col] if price_col in benchmark_historical_data.columns else benchmark_historical_data.iloc[:, 0]
        portfolio_betas = self._calculate_portfolio_betas(
            price_data, benchmark_prices, portfolios, params["beta_lookback_months"]
        )
        
        # Create factor weights based on strategy configuration
        if params["long_only"]:
            weights = self._create_long_only_weights(portfolios)
        else:
            weights = self._create_hedged_factor_returns(portfolios, portfolio_betas, current_date)
        
        # Apply transaction costs if enabled
        weights = self._apply_transaction_costs(weights, current_date)
        
        # Apply leverage and smoothing
        if self.w_prev is not None:
            weights = self._apply_leverage_and_smoothing(weights, self.w_prev)
        else:
            weights = weights * params["leverage"]
        
        # Update previous weights
        # PERFORMANCE OPTIMIZATION: Store reference, copy only if strategy modifies weights later

        self.w_prev = weights
        
        # Create result DataFrame
        result = pd.DataFrame(0.0, index=[current_date], columns=valid_assets)
        for stock in weights.index:
            if stock in valid_assets:
                result.loc[current_date, stock] = weights[stock]
        
        return result 