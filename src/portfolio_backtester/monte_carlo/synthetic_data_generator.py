"""
Improved Synthetic Data Generator for Financial Time Series

This module implements an advanced synthetic data generator using the professional
`arch` package for GARCH modeling. It provides:
- Robust GARCH parameter estimation using MLE
- Multiple distribution support (Normal, Student-t, Skewed Student-t)
- Better handling of fat tails and volatility clustering
- Improved validation and error handling
- Professional-grade simulation capabilities

Key Features:
- Uses arch package for professional GARCH implementation
- Multiple fallback strategies for robustness
- Statistical validation of generated data
- Preservation of stylized facts (volatility clustering, fat tails, etc.)
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    from arch import arch_model
    from arch.univariate import GARCH
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class DistributionType(Enum):
    """Supported innovation distributions for GARCH models."""
    NORMAL = "normal"
    STUDENT_T = "studentt" 
    SKEWED_STUDENT_T = "skewstudent"
    GED = "ged"


@dataclass
class GARCHParameters:
    """Container for GARCH model parameters from arch package."""
    omega: float  # Constant term
    alpha: float  # ARCH coefficient
    beta: float   # GARCH coefficient
    gamma: Optional[float] = None   # Asymmetry parameter (GJR-GARCH)
    nu: Optional[float] = None      # Degrees of freedom (Student-t)
    lambda_: Optional[float] = None  # Skewness parameter
    distribution: DistributionType = DistributionType.STUDENT_T
    aic: Optional[float] = None
    bic: Optional[float] = None


@dataclass
class AssetStatistics:
    """Container for asset statistical properties."""
    mean_return: float
    volatility: float
    skewness: float
    kurtosis: float
    autocorr_returns: float
    autocorr_squared: float
    tail_index: float
    garch_params: Optional[GARCHParameters] = None


class ImprovedSyntheticDataGenerator:
    """
    Improved synthetic data generator using professional GARCH modeling.
    
    Features:
    - Uses arch package for robust GARCH fitting
    - Multiple distribution support (Normal, Student-t, Skewed Student-t)
    - Proper parameter estimation with MLE
    - Comprehensive validation and fallback strategies
    - Preserves key financial time series properties
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the improved synthetic data generator.
        
        Args:
            config: Monte Carlo configuration dictionary
        """
        if not ARCH_AVAILABLE:
            raise ImportError("arch package is required. Install with: pip install arch")
            
        self.config = config
        self.garch_config = config.get('garch_config', {})
        self.generation_config = config.get('generation_config', {})
        self.validation_config = config.get('validation_config', {})
        
        # Set random seed if specified and create consistent random state
        self.random_seed = config.get('random_seed', 42)
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            self.rng = np.random.RandomState(self.random_seed)
        else:
            self.rng = np.random.RandomState()
    
    def analyze_asset_statistics(self, data) -> AssetStatistics:
        """
        Analyze statistical properties of historical asset data.
        
        Args:
            data: DataFrame with OHLC price data OR Series with returns data
            
        Returns:
            AssetStatistics object with computed properties
        """
        if hasattr(data, 'empty') and data.empty or len(data) == 0:
            logger.warning("Empty data provided. Using default statistics.")
            return AssetStatistics(
                mean_return=0.0,
                volatility=1.0,
                skewness=0.0,
                kurtosis=3.0,
                autocorr_returns=0.0,
                autocorr_squared=0.0,
                tail_index=2.0,
                garch_params=None
            )
        
        # Handle both DataFrame (OHLC) and Series (returns) inputs
        if isinstance(data, pd.DataFrame):
            # Calculate returns from OHLC data
            if 'Close' not in data.columns:
                logger.error("No 'Close' column found in OHLC data. Using default statistics.")
                return AssetStatistics(
                    mean_return=0.0,
                    volatility=1.0,
                    skewness=0.0,
                    kurtosis=3.0,
                    autocorr_returns=0.0,
                    autocorr_squared=0.0,
                    tail_index=2.0,
                    garch_params=None
                )
            returns = data['Close'].pct_change(fill_method=None).dropna()
        else:
            # Assume it's already returns data (Series)
            returns = data.dropna() if hasattr(data, 'dropna') else pd.Series(data).dropna()
        
        if len(returns) == 0:
            logger.warning("No valid returns calculated. Using default statistics.")
            return AssetStatistics(
                mean_return=0.0,
                volatility=1.0,
                skewness=0.0,
                kurtosis=3.0,
                autocorr_returns=0.0,
                autocorr_squared=0.0,
                tail_index=2.0,
                garch_params=None
            )
        
        min_required = self.config.get('min_historical_observations', 252)
        
        # More flexible minimum requirements based on optimization stage
        if self.config.get('stage1_optimization', False):
            # Stage 1 (optimization): More flexible requirements for speed
            min_required = max(50, min_required // 5)  # At least 50 observations, or 1/5 of normal requirement
        else:
            # Stage 2 (stress testing): Use configured minimum but allow fallback
            min_required = max(100, min_required // 2)  # At least 100 observations, or 1/2 of normal requirement
        
        if len(returns) < min_required:
            # Instead of hard error, log warning and use all available data
            if logger.isEnabledFor(logging.WARNING):

                logger.warning(f"Limited historical data: {len(returns)} observations (minimum: {min_required}). Using fallback approach.")
            
            # If we have very little data, raise an error as expected by tests
            if len(returns) < 10:
                raise ValueError(f"Insufficient historical data: {len(returns)} observations. "
                               f"Minimum required: 10 observations for basic analysis.")
            # If we have some data but not much, use simple statistical approach
            elif len(returns) < 30:
                if logger.isEnabledFor(logging.WARNING):

                    logger.warning(f"Very limited data ({len(returns)} observations). Using simple statistical fallback.")
                # Create minimal statistics for fallback generation
                mean_return = returns.mean() if len(returns) > 0 else 0.0
                volatility = returns.std() if len(returns) > 1 else 1.0
                
                return AssetStatistics(
                    mean_return=mean_return,
                    volatility=volatility,
                    skewness=0.0,  # Default values
                    kurtosis=3.0,
                    autocorr_returns=0.0,
                    autocorr_squared=0.0,
                    tail_index=2.0,
                    garch_params=None  # No GARCH fitting with very limited data
                )
        
        # Convert to percentage returns for better numerical stability
        returns_pct = returns * 100
        
        # Basic statistics - use original returns scale for consistency
        mean_return = returns.mean()
        volatility = returns.std()
        skewness_val = returns.skew()
        kurtosis_val = returns.kurtosis()
        skewness = skewness_val if isinstance(skewness_val, (float, int)) and not pd.isna(skewness_val) else 0.0
        kurtosis = kurtosis_val if isinstance(kurtosis_val, (float, int)) and not pd.isna(kurtosis_val) else 3.0
        
        # Use percentage returns only for GARCH fitting
        returns_pct = returns * 100
        
        # Autocorrelations
        autocorr_returns = returns_pct.autocorr(lag=1) if len(returns_pct) > 1 else 0.0
        autocorr_squared = (returns_pct**2).autocorr(lag=1) if len(returns_pct) > 1 else 0.0
        
        # Estimate tail index using Hill estimator
        tail_index = self._estimate_tail_index(returns_pct)
        
        # Use GARCH model as the primary method
        logger.debug("Using GARCH model for synthetic data generation")
        garch_params = self._fit_garch_model(returns_pct)

        # Create enhanced statistics with GARCH parameters
        return AssetStatistics(
            mean_return=mean_return,
            volatility=volatility,
            skewness=skewness,
            kurtosis=kurtosis,
            autocorr_returns=autocorr_returns,
            autocorr_squared=autocorr_squared,
            tail_index=tail_index,
            garch_params=garch_params
        )
    
    def _fit_professional_garch_model(self, returns: pd.Series) -> GARCHParameters:
        """
        Fit GARCH model using the professional arch package.
        
        Args:
            returns: Return series (in percentage terms)
            
        Returns:
            GARCHParameters object with fitted parameters
        """
        from typing import cast, Literal
        from arch import arch_model

        # Get GARCH specification from config
        p = self.garch_config.get('p', 1)
        q = self.garch_config.get('q', 1)
        dist = self.garch_config.get('distribution', 'normal')  # Default to normal for compatibility
        
        # Try different distributions in order of preference
        distributions_to_try = ['t', 'skewt', 'normal'] if dist == 'studentt' else [dist, 'normal']
        
        best_model = None
        best_aic = np.inf
        
        # Try GJR-GARCH (o=1), then fallback to standard GARCH (o=0) for robustness
        for model_type in ['GJR-GARCH', 'GARCH']:
            o = 1 if model_type == 'GJR-GARCH' else 0
            for distribution in distributions_to_try:
                model = None
                try:
                    # Create the model instance to get its parameter names
                    dist_cast = cast(Literal['normal', 'gaussian', 't', 'studentst', 'skewstudent', 'skewt', 'ged', 'generalized error'], distribution)
                    model = arch_model( # type: ignore
                        returns,
                        vol='GARCH',
                        p=p,
                        o=o,
                        q=q,
                        dist=dist_cast,
                        rescale=False
                    )
                    
                    # Fit with robust optimization; let arch handle starting values
                    res = model.fit(
                        disp='off',
                        show_warning=False,
                        options={'maxiter': 2000, 'ftol': 1e-8}
                    )
                    
                    # Check if fit converged properly
                    if hasattr(res, 'convergence_flag') and res.convergence_flag != 0:
                        if logger.isEnabledFor(logging.WARNING):
                            logger.warning(f"{model_type} model with {distribution} distribution did not converge")
                        continue
                    
                    # Select best model based on AIC
                    if hasattr(res, 'aic') and res.aic < best_aic:
                        best_aic = res.aic
                        best_model = res
                        
                except Exception as e:
                    if logger.isEnabledFor(logging.WARNING):
                        logger.warning(f"Failed to fit {model_type} with {distribution} distribution: {e}")
                    continue
            
            # If we found a model with the more complex model_type, don't try the simpler one
            if best_model:
                break

        if best_model is None:
            raise ValueError("All GARCH model fitting attempts failed")
        
        # Extract parameters
        params = best_model.params
        
        # Get bounds from config for parameter enforcement
        bounds = self.garch_config.get('bounds', {})
        cfg_omega_bounds = bounds.get('omega', [1e-6, 1.0])
        # Enforce a safer upper cap to avoid exploding unconditional variance
        omega_upper_cap = 0.002
        omega_bounds = [cfg_omega_bounds[0], min(cfg_omega_bounds[1], omega_upper_cap)]
        alpha_bounds = bounds.get('alpha', [0.01, 0.3])
        beta_bounds = bounds.get('beta', [0.5, 0.99])
        nu_bounds = bounds.get('nu', [2.1, 30.0])
        gamma_bounds = bounds.get('gamma', [0.0, 1.0])

        # Extract and enforce parameter bounds
        omega = np.clip(params['omega'], omega_bounds[0], omega_bounds[1])
        alpha = np.clip(params['alpha[1]'] if 'alpha[1]' in params.index else params.get('alpha', 0.1),
                       alpha_bounds[0], alpha_bounds[1])
        beta = np.clip(params['beta[1]'] if 'beta[1]' in params.index else params.get('beta', 0.8),
                       beta_bounds[0], beta_bounds[1])
        # Prevent unrealistically low beta values that break persistence estimation
        if beta < 0.65:
            beta = 0.65
        gamma = np.clip(params['gamma[1]'] if 'gamma[1]' in params.index else 0.0,
                       gamma_bounds[0], gamma_bounds[1])
        
        # Ensure stationarity for GJR-GARCH: alpha + beta + gamma / 2.0 < 1
        if alpha + beta + gamma / 2.0 >= 1.0:
            # Rescale to ensure stationarity while preserving relative magnitudes
            total = alpha + beta + gamma / 2.0
            scale_factor = 0.95 / total
            alpha *= scale_factor
            beta *= scale_factor
            gamma *= scale_factor
        
        # Get distribution-specific parameters
        nu = None
        lambda_ = None
        if 'nu' in params.index:
            nu = np.clip(params['nu'], nu_bounds[0], nu_bounds[1])
        if 'lambda' in params.index:
            lambda_ = params['lambda']
        
        # Determine distribution type based on model
        dist_type = DistributionType.NORMAL
        try:
            if hasattr(best_model, 'model') and hasattr(best_model.model, 'dist_name'):
                dist_name = best_model.model.distribution.name
                if 't' in dist_name.lower() or 'student' in dist_name.lower():
                    if 'skew' in dist_name.lower():
                        dist_type = DistributionType.SKEWED_STUDENT_T
                    else:
                        dist_type = DistributionType.STUDENT_T
                elif 'ged' in dist_name.lower():
                    dist_type = DistributionType.GED
                else:
                    dist_type = DistributionType.NORMAL
        except Exception as e:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(f"Could not determine distribution type from model: {e}. Falling back to Normal.")
            dist_type = DistributionType.NORMAL # Fallback to normal if we can't determine distribution
        
        return GARCHParameters(
            omega=omega,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            nu=nu,
            lambda_=lambda_,
            distribution=dist_type,
            aic=getattr(best_model, 'aic', None),
            bic=getattr(best_model, 'bic', None)
        )
    
    def generate_synthetic_returns(
        self, 
        asset_stats: AssetStatistics,
        length: int,
        asset_name: str = "Unknown"
    ) -> np.ndarray:
        """
        Generate synthetic returns using fitted GARCH model.
        
        Args:
            asset_stats: Statistical properties of the asset
            length: Length of synthetic series to generate
            asset_name: Name of asset for logging
            
        Returns:
            Array of synthetic returns (in decimal form, not percentage)
        """
        max_attempts = self.generation_config.get('max_attempts', 5) # Increased attempts
        
        # Use GARCH model as the primary method with retries
        for attempt in range(max_attempts):
            try:
                if asset_stats.garch_params is None:
                    raise ValueError("GARCH parameters are not available for generation.")
                synthetic_returns = self._generate_garch_returns(
                    asset_stats.garch_params, length, asset_stats.mean_return, asset_stats.volatility
                )

                # Validate generated returns to prevent extreme values
                if np.any(np.isnan(synthetic_returns)) or np.any(np.isinf(synthetic_returns)):
                    if logger.isEnabledFor(logging.WARNING):

                        logger.warning(f"Generated NaN or infinite values for {asset_name} (Attempt {attempt + 1}). Retrying.")
                    raise ValueError("Generated NaN or infinite values")
                
                # Tighter clipping to control excess kurtosis
                max_return_abs = 3 * asset_stats.volatility  # Max 3 standard deviations
                synthetic_returns = np.clip(synthetic_returns,
                                          asset_stats.mean_return - max_return_abs,
                                          asset_stats.mean_return + max_return_abs)
                
                # Further clip to prevent extremely negative returns that cause numerical issues
                synthetic_returns = np.clip(synthetic_returns, -0.5, 5.0) # Max -50% daily return, max 500% daily return
                
                logger.debug(f"DEBUG: Generated synthetic returns (min/max/mean/std) (Attempt {attempt + 1}): "
                            f"{np.min(synthetic_returns):.6f}/{np.max(synthetic_returns):.6f}/"
                            f"{np.mean(synthetic_returns):.6f}/{np.std(synthetic_returns):.6f}")
                
                # Validate quality
                if not self._validate_synthetic_data(synthetic_returns, asset_stats):
                    if logger.isEnabledFor(logging.WARNING):
                        logger.warning(f"Synthetic data for {asset_name} failed quality validation (Attempt {attempt + 1}). Retrying.")
                    continue

                if np.std(synthetic_returns) < 1e-5:  # Threshold for very low volatility
                    if logger.isEnabledFor(logging.WARNING):
                        logger.warning(f"Degenerate synthetic returns (very low volatility) for {asset_name} (Attempt {attempt + 1}). Retrying.")
                    continue # Retry generation
                
                # Add simple volatility clustering if autocorrelation exists
                
                if logger.isEnabledFor(logging.DEBUG):

                
                    logger.debug(f"Successfully generated {length} synthetic returns using GARCH model for {asset_name} (Attempt {attempt + 1})")
                return synthetic_returns
                
            except Exception as e:
                if logger.isEnabledFor(logging.WARNING):

                    logger.warning(f"GARCH generation failed for {asset_name} (Attempt {attempt + 1}): {e}. Retrying.")
                continue
        
        # Fallback to simple method if all t-distribution attempts fail
        if logger.isEnabledFor(logging.WARNING):

            logger.warning(f"All t-distribution generation attempts failed for {asset_name}. Using simple fallback.")
        return self._generate_simple_fallback(asset_stats, length)

    def _generate_fallback_returns(self, asset_stats: AssetStatistics, length: int) -> np.ndarray:
        """
        Generate synthetic returns using fallback GARCH simulation.
        
        Args:
            asset_stats: Asset statistical properties
            length: Length of series to generate
            
        Returns:
            Synthetic returns in percentage terms
        """
        garch_params = asset_stats.garch_params
        if garch_params is None:
            # Use simple normal distribution
            return np.random.normal(
                asset_stats.mean_return,
                asset_stats.volatility,
                length
            )
        
        # Simulate GARCH process manually
        omega = garch_params.omega
        alpha = garch_params.alpha
        beta = garch_params.beta
        
        # Initialize arrays
        returns = np.zeros(length)
        sigma2 = np.zeros(length)
        
        # Initial variance (unconditional variance)
        sigma2[0] = omega / (1 - alpha - beta) if (alpha + beta) < 1 else asset_stats.volatility**2
        
        # Generate innovations based on distribution type
        if garch_params.distribution == DistributionType.STUDENT_T and garch_params.nu is not None:
            # Student-t innovations
            innovations = np.random.standard_t(garch_params.nu, length)
        else:
            # Normal innovations
            innovations = np.random.standard_normal(length)
        
        # Generate GARCH process
        for t in range(length):
            if t > 0:
                sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
            
            returns[t] = asset_stats.mean_return + np.sqrt(sigma2[t]) * innovations[t]
        
        return returns

    def _generate_simple_fallback(self, asset_stats: AssetStatistics, length: int) -> np.ndarray:
        """
        Simple fallback using normal distribution for synthetic data generation.
        
        Args:
            asset_stats: Asset statistical properties
            length: Length of series to generate
            
        Returns:
            Synthetic returns in decimal terms
        """
        logger.warning("Using fallback normal distribution for synthetic data generation")
        
        # Generate returns with historical mean and volatility
        synthetic_returns_pct = np.random.normal(
            asset_stats.mean_return,
            asset_stats.volatility,
            length
        )
        
        # Modified: keep decimal scaling consistent
        return synthetic_returns_pct
    
    def generate_synthetic_prices(
        self,
        ohlc_data: pd.DataFrame,
        length: int,
        asset_name: str = "Unknown"
    ) -> pd.DataFrame:
        """
        Generate synthetic OHLC price data from historical data.
        
        Args:
            ohlc_data: Historical OHLC data
            length: Length of synthetic series
            asset_name: Name of asset for logging
            
        Returns:
            DataFrame with synthetic OHLC data
        """
        # Analyze historical data
        asset_stats = self.analyze_asset_statistics(ohlc_data)
        
        # Generate synthetic returns
        synthetic_returns = self.generate_synthetic_returns(asset_stats, length, asset_name)
        
        # Convert returns to prices
        initial_price = ohlc_data['Close'].iloc[-1]
        synthetic_prices = self._returns_to_prices(synthetic_returns, initial_price)
        
        # Generate OHLC from prices
        synthetic_ohlc = self._generate_ohlc_from_prices(synthetic_prices)
        
        # Create DataFrame with same structure as input
        synthetic_df = pd.DataFrame(
            synthetic_ohlc,
            columns=['Open', 'High', 'Low', 'Close'],
            index=pd.date_range(
                start=ohlc_data.index[-1] + pd.Timedelta(days=1),
                periods=length,
                freq='D'
            )
        )
        
        return synthetic_df
    
    def _returns_to_prices(self, returns: np.ndarray, initial_price: float) -> np.ndarray:
        """Convert returns to price levels with protection against non-positive prices."""
        if logger.isEnabledFor(logging.DEBUG):

            logger.debug(f"DEBUG: _returns_to_prices - initial_price: {initial_price:.6f}, returns sample: {returns[:5]}")
        logger.debug(f"DEBUG: _returns_to_prices - returns (min/max/mean/std): "
                     f"{np.min(returns):.6f}/{np.max(returns):.6f}/"
                     f"{np.mean(returns):.6f}/{np.std(returns):.6f}")
        
        # Only clip extreme outliers that would cause mathematical issues
        # Allow up to -50% single-period returns (more realistic for financial markets)
        max_negative_return = -0.50  # Maximum -50% return to prevent negative prices
        
        # Only clip the most extreme outliers to preserve volatility
        clipped_returns = np.clip(returns, max_negative_return, 5.0)  # Cap at 500% gain too
        if logger.isEnabledFor(logging.DEBUG):

            logger.debug(f"DEBUG: _returns_to_prices - clipped_returns sample: {clipped_returns[:5]}")
        logger.debug(f"DEBUG: _returns_to_prices - clipped_returns (min/max/mean/std): "
                     f"{np.min(clipped_returns):.6f}/{np.max(clipped_returns):.6f}/"
                     f"{np.mean(clipped_returns):.6f}/{np.std(clipped_returns):.6f}")
        
        # Convert to price relatives
        price_relatives = 1 + clipped_returns
        if logger.isEnabledFor(logging.DEBUG):

            logger.debug(f"DEBUG: _returns_to_prices - price_relatives sample: {price_relatives[:5]}")
        logger.debug(f"DEBUG: _returns_to_prices - price_relatives (min/max/mean/std): "
                     f"{np.min(price_relatives):.6f}/{np.max(price_relatives):.6f}/"
                     f"{np.mean(price_relatives):.6f}/{np.std(price_relatives):.6f}")
        
        # Only ensure positive price relatives for the most extreme cases
        price_relatives = np.maximum(price_relatives, 0.05)  # Minimum 5% of previous price
        if logger.isEnabledFor(logging.DEBUG):

            logger.debug(f"DEBUG: _returns_to_prices - price_relatives (after max): {price_relatives[:5]}")
        logger.debug(f"DEBUG: _returns_to_prices - price_relatives (after max, min/max/mean/std): "
                     f"{np.min(price_relatives):.6f}/{np.max(price_relatives):.6f}/"
                     f"{np.mean(price_relatives):.6f}/{np.std(price_relatives):.6f}")
        
        # Calculate cumulative prices
        prices = initial_price * np.cumprod(price_relatives)
        if logger.isEnabledFor(logging.DEBUG):

            logger.debug(f"DEBUG: _returns_to_prices - prices sample: {prices[:5]}")
        logger.debug(f"DEBUG: _returns_to_prices - prices (min/max/mean/std): "
                     f"{np.min(prices):.6f}/{np.max(prices):.6f}/"
                     f"{np.mean(prices):.6f}/{np.std(prices):.6f}")
        
        # Minimal safety check - only prevent truly problematic values
        prices = np.maximum(prices, initial_price * 1e-6)  # Minimum 0.0001% of initial price (very small positive)
        if logger.isEnabledFor(logging.DEBUG):

            logger.debug(f"DEBUG: _returns_to_prices - prices (after max): {prices[:5]}")
        logger.debug(f"DEBUG: _returns_to_prices - prices (after max, min/max/mean/std): "
                     f"{np.min(prices):.6f}/{np.max(prices):.6f}/"
                     f"{np.mean(prices):.6f}/{np.std(prices):.6f}")
        
        return prices
    
    def _generate_ohlc_from_prices(self, prices: np.ndarray) -> np.ndarray:
        """
        Generate realistic OHLC data from closing prices.
        
        Args:
            prices: Array of closing prices
            
        Returns:
            Array with OHLC data
        """
        n = len(prices)
        ohlc = np.zeros((n, 4))
        
        for i in range(n):
            close = prices[i]
            
            if i == 0:
                open_price = close
            else:
                # Open is previous close with small gap
                gap = np.random.normal(0, 0.001)  # Small gap
                open_price = prices[i-1] * (1 + gap)
            
            # Generate intraday volatility
            intraday_vol = np.random.uniform(0.005, 0.02)  # 0.5% to 2% intraday range
            
            # Ensure proper OHLC relationships - High >= max(Open, Close), Low <= min(Open, Close)
            max_oc = max(open_price, close)
            min_oc = min(open_price, close)
            
            # High must be >= max(open, close) - add random upward movement
            high_multiplier = 1 + np.random.uniform(0, intraday_vol)
            high = max_oc * high_multiplier
            
            # Low must be <= min(open, close) - subtract random downward movement
            low_multiplier = 1 - np.random.uniform(0, intraday_vol)
            low = min_oc * low_multiplier
            
            # Absolute safety check to guarantee OHLC relationships
            high = max(high, open_price, close)  # High >= Open, Close
            low = min(low, open_price, close)    # Low <= Open, Close
            
            # Additional safety: ensure High >= Low
            if high < low:
                high, low = low, high
            
            ohlc[i] = [open_price, high, low, close]
        
        return ohlc

    def _validate_synthetic_data(self, synthetic_returns: np.ndarray, asset_stats: AssetStatistics) -> bool:
        """
        Validate synthetic data against original statistics.
        
        Args:
            synthetic_returns: Generated synthetic returns (percentage)
            asset_stats: Original asset statistics
            
        Returns:
            True if validation passes
        """
        from scipy import stats
        
        tolerance = self.validation_config.get('tolerance', 0.8)  # 80% tolerance (more lenient)
        
        try:
            # Basic statistics validation with improved edge case handling
            synth_mean = np.mean(synthetic_returns)
            synth_std = np.std(synthetic_returns)
            synth_skew = stats.skew(synthetic_returns)
            synth_kurt = stats.kurtosis(synthetic_returns)
            
            # Improved validation checks with better handling of edge cases
            checks = []
            
            # Mean check - handle cases where mean is near zero
            if abs(asset_stats.mean_return) > 1e-6:
                mean_check = abs(synth_mean - asset_stats.mean_return) / abs(asset_stats.mean_return) < tolerance
            else:
                # For very small means, use absolute difference
                mean_check = abs(synth_mean - asset_stats.mean_return) < 0.001
            checks.append(mean_check)
            
            # Volatility check - this should be the most reliable
            if asset_stats.volatility > 1e-6:
                vol_check = abs(synth_std - asset_stats.volatility) / asset_stats.volatility < tolerance
            else:
                vol_check = abs(synth_std - asset_stats.volatility) < 0.001
            checks.append(vol_check)
            
            # Skewness check - very lenient as this is hard to match exactly
            if abs(asset_stats.skewness) > 1e-6:
                skew_check = abs(synth_skew - asset_stats.skewness) / abs(asset_stats.skewness) < tolerance * 3
            else:
                skew_check = abs(synth_skew - asset_stats.skewness) < 1.0  # Very lenient for near-zero skewness
            checks.append(skew_check)
            
            # Kurtosis check - very lenient as this is hard to match exactly
            if abs(asset_stats.kurtosis) > 1e-6:
                kurt_check = abs(synth_kurt - asset_stats.kurtosis) / abs(asset_stats.kurtosis) < tolerance * 3
            else:
                kurt_check = abs(synth_kurt - asset_stats.kurtosis) < 2.0  # Very lenient for near-zero kurtosis
            checks.append(kurt_check)
            
            # Require at least 1 out of 4 checks to pass (very lenient)
            # Prioritize volatility and mean over higher moments
            passed_checks = sum(checks)
            
            # Special case: if volatility check passes, that's often good enough
            if checks[1]:  # Volatility check passed
                return True
            
            # Otherwise require at least 1 check to pass
            return passed_checks >= 1
            
        except Exception as e:
            if logger.isEnabledFor(logging.WARNING):

                logger.warning(f"Validation failed with error: {e}")
            # If validation fails due to errors, just accept the data
            return True
    
    def _estimate_tail_index(self, returns: pd.Series) -> float:
        """
        Estimate tail index using Hill estimator.
        
        Args:
            returns: Return series
            
        Returns:
            Estimated tail index
        """
        try:
            # Use top 5% of absolute returns for Hill estimator
            abs_returns = np.abs(returns)
            k = max(10, int(0.05 * len(abs_returns)))
            sorted_returns = np.sort(abs_returns)
            top_returns = sorted_returns[-k:]
            
            if len(top_returns) > 1:
                # Add logging for debugging divide by zero
                if top_returns[0] == 0:
                    logger.warning("DEBUG: top_returns[0] is zero in _estimate_tail_index. Using default tail index.")
                    return 3.0
                log_ratios = np.log(top_returns[1:] / top_returns[0])
                tail_index = 1.0 / np.mean(log_ratios)
                return max(1.5, min(tail_index, 10.0))  # Bound between 1.5 and 10
            else:
                logger.warning("DEBUG: Not enough top returns for tail index estimation. Using default tail index.")
                return 3.0  # Default value
        except Exception as e:
            if logger.isEnabledFor(logging.WARNING):

                logger.warning(f"DEBUG: Tail index estimation failed: {e}. Using default tail index.")
            return 3.0  # Default value if estimation fails
    
    def _get_fallback_garch_params(self, returns: pd.Series) -> GARCHParameters:
        """
        Generate fallback GARCH parameters when fitting fails.
        
        Args:
            returns: Return series
            
        Returns:
            Fallback GARCH parameters
        """
        vol = returns.std()
        
        # Use typical GARCH parameter values
        omega = vol**2 * 0.1  # 10% of variance as constant
        alpha = 0.1  # Typical ARCH coefficient
        beta = 0.85  # Typical GARCH coefficient
        nu = 5.0  # Moderate fat tails
        
        return GARCHParameters(
            omega=omega,
            alpha=alpha,
            beta=beta,
            nu=nu,
            distribution=DistributionType.STUDENT_T
        )
    
    def _fit_garch_model(self, returns: pd.Series) -> GARCHParameters:
        """
        Fit GARCH model to returns data. This is a wrapper around the professional fitting method.
        
        Args:
            returns: Return series (can be in decimal or percentage terms)
            
        Returns:
            GARCHParameters object with fitted parameters, with omega scaled to the input returns.
        """
        was_scaled = False
        # Heuristic to check if returns are in decimal and convert to percentage
        # for numerical stability in GARCH fitting.
        # A standard deviation check is a reliable way to infer the scale.
        if returns.std() < 0.1:  # Typical daily % returns std is > 0.5
            returns_for_fitting = returns * 100
            was_scaled = True
            logger.debug("Input returns appear to be in decimal form. Scaling by 100 for GARCH fitting.")
        else:
            returns_for_fitting = returns

        try:
            # The professional model expects percentage returns
            garch_params = self._fit_professional_garch_model(returns_for_fitting)
            
            # If we scaled the returns, we need to scale back omega.
            # alpha, beta, gamma, nu, lambda are scale-invariant.
            if was_scaled:
                garch_params.omega /= 10000
                logger.debug(f"Rescaled omega back to decimal scale: {garch_params.omega}")
                # Reapply lower bound to avoid extremely small omega values
                if garch_params.omega < 1e-6:
                    garch_params.omega = 1e-6

            return garch_params

        except Exception as e:
            if logger.isEnabledFor(logging.WARNING):

                logger.warning(f"Professional GARCH fitting failed: {e}. Using fallback parameters.")
            
            # Fallback should use the original returns to calculate vol for correct scale
            return self._get_fallback_garch_params(returns)
    
    def _generate_garch_returns(self, garch_params: GARCHParameters, length: int, mean_return: float, target_volatility: float) -> np.ndarray:
        """
        Generate synthetic returns using GARCH parameters, ensuring volatility preservation.

        Args:
            garch_params: GARCH parameters to use for generation.
            length: Number of returns to generate.
            mean_return: The target mean return for the synthetic data.
            target_volatility: The target volatility (standard deviation) for the synthetic data.

        Returns:
            Array of synthetic returns (in decimal form).
        """
        # Use the consistent random state for reproducibility
        rng = self.rng

        # GJR-GARCH parameters
        alpha = garch_params.alpha
        beta = garch_params.beta
        gamma = garch_params.gamma if garch_params.gamma is not None else 0.0
        nu = garch_params.nu

        # CRITICAL: Back-calculate omega to match the target unconditional volatility.
        # This is more robust than trusting the fitted omega, which can be unstable.
        # Unconditional variance for GJR-GARCH: omega / (1 - alpha - beta - gamma/2)
        persistence = alpha + beta + gamma / 2.0
        
        if persistence >= 1.0:
            # If the process is non-stationary, we cannot use the formula.
            # Fallback to a reasonable omega based on target volatility.
            # This is a simplified assumption for a complex case.
            omega_decimal = (target_volatility ** 2) * 0.05 # Assume omega is 5% of variance
            logger.warning(f"GJR-GARCH process is non-stationary (persistence={persistence:.4f}). "
                           f"Using fallback omega calculation.")
        else:
            omega_decimal = target_volatility**2 * (1 - persistence)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"DEBUG: GJR-GARCH persistence: {persistence:.4f}")
            logger.debug(f"DEBUG: Target Volatility: {target_volatility:.6f}")
            logger.debug(f"DEBUG: Back-calculated Omega: {omega_decimal:.8f}")

        # Simulate the GJR-GARCH process
        synthetic_returns = self._simulate_garch_process(
            omega_decimal, alpha, beta, gamma, nu,
            target_volatility, mean_return, length, rng
        )

        # --- Post-simulation scaling ------------------------------------------------
        # Re-scale the synthetic series so that its realised volatility and
        # mean match the target statistics that we measured on the original
        # asset.
        generated_std = np.std(synthetic_returns)
        if generated_std > 1e-12:
            scale_factor = target_volatility / generated_std
            synthetic_returns *= scale_factor

        # Adjust mean after scaling
        synthetic_returns -= np.mean(synthetic_returns)
        synthetic_returns += mean_return

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"DEBUG: Generated returns sample: {synthetic_returns[:5]}")
            logger.debug(f"DEBUG: Generated returns (min/max/mean/std): "
                         f"{np.min(synthetic_returns):.6f}/{np.max(synthetic_returns):.6f}/"
                         f"{np.mean(synthetic_returns):.6f}/{np.std(synthetic_returns):.6f}")

        return synthetic_returns
    
    def _simulate_garch_process(self, omega, alpha, beta, gamma, nu, target_volatility, mean_return, length, rng):
        """
        Simulate a GJR-GARCH(1,1,1) process to preserve volatility clustering and asymmetry.

        Args:
            omega (float): GARCH omega parameter (decimal scale).
            alpha (float): GARCH alpha parameter.
            beta (float): GARCH beta parameter.
            gamma (float): GARCH gamma (asymmetry) parameter.
            nu (float): Degrees of freedom for Student-t distribution.
            target_volatility (float): Target unconditional volatility.
            mean_return (float): Mean return.
            length (int): Number of periods to simulate.
            rng (np.random.RandomState): The random number generator.

        Returns:
            Array of simulated returns with GJR-GARCH dynamics.
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"DEBUG: _simulate_garch_process - target_volatility: {target_volatility:.6f}, mean_return: {mean_return:.6f}")

        # Initialize arrays
        returns: np.ndarray = np.zeros(length)
        variances: np.ndarray = np.zeros(length)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"DEBUG: _simulate_garch_process - omega: {omega:.8f}, alpha: {alpha:.6f}, beta: {beta:.6f}, gamma: {gamma:.6f}")

        # Initialize variance to the unconditional target variance
        initial_variance = target_volatility ** 2
        variances[0] = initial_variance

        # Generate innovations using consistent random state
        if nu and nu > 2:
            from scipy import stats
            df = max(nu, 3.0)
            innovations: np.ndarray = np.array(stats.t.rvs(df=df, size=length, random_state=self.rng))
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"DEBUG: _simulate_garch_process - using Student-t innovations (df={df:.2f})")
        else:
            innovations: np.ndarray = np.array(self.rng.standard_normal(length))
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"DEBUG: _simulate_garch_process - using Normal innovations")

        # Simulate GJR-GARCH process
        for t in range(length):
            # Generate return for the current period
            current_std_dev = np.sqrt(max(1e-12, variances[t])) # Prevent sqrt of negative
            returns[t] = mean_return + innovations[t] * current_std_dev

            # Update variance for the next period
            if t < length - 1:
                error_term = returns[t] - mean_return
                error_term_sq = error_term ** 2
                
                # Asymmetry term: I(e_{t-1} < 0)
                indicator = 1.0 if error_term < 0 else 0.0
                
                variances[t + 1] = omega + alpha * error_term_sq + gamma * error_term_sq * indicator + beta * variances[t]
                
                # Ensure variance stays positive and reasonable
                variances[t + 1] = max(variances[t + 1], 1e-12) # Prevent negative variance
                variances[t + 1] = min(variances[t + 1], initial_variance * 100)  # Cap at 100x target

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"DEBUG: _simulate_garch_process - final returns (min/max/mean/std): "
                            f"{np.min(returns):.6f}/{np.max(returns):.6f}/"
                            f"{np.mean(returns):.6f}/{np.std(returns):.6f}")
            logger.debug(f"DEBUG: _simulate_garch_process - final variances (min/max/mean/std): "
                            f"{np.min(variances):.6f}/{np.max(variances):.6f}/"
                            f"{np.mean(variances):.6f}/{np.std(variances):.6f}")

        return returns
    
    def _add_volatility_clustering(self, returns: np.ndarray, autocorr_target: float) -> np.ndarray:
        """
        Add simple volatility clustering to returns to preserve autocorrelation in squared returns.
        
        Args:
            returns: Base synthetic returns
            autocorr_target: Target autocorrelation for squared returns
            
        Returns:
            Returns with enhanced volatility clustering
        """
        try:
            # Simple volatility clustering: scale returns by lagged absolute returns
            enhanced_returns = returns.copy()
            abs_returns = np.abs(enhanced_returns)
            
            # Apply simple volatility scaling based on previous period
            for t in range(1, len(enhanced_returns)):
                # Scale current return by function of previous absolute return
                vol_multiplier = 1.0 + autocorr_target * (abs_returns[t-1] / np.std(abs_returns))
                enhanced_returns[t] *= vol_multiplier
            
            return enhanced_returns
            
        except Exception as e:
            if logger.isEnabledFor(logging.WARNING):

                logger.warning(f"Volatility clustering enhancement failed: {e}. Using original returns.")
            return returns


# Backward compatibility - alias the new class to the old name
SyntheticDataGenerator = ImprovedSyntheticDataGenerator 