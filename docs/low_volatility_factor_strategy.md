# Low-Volatility Factor Strategy

## Overview

The Low-Volatility Factor Strategy is implemented based on the research paper "Factoring in the Low-Volatility Factor" (SSRN-5295002) by Amar Soebhag, Guido Baltussen, and Pim van Vliet (June 2025). This strategy exploits the well-documented low-volatility anomaly where low-volatility stocks have historically delivered higher risk-adjusted returns than high-volatility stocks.

## Key Research Findings

The paper demonstrates that:

1. **Factor Asymmetry**: The long leg (low-volatility stocks) drives most of the factor's performance, while the short leg (high-volatility stocks) contributes less.

2. **Transaction Costs Matter**: When accounting for real-world trading costs and shorting fees, the low-volatility factor remains robust, especially on the long side.

3. **Market Hedging**: Using market-hedged portfolios (beta-neutral construction) improves the factor's performance and reduces unwanted market exposure.

4. **Significant Improvements**: The low-volatility factor improves existing asset pricing models by 11.9% to 17.0% when accounting for factor asymmetry and investment frictions.

## Strategy Implementation

### Core Methodology

The strategy follows the paper's approach:

1. **2×3 Sorting Procedure**: 
   - Stocks are sorted into Small/Big based on market cap (NYSE median)
   - Independently sorted into Low/Medium/High volatility (30th/70th percentiles)
   - Creates 6 portfolios: SLV, SMV, SHV, BLV, BMV, BHV

2. **Volatility Calculation**:
   - Uses 252-day rolling standard deviation of daily returns
   - Annualized volatility: σ = std(daily_returns) × √252

3. **Beta-Neutral Construction**:
   - Calculates 36-month rolling betas for each portfolio
   - Applies beta caps (0.25 to 2.0) to avoid excessive hedging
   - Creates market-hedged long and short legs

4. **Factor Construction**:
   - VOL_t = r_L,t - r_H,t - (β_L,t-1 - β_H,t-1) × MKT_t
   - Where L = Low volatility, H = High volatility, MKT = Market return

### Implementation Variants

The strategy supports multiple specifications from the paper:

#### 1. Long-minus-Short (Classic)
- Traditional long-short factor construction
- Equal weight to long and short legs
- Market-hedged to be beta-neutral

#### 2. Long-plus-Short (Factor Asymmetry)
- Separate weights for long and short legs
- Accounts for different risk characteristics
- Typically assigns higher weight to long leg

#### 3. Net Long-plus-Short (With Costs)
- Includes transaction costs and shorting fees
- Reduces attractiveness of short leg
- More realistic for practical implementation

#### 4. Net Long-Market (Long-Only)
- Focuses only on low-volatility stocks
- Eliminates shorting constraints and costs
- Suitable for long-only mandates

## Configuration Parameters

### Core Parameters

- **volatility_lookback_days** (default: 252): Days for volatility calculation
- **size_percentile** (default: 50): NYSE percentile for size breakpoint
- **vol_percentile_low** (default: 30): Low volatility percentile threshold
- **vol_percentile_high** (default: 70): High volatility percentile threshold

### Beta and Risk Management

- **beta_lookback_months** (default: 36): Months for beta estimation
- **beta_min_cap** (default: 0.25): Minimum beta cap
- **beta_max_cap** (default: 2.0): Maximum beta cap
- **beta_max_low_vol** (default: 1.0): Maximum beta for low-vol portfolios

### Strategy Variants

- **long_only** (default: False): Use long-only implementation
- **use_hedged_legs** (default: True): Use market-hedged portfolios
- **account_for_costs** (default: False): Include transaction costs

### Portfolio Management

- **leverage** (default: 1.0): Portfolio leverage
- **smoothing_lambda** (default: 0.5): Weight smoothing parameter
- **shorting_fee_annual** (default: 0.01): Annual shorting fee (1%)

## Usage Examples

### Basic Long-Short Strategy

```yaml
LowVolatilityFactor_Classic:
  strategy_class: "LowVolatilityFactorStrategy"
  strategy_params:
    volatility_lookback_days: 252
    long_only: false
    account_for_costs: false
    leverage: 1.0
```

### Long-Only Strategy

```yaml
LowVolatilityFactor_LongOnly:
  strategy_class: "LowVolatilityFactorStrategy"
  strategy_params:
    volatility_lookback_days: 252
    long_only: true
    account_for_costs: false
    leverage: 1.0
```

### With Transaction Costs

```yaml
LowVolatilityFactor_WithCosts:
  strategy_class: "LowVolatilityFactorStrategy"
  strategy_params:
    volatility_lookback_days: 252
    long_only: false
    account_for_costs: true
    shorting_fee_annual: 0.01
    leverage: 1.0
```

## Performance Characteristics

Based on the paper's findings:

### Historical Performance (1972-2023)
- **Long-Short Return**: 5.01% per annum (gross)
- **Long-Only Return**: 3.18% per annum (gross)
- **Net Return (with costs)**: 3.22% per annum (long-short)
- **Long Leg Net Return**: 2.70% per annum

### Risk Characteristics
- **Low correlation** with traditional factors when using hedged legs
- **Robust performance** across different market conditions
- **Reduced during high volatility periods** (as expected)

### Factor Model Improvements
- **11.9% improvement** in Sharpe ratio (Long-plus-Short)
- **13.0% improvement** with transaction costs
- **17.0% improvement** in long-only version

## Implementation Notes

### Data Requirements
- **Minimum periods**: 38 months for reliable calculations
- **Daily price data**: Required for volatility calculations
- **Market cap data**: For size-based sorting (simplified as price proxy)

### Computational Considerations
- **Beta calculations**: Use 36-month rolling windows
- **Portfolio rebalancing**: Monthly frequency recommended
- **Memory usage**: Stores historical breakpoints and betas

### Limitations
- **Simplified market cap**: Uses price as proxy (should use shares outstanding)
- **Transaction cost model**: Simplified approach (could use bid-ask spreads)
- **Shorting availability**: Assumes all stocks can be shorted

## Research Background

### Low-Volatility Anomaly
The low-volatility anomaly is one of the most robust findings in finance:
- Documented since Black, Jensen, and Scholes (1972)
- Persists across markets, time periods, and risk measures
- Contradicts traditional risk-return relationship

### Economic Explanations
- **Leverage constraints**: Institutional investors can't use leverage
- **Behavioral biases**: Preference for lottery-like stocks
- **Agency issues**: Fund managers chase high-volatility stocks
- **Arbitrage limits**: Shorting constraints limit correction

### Factor Asymmetry Discovery
The paper's key contribution is showing that:
- Traditional factor models miss the low-volatility factor
- Factor asymmetry explains this paradox
- Long leg contains most of the economic value
- Short leg is correlated with other factors

## References

1. Soebhag, A., Baltussen, G., & van Vliet, P. (2025). "Factoring in the Low-Volatility Factor." SSRN-5295002.

2. Black, F., Jensen, M. C., & Scholes, M. (1972). "The Capital Asset Pricing Model: Some Empirical Tests."

3. Frazzini, A., & Pedersen, L. H. (2014). "Betting Against Beta." Journal of Financial Economics.

4. Blitz, D. (2007). "The Volatility Effect: Lower Risk without Lower Return." Journal of Portfolio Management.

## Implementation Status

✅ **Completed Features:**
- 2×3 sorting procedure
- Market-hedged portfolio construction
- Beta-neutral factor creation
- Long-only and long-short variants
- Transaction cost handling
- Comprehensive parameter tuning
- Multiple configuration scenarios

🔄 **Future Enhancements:**
- Real market cap data integration
- Advanced transaction cost models
- Shorting availability constraints
- Multi-factor model integration
- Performance attribution analysis 