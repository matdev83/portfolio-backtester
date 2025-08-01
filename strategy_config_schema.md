# Strategy Configuration Schema

This document outlines the schema for the strategy configuration YAML files, based on an analysis of all existing configuration files.

## Top-Level Parameters

| Parameter | Type | Observed Values/Ranges | Description |
|---|---|---|---|
| `name` | string | various | The name of the scenario. |
| `strategy` | string | various | The name of the strategy to use. |
| `rebalance_frequency` | string | `B`, `D`, `W`, `M`, `ME`, `Q`, `QE`, `A`, `YE`, `6ME` | The frequency at which to rebalance the portfolio. |
| `position_sizer` | string | `equal_weight`, `direct`, `rolling_beta`, `rolling_downside_volatility`, `rolling_sharpe`, `rolling_sortino`, `rolling_benchmark_corr` | The position sizing algorithm to use. |
| `train_window_months` | integer | 12 - 48 | The number of months to use for the training window. |
| `test_window_months` | integer | 12 - 48 | The number of months to use for the testing window. |
| `optimization_metric` | string | `Sharpe`, `Sortino`, `Calmar`, `Total Return`, `Sharpe Ratio` | The metric to use for optimization. |
| `optimization_target` | object | | The target for optimization. |
| `optimization_target.name` | string | `Sharpe Ratio`, `Sortino`, `Max Drawdown` | The name of the optimization target. |
| `optimization_target.direction` | string | `maximize`, `minimize` | The direction of the optimization. |
| `optimization_constraints` | array of objects | | Constraints for the optimization. |
| `optimization_constraints[].metric` | string | `Ann. Vol` | The metric to constrain. |
| `optimization_constraints[].max_value` | float | 0.2, 0.25 | The maximum value for the constrained metric. |
| `optimizers` | object | | Configuration for the optimizers. |
| `strategy_params` | object | | Parameters for the strategy. |
| `universe_config` | object | | Configuration for the universe. |
| `universe` | array of strings | | A list of tickers to use as the universe. |
| `mc_simulations` | integer | 50 - 1000 | The number of Monte Carlo simulations to run. |
| `mc_years` | integer | 3 - 20 | The number of years to simulate in the Monte Carlo simulation. |
| `start_date` | string | `2018-01-01` | The start date for the backtest. |
| `end_date` | string | `2023-12-31` | The end date for the backtest. |
| `description` | string | various | A description of the scenario. |
| `strategy_class` | string | `LowVolatilityFactorStrategy` | The class name of the strategy. |
| `walk_forward_type` | string | `expanding` | The type of walk-forward optimization to use. |
| `enable_monte_carlo_during_optimization` | boolean | `false` | Whether to enable Monte Carlo simulation during optimization. |

## Optimizer Parameters

| Parameter | Type | Observed Values/Ranges | Description |
|---|---|---|---|
| `optimizers.genetic` | object | | Configuration for the genetic optimizer. |
| `optimizers.genetic.ga_num_generations` | integer | 2 - 10 | The number of generations for the genetic optimizer. |
| `optimizers.genetic.ga_sol_per_pop` | integer | 3 - 10 | The number of solutions per population for the genetic optimizer. |
| `optimizers.genetic.ga_num_parents_mating` | integer | 2 - 5 | The number of parents to use for mating in the genetic optimizer. |
| `optimizers.genetic.advanced_crossover_type` | string | `simulated_binary`, `multi_point`, `uniform_variant`, `arithmetic` | The type of crossover to use in the genetic optimizer. |
| `optimizers.genetic.sbx_distribution_index` | float | 15.0 | The distribution index for the simulated binary crossover. |
| `optimizers.genetic.num_crossover_points` | integer | 4 | The number of crossover points for the multi-point crossover. |
| `optimizers.genetic.uniform_crossover_bias` | float | 0.3 | The bias for the uniform crossover. |
| `optimizers.genetic.optimize` | array of objects | | The parameters to optimize with the genetic optimizer. |
| `optimizers.optuna` | object | | Configuration for the Optuna optimizer. |
| `optimizers.optuna.n_trials` | integer | 10 - 20 | The number of trials for the Optuna optimizer. |
| `optimizers.optuna.optimize` | array of objects | | The parameters to optimize with the Optuna optimizer. |

## Optimization Parameters

| Parameter | Type | Observed Values/Ranges | Description |
|---|---|---|---|
| `parameter` | string | various | The name of the parameter to optimize. |
| `type` | string | `int`, `float`, `categorical`, `multi-categorical` | The type of the parameter to optimize. |
| `min_value` | integer, float | various | The minimum value for the parameter. |
| `max_value` | integer, float | various | The maximum value for the parameter. |
| `step` | integer, float | various | The step size for the parameter. |
| `values` | array | various | The possible values for a categorical parameter. |
| `choices` | array | `[true, false]` | The possible choices for a categorical parameter. |
| `log` | boolean | `false` | Whether to use a logarithmic scale for the parameter. |

## Strategy Parameters

| Parameter | Type | Observed Values/Ranges | Description |
|---|---|---|---|
| `long_only` | boolean | `true`, `false` | Whether to only take long positions. |
| `lookback_months` | integer | 3 - 24 | The number of months to look back for momentum calculation. |
| `num_holdings` | integer | 5 - 35 | The number of holdings in the portfolio. |
| `top_decile_fraction` | float | 0.05 - 0.3 | The fraction of the top decile to use. |
| `smoothing_lambda` | float | 0.0 - 1.0 | The smoothing lambda for the momentum calculation. |
| `leverage` | float | 0.1 - 2.0 | The leverage to use. |
| `sma_filter_window` | integer | 2 - 50 | The window for the SMA filter. |
| `derisk_days_under_sma` | integer | 0 - 30 | The number of days to derisk under the SMA. |
| `sizer_dvol_window` | integer | 2 - 12 | The window for the downside volatility sizer. |
| `target_volatility` | float | 0.05 - 0.3 | The target volatility for the downside volatility sizer. |
| `max_leverage` | float | 1.5 - 3.0 | The maximum leverage for the downside volatility sizer. |
| `sizer_beta_window` | integer | 2 - 12 | The window for the beta sizer. |
| `beta_lookback_days` | integer | 21 | The number of days to look back for beta calculation. |
| `num_high_beta_to_exclude` | integer | 3 | The number of high beta stocks to exclude. |
| `rsi_length` | integer | 2 | The length of the RSI. |
| `rsi_overbought` | integer | 70 | The overbought threshold for the RSI. |
| `short_max_holding_days` | integer | 30 | The maximum number of days to hold a short position. |
| `atr_length` | integer | 10 - 20 | The length of the ATR. |
| `atr_multiple` | float | 2.0 - 3.0 | The multiple of the ATR to use for the stop loss. |
| `rolling_window` | integer | 1 - 12 | The rolling window for the Sharpe and Sortino momentum strategies. |
| `alpha` | float | 0.0 - 1.0 | The alpha for the VAMS momentum strategy. |
| `spy_weight` | float | 0.1 - 0.9 | The weight for SPY in the fixed weight strategy. |
| `gld_weight` | float | 0.1 - 0.9 | The weight for GLD in the fixed weight strategy. |
| `vol_window` | integer | 63 - 252 | The window for the volatility in the volatility targeted fixed weight strategy. |
| `dummy_param_1` | integer | 1 - 10 | A dummy parameter for testing. |
| `dummy_param_2` | integer | 1 - 2 | A dummy parameter for testing. |
| `fast_ema_days` | integer | 7 - 32 | The number of days for the fast EMA. |
| `slow_ema_days` | integer | 64 - 128 | The number of days for the slow EMA. |
| `entry_day` | integer | -14 - 14 | The entry day for the intramonth seasonal strategy. |
| `hold_days` | integer | 1 - 14 | The number of days to hold for the intramonth seasonal strategy. |
| `trade_month_1` - `trade_month_12` | boolean | `true`, `false` | Whether to trade in the corresponding month for the intramonth seasonal strategy. |
| `direction` | string | `long`, `short` | The direction to trade for the intramonth seasonal strategy. |
| `use_ema_filter` | boolean | `true`, `false` | Whether to use an EMA filter for the intramonth seasonal strategy. |
| `fast_ema_len` | integer | 0, 14 | The length of the fast EMA for the intramonth seasonal strategy. |
| `slow_ema_multiplier` | float | 2.0, 4.0 | The multiplier for the slow EMA for the intramonth seasonal strategy. |
| `rsi_threshold` | float | 20.0 - 40.0 | The RSI threshold for the UVXY RSI strategy. |
| `rsi_period` | integer | 2 - 5 | The RSI period for the UVXY RSI strategy. |
| `price_column_asset` | string | `Close` | The price column to use for the asset. |
| `price_column_benchmark` | string | `Close` | The price column to use for the benchmark. |
| `stop_loss_type` | string | `AtrBasedStopLoss` | The type of stop loss to use. |
| `initial_capital` | integer | 1000000, 2000000 | The initial capital for the meta strategy. |
| `min_allocation` | float | 0.05 | The minimum allocation for the meta strategy. |
| `allocations` | array of objects | | The allocations for the meta strategy. |
| `volatility_lookback_days` | integer | 126, 252 | The number of days to look back for volatility calculation in the low volatility factor strategy. |
| `size_percentile` | integer | 50 | The percentile to use for size in the low volatility factor strategy. |
| `vol_percentile_low` | integer | 20 - 30 | The low percentile for volatility in the low volatility factor strategy. |
| `vol_percentile_high` | integer | 70 - 80 | The high percentile for volatility in the low volatility factor strategy. |
| `beta_lookback_months` | integer | 24, 36 | The number of months to look back for beta calculation in the low volatility factor strategy. |
| `beta_min_cap` | float | 0.25 | The minimum cap for beta in the low volatility factor strategy. |
| `beta_max_cap` | float | 2.0 | The maximum cap for beta in the low volatility factor strategy. |
| `beta_max_low_vol` | float | 1.0 | The maximum beta for low volatility stocks in the low volatility factor strategy. |
| `use_hedged_legs` | boolean | `true` | Whether to use hedged legs in the low volatility factor strategy. |
| `price_column` | string | `Close` | The price column to use in the low volatility factor strategy. |
| `apply_trading_lag` | boolean | `false` | Whether to apply a trading lag. |
| `target_return` | float | 0.0, 0.005, 0.02 | The target return for the Sortino momentum strategy. |
| `volatility_targeting` | object | | Configuration for volatility targeting. |
| `volatility_targeting.name` | string | `annualized` | The name of the volatility targeting. |
| `volatility_targeting.target_annual_vol` | float | 0.1 | The target annual volatility. |
| `volatility_targeting.lookback_days` | integer | 60 | The number of days to look back for volatility targeting. |
| `volatility_targeting.max_leverage` | float | 1.5 | The maximum leverage for volatility targeting. |
| `volatility_targeting.min_leverage` | float | 0.5 | The minimum leverage for volatility targeting. |
