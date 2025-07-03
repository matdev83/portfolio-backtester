# =============================================================================
# MAIN CONFIGURATION
# =============================================================================
# This list defines all the backtest scenarios to be run and compared.
# Each dictionary represents one complete backtest run.
# =============================================================================
BACKTEST_SCENARIOS = [
    {
        "name": "Momentum_Unfiltered",
        "strategy": "momentum",
        "rebalance_frequency": "ME",
        "position_sizer": "equal_weight",
        "transaction_costs_bps": 10,
        "train_window_months": 60,
        "test_window_months": 24,
        # Example of new multi-objective optimization
        "optimization_targets": [
            {"name": "Sortino", "direction": "maximize"},
            {"name": "Max Drawdown", "direction": "minimize"}
        ],
        # "optimization_metric": "Sortino", # Replaced by optimization_targets
        "optimize": [
            {"parameter": "num_holdings", "min_value": 10, "max_value": 35, "step": 1},
            {"parameter": "top_decile_fraction", "min_value": 0.05, "max_value": 0.3, "step": 0.01},
            {"parameter": "lookback_months", "min_value": 3, "max_value": 14, "step": 1},
            {"parameter": "smoothing_lambda", "min_value": 0.0, "max_value": 1.0, "step": 0.05},
            {"parameter": "leverage", "min_value": 0.1, "max_value": 2.0, "step": 0.1},
            {"parameter": "sma_filter_window", "min_value": 2, "max_value": 24, "step": 1},
            {"parameter": "derisk_days_under_sma", "min_value": 0, "max_value": 30, "step": 1}
        ],
        "strategy_params": {
            # All optimized parameters are omitted here
            "long_only": True
            # Example: No volatility targeting (default behavior)
            # "volatility_targeting": {"name": "none"}
        },
        "mc_simulations": 1000,
        "mc_years": 10
    },
    {
        "name": "Momentum_Vol_Targeted_10pct",
        "strategy": "momentum",
        "rebalance_frequency": "ME",
        "position_sizer": "equal_weight",
        "transaction_costs_bps": 10,
        "train_window_months": 60,
        "test_window_months": 24,
        "strategy_params": {
            "lookback_months": 6,
            "num_holdings": 20,
            "smoothing_lambda": 0.5,
            "long_only": True,
            "volatility_targeting": {
                "name": "annualized",
                "target_annual_vol": 0.10, # Target 10% annualized volatility
                "lookback_days": 60,       # Use 60 trading days for vol calculation
                "max_leverage": 1.5,       # Max leverage capped at 1.5x
                "min_leverage": 0.5        # Min leverage at 0.5x
            }
        },
        "mc_simulations": 1000,
        "mc_years": 10
        # "optimize" section could be added here too if desired
    },
    {
        "name": "Momentum_Beta_Sized",
        "strategy": "momentum",
        "rebalance_frequency": "ME",
        "position_sizer": "rolling_beta",
        "transaction_costs_bps": 10,
        "train_window_months": 60,
        "test_window_months": 24,
        "optimization_metric": "Sortino", # Added scenario-level metric
        "optimize": [
            {"parameter": "num_holdings", "min_value": 10, "max_value": 35, "step": 1},
            {"parameter": "top_decile_fraction", "min_value": 0.05, "max_value": 0.3, "step": 0.01},
            {"parameter": "lookback_months", "min_value": 3, "max_value": 14, "step": 1},
            {"parameter": "smoothing_lambda", "min_value": 0.0, "max_value": 1.0, "step": 0.05},
            {"parameter": "leverage", "min_value": 0.1, "max_value": 2.0, "step": 0.1},
            {"parameter": "sma_filter_window", "min_value": 2, "max_value": 24, "step": 1},
            {"parameter": "derisk_days_under_sma", "min_value": 0, "max_value": 30, "step": 1},
            {"parameter": "sizer_beta_window", "min_value": 2, "max_value": 12, "step": 1}
        ],
        "strategy_params": {
            "long_only": True,
            "sizer_beta_window": 3
        },
        "mc_simulations": 1000,
        "mc_years": 10
    },
    {
        "name": "Momentum_DVOL_Sizer",
        "strategy": "momentum_dvol_sizer",
        "rebalance_frequency": "ME",
        "position_sizer": "rolling_downside_volatility",
        "transaction_costs_bps": 10,
        "train_window_months": 60,
        "test_window_months": 24,
        "optimization_metric": "Total Return", # Added scenario-level metric
        "optimize": [
            {"parameter": "num_holdings", "min_value": 10, "max_value": 35, "step": 1},
            {"parameter": "top_decile_fraction", "min_value": 0.05, "max_value": 0.3, "step": 0.01},
            {"parameter": "lookback_months", "min_value": 3, "max_value": 14, "step": 1},
            {"parameter": "smoothing_lambda", "min_value": 0.0, "max_value": 1.0, "step": 0.05},
            {"parameter": "leverage", "min_value": 0.1, "max_value": 2.0, "step": 0.1},
            {"parameter": "sma_filter_window", "min_value": 2, "max_value": 24, "step": 1},
            {"parameter": "derisk_days_under_sma", "min_value": 0, "max_value": 30, "step": 1},
            {"parameter": "sizer_dvol_window", "min_value": 2, "max_value": 12, "step": 1},
            {"parameter": "target_volatility", "min_value": 0.05, "max_value": 0.3, "step": 0.01}
        ],
        "strategy_params": {
            "long_only": True
        },
        "mc_simulations": 1000,
        "mc_years": 10
    },
    {
        "name": "Momentum_SMA_Filtered",
        "strategy": "momentum",
        "rebalance_frequency": "ME",
        "position_sizer": "equal_weight",
        "transaction_costs_bps": 10,
        "train_window_months": 24,
        "test_window_months": 12,
        "optimization_metric": "Total Return", # Added scenario-level metric
        "optimize": [
            {
                "parameter": "num_holdings",
                "min_value": 10,
                "max_value": 30,
                "step": 1
            },
            {
                "parameter": "derisk_days_under_sma",
                "min_value": 1,
                "max_value": 30,
                "step": 1
            }
        ],
        "strategy_params": {
            "lookback_months": 11,
            "top_decile_fraction": 0.10,
            "smoothing_lambda": 0.5,
            "leverage": 0.50,
            "long_only": True,
            "sma_filter_window": 10, # 10-month SMA filter
            "derisk_days_under_sma": 10,
        },
        "mc_simulations": 1000,
        "mc_years": 10
    },
    {
        "name": "Sharpe_Momentum",
        "strategy": "sharpe_momentum",
        "rebalance_frequency": "ME",
        "position_sizer": "equal_weight",
        "transaction_costs_bps": 10,
        "train_window_months": 24,
        "test_window_months": 12,
        "optimization_metric": "Sharpe", # Added scenario-level metric
        "optimize": [
            {
                "parameter": "num_holdings",
                "min_value": 10,
                "max_value": 30,
                "step": 1
            },
            {
                "parameter": "rolling_window",
                "min_value": 1,
                "max_value": 6,
                "step": 1
            }
        ],
        "strategy_params": {
            "lookback_months": 11, # This will be used for mom_look/pred, but based on Sharpe
            "top_decile_fraction": 0.10,
            "smoothing_lambda": 0.5,
            "leverage": 0.50,
            "long_only": True,
            "sma_filter_window": None,
        },
        "mc_simulations": 1000,
        "mc_years": 10
    },
    {
        "name": "VAMS_Downside_Penalized",
        "strategy": "vams_momentum",
        "rebalance_frequency": "ME",
        "position_sizer": "equal_weight",
        "transaction_costs_bps": 10,
        "train_window_months": 24, # Use a 24-month training window
        "test_window_months": 12, # Test on the next 12 months
        "optimization_metric": "Sortino", # Added scenario-level metric
        "strategy_params": {
            "long_only": True,
            "sma_filter_window": None,
        },
        "mc_simulations": 1000,
        "mc_years": 10,
        "optimize": [
            {
                "parameter": "num_holdings",
                "min_value": 10,
                "max_value": 30,
                "step": 1
            },
            {
                "parameter": "lookback_months",
                "min_value": 3,
                "max_value": 24,
                "step": 1,
            },
            {
                "parameter": "top_decile_fraction",
                "min_value": 0.05,
                "max_value": 0.20,
                "step": 0.05,
            },
            {
                "parameter": "smoothing_lambda",
                "min_value": 0.1,
                "max_value": 0.9,
                "step": 0.1,
            },
            {
                "parameter": "leverage",
                "min_value": 0.1,
                "max_value": 2.0,
                "step": 0.1,
            },
            {
                "parameter": "alpha",
                "min_value": 0.0,
                "max_value": 1.0,
                "step": 0.1,
            }
        ]
    },
    {
        "name": "VAMS_No_Downside",
        "strategy": "vams_no_downside",
        "rebalance_frequency": "ME",
        "position_sizer": "equal_weight",
        "transaction_costs_bps": 10,
        "train_window_months": 24, # Use a 24-month training window
        "test_window_months": 12, # Test on the next 12 months
        "optimization_metric": "Sortino", # Added scenario-level metric
        "strategy_params": {
            "long_only": True,
            "sma_filter_window": None,
        },
        "mc_simulations": 1000,
        "mc_years": 10,
        "optimize": [
            {
                "parameter": "num_holdings",
                "min_value": 10,
                "max_value": 30,
                "step": 1
            },
            {
                "parameter": "lookback_months",
                "min_value": 6,
                "max_value": 24,
                "step": 1,
            },
            {
                "parameter": "top_decile_fraction",
                "min_value": 0.05,
                "max_value": 0.20,
                "step": 0.05,
            },
            {
                "parameter": "smoothing_lambda",
                "min_value": 0.1,
                "max_value": 0.9,
                "step": 0.1,
            },
            {
                "parameter": "leverage",
                "min_value": 0.1,
                "max_value": 1.0,
                "step": 0.1,
            }
        ]
    },
    {
        "name": "Sortino_Momentum",
        "strategy": "sortino_momentum",
        "rebalance_frequency": "ME",
        "position_sizer": "equal_weight",
        "transaction_costs_bps": 10,
        "train_window_months": 24,
        "test_window_months": 12,
        "optimization_metric": "Sortino", # Added scenario-level metric
        "optimize": [
            {
                "parameter": "num_holdings",
                "min_value": 10,
                "max_value": 30,
                "step": 1
            },
            {
                "parameter": "rolling_window",
                "min_value": 1,
                "max_value": 6,
                "step": 1
            }
        ],
        "strategy_params": {
            "lookback_months": 11, # This will be used for mom_look/pred, but based on Sortino
            "top_decile_fraction": 0.10,
            "smoothing_lambda": 0.5,
            "leverage": 0.50,
            "long_only": True,
            "sma_filter_window": None,
            "target_return": 0.0,  # Target return for Sortino calculation
        },
        "mc_simulations": 1000,
        "mc_years": 10
    },
    {
        "name": "Calmar_Momentum",
        "strategy": "calmar_momentum",
        "rebalance_frequency": "ME",
        "position_sizer": "equal_weight",
        "transaction_costs_bps": 10,
        "train_window_months": 24,
        "test_window_months": 12,
        "optimization_metric": "Calmar", # Added scenario-level metric
        "optimize": [
            {
                "parameter": "num_holdings",
                "min_value": 10,
                "max_value": 30,
                "step": 1
            },
            {
                "parameter": "rolling_window",
                "min_value": 1,
                "max_value": 6,
                "step": 1
            }
        ],
        "strategy_params": {
            "lookback_months": 11, # This will be used for mom_look/pred, but based on Calmar
            "top_decile_fraction": 0.10,
            "smoothing_lambda": 0.5,
            "leverage": 0.50,
            "long_only": True,
            "sma_filter_window": None,
        },
        "mc_simulations": 500,
        "mc_years": 20
    },
    {
        "name": "Sharpe_Sized_Momentum",
        "strategy": "sharpe_momentum",
        "rebalance_frequency": "ME",
        "position_sizer": "rolling_sharpe",
        "transaction_costs_bps": 10,
        "train_window_months": 24,
        "test_window_months": 12,
        "optimization_metric": "Sharpe", # Added scenario-level metric
        "optimize": [
            {
                "parameter": "num_holdings",
                "min_value": 10,
                "max_value": 30,
                "step": 1
            },
            {
                "parameter": "rolling_window",
                "min_value": 1,
                "max_value": 6,
                "step": 1
            },
            {
                "parameter": "sizer_sharpe_window",
                "min_value": 2,
                "max_value": 12,
                "step": 1
            }
        ],
        "strategy_params": {
            "lookback_months": 11,
            "top_decile_fraction": 0.10,
            "smoothing_lambda": 0.5,
            "leverage": 0.50,
            "long_only": True,
            "sma_filter_window": None,
        },
        "mc_simulations": 1000,
        "mc_years": 10
    }
]

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
# These settings are shared across all backtest scenarios.
# =============================================================================
GLOBAL_CONFIG = {
    "data_source": "stooq",
    "universe": [
        "AAPL","AMZN","MSFT","GOOGL","NVDA","TSLA","V","MCD","AVGO","AMD",
        "WMT","COST","JPM","MA","MU","LLY","TGT","META","ORCL","PG","HD",
        "JNJ","BAC","ABBV","XOM","CVX","WFC","CRM","GE","BA","ABT","MS",
        "GS","DIS","LIN","MRK","RTX","BKNG","CAT","PEP","KO","INTU","TMUS",
        "ACN","SCHW","TMO","SYK","AMGN","HON","AMAT","DHR","NEE","PLTR","UNH", "GBTC"
    ],
    "benchmark": "SPY",
    "start_date": "2010-01-01",
    "end_date": "2025-06-30",
    "ibkr_commission_per_share": 0.005,
    "ibkr_commission_min_per_order": 1.0,
    "ibkr_commission_max_percent_of_trade": 0.005, # 0.5% of trade value
    "slippage_bps": 1, # 1 basis point (0.01%) of trade value
}

# =============================================================================
# OPTIMIZER PARAMETER DEFAULTS
# =============================================================================
# Defines the default search space for all optimizable parameters.
# These values can be overridden within the 'optimize' section of individual
# BACKTEST_SCENARIOS.
# =============================================================================
OPTIMIZER_PARAMETER_DEFAULTS = {
  "max_lookback": {
    "type": "int",
    "low": 20,
    "high": 252,
    "step": 10
  },
  "calmar_lookback": {
    "type": "int",
    "low": 20,
    "high": 252,
    "step": 10
  },
  "leverage": {
    "type": "float",
    "low": 0.5,
    "high": 2.0,
    "step": 0.1
  },
  "smoothing_lambda": {
    "type": "float",
    "low": 0.0,
    "high": 0.9
  },
  "num_holdings": {
    "type": "int",
    "low": 1,
    "high": 50,
    "step": 1
  },
  "rolling_window": {
    "type": "int",
    "low": 20,
    "high": 252,
    "step": 10
  },
  "sma_filter_window": {
    "type": "int",
    "low": 20,
    "high": 252,
    "step": 10
  },
  "target_return": {
    "type": "float",
    "low": 0.0,
    "high": 0.1,
    "step": 0.01
  },
  "lookback_months": {
    "type": "int",
    "low": 1,
    "high": 12,
    "step": 1
  },
  "alpha": {
    "type": "float",
    "low": 0.1,
    "high": 0.9,
    "step": 0.1
  },
  "top_decile_fraction": {
    "type": "float",
    "low": 0.1,
    "high": 0.5,
    "step": 0.1
  },
  "derisk_days_under_sma": {
    "type": "int",
    "low": 4,
    "high": 18,
    "step": 2
  },
  "sizer_sharpe_window": {"type": "int", "low": 2, "high": 12, "step": 1},
  "sizer_sortino_window": {"type": "int", "low": 2, "high": 12, "step": 1},
  "sizer_beta_window": {"type": "int", "low": 2, "high": 12, "step": 1},
  "sizer_corr_window": {"type": "int", "low": 2, "high": 12, "step": 1},
  "sizer_dvol_window": {"type": "int", "low": 2, "high": 12, "step": 1},
  "long_only": {"type": "int", "low": 0, "high": 1, "step": 1},
  "target_volatility": {"type": "float", "low": 0.05, "high": 0.3, "step": 0.01}
}


def _get_strategy_tunable_params(strategy_name: str) -> set[str]:
    """Resolves strategy and returns its tunable parameters."""
    from .utils import _resolve_strategy # Local import to avoid circular dependency issues at module load time
    strat_cls = _resolve_strategy(strategy_name)
    if strat_cls:
        return set(strat_cls.tunable_parameters())
    return set()

def _get_sizer_tunable_param(sizer_name: str | None, sizer_param_map: dict) -> str | None:
    """Returns the tunable parameter name for a given sizer, if applicable."""
    if sizer_name:
        return sizer_param_map.get(sizer_name)
    return None

def populate_default_optimizations():
    """Ensure each scenario has an optimize section covering all tunable
    parameters of its strategy and dynamic position sizer.
    Min/max/step values for these parameters are sourced from
    OPTIMIZER_PARAMETER_DEFAULTS at runtime by the optimizer.
    """
    sizer_param_map = {
        "rolling_sharpe": "sizer_sharpe_window",
        "rolling_sortino": "sizer_sortino_window",
        "rolling_beta": "sizer_beta_window",
        "rolling_benchmark_corr": "sizer_corr_window",
        "rolling_downside_volatility": "sizer_dvol_window",
    }

    for scenario_config in BACKTEST_SCENARIOS:
        # Ensure "optimize" list exists
        if "optimize" not in scenario_config:
            scenario_config["optimize"] = []

        # Note: 'optimization_metric' or 'optimization_targets' are expected to be
        # manually defined in the scenario if optimization is intended.
        # This function focuses on populating the 'parameter' names.

        optimized_parameters_in_scenario = {opt_spec["parameter"] for opt_spec in scenario_config["optimize"]}

        # Get tunable parameters from the strategy
        strategy_params_to_add = _get_strategy_tunable_params(scenario_config["strategy"])

        # Get tunable parameter from the sizer, if any
        sizer_param_to_add = _get_sizer_tunable_param(scenario_config.get("position_sizer"), sizer_param_map)

        # Combine all potential parameters to be added
        all_potential_params = strategy_params_to_add
        if sizer_param_to_add:
            all_potential_params.add(sizer_param_to_add)

        # Add missing parameters to the scenario's "optimize" list
        for param_name in all_potential_params:
            if param_name not in optimized_parameters_in_scenario:
                scenario_config["optimize"].append({"parameter": param_name})
                # Default min/max/step will be picked up from OPTIMIZER_PARAMETER_DEFAULTS later

# Populate optimization lists on import
populate_default_optimizations()
