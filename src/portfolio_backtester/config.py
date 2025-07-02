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
        "optimize": [
            {"parameter": "num_holdings", "metric": "Sortino", "min_value": 10, "max_value": 35, "step": 1},
            {"parameter": "top_decile_fraction", "metric": "Sortino", "min_value": 0.05, "max_value": 0.3, "step": 0.01},
            {"parameter": "lookback_months", "metric": "Sortino", "min_value": 3, "max_value": 14, "step": 1},
            {"parameter": "smoothing_lambda", "metric": "Sortino", "min_value": 0.0, "max_value": 1.0, "step": 0.05},
            {"parameter": "leverage", "metric": "Sortino", "min_value": 0.1, "max_value": 2.0, "step": 0.1},
            {"parameter": "sma_filter_window", "metric": "Sortino", "min_value": 2, "max_value": 24, "step": 1},
            {"parameter": "derisk_days_under_sma", "metric": "Sortino", "min_value": 0, "max_value": 30, "step": 1}
        ],
        "strategy_params": {
            # All optimized parameters are omitted here
            "long_only": True
        },
        "mc_simulations": 1000,
        "mc_years": 10
    },
    {
        "name": "Momentum_Beta_Sized",
        "strategy": "momentum",
        "rebalance_frequency": "ME",
        "position_sizer": "rolling_beta",
        "transaction_costs_bps": 10,
        "train_window_months": 60,
        "test_window_months": 24,
        "optimize": [
            {"parameter": "num_holdings", "metric": "Sortino", "min_value": 10, "max_value": 35, "step": 1},
            {"parameter": "top_decile_fraction", "metric": "Sortino", "min_value": 0.05, "max_value": 0.3, "step": 0.01},
            {"parameter": "lookback_months", "metric": "Sortino", "min_value": 3, "max_value": 14, "step": 1},
            {"parameter": "smoothing_lambda", "metric": "Sortino", "min_value": 0.0, "max_value": 1.0, "step": 0.05},
            {"parameter": "leverage", "metric": "Sortino", "min_value": 0.1, "max_value": 2.0, "step": 0.1},
            {"parameter": "sma_filter_window", "metric": "Sortino", "min_value": 2, "max_value": 24, "step": 1},
            {"parameter": "derisk_days_under_sma", "metric": "Sortino", "min_value": 0, "max_value": 30, "step": 1},
            {"parameter": "sizer_beta_window", "metric": "Sortino", "min_value": 2, "max_value": 12, "step": 1}
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
        "strategy": "momentum",
        "rebalance_frequency": "ME",
        "position_sizer": "rolling_downside_volatility",
        "transaction_costs_bps": 10,
        "train_window_months": 60,
        "test_window_months": 24,
        "optimize": [
            {"parameter": "num_holdings", "metric": "Total Return", "min_value": 10, "max_value": 35, "step": 1},
            {"parameter": "top_decile_fraction", "metric": "Total Return", "min_value": 0.05, "max_value": 0.3, "step": 0.01},
            {"parameter": "lookback_months", "metric": "Total Return", "min_value": 3, "max_value": 14, "step": 1},
            {"parameter": "smoothing_lambda", "metric": "Total Return", "min_value": 0.0, "max_value": 1.0, "step": 0.05},
            {"parameter": "leverage", "metric": "Total Return", "min_value": 0.1, "max_value": 2.0, "step": 0.1},
            {"parameter": "sma_filter_window", "metric": "Total Return", "min_value": 2, "max_value": 24, "step": 1},
            {"parameter": "derisk_days_under_sma", "metric": "Total Return", "min_value": 0, "max_value": 30, "step": 1},
            {"parameter": "sizer_dvol_window", "metric": "Total Return", "min_value": 2, "max_value": 12, "step": 1}
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
        "optimize": [
            {
                "parameter": "num_holdings",
                "metric": "Total Return",
                "min_value": 10,
                "max_value": 30,
                "step": 1
            },
            {
                "parameter": "derisk_days_under_sma",
                "metric": "Total Return",
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
        "optimize": [
            {
                "parameter": "num_holdings",
                "metric": "Sharpe",
                "min_value": 10,
                "max_value": 30,
                "step": 1
            },
            {
                "parameter": "rolling_window",
                "metric": "Sharpe",
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
        "strategy_params": {
            "long_only": True,
            "sma_filter_window": None,
        },
        "mc_simulations": 1000,
        "mc_years": 10,
        "optimize": [
            {
                "parameter": "num_holdings",
                "metric": "Sortino",
                "min_value": 10,
                "max_value": 30,
                "step": 1
            },
            {
                "parameter": "lookback_months",
                "metric": "Sortino",
                "min_value": 3,
                "max_value": 24,
                "step": 1,
            },
            {
                "parameter": "top_decile_fraction",
                "metric": "Sortino",
                "min_value": 0.05,
                "max_value": 0.20,
                "step": 0.05,
            },
            {
                "parameter": "smoothing_lambda",
                "metric": "Sortino",
                "min_value": 0.1,
                "max_value": 0.9,
                "step": 0.1,
            },
            {
                "parameter": "leverage",
                "metric": "Sortino",
                "min_value": 0.1,
                "max_value": 2.0,
                "step": 0.1,
            },
            {
                "parameter": "alpha",
                "metric": "Sortino",
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
        "strategy_params": {
            "long_only": True,
            "sma_filter_window": None,
        },
        "mc_simulations": 1000,
        "mc_years": 10,
        "optimize": [
            {
                "parameter": "num_holdings",
                "metric": "Sortino",
                "min_value": 10,
                "max_value": 30,
                "step": 1
            },
            {
                "parameter": "lookback_months",
                "metric": "Sortino",
                "min_value": 6,
                "max_value": 24,
                "step": 1,
            },
            {
                "parameter": "top_decile_fraction",
                "metric": "Sortino",
                "min_value": 0.05,
                "max_value": 0.20,
                "step": 0.05,
            },
            {
                "parameter": "smoothing_lambda",
                "metric": "Sortino",
                "min_value": 0.1,
                "max_value": 0.9,
                "step": 0.1,
            },
            {
                "parameter": "leverage",
                "metric": "Sortino",
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
        "optimize": [
            {
                "parameter": "num_holdings",
                "metric": "Sortino",
                "min_value": 10,
                "max_value": 30,
                "step": 1
            },
            {
                "parameter": "rolling_window",
                "metric": "Sortino",
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
        "optimize": [
            {
                "parameter": "num_holdings",
                "metric": "Calmar",
                "min_value": 10,
                "max_value": 30,
                "step": 1
            },
            {
                "parameter": "rolling_window",
                "metric": "Calmar",
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
        "optimize": [
            {
                "parameter": "num_holdings",
                "metric": "Sharpe",
                "min_value": 10,
                "max_value": 30,
                "step": 1
            },
            {
                "parameter": "rolling_window",
                "metric": "Sharpe",
                "min_value": 1,
                "max_value": 6,
                "step": 1
            },
            {
                "parameter": "sizer_sharpe_window",
                "metric": "Sharpe",
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
  "long_only": {"type": "int", "low": 0, "high": 1, "step": 1}
}


def populate_default_optimizations():
    """Ensure each scenario has an optimize section covering all tunable
    parameters of its strategy and dynamic position sizer."""
    from .utils import _resolve_strategy

    sizer_param_map = {
        "rolling_sharpe": "sizer_sharpe_window",
        "rolling_sortino": "sizer_sortino_window",
        "rolling_beta": "sizer_beta_window",
        "rolling_benchmark_corr": "sizer_corr_window",
        "rolling_downside_volatility": "sizer_dvol_window",
    }

    for scen in BACKTEST_SCENARIOS:
        if "optimize" not in scen:
            scen["optimize"] = []
        existing = {opt["parameter"] for opt in scen["optimize"]}

        strat_cls = _resolve_strategy(scen["strategy"])
        if strat_cls is not None:
            for param in strat_cls.tunable_parameters():
                if param not in existing:
                    scen["optimize"].append({"parameter": param})

        position_sizer_name = scen.get("position_sizer")
        if position_sizer_name:
            sizer_param = sizer_param_map.get(position_sizer_name)
            if sizer_param and sizer_param not in existing:
                scen["optimize"].append({"parameter": sizer_param})


# Populate optimization lists on import
populate_default_optimizations()
