from datetime import datetime

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
        "strategy_params": {
            "lookback_months": 11,
            "top_decile_fraction": 0.10,
            "smoothing_lambda": 0.5,
            "leverage": 0.50,
            "long_only": True,
            "sma_filter_window": None, # No SMA filter for this scenario
        }
    },
    {
        "name": "Momentum_SMA_Filtered",
        "strategy": "momentum",
        "rebalance_frequency": "ME",
        "position_sizer": "equal_weight",
        "transaction_costs_bps": 10,
        "strategy_params": {
            "lookback_months": 11,
            "top_decile_fraction": 0.10,
            "smoothing_lambda": 0.5,
            "leverage": 0.50,
            "long_only": True,
            "sma_filter_window": 10, # 10-month SMA filter
        }
    },
    {
        "name": "Momentum_Optimized",
        "strategy": "momentum",
        "rebalance_frequency": "ME",
        "position_sizer": "equal_weight",
        "transaction_costs_bps": 10,
        "strategy_params": {
            "lookback_months": 11,
            "top_decile_fraction": 0.10,
            "smoothing_lambda": 0.5,
            "leverage": 0.50,
            "long_only": True,
            "sma_filter_window": None,
        }
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
                "parameter": "rolling_window",
                "metric": "Sortino",
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
        }
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
        "optimize": [
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
        "optimize": [
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
        }
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
        }
    }
]

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
# These settings are shared across all backtest scenarios.
# =============================================================================
GLOBAL_CONFIG = {
    "data_source": "yfinance",
    "universe": [
        "AAPL","AMZN","MSFT","GOOGL","NVDA","TSLA","V","MCD","AVGO","AMD",
        "WMT","COST","JPM","MA","MU","LLY","TGT","META","ORCL","PG","HD",
        "JNJ","BAC","ABBV","XOM","CVX","WFC","CRM","GE","BA","ABT","MS",
        "GS","DIS","LIN","MRK","RTX","BKNG","CAT","PEP","KO","INTU","TMUS",
        "ACN","SCHW","TMO","SYK","AMGN","HON","AMAT","DHR","NEE","PLTR","UNH", "GBTC"
    ],
    "benchmark": "^GSPC",
    "start_date": "2010-01-01",
    "end_date": "2025-06-30",
    "ibkr_commission_per_share": 0.005,
    "ibkr_commission_min_per_order": 1.0,
    "ibkr_commission_max_percent_of_trade": 0.005, # 0.5% of trade value
    "slippage_bps": 1, # 1 basis point (0.01%) of trade value
}
