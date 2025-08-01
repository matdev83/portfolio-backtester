import argparse
from src.portfolio_backtester.core import Backtester
from src.portfolio_backtester.config_loader import GLOBAL_CONFIG, BACKTEST_SCENARIOS

def main():
    parser = argparse.ArgumentParser(description="Run portfolio backtester.")
    parser.add_argument("--scenario-name", type=str, default="Test_Dummy_WFO", help="Name of the scenario to run/optimize from BACKTEST_SCENARIOS. Required for all modes.")
    parser.add_argument("--optuna-trials", type=int, default=100, help="Maximum trials per optimization.")
    parser.add_argument("--mode", type=str, default="optimize", help="Mode to run the backtester in.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel worker processes to use (-1 ⇒ all cores).")
    parser.add_argument("--timeout", type=int, default=None, help="Global timeout in seconds for the entire run.")
    args = parser.parse_args()

    scenarios = [
            {
                "name": "Test_Dummy_WFO",
                "strategy": "dummy_strategy_for_testing",
                "strategy_params": {
                    "symbol": "SPY",
                    "long_only": True,
                    "open_long_prob": 0.1,
                    "close_long_prob": 0.01,
                    "dummy_param_1": 10,
                    "dummy_param_2": 20,
                    "seed": 42
                },
                "rebalance_frequency": "D",
                "position_sizer": "equal_weight",
                "transaction_costs_bps": 0,
                "train_window_months": 12,
                "test_window_months": 3,
                "optimization_metric": "Sharpe",
                "optimize": [
                    {
                        "parameter": "open_long_prob",
                        "min_value": 0.01,
                        "max_value": 0.5,
                        "step": 0.05
                    },
                    {
                        "parameter": "close_long_prob",
                        "min_value": 0.01,
                        "max_value": 0.2,
                        "step": 0.02
                    }
                ]
            }
        ]

    backtester = Backtester(GLOBAL_CONFIG, scenarios, args)
    backtester.run()
    print(backtester.results["Test_Dummy_WFO_Optimized"])

if __name__ == "__main__":
    main()
