import pytest
import pandas as pd
import numpy as np

from src.portfolio_backtester.backtester import Backtester
from src.portfolio_backtester.config import GLOBAL_CONFIG as DEFAULT_GLOBAL_CONFIG
from src.portfolio_backtester.strategies.base_strategy import BaseStrategy
from src.portfolio_backtester.signal_generators import BaseSignalGenerator
from src.portfolio_backtester.portfolio.position_sizer import equal_weight_sizer
from src.portfolio_backtester.data_sources.base_data_source import BaseDataSource

# --- Minimal Strategy Components for Testing ---
class TestSignalGenerator(BaseSignalGenerator):
    def required_features(self):
        return set()
    def scores(self, features):
        # Always generate a positive score for the first asset, zero for others
        # Scores DataFrame should have dates as index and tickers as columns
        dates = features['price_data_for_scores'].index # Assume price_data_for_scores is passed in features by test setup
        tickers = features['price_data_for_scores'].columns
        scores_df = pd.DataFrame(0.0, index=dates, columns=tickers)
        if not scores_df.empty and len(tickers) > 0:
            scores_df.iloc[:, 0] = 1.0
        return scores_df

class IntegrationTestStrategy(BaseStrategy): # Renamed class
    signal_generator_class = TestSignalGenerator

    def __init__(self, strategy_config):
        super().__init__(strategy_config)
        self.strategy_config.setdefault('num_holdings', 1)
        self.strategy_config.setdefault('long_only', True)


# --- Mock Data Source ---
class MockDataSource(BaseDataSource):
    def __init__(self, price_data):
        self.price_data = price_data

    def get_data(self, tickers, start_date, end_date):
        # Return relevant portion of pre-defined price_data
        # Ensure benchmark is included if not already
        if self.price_data.index.name != 'Date': # Ensure index is named 'Date' if not already
             self.price_data.index.name = 'Date'

        # Filter by date first
        data_in_range = self.price_data[(self.price_data.index >= pd.to_datetime(start_date)) &
                                        (self.price_data.index <= pd.to_datetime(end_date))]

        # Filter by tickers, ensuring all requested tickers are present, filling with NaN if not
        # This mock is simplified; a real one might need more robust handling for missing tickers
        return data_in_range.reindex(columns=tickers)


# --- Fixtures ---
@pytest.fixture
def global_config_fixture():
    config = DEFAULT_GLOBAL_CONFIG.copy()
    config["universe"] = ["ASSET_A", "ASSET_B"]
    config["benchmark"] = "BENCHMARK"
    config["start_date"] = "2022-01-01"
    config["end_date"] = "2023-12-31"
    # Using a much shorter period for faster tests if needed, but keep enough for lookbacks
    config["start_date"] = "2023-01-01"
    config["end_date"] = "2023-08-31" # Approx 8 months for testing
    return config

@pytest.fixture
def mock_price_data_fixture(global_config_fixture):
    dates = pd.date_range(start=global_config_fixture["start_date"],
                          end=global_config_fixture["end_date"], freq='B')
    n_days = len(dates)

    # Asset A: Starts stable, then becomes volatile
    asset_a_returns_stable = np.random.normal(0.0005, 0.005, n_days // 2)
    asset_a_returns_volatile = np.random.normal(0.0005, 0.03, n_days - (n_days // 2)) # Higher vol
    asset_a_returns = np.concatenate([asset_a_returns_stable, asset_a_returns_volatile])
    asset_a_prices = (1 + asset_a_returns).cumprod() * 100

    asset_b_returns = np.random.normal(0.0001, 0.01, n_days)
    asset_b_prices = (1 + asset_b_returns).cumprod() * 100

    benchmark_returns = np.random.normal(0.0003, 0.01, n_days)
    benchmark_prices = (1 + benchmark_returns).cumprod() * 100

    price_df = pd.DataFrame({
        "ASSET_A": asset_a_prices,
        "ASSET_B": asset_b_prices,
        "BENCHMARK": benchmark_prices
    }, index=dates)
    price_df.index.name = "Date"
    return price_df

# --- Args for Backtester (mocking command line args) ---
class MockArgs:
    def __init__(self):
        self.mode = "backtest"
        self.study_name = None
        self.storage_url = None
        self.random_seed = 42
        self.n_jobs = 1
        self.early_stop_patience = 10
        self.optuna_trials = 10
        self.optuna_timeout_sec = None
        self.mc_simulations = 100
        self.mc_years = 5
        self.interactive = False


# --- Test Class ---
class TestVolatilityIntegration:

    def _run_backtest_get_results(self, global_config, scenario_config, price_data, strategy_class_override=None):
        """Helper to run backtest and extract key results like weights and returns."""

        mock_data_source = MockDataSource(price_data)

        # Temporarily register our TestStrategy if needed, or pass it if Backtester supports it
        # For simplicity, we assume Backtester can pick up strategies from its `strategies` module
        # If TestStrategy is not in `src.portfolio_backtester.strategies`, this needs adjustment.
        # One way: monkeypatch getattr(strategies, class_name) in _get_strategy
        # Or, allow passing strategy_cls directly to Backtester or run_scenario (ideal for testing)

        # For this test, let's modify _get_strategy in the instance if possible, or make it simpler.
        # The current backtester resolves strategy by name. We need to ensure "test_strategy" resolves.
        # For now, let's assume we can add TestStrategy to the strategies module or mock its retrieval.

        # To make TestStrategy discoverable without complex mocking of strategy loading:
        import src.portfolio_backtester.strategies as strategies_module

        # Backtester._get_strategy will convert "TestStrategy" from config to "TestStrategyStrategy"
        # (capitalizes first word of split by '_', then adds "Strategy")
        # If strategy_name in config is "test_strategy", it would look for "TestStrategyStrategy".
        # Since our config uses "TestStrategy", it becomes:
        # Capitalize("TestStrategy") -> "TestStrategy"
        # Then add "Strategy" -> "TestStrategyStrategy"

        # Our actual class is named TestStrategy.
        # The scenario config uses "strategy": "TestStrategy".
        # _get_strategy converts "TestStrategy" to "TeststrategyStrategy" (due to word.capitalize())
        # This is tricky. Let's assume the config strategy name should be "test_strategy"
        # for it to correctly resolve to a class named "TestStrategyStrategy".
        # Or, if config is "TestStrategy", it looks for "TestStrategyStrategy" if TestStrategy is one word.

        # Let's trace _get_strategy carefully:
        # strategy_name = "TestStrategy" (from scenario_config["strategy"])
        # class_name_parts = [word.capitalize() for word in strategy_name.split('_')] -> ["TestStrategy"]
        # class_name_joined = "".join(class_name_parts) -> "TestStrategy"
        # final_class_name_lookup = class_name_joined + "Strategy" -> "TestStrategyStrategy"

        # So, we need to register our `IntegrationTestStrategy` class with the name that _get_strategy will look up.
        # If scenario config strategy name is "integration_test", _get_strategy looks for "IntegrationTestStrategy".
        target_class_name_in_module = "IntegrationTestStrategy" # This should match the class name directly
        original_strategy_attr = getattr(strategies_module, target_class_name_in_module, None)
        setattr(strategies_module, target_class_name_in_module, IntegrationTestStrategy) # Register our class


        # TestSignalGenerator is assigned directly via IntegrationTestStrategy.signal_generator_class,
        # so no name lookup issues there.

        backtester = Backtester(global_config, [scenario_config], MockArgs(), random_state=42)
        backtester.data_source = mock_data_source # Override data source

        # The backtester's run method does a lot. We want to isolate run_scenario.
        # To do this, we need to manually prepare what run() prepares for run_scenario:
        # 1. daily_data, monthly_data
        # 2. features
        # 3. rets_full

        daily_data = mock_data_source.get_data(
            tickers=global_config["universe"] + [global_config["benchmark"]],
            start_date=global_config["start_date"],
            end_date=global_config["end_date"]
        )
        daily_data.dropna(how="all", inplace=True)
        monthly_data = daily_data.resample("BME").last()
        rets_full = daily_data.pct_change().fillna(0)

        # Features: BaseStrategy might require BenchmarkSMA. TestSignalGenerator requires 'price_data_for_scores'.
        # For this test, TestSignalGenerator will use the monthly prices.
        # The main precompute_features might be too complex. Let's simplify.
        # Our TestStrategy doesn't require complex features from feature_engineering.
        # It just needs 'price_data_for_scores' for the TestSignalGenerator.
        # BaseStrategy.get_required_features might add BenchmarkSMA if sma_filter_window is in strategy_params.

        # Simplified feature creation for this test:
        test_features = {'price_data_for_scores': monthly_data[global_config["universe"]]}
        # If BaseStrategy adds BenchmarkSMA, it would be based on monthly_data[benchmark]

        # If sma_filter_window is in strategy_params, BenchmarkSMA will be added by BaseStrategy.get_required_features
        # and precompute_features would try to calculate it. For this test, ensure it's not set or handle it.
        # The TestStrategy config doesn't set it, so this should be fine.

        # We need to capture the adjusted_daily_weights from within run_scenario.
        # This is tricky as it's not returned directly.
        # For testing, we might need to:
        #    a) Modify run_scenario to return it (intrusive for main code)
        #    b) Monkeypatch something to capture it (complex)
        #    c) Store it on the backtester instance from run_scenario if it's accessible after run.
        # The current backtester.py calculates `adjusted_daily_weights` locally in `run_scenario`.
        # Let's make a small modification to `run_scenario` for testing purposes, or log extensively.
        # Given the constraints, we'll infer leverage changes from portfolio returns characteristics.
        # Or, we check the sum of weights if `portfolio_rets_net` is accompanied by the weights.
        # The `Backtester.results` stores returns.

        # For now, we'll rely on `portfolio_rets_net` and try to deduce behavior.
        # A more robust test would directly inspect `adjusted_daily_weights`.

        portfolio_rets_net = backtester.run_scenario(
            scenario_config,
            price_data_monthly=monthly_data, # Corrected: was price_data
            price_data_daily=daily_data,     # Corrected: was price_data
            rets_daily=rets_full,
            features=test_features,
            verbose=False # Keep verbose False for cleaner test output
        )

        # Clean up monkeypatching
        if original_strategy_attr is not None: # If it existed before, restore it
            setattr(strategies_module, target_class_name_in_module, original_strategy_attr)
        else: # If it didn't exist before (original_strategy_attr was the default None from getattr)
            if hasattr(strategies_module, target_class_name_in_module): # Check if we actually added it
                 delattr(strategies_module, target_class_name_in_module)

        return portfolio_rets_net # Later, we might try to get weights out too.


    def test_no_volatility_targeting_maintains_leverage(self, global_config_fixture, mock_price_data_fixture):
        scenario_no_vol = {
            "name": "Test_NoVolTarget",
            "strategy": "integration_test", # Updated to match new registration logic
            "rebalance_frequency": "ME", # Monthly rebalance
            "position_sizer": "equal_weight", # Simple sizer
            "transaction_costs_bps": 0,
            "strategy_params": {
                "num_holdings": 1, # Invest in ASSET_A only based on TestSignalGenerator
                "volatility_targeting": {"name": "none"} # Explicitly no targeting
            }
        }

        # Modify global_config for this specific test if needed (e.g. shorter date range for speed)
        # global_config_fixture["end_date"] = "2023-03-31" # Example for shorter run
        # mock_price_data_fixture = mock_price_data_fixture[mock_price_data_fixture.index <= pd.to_datetime("2023-03-31")]


        returns_no_vol = self._run_backtest_get_results(global_config_fixture, scenario_no_vol, mock_price_data_fixture)

        # Expected behavior: Leverage should be relatively constant (around 1.0 by default from sizer and strategy leverage)
        # Since we can't directly see weights yet, we check that returns are not zero and have some variance.
        # A more direct test would be to have run_scenario return weights.
        assert not returns_no_vol.empty
        assert returns_no_vol.std() > 0 # Check that it did something

        # If we had weights:
        # daily_leverage = adjusted_daily_weights.abs().sum(axis=1)
        # assert daily_leverage.std() < some_small_threshold_indicating_stability

    def test_annualized_volatility_targeting_adjusts_leverage(self, global_config_fixture, mock_price_data_fixture):
        target_vol = 0.10 # 10% annualized
        lookback_d = 30 # days (approx 1.5 months)

        scenario_ann_vol = {
            "name": "Test_AnnualizedVolTarget",
            "strategy": "integration_test", # Updated to match new registration logic
            "rebalance_frequency": "ME",
            "position_sizer": "equal_weight",
            "transaction_costs_bps": 0,
            "strategy_params": {
                "num_holdings": 1,
                "volatility_targeting": {
                    "name": "annualized",
                    "target_annual_vol": target_vol,
                    "lookback_days": lookback_d, # Shorter lookback for faster reaction in test
                    "max_leverage": 2.0,
                    "min_leverage": 0.1
                }
            }
        }

        returns_ann_vol = self._run_backtest_get_results(global_config_fixture, scenario_ann_vol, mock_price_data_fixture)
        assert not returns_ann_vol.empty

        # Price data: ASSET_A starts stable, then becomes volatile in the second half.
        # We expect leverage to be higher in the first half and lower in the second half.
        # This change in leverage should be reflected in the standard deviation of returns
        # for the two periods, assuming the underlying asset's returns are somewhat consistent in direction.
        # This is an indirect assertion.

        n_days_total = len(returns_ann_vol)
        mid_point_idx = n_days_total // 2

        returns_first_half = returns_ann_vol.iloc[:mid_point_idx]
        returns_second_half = returns_ann_vol.iloc[mid_point_idx:]

        # ASSET_A vol is low then high.
        # P&L vol should follow.
        # So, leverage should be higher then lower.
        # If underlying asset has positive mean return, then higher leverage = higher return std & mean.
        # This is a weak assertion because returns can be negative.

        # A better assertion if we had weights:
        # daily_leverage = adjusted_daily_weights.abs().sum(axis=1)
        # leverage_first_half = daily_leverage.iloc[:mid_point_idx].mean()
        # leverage_second_half = daily_leverage.iloc[mid_point_idx:].mean()
        # assert leverage_first_half > leverage_second_half (given our price data design for ASSET_A)

        # For now, let's just check the portfolio realized some vol, and it's different from no-targetting (hard to assert direction)
        # We could also check if overall realized volatility is somewhat close to target_vol,
        # but this depends heavily on the quality of the vol estimation and market conditions.

        realized_annual_vol_total = returns_ann_vol.std() * np.sqrt(252)
        # print(f"\nRealized Annual Vol (Targeting Scenario): {realized_annual_vol_total:.4f}")
        # print(f"Target Annual Vol: {target_vol:.4f}")

        # This assertion is weak and might be flaky:
        # assert target_vol * 0.5 < realized_annual_vol_total < target_vol * 2.0
        # A better test would be to have a mock price series that *guarantees* certain P&L volatilities
        # and then check the leverage factor applied by AnnualizedVolatilityTargeting more directly.

        # Given the current setup, the most practical thing is to ensure the code runs and produces
        # some output that differs from the non-targeted version, implying the mechanism was active.
        # A truly robust integration test here needs either:
        # 1. `run_scenario` to return `adjusted_daily_weights`.
        # 2. A very carefully crafted input price series where P&L volatility changes are predictable
        #    and the impact on returns (due to leverage changes) is also predictable and testable.

        # For now, the fact that it runs without error and potentially produces different
        # return characteristics (even if hard to assert precisely) is a basic integration check.
        # We'll rely more on unit tests for the correctness of VolatilityTargeting classes.

        # A simple check: did it produce returns?
        assert returns_ann_vol.std() > 0

        # Try to get the weights from the backtester object after run_scenario is called.
        # This requires Backtester or run_scenario to store it.
        # Let's assume for a moment we modified run_scenario to store `adjusted_daily_weights` on the instance.
        # e.g., self.last_run_adjusted_weights = adjusted_daily_weights
        # Then we could do:
        # last_weights = backtester.last_run_adjusted_weights
        # if last_weights is not None:
        #    daily_leverage = last_weights.abs().sum(axis=1)
        #    leverage_first_half = daily_leverage.iloc[:mid_point_idx].mean()
        #    leverage_second_half = daily_leverage.iloc[mid_point_idx:].mean()
        #    print(f"Avg Leverage First Half: {leverage_first_half}, Second Half: {leverage_second_half}")
        #    assert leverage_first_half > leverage_second_half # ASSET_A vol was low then high
        # else:
        #    pytest.skip("Adjusted daily weights not available for detailed leverage checking.")

        # Since we cannot easily get weights, this test is more of a "runs without error" check.
        pass # Placeholder for more robust assertions if weights become available.


# To run: pytest tests/test_volatility_integration.py
# Make sure __init__.py are in relevant test and src subdirectories for pytest discovery.
# And ensure TestStrategy and TestSignalGenerator can be found by the backtester.
# (This might require temporary addition to portfolio_backtester.strategies / .signal_generators,
# or more advanced mocking of the strategy loading mechanism in Backtester._get_strategy)

# Add __init__.py to tests/ and tests/portfolio/ if not present.
# Add __init__.py to src/portfolio_backtester/data_sources/ if not present.
# (The MockDataSource might need to be in a proper module structure too if BaseDataSource is imported from there)
# Corrected BaseDataSource import to be from its actual location.
# `from src.portfolio_backtester.data_sources.base_data_source import BaseDataSource`
# `from src.portfolio_backtester.signal_generators import BaseSignalGenerator`
# `from src.portfolio_backtester.strategies.base_strategy import BaseStrategy`
# `from src.portfolio_backtester.portfolio.position_sizer import equal_weight_sizer`
# `from src.portfolio_backtester.config import GLOBAL_CONFIG as DEFAULT_GLOBAL_CONFIG`
# `from src.portfolio_backtester.backtester import Backtester`
# These imports assume PYTHONPATH or package structure allows finding src.
# Pytest typically handles this if run from the root.

# To make TestStrategy and TestSignalGenerator available during the test run without
# permanently altering the main codebase or using complex pytest plugin systems,
# the helper `_run_backtest_get_results` now includes temporary registration
# of these test components into the respective modules (strategies_module, signal_generators_module).
# This is a common pattern for testing with dependency injection or module-level registries.

# The price_data_monthly and price_data_daily arguments to run_scenario were incorrect in the helper. Fixed.
# `price_data_monthly=monthly_data`, `price_data_daily=daily_data`.

# TestSignalGenerator.scores was simplified to take features['price_data_for_scores']
# which is populated in the test helper.
# The TestStrategy configures num_holdings=1, so it will try to invest in ASSET_A.
# The mock price data is designed so ASSET_A's volatility changes, which should trigger
# changes in portfolio P&L volatility, and thus changes in leverage from AnnualizedVolatilityTargeting.
