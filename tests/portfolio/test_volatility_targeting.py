import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.portfolio_backtester.portfolio.volatility_targeting import (
    NoVolatilityTargeting,
    AnnualizedVolatilityTargeting,
    VolatilityTargetingBase,
)

# Helper to create a sample price series
def create_price_series(start_val, vol, num_days, start_date_str="2023-01-01"):
    re_turns = np.random.normal(loc=0, scale=vol / np.sqrt(252), size=num_days) # daily vol
    start_date = pd.to_datetime(start_date_str)
    dates = pd.date_range(start_date, periods=num_days, freq='B')
    prices = pd.Series((1 + re_turns).cumprod() * start_val, index=dates)
    return prices

# Helper to create dummy portfolio returns
def create_dummy_portfolio_returns(mean_ret, std_dev_daily, num_days, start_date_str="2023-01-01"):
    start_date = pd.to_datetime(start_date_str)
    dates = pd.date_range(start_date, periods=num_days, freq='B')
    returns = np.random.normal(loc=mean_ret, scale=std_dev_daily, size=num_days)
    return pd.Series(returns, index=dates)

class TestNoVolatilityTargeting:
    def test_always_returns_one(self):
        mechanism = NoVolatilityTargeting()
        dummy_weights = pd.Series([0.5, 0.5], index=['A', 'B'])
        dummy_returns_history = pd.Series([0.01, -0.01], index=[pd.to_datetime("2023-01-01"), pd.to_datetime("2023-01-02")])
        current_date = pd.to_datetime("2023-01-03")
        dummy_prices = pd.DataFrame({'A': [100,101], 'B': [100,99]}, index=dummy_returns_history.index)

        factor = mechanism.calculate_leverage_factor(
            current_raw_weights=dummy_weights,
            portfolio_returns_history=dummy_returns_history,
            current_date=current_date,
            daily_prices=dummy_prices,
            lookback_period_days=60
        )
        assert factor == 1.0

class TestAnnualizedVolatilityTargeting:

    @pytest.fixture
    def dummy_args(self):
        # Dummy args that are often needed but whose values might not matter for some tests
        return {
            "current_raw_weights": pd.Series([1.0], index=['A']),
            "current_date": pd.to_datetime("2023-06-01"),
            "daily_prices": pd.DataFrame({'A': create_price_series(100, 0.2, 100, "2023-01-01")})
        }

    def test_initialization(self):
        mechanism = AnnualizedVolatilityTargeting(
            target_annual_volatility=0.15,
            lookback_period_days=60,
            max_leverage=2.0,
            min_leverage=0.5
        )
        assert mechanism.target_annual_volatility == 0.15
        assert mechanism.lookback_period_days == 60
        assert mechanism.max_leverage == 2.0
        assert mechanism.min_leverage == 0.5
        assert mechanism.annualization_factor == 252.0
        assert mechanism.min_data_points == 30 # 60 * 0.5

    def test_initialization_invalid_params(self):
        with pytest.raises(ValueError):
            AnnualizedVolatilityTargeting(target_annual_volatility=0, lookback_period_days=60)
        with pytest.raises(ValueError):
            AnnualizedVolatilityTargeting(target_annual_volatility=0.1, lookback_period_days=0)
        with pytest.raises(ValueError):
            AnnualizedVolatilityTargeting(target_annual_volatility=0.1, lookback_period_days=60, max_leverage=0)
        with pytest.raises(ValueError):
            AnnualizedVolatilityTargeting(target_annual_volatility=0.1, lookback_period_days=60, min_leverage=-1)
        with pytest.raises(ValueError): # min_leverage >= max_leverage
            AnnualizedVolatilityTargeting(target_annual_volatility=0.1, lookback_period_days=60, min_leverage=2.0, max_leverage=1.0)


    def test_calculate_annualized_volatility_helper(self, dummy_args):
        mechanism = AnnualizedVolatilityTargeting(target_annual_volatility=0.10, lookback_period_days=20)

        # Test with insufficient data
        returns_short = create_dummy_portfolio_returns(0, 0.01, mechanism.min_data_points -1)
        assert mechanism._calculate_annualized_volatility(returns_short) is None

        # Test with sufficient data, known std dev
        # Daily std dev of 0.01 -> annual vol = 0.01 * sqrt(252) approx 0.1587
        returns_known_std = pd.Series([0.01, -0.01] * (mechanism.lookback_period_days // 2)) # std will be 0.01
        # Ensure length is enough for min_data_points
        if len(returns_known_std) < mechanism.min_data_points:
            returns_known_std = pd.Series([0.01, -0.01] * (mechanism.min_data_points))

        # Pad with some leading values to test slicing
        padding = pd.Series([0.0] * 10)
        returns_padded = pd.concat([padding, returns_known_std])

        calculated_vol = mechanism._calculate_annualized_volatility(returns_padded)
        # std of [0.01, -0.01] is sqrt( ((0.01-0)^2 + (-0.01-0)^2) / 1 ) = sqrt(2*0.0001) = 0.01 if ddof=0
        # for series [0.01, -0.01, 0.01, -0.01 ...], std is approx 0.01
        # Manually calculate expected for [0.01, -0.01] series for ddof=1:
        # mean = 0. var = ( (0.01-0)^2 + (-0.01-0)^2 ) / (2-1) = 0.0001 + 0.0001 = 0.0002. std = sqrt(0.0002) = 0.0141421356
        # This is if series is just [0.01, -0.01]. If it's longer, it's just 0.01.
        # Let's create a series with exact std_dev
        np.random.seed(42)
        exact_std_returns = pd.Series(np.random.normal(0, 0.01, mechanism.lookback_period_days * 2)) # ensure enough points

        # Slice to lookback_period_days for the _calculate_annualized_volatility function
        sliced_returns = exact_std_returns.iloc[-mechanism.lookback_period_days:]
        expected_std = sliced_returns.std(ddof=1)
        expected_annual_vol = expected_std * np.sqrt(mechanism.annualization_factor)

        calculated_vol_exact = mechanism._calculate_annualized_volatility(exact_std_returns)
        assert calculated_vol_exact == pytest.approx(expected_annual_vol, abs=1e-4)

        # Test with zero volatility
        returns_zero_vol = pd.Series([0.0] * mechanism.lookback_period_days)
        assert mechanism._calculate_annualized_volatility(returns_zero_vol) == 0.0

    def test_insufficient_data_returns_default_leverage(self, dummy_args):
        mechanism = AnnualizedVolatilityTargeting(target_annual_volatility=0.10, lookback_period_days=60)

        # History shorter than min_data_points
        short_history = create_dummy_portfolio_returns(0, 0.01, mechanism.min_data_points - 5, "2023-01-01")
        short_history = short_history[short_history.index < dummy_args["current_date"]] # Ensure history is past

        factor = mechanism.calculate_leverage_factor(
            portfolio_returns_history=short_history,
            lookback_period_days=mechanism.lookback_period_days,
            **dummy_args
        )
        assert factor == 1.0

    def test_zero_volatility_returns_default_leverage(self, dummy_args):
        mechanism = AnnualizedVolatilityTargeting(target_annual_volatility=0.10, lookback_period_days=30, max_leverage=2.0)
        zero_vol_returns = create_dummy_portfolio_returns(0, 0, 50, "2023-04-01") # 50 days of zero returns
        zero_vol_returns = zero_vol_returns[zero_vol_returns.index < dummy_args["current_date"]]

        factor = mechanism.calculate_leverage_factor(
            portfolio_returns_history=zero_vol_returns,
            lookback_period_days=mechanism.lookback_period_days,
            **dummy_args
        )
        assert factor == 1.0 # Current behavior for safety

    def test_leverage_adjustment_logic(self, dummy_args):
        target_vol = 0.15
        lookback = 30 # days
        mechanism = AnnualizedVolatilityTargeting(
            target_annual_volatility=target_vol,
            lookback_period_days=lookback,
            max_leverage=3.0,
            min_leverage=0.1
        )

        # Scenario 1: Current vol = 0.10 (lower than target) -> expect increased leverage
        # daily_std = 0.10 / sqrt(252) approx 0.0063
        returns_low_vol = create_dummy_portfolio_returns(0, 0.10 / np.sqrt(252), lookback + 20, "2023-04-01")
        returns_low_vol_history = returns_low_vol[returns_low_vol.index < dummy_args["current_date"]]

        factor_low_vol = mechanism.calculate_leverage_factor(
            portfolio_returns_history=returns_low_vol_history,
            lookback_period_days=mechanism.lookback_period_days,
            **dummy_args
        )
        # actual_vol_low = mechanism._calculate_annualized_volatility(returns_low_vol_history.tail(lookback))
        # print(f"Low vol scenario: Actual vol={actual_vol_low}, Expected factor={target_vol/actual_vol_low if actual_vol_low else -1}, Got factor={factor_low_vol}")
        assert factor_low_vol > 1.0 # Should increase leverage
        # Recalculate to check:
        hist_for_calc = returns_low_vol_history.tail(lookback)
        if len(hist_for_calc) >= mechanism.min_data_points :
            current_ann_vol = hist_for_calc.std(ddof=1) * np.sqrt(mechanism.annualization_factor)
            if current_ann_vol > 1e-8:
                 expected_factor = target_vol / current_ann_vol
                 assert factor_low_vol == pytest.approx(max(mechanism.min_leverage, min(expected_factor, mechanism.max_leverage)))


        # Scenario 2: Current vol = 0.20 (higher than target) -> expect decreased leverage
        # daily_std = 0.20 / sqrt(252) approx 0.0126
        returns_high_vol = create_dummy_portfolio_returns(0, 0.20 / np.sqrt(252), lookback + 20, "2023-04-01")
        returns_high_vol_history = returns_high_vol[returns_high_vol.index < dummy_args["current_date"]]

        factor_high_vol = mechanism.calculate_leverage_factor(
            portfolio_returns_history=returns_high_vol_history,
            lookback_period_days=mechanism.lookback_period_days,
            **dummy_args
        )
        # actual_vol_high = mechanism._calculate_annualized_volatility(returns_high_vol_history.tail(lookback))
        # print(f"High vol scenario: Actual vol={actual_vol_high}, Expected factor={target_vol/actual_vol_high if actual_vol_high else -1}, Got factor={factor_high_vol}")
        assert factor_high_vol < 1.0 # Should decrease leverage
        hist_for_calc_high = returns_high_vol_history.tail(lookback)
        if len(hist_for_calc_high) >= mechanism.min_data_points:
            current_ann_vol_high = hist_for_calc_high.std(ddof=1) * np.sqrt(mechanism.annualization_factor)
            if current_ann_vol_high > 1e-8:
                expected_factor_high = target_vol / current_ann_vol_high
                assert factor_high_vol == pytest.approx(max(mechanism.min_leverage, min(expected_factor_high, mechanism.max_leverage)))


        # Scenario 3: Current vol = target_vol (0.15) -> expect leverage factor around 1.0
        returns_target_vol = create_dummy_portfolio_returns(0, target_vol / np.sqrt(252), lookback + 20, "2023-04-01")
        returns_target_vol_history = returns_target_vol[returns_target_vol.index < dummy_args["current_date"]]
        factor_target_vol = mechanism.calculate_leverage_factor(
            portfolio_returns_history=returns_target_vol_history,
            lookback_period_days=mechanism.lookback_period_days,
            **dummy_args
        )
        # actual_vol_target = mechanism._calculate_annualized_volatility(returns_target_vol_history.tail(lookback))
        # print(f"Target vol scenario: Actual vol={actual_vol_target}, Expected factor={target_vol/actual_vol_target if actual_vol_target else -1}, Got factor={factor_target_vol}")
        assert factor_target_vol == pytest.approx(1.0, abs=0.2) # Increased tolerance

    def test_leverage_capping(self, dummy_args):
        target_vol = 0.15
        lookback = 30
        max_lev = 1.5
        min_lev = 0.5
        mechanism = AnnualizedVolatilityTargeting(
            target_annual_volatility=target_vol,
            lookback_period_days=lookback,
            max_leverage=max_lev,
            min_leverage=min_lev
        )

        # Scenario 1: Calculated factor > max_leverage
        # Current vol = 0.05 (target_vol / 0.05 = 3.0, but capped at 1.5)
        returns_very_low_vol = create_dummy_portfolio_returns(0, 0.05 / np.sqrt(252), lookback + 20, "2023-04-01")
        returns_very_low_vol_history = returns_very_low_vol[returns_very_low_vol.index < dummy_args["current_date"]]

        factor_very_low = mechanism.calculate_leverage_factor(
            portfolio_returns_history=returns_very_low_vol_history,
            lookback_period_days=mechanism.lookback_period_days,
            **dummy_args
        )
        assert factor_very_low == pytest.approx(max_lev, abs=0.1) # Check it's capped at max_leverage, allow slight variation

        # Scenario 2: Calculated factor < min_leverage
        # Current vol = 0.50 (target_vol / 0.50 = 0.3, but floored at 0.5)
        returns_very_high_vol = create_dummy_portfolio_returns(0, 0.50 / np.sqrt(252), lookback + 20, "2023-04-01")
        returns_very_high_vol_history = returns_very_high_vol[returns_very_high_vol.index < dummy_args["current_date"]]

        factor_very_high = mechanism.calculate_leverage_factor(
            portfolio_returns_history=returns_very_high_vol_history,
            lookback_period_days=mechanism.lookback_period_days,
            **dummy_args
        )
        assert factor_very_high == pytest.approx(min_lev, abs=0.1) # Check it's floored at min_leverage

    def test_real_usage_slicing(self, dummy_args):
        # Test that portfolio_returns_history is correctly sliced by current_date and lookback_period_days
        target_vol = 0.10
        lookback = 20
        mechanism = AnnualizedVolatilityTargeting(target_annual_volatility=target_vol, lookback_period_days=lookback)

        # Create a history that spans well before and up to the day before current_date
        # current_date is "2023-06-01"
        # History should end on "2023-05-31"
        # Lookback is 20 days, so history from "2023-05-04" to "2023-05-31" (approx, due to weekends)
        # Let's make history from 2023-03-01 to 2023-05-31 (more than enough)
        num_hist_days = (pd.to_datetime("2023-05-31") - pd.to_datetime("2023-03-01")).days + 30 # estimate

        full_history = create_dummy_portfolio_returns(0, 0.01, num_hist_days, "2023-03-01")
        full_history = full_history[full_history.index <= pd.to_datetime("2023-05-31")] # Ensure it ends correctly

        # Mock _calculate_annualized_volatility to check the input it receives
        class MockAnnualizedVolTargeting(AnnualizedVolatilityTargeting):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.received_history_for_vol_calc = None

            def _calculate_annualized_volatility(self, returns_series: pd.Series) -> float | None:
                self.received_history_for_vol_calc = returns_series
                # Return a dummy value so the main function doesn't break
                return super()._calculate_annualized_volatility(returns_series)

        mock_mechanism = MockAnnualizedVolTargeting(target_annual_volatility=target_vol, lookback_period_days=lookback)

        mock_mechanism.calculate_leverage_factor(
            portfolio_returns_history=full_history,
            lookback_period_days=mock_mechanism.lookback_period_days,
            **dummy_args # current_date is 2023-06-01
        )

        received_hist = mock_mechanism.received_history_for_vol_calc
        assert received_hist is not None

        # Expected end date for history used in vol calc is one day before current_date
        expected_last_hist_date = dummy_args["current_date"] - pd.Timedelta(days=1)
        # Find the closest business day if it's a weekend/holiday
        while expected_last_hist_date not in full_history.index and expected_last_hist_date > full_history.index.min():
            expected_last_hist_date -= pd.Timedelta(days=1)

        if not received_hist.empty :
             assert received_hist.index.max() <= expected_last_hist_date

        # The relevant_history inside calculate_leverage_factor should be:
        # full_history[full_history.index < dummy_args["current_date"]].tail(lookback)
        # So, received_hist should be this tail part.
        expected_relevant_history = full_history[full_history.index < dummy_args["current_date"]].tail(lookback)

        if not expected_relevant_history.empty and not received_hist.empty:
            pd.testing.assert_series_equal(received_hist, expected_relevant_history, check_dtype=False)
            assert len(received_hist) <= lookback
        elif len(expected_relevant_history) < mock_mechanism.min_data_points:
            # If expected relevant history is too short, _calculate_annualized_volatility might not be called
            # or called with an empty/short series leading to None.
            # Factor calculation returns 1.0 in this case. Let's ensure that.
             factor = mechanism.calculate_leverage_factor(portfolio_returns_history=full_history.head(mock_mechanism.min_data_points-1), **dummy_args)
             assert factor == 1.0
        else:
             assert len(received_hist) == 0 and len(expected_relevant_history) == 0


# To run these tests: pytest tests/portfolio/test_volatility_targeting.py
# from the root directory of the project.
# Ensure __init__.py files are present in test directories if needed by pytest structure.

# Adding a simple test for VolatilityTargetingBase (it's abstract, so mostly for completeness)
def test_volatility_targeting_base_is_abc():
    with pytest.raises(TypeError): # Cannot instantiate ABC
        VolatilityTargetingBase()

    class ConcreteImpl(VolatilityTargetingBase):
        def calculate_leverage_factor(self, current_raw_weights, portfolio_returns_history, current_date, daily_prices, lookback_period_days):
            return 1.0

    impl = ConcreteImpl()
    assert isinstance(impl, VolatilityTargetingBase)
