from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Set, Callable

import numpy as np
import pandas as pd

from ..feature import Feature, BenchmarkSMA
from ..portfolio.position_sizer import get_position_sizer
from ..signal_generators import BaseSignalGenerator
from ..roro_signals import BaseRoRoSignal # Import BaseRoRoSignal
from .stop_loss import BaseStopLoss, NoStopLoss, AtrBasedStopLoss # Stop Loss imports


class BaseStrategy(ABC):
    """Base class for trading strategies using pluggable signal generators."""

    #: class attribute specifying which signal generator to use
    signal_generator_class: type[BaseSignalGenerator] | None = None
    #: class attribute specifying which RoRo signal generator to use
    roro_signal_class: type[BaseRoRoSignal] | None = None
    #: class attribute specifying which Stop Loss handler to use by default
    stop_loss_handler_class: type[BaseStopLoss] = NoStopLoss


    def __init__(self, strategy_config: dict):
        self.strategy_config = strategy_config
        self._roro_signal_instance: BaseRoRoSignal | None = None
        self._stop_loss_handler_instance: BaseStopLoss | None = None
        # Initialize entry_prices Series, to be populated during generate_signals
        # It will store the entry price for the current holding period of an asset.
        # Needs to be persistent across calls to generate_signals if strategy is stateful,
        # but for backtesting, it's typically re-evaluated or carried over per time step.
        # For simplicity in this loop, it's re-initialized per call to generate_signals
        # and updated based on w_prev.
        self.entry_prices: pd.Series | None = None


    # ------------------------------------------------------------------ #
    # Hooks to override in subclasses
    # ------------------------------------------------------------------ #

    def get_signal_generator(self) -> BaseSignalGenerator:
        if self.signal_generator_class is None:
            raise NotImplementedError("signal_generator_class must be set")
        return self.signal_generator_class(self.strategy_config)

    def get_roro_signal(self) -> BaseRoRoSignal | None:
        if self.roro_signal_class is None:
            return None
        if self._roro_signal_instance is None:
            # Assuming RoRo signal might have its own config under "roro_signal_params"
            roro_config = self.strategy_config.get("roro_signal_params", self.strategy_config)
            self._roro_signal_instance = self.roro_signal_class(roro_config)
        return self._roro_signal_instance

    def get_stop_loss_handler(self) -> BaseStopLoss:
        if self._stop_loss_handler_instance is None:
            sl_config = self.strategy_config.get("stop_loss_config", {})
            sl_type_name = sl_config.get("type", "NoStopLoss") # Default to NoStopLoss

            handler_class = NoStopLoss # Default
            if sl_type_name == "AtrBasedStopLoss":
                handler_class = AtrBasedStopLoss
            # Can add more else-if for other stop-loss types here

            # Pass both general strategy_config and specific sl_config to the handler
            self._stop_loss_handler_instance = handler_class(self.strategy_config, sl_config)
        return self._stop_loss_handler_instance

    def get_position_sizer(self) -> Callable[[pd.DataFrame], pd.DataFrame]:
        name = self.strategy_config.get("position_sizer", "equal_weight")
        return get_position_sizer(name)

    # ------------------------------------------------------------------ #
    # Required features
    # ------------------------------------------------------------------ #
    @classmethod
    def get_required_features(cls, strategy_config: dict) -> Set[Feature]:
        features: Set[Feature] = set()

        # Features from signal generator
        signal_gen_cls = getattr(cls, "signal_generator_class", None)
        if signal_gen_cls is not None:
            generator = signal_gen_cls(strategy_config)
            features.update(generator.required_features())

        # Features from RoRo signal generator
        roro_gen_cls = getattr(cls, "roro_signal_class", None)
        if roro_gen_cls is not None:
            # Assuming RoRo signal might have its own config under "roro_signal_params"
            roro_config = strategy_config.get("roro_signal_params", strategy_config)
            roro_generator = roro_gen_cls(roro_config)
            features.update(roro_generator.get_required_features())

        params = strategy_config.get("strategy_params", strategy_config)
        sma_window = params.get("sma_filter_window")
        if sma_window is not None:
            features.add(BenchmarkSMA(sma_filter_window=sma_window))

        # Features from stop loss handler
        # To get the handler class, we need to temporarily instantiate one or access class attribute
        # This logic assumes stop_loss_handler_class is appropriately set or configured.
        sl_conf = strategy_config.get("stop_loss_config", {})
        sl_type_name = sl_conf.get("type", "NoStopLoss")

        sl_handler_cls = NoStopLoss # Default
        if sl_type_name == "AtrBasedStopLoss":
            sl_handler_cls = AtrBasedStopLoss
        # Add more handlers here if needed

        # Instantiate with potentially minimal config just for feature extraction
        # Some handlers might not need full config for get_required_features if it's static
        temp_sl_handler = sl_handler_cls(strategy_config, sl_conf)
        features.update(temp_sl_handler.get_required_features())

        return features

    # ------------------------------------------------------------------ #
    # Universe helper
    # ------------------------------------------------------------------ #
    def get_universe(self, global_config: dict) -> list[tuple[str, float]]:
        default_universe = global_config.get("universe", [])
        return [(ticker, 1.0) for ticker in default_universe]


    # ------------------------------------------------------------------ #
    # Optimiser-introspection hook                                       #
    # ------------------------------------------------------------------ #
    @classmethod
    def tunable_parameters(cls) -> set[str]:
        """Names of hyper-parameters this strategy understands."""
        return set()

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #
    def _calculate_candidate_weights(self, look: pd.Series) -> pd.Series:
        num_holdings = self.strategy_config.get("num_holdings")
        if num_holdings is not None and num_holdings > 0:
            nh = int(num_holdings)
        else:
            nh = max(
                int(
                    np.ceil(
                        self.strategy_config.get("top_decile_fraction", 0.1)
                        * look.count()
                    )
                ),
                1,
            )

        winners = look.nlargest(nh).index
        losers = look.nsmallest(nh).index

        cand = pd.Series(index=look.index, dtype=float).fillna(0.0)
        if len(winners) > 0:
            cand[winners] = 1 / len(winners)
        if not self.strategy_config.get("long_only", True) and len(losers) > 0:
            cand[losers] = -1 / len(losers)
        return cand

    def _apply_leverage_and_smoothing(
        self, cand: pd.Series, w_prev: pd.Series
    ) -> pd.Series:
        leverage = self.strategy_config.get("leverage", 1.0)
        smoothing_lambda = self.strategy_config.get("smoothing_lambda", 0.5)

        w_new = smoothing_lambda * w_prev + (1 - smoothing_lambda) * cand

        if cand.abs().sum() > 1e-9:
            long_lev = w_new[w_new > 0].sum()
            short_lev = -w_new[w_new < 0].sum()

            if long_lev > leverage:
                w_new[w_new > 0] *= leverage / long_lev
            if short_lev > leverage:
                w_new[w_new < 0] *= leverage / short_lev

        return w_new

    # ------------------------------------------------------------------ #
    # Default signal generation pipeline
    # ------------------------------------------------------------------ #
    def generate_signals(
        self,
        prices: pd.DataFrame,
        features: dict,
        benchmark_data: pd.Series,
    ) -> pd.DataFrame:
        generator = self.get_signal_generator()
        scores = generator.scores(features)

        weights = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
        w_prev = pd.Series(index=prices.columns, dtype=float).fillna(0.0) # Previous period's weights

        # Initialize or clear entry_prices for this run of generate_signals
        # It's a Series: index=asset_tickers, values=entry_price
        # Needs to be managed carefully if generate_signals is called multiple times for a stateful strategy object.
        # For typical backtesting loop, this initialization per call is fine.
        if self.entry_prices is None or not isinstance(self.entry_prices, pd.Series):
             self.entry_prices = pd.Series(np.nan, index=prices.columns)
        else:
            # If it exists, ensure it covers all current columns, fill new ones with NaN
            self.entry_prices = self.entry_prices.reindex(prices.columns).fillna(np.nan)


        # --- Stop Loss Handler ---
        sl_handler = self.get_stop_loss_handler()

        # --- SMA-based risk filter ---
        sma_window = self.strategy_config.get("sma_filter_window")
        derisk_days = self.strategy_config.get("derisk_days_under_sma", 10)
        use_sma_derisk = sma_window and derisk_days > 0

        if sma_window is not None:
            sma_name = f"benchmark_sma_{sma_window}m"
            # Ensure sma_risk_on_series has a boolean dtype after reindexing and filling
            sma_risk_on_series = features[sma_name].reindex(prices.index, fill_value=True).astype(bool)
        else:
            sma_risk_on_series = pd.Series(True, index=prices.index, name="sma_risk_on")

        derisk_flags = self._calculate_derisk_flags(sma_risk_on_series, derisk_days) if use_sma_derisk else pd.Series(False, index=prices.index)

        # --- RoRo signal based risk filter ---
        roro_signal_instance = self.get_roro_signal()
        if roro_signal_instance:
            # Generate RoRo signal for the prices.index (typically monthly dates)
            roro_signal_values = roro_signal_instance.generate_signal(prices.index)
            # Ensure it's boolean: 1 (risk-on) -> True, 0 (risk-off) -> False
            roro_risk_on_series = roro_signal_values.astype(bool)
        else:
            roro_risk_on_series = pd.Series(True, index=prices.index, name="roro_risk_on")


        for date in prices.index:
            # Handle case where scores doesn't have data for this date
            if date not in scores.index:
                weights.loc[date] = w_prev
                continue
                
            look = scores.loc[date]

            if generator.zero_if_nan and look.isna().any():
                weights.loc[date] = 0.0
                w_prev = weights.loc[date]
                continue

            if look.count() == 0:
                weights.loc[date] = w_prev
                continue

            look = look.dropna()

            cand = self._calculate_candidate_weights(look)
            # These are the preliminary target weights for the period, *before* stop loss or risk filters
            w_target_pre_filter = self._apply_leverage_and_smoothing(cand, w_prev)

            # --- Update Entry Prices ---
            # If a position is newly initiated (w_prev was 0, w_target_pre_filter is not)
            # or direction flips (sign changes), update entry price.
            # `prices` DataFrame here is typically monthly close prices.
            current_period_prices = prices.loc[date] # Series of prices for the current date

            for asset in w_target_pre_filter.index:
                # New position or flip in direction
                if (w_prev[asset] == 0 and w_target_pre_filter[asset] != 0) or \
                   (np.sign(w_prev[asset]) != np.sign(w_target_pre_filter[asset]) and w_target_pre_filter[asset] != 0):
                    self.entry_prices[asset] = current_period_prices[asset]
                # If position is closed, reset entry price
                elif w_target_pre_filter[asset] == 0:
                    self.entry_prices[asset] = np.nan
                # If position maintained (w_prev !=0 and w_target_pre_filter has same sign), entry price remains.

            # --- Apply Stop Loss ---
            # Stop loss is applied to w_target_pre_filter based on w_prev (active positions at start of period)
            # and their respective entry prices.
            stop_levels = sl_handler.calculate_stop_levels(
                date, prices, features, w_prev, self.entry_prices
            )
            # The `apply_stop_loss` will use `current_period_prices` (which are month-end closes here)
            # to check against the calculated stop_levels.
            w_after_sl = sl_handler.apply_stop_loss(
                date, current_period_prices, w_target_pre_filter, self.entry_prices, stop_levels
            )

            # If stop loss zeros out a position, update entry price to NaN
            for asset in w_after_sl.index:
                if w_target_pre_filter[asset] != 0 and w_after_sl[asset] == 0: # Position was closed by SL
                    self.entry_prices[asset] = np.nan


            # --- Apply Risk Filters (SMA, RoRo) to weights *after* stop loss ---
            w_final = w_after_sl.copy()
            if use_sma_derisk and derisk_flags.loc[date]:
                w_final[:] = 0.0
            if sma_window is not None and not sma_risk_on_series.loc[date]: # General SMA filter
                w_final[:] = 0.0
            if not roro_risk_on_series.loc[date]: # RoRo filter
                w_final[:] = 0.0

            # If risk filters zero out positions that SL didn't, also update entry prices
            for asset in w_final.index:
                if w_after_sl[asset] != 0 and w_final[asset] == 0:
                    self.entry_prices[asset] = np.nan

            weights.loc[date] = w_final
            w_prev = w_final # Update w_prev for the next iteration

        # Check for 'apply_trading_lag' potentially nested in 'strategy_params' or at top level of config
        strategy_params = self.strategy_config.get("strategy_params", self.strategy_config)
        apply_lag = strategy_params.get("apply_trading_lag", False)
        if not apply_lag and "apply_trading_lag" in self.strategy_config : # Check top level if not in strategy_params
             apply_lag = self.strategy_config.get("apply_trading_lag", False)


        if apply_lag:
            # If trading lag is applied, the weights for the first period will be NaN after shift.
            # The backtester itself also applies a shift when converting monthly to daily weights.
            # This ensures that signals generated on day D are acted upon on D+1.
            # Applying it here makes the strategy's direct output reflect the lag.
            weights = weights.shift(1)

        return weights

    def _calculate_derisk_flags(self, sma_risk_on_series: pd.Series, derisk_days: int) -> pd.Series:
        """
        Calculates flags indicating when to derisk based on consecutive days under SMA.

        Parameters:
        - sma_risk_on_series (pd.Series): Boolean series indicating if risk is on (True, price >= SMA)
                                          or off (False, price < SMA). Index must match prices.index.
        - derisk_days (int): Number of consecutive days asset must be under SMA to trigger derisking.

        Returns:
        - pd.Series: Boolean series with True where derisking should occur.
        """
        if not isinstance(sma_risk_on_series, pd.Series) or not isinstance(derisk_days, int) or derisk_days <= 0:
            # This case should ideally be handled by the caller (use_sma_derisk check)
            # but as a safeguard, return an all-false series.
            return pd.Series(False, index=sma_risk_on_series.index)

        derisk_flags = pd.Series(False, index=sma_risk_on_series.index)
        consecutive_days_risk_off = 0

        for date in sma_risk_on_series.index:
            if sma_risk_on_series.loc[date]:  # Price is >= SMA (Risk is ON based on SMA)
                consecutive_days_risk_off = 0
            else:  # Price is < SMA (Risk is OFF based on SMA)
                consecutive_days_risk_off += 1

            if consecutive_days_risk_off > derisk_days: # Note: strictly greater
                derisk_flags.loc[date] = True
            # else: # If it's not > derisk_days, the flag remains False (or becomes False if it was True and now sma_risk_on_series is True)
            #    if derisk_flags.loc[date]: # This logic is not needed as we reset consecutive_days_risk_off
            #        pass

        return derisk_flags
