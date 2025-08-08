import logging
import pandas as pd

from ..interfaces.strategy_resolver import StrategyResolverFactory
from ..interfaces.enforcement import enforce_strategy_parameter

# Removed legacy position sizer imports - now using strategy provider interfaces

logger = logging.getLogger(__name__)


def generate_signals(
    strategy,
    scenario_config,
    price_data_daily_ohlc,
    universe_tickers,
    benchmark_ticker,
    has_timed_out,
):
    # Check if this is a meta strategy - if so, use trade-based approach
    strategy_resolver = StrategyResolverFactory.create()
    if strategy_resolver.is_meta_strategy(type(strategy)):
        return _generate_meta_strategy_signals(
            strategy,
            scenario_config,
            price_data_daily_ohlc,
            universe_tickers,
            benchmark_ticker,
            has_timed_out,
        )

    # Standard strategy signal generation
    timing_controller = strategy.get_timing_controller()
    timing_controller.reset_state()

    start_date = price_data_daily_ohlc.index.min()
    end_date = price_data_daily_ohlc.index.max()

    wfo_start_date = pd.to_datetime(scenario_config.get("wfo_start_date", None))
    wfo_end_date = pd.to_datetime(scenario_config.get("wfo_end_date", None))

    if wfo_start_date is not None:
        start_date = max(start_date, wfo_start_date)
    if wfo_end_date is not None:
        end_date = min(end_date, wfo_end_date)

    rebalance_dates = timing_controller.get_rebalance_dates(
        start_date=start_date,
        end_date=end_date,
        available_dates=price_data_daily_ohlc.index,
        strategy_context=strategy,
    )

    # Honor scenario-configured daily rebalance for meta strategies in single-path architecture
    try:
        configured_freq = (
            scenario_config.get("timing_config", {}).get("rebalance_frequency")
            if isinstance(scenario_config, dict)
            else None
        )
        if configured_freq == "D":
            rebalance_dates = price_data_daily_ohlc.index
    except Exception:
        # Fall back silently to timing controller dates on any issue
        pass

    all_monthly_weights = []

    for current_rebalance_date in rebalance_dates:
        if has_timed_out():
            logger.warning("Timeout reached during scenario run. Halting signal generation.")
            break

        should_generate = timing_controller.should_generate_signal(
            current_date=current_rebalance_date, strategy_context=strategy
        )

        if not should_generate:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Timing controller skipped signal generation for date: {current_rebalance_date}"
                )
            continue

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Generating signals for date: {current_rebalance_date}")

        if (
            isinstance(price_data_daily_ohlc.columns, pd.MultiIndex)
            and "Ticker" in price_data_daily_ohlc.columns.names
        ):
            asset_hist_data_cols = pd.MultiIndex.from_product(
                [
                    universe_tickers,
                    list(price_data_daily_ohlc.columns.get_level_values("Field").unique()),
                ],
                names=["Ticker", "Field"],
            )
            # Use intersection to preserve MultiIndex dtype
            asset_hist_data_cols = asset_hist_data_cols.intersection(price_data_daily_ohlc.columns)
            all_historical_data_for_strat = price_data_daily_ohlc.loc[
                price_data_daily_ohlc.index <= current_rebalance_date, asset_hist_data_cols
            ]

            benchmark_hist_data_cols = pd.MultiIndex.from_product(
                [
                    [benchmark_ticker],
                    list(price_data_daily_ohlc.columns.get_level_values("Field").unique()),
                ],
                names=["Ticker", "Field"],
            )
            benchmark_hist_data_cols = benchmark_hist_data_cols.intersection(
                price_data_daily_ohlc.columns
            )
            benchmark_historical_data_for_strat = price_data_daily_ohlc.loc[
                price_data_daily_ohlc.index <= current_rebalance_date, benchmark_hist_data_cols
            ]
        else:
            all_historical_data_for_strat = price_data_daily_ohlc.loc[
                price_data_daily_ohlc.index <= current_rebalance_date, universe_tickers
            ]
            benchmark_historical_data_for_strat = price_data_daily_ohlc.loc[
                price_data_daily_ohlc.index <= current_rebalance_date, [benchmark_ticker]
            ]

        non_universe_tickers = strategy.get_non_universe_data_requirements()
        non_universe_historical_data_for_strat = pd.DataFrame()
        if non_universe_tickers:
            if (
                isinstance(price_data_daily_ohlc.columns, pd.MultiIndex)
                and "Ticker" in price_data_daily_ohlc.columns.names
            ):
                non_universe_hist_data_cols = pd.MultiIndex.from_product(
                    [
                        non_universe_tickers,
                        list(price_data_daily_ohlc.columns.get_level_values("Field").unique()),
                    ],
                    names=["Ticker", "Field"],
                )
                non_universe_hist_data_cols = non_universe_hist_data_cols.intersection(
                    price_data_daily_ohlc.columns
                )
                non_universe_historical_data_for_strat = price_data_daily_ohlc.loc[
                    price_data_daily_ohlc.index <= current_rebalance_date,
                    non_universe_hist_data_cols,
                ]
            else:
                non_universe_historical_data_for_strat = price_data_daily_ohlc.loc[
                    price_data_daily_ohlc.index <= current_rebalance_date, non_universe_tickers
                ]

        import inspect

        sig = inspect.signature(strategy.generate_signals)
        if "non_universe_historical_data" in sig.parameters:
            current_weights_df = strategy.generate_signals(
                all_historical_data=all_historical_data_for_strat,
                benchmark_historical_data=benchmark_historical_data_for_strat,
                non_universe_historical_data=non_universe_historical_data_for_strat,
                current_date=current_rebalance_date,
                start_date=wfo_start_date,
                end_date=wfo_end_date,
            )
        else:
            current_weights_df = strategy.generate_signals(
                all_historical_data=all_historical_data_for_strat,
                benchmark_historical_data=benchmark_historical_data_for_strat,
                current_date=current_rebalance_date,
                start_date=wfo_start_date,
                end_date=wfo_end_date,
            )

        if current_weights_df is not None and not current_weights_df.empty:
            if len(current_weights_df) > 0:
                current_weights_series = current_weights_df.iloc[0]
                timing_controller.update_signal_state(
                    current_rebalance_date, current_weights_series
                )

                try:
                    if isinstance(
                        price_data_daily_ohlc.columns, pd.MultiIndex
                    ) and "Close" in price_data_daily_ohlc.columns.get_level_values(1):
                        current_prices = price_data_daily_ohlc.loc[current_rebalance_date].xs(
                            "Close", level="Field"
                        )
                    elif not isinstance(price_data_daily_ohlc.columns, pd.MultiIndex):
                        current_prices = price_data_daily_ohlc.loc[current_rebalance_date]
                    else:
                        try:
                            current_prices = price_data_daily_ohlc.loc[current_rebalance_date].xs(
                                "Close", level=-1
                            )
                        except Exception:
                            current_prices = price_data_daily_ohlc.loc[current_rebalance_date].iloc[
                                : len(universe_tickers)
                            ]

                    universe_prices = current_prices.reindex(universe_tickers).ffill()

                    timing_controller.update_position_state(
                        current_rebalance_date, current_weights_series, universe_prices
                    )

                except Exception as e:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"Could not update position state for {current_rebalance_date}: {e}"
                        )

        all_monthly_weights.append(current_weights_df)

    if not all_monthly_weights:
        if logger.isEnabledFor(logging.WARNING):
            logger.warning(
                f"No signals generated for scenario {scenario_config['name']}. This might be due to WFO window or other issues."
            )
        signals = pd.DataFrame(columns=universe_tickers, index=rebalance_dates)
    else:
        signals = pd.concat(all_monthly_weights)
        signals = signals.reindex(rebalance_dates).fillna(0.0)

    # DEPRECATED: Framework-level trade direction filtering has been replaced
    # by strict enforcement at the strategy level. Strategies now throw
    # TradeDirectionViolationError if they violate trade_longs/trade_shorts constraints.
    # This prevents coding errors and ensures immediate failure rather than silent filtering.
    # The framework filtering is kept for backward compatibility but should not be needed.

    return signals


def _generate_meta_strategy_signals(
    strategy,
    scenario_config,
    price_data_daily_ohlc,
    universe_tickers,
    benchmark_ticker,
    has_timed_out,
):
    """
    Generate signals for meta strategies using trade-based approach.

    Meta strategies track actual trades from sub-strategies rather than just aggregating signals.
    This function coordinates the meta strategy's signal generation and returns signals that
    represent the actual trades executed by sub-strategies.
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Generating meta strategy signals for {strategy.__class__.__name__}")

    # Reset interceptor state for clean run
    strategy.reset_interceptor_state()

    timing_controller = strategy.get_timing_controller()
    timing_controller.reset_state()

    start_date = price_data_daily_ohlc.index.min()
    end_date = price_data_daily_ohlc.index.max()

    wfo_start_date = pd.to_datetime(scenario_config.get("wfo_start_date", None))
    wfo_end_date = pd.to_datetime(scenario_config.get("wfo_end_date", None))

    if wfo_start_date is not None:
        start_date = max(start_date, wfo_start_date)
    if wfo_end_date is not None:
        end_date = min(end_date, wfo_end_date)

    rebalance_dates = timing_controller.get_rebalance_dates(
        start_date=start_date,
        end_date=end_date,
        available_dates=price_data_daily_ohlc.index,
        strategy_context=strategy,
    )

    # Honor scenario-configured daily rebalance for meta strategies
    try:
        configured_freq = (
            scenario_config.get("timing_config", {}).get("rebalance_frequency")
            if isinstance(scenario_config, dict)
            else None
        )
        is_daily = configured_freq == "D"
        if is_daily:
            rebalance_dates = price_data_daily_ohlc.index
    except Exception:
        pass

    all_monthly_weights = []

    for current_rebalance_date in rebalance_dates:
        if has_timed_out():
            logger.warning(
                "Timeout reached during meta strategy scenario run. Halting signal generation."
            )
            break

        # For daily-configured meta strategies, generate every day
        try:
            if is_daily:
                should_generate = True
            else:
                should_generate = timing_controller.should_generate_signal(
                    current_date=current_rebalance_date, strategy_context=strategy
                )
        except NameError:
            # If is_daily not defined due to earlier exception, fall back to controller
            should_generate = timing_controller.should_generate_signal(
                current_date=current_rebalance_date, strategy_context=strategy
            )

        if not should_generate:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Timing controller skipped meta strategy signal generation for date: {current_rebalance_date}"
                )
            continue

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Generating meta strategy signals for date: {current_rebalance_date}")

        # Prepare data for meta strategy (same as regular strategy)
        if (
            isinstance(price_data_daily_ohlc.columns, pd.MultiIndex)
            and "Ticker" in price_data_daily_ohlc.columns.names
        ):
            asset_hist_data_cols = pd.MultiIndex.from_product(
                [
                    universe_tickers,
                    list(price_data_daily_ohlc.columns.get_level_values("Field").unique()),
                ],
                names=["Ticker", "Field"],
            )
            asset_hist_data_cols = asset_hist_data_cols.intersection(price_data_daily_ohlc.columns)
            all_historical_data_for_strat = price_data_daily_ohlc.loc[
                price_data_daily_ohlc.index <= current_rebalance_date, asset_hist_data_cols
            ]

            benchmark_hist_data_cols = pd.MultiIndex.from_product(
                [
                    [benchmark_ticker],
                    list(price_data_daily_ohlc.columns.get_level_values("Field").unique()),
                ],
                names=["Ticker", "Field"],
            )
            benchmark_hist_data_cols = benchmark_hist_data_cols.intersection(
                price_data_daily_ohlc.columns
            )
            benchmark_historical_data_for_strat = price_data_daily_ohlc.loc[
                price_data_daily_ohlc.index <= current_rebalance_date, benchmark_hist_data_cols
            ]
        else:
            all_historical_data_for_strat = price_data_daily_ohlc.loc[
                price_data_daily_ohlc.index <= current_rebalance_date, universe_tickers
            ]
            benchmark_historical_data_for_strat = price_data_daily_ohlc.loc[
                price_data_daily_ohlc.index <= current_rebalance_date, [benchmark_ticker]
            ]

        non_universe_tickers = strategy.get_non_universe_data_requirements()
        non_universe_historical_data_for_strat = pd.DataFrame()
        if non_universe_tickers:
            if (
                isinstance(price_data_daily_ohlc.columns, pd.MultiIndex)
                and "Ticker" in price_data_daily_ohlc.columns.names
            ):
                non_universe_hist_data_cols = pd.MultiIndex.from_product(
                    [
                        non_universe_tickers,
                        list(price_data_daily_ohlc.columns.get_level_values("Field").unique()),
                    ],
                    names=["Ticker", "Field"],
                )
                non_universe_hist_data_cols = non_universe_hist_data_cols.intersection(
                    price_data_daily_ohlc.columns
                )
                non_universe_historical_data_for_strat = price_data_daily_ohlc.loc[
                    price_data_daily_ohlc.index <= current_rebalance_date,
                    non_universe_hist_data_cols,
                ]
            else:
                non_universe_historical_data_for_strat = price_data_daily_ohlc.loc[
                    price_data_daily_ohlc.index <= current_rebalance_date, non_universe_tickers
                ]

        # Generate signals from meta strategy (this will trigger trade interceptors)
        import inspect

        sig = inspect.signature(strategy.generate_signals)
        if "non_universe_historical_data" in sig.parameters:
            current_weights_df = strategy.generate_signals(
                all_historical_data=all_historical_data_for_strat,
                benchmark_historical_data=benchmark_historical_data_for_strat,
                non_universe_historical_data=non_universe_historical_data_for_strat,
                current_date=current_rebalance_date,
                start_date=wfo_start_date,
                end_date=wfo_end_date,
            )
        else:
            current_weights_df = strategy.generate_signals(
                all_historical_data=all_historical_data_for_strat,
                benchmark_historical_data=benchmark_historical_data_for_strat,
                current_date=current_rebalance_date,
                start_date=wfo_start_date,
                end_date=wfo_end_date,
            )

        if current_weights_df is not None and not current_weights_df.empty:
            if len(current_weights_df) > 0:
                current_weights_series = current_weights_df.iloc[0]
                timing_controller.update_signal_state(
                    current_rebalance_date, current_weights_series
                )

                try:
                    if isinstance(
                        price_data_daily_ohlc.columns, pd.MultiIndex
                    ) and "Close" in price_data_daily_ohlc.columns.get_level_values(1):
                        current_prices = price_data_daily_ohlc.loc[current_rebalance_date].xs(
                            "Close", level="Field"
                        )
                    elif not isinstance(price_data_daily_ohlc.columns, pd.MultiIndex):
                        current_prices = price_data_daily_ohlc.loc[current_rebalance_date]
                    else:
                        try:
                            current_prices = price_data_daily_ohlc.loc[current_rebalance_date].xs(
                                "Close", level=-1
                            )
                        except Exception:
                            current_prices = price_data_daily_ohlc.loc[current_rebalance_date].iloc[
                                : len(universe_tickers)
                            ]

                    universe_prices = current_prices.reindex(universe_tickers).ffill()

                    timing_controller.update_position_state(
                        current_rebalance_date, current_weights_series, universe_prices
                    )

                except Exception as e:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"Could not update position state for meta strategy {current_rebalance_date}: {e}"
                        )

        all_monthly_weights.append(current_weights_df)

    if not all_monthly_weights:
        if logger.isEnabledFor(logging.WARNING):
            logger.warning(
                f"No signals generated for meta strategy scenario {scenario_config['name']}. This might be due to WFO window or other issues."
            )
        signals = pd.DataFrame(columns=universe_tickers, index=rebalance_dates)
    else:
        signals = pd.concat(all_monthly_weights)
        signals = signals.reindex(rebalance_dates).fillna(0.0)

    # For meta strategies, we want to return signals that represent actual trades
    # The trade interceptors have already captured the actual trades
    # Now we need to convert those trades back to signal format for framework compatibility

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Meta strategy generated {len(strategy.get_aggregated_trades())} trades")
        logger.debug(f"Returning signals with shape: {signals.shape}")

    return signals


@enforce_strategy_parameter
def size_positions(
    signals,
    scenario_config,
    price_data_monthly_closes,
    price_data_daily_ohlc,
    universe_tickers,
    benchmark_ticker,
    strategy,
):
    """
    Size positions using the strategy's position sizer provider.

    Args:
        signals: Trading signals DataFrame
        scenario_config: Scenario configuration (for direct bypass check)
        price_data_monthly_closes: Monthly price data
        price_data_daily_ohlc: Daily OHLC price data
        universe_tickers: List of universe tickers
        benchmark_ticker: Benchmark ticker
        strategy: Strategy instance with position sizer provider

    Returns:
        Sized signals DataFrame
    """
    # If the position_sizer is set to "direct", bypass the sizing logic
    if scenario_config.get("position_sizer") == "direct":
        return signals

    # Use position sizer provider from strategy
    position_sizer_provider = strategy.get_position_sizer_provider()
    sizer = position_sizer_provider.get_position_sizer()
    sizer_config = position_sizer_provider.get_position_sizer_config()

    # Extract position sizer parameters from provider config
    filtered_sizer_params = {k: v for k, v in sizer_config.items() if k != "position_sizer"}

    strategy_monthly_closes = price_data_monthly_closes[universe_tickers]
    benchmark_monthly_closes = price_data_monthly_closes[benchmark_ticker]

    # Prepare the arguments for the sizer's calculate_weights method
    sizer_kwargs = {
        "signals": signals,
        "prices": strategy_monthly_closes,
        "benchmark": benchmark_monthly_closes,
        **filtered_sizer_params,
    }

    # Special handling for rolling_downside_volatility sizer
    position_sizer_name = sizer_config.get("position_sizer", "equal_weight")
    if position_sizer_name == "rolling_downside_volatility":
        if isinstance(
            price_data_daily_ohlc.columns, pd.MultiIndex
        ) and "Close" in price_data_daily_ohlc.columns.get_level_values(1):
            daily_closes_for_sizer = price_data_daily_ohlc.xs("Close", level="Field", axis=1)[
                universe_tickers
            ]
        elif not isinstance(price_data_daily_ohlc.columns, pd.MultiIndex):
            daily_closes_for_sizer = price_data_daily_ohlc[universe_tickers]
        else:
            raise ValueError(
                "rolling_downside_volatility sizer: Could not extract daily close prices from price_data_daily_ohlc."
            )
        sizer_kwargs["daily_prices_for_vol"] = daily_closes_for_sizer

    sized_signals = sizer.calculate_weights(**sizer_kwargs)

    return sized_signals
