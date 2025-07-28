import logging
import pandas as pd

from ..portfolio.position_sizer import get_position_sizer, SIZER_PARAM_MAPPING

logger = logging.getLogger(__name__)

def generate_signals(strategy, scenario_config, price_data_daily_ohlc, universe_tickers, benchmark_ticker, has_timed_out):
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
        strategy_context=strategy
    )

    all_monthly_weights = []

    for current_rebalance_date in rebalance_dates:
        if has_timed_out():
            logger.warning("Timeout reached during scenario run. Halting signal generation.")
            break

        should_generate = timing_controller.should_generate_signal(
            current_date=current_rebalance_date,
            strategy_context=strategy
        )

        if not should_generate:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Timing controller skipped signal generation for date: {current_rebalance_date}")
            continue

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Generating signals for date: {current_rebalance_date}")

        if isinstance(price_data_daily_ohlc.columns, pd.MultiIndex) and 'Ticker' in price_data_daily_ohlc.columns.names:
            asset_hist_data_cols = pd.MultiIndex.from_product([universe_tickers, list(price_data_daily_ohlc.columns.get_level_values('Field').unique())], names=['Ticker', 'Field'])
            asset_hist_data_cols = [col for col in asset_hist_data_cols if col in price_data_daily_ohlc.columns]
            all_historical_data_for_strat = price_data_daily_ohlc.loc[price_data_daily_ohlc.index <= current_rebalance_date, asset_hist_data_cols]

            benchmark_hist_data_cols = pd.MultiIndex.from_product([[benchmark_ticker], list(price_data_daily_ohlc.columns.get_level_values('Field').unique())], names=['Ticker', 'Field'])
            benchmark_hist_data_cols = [col for col in benchmark_hist_data_cols if col in price_data_daily_ohlc.columns]
            benchmark_historical_data_for_strat = price_data_daily_ohlc.loc[price_data_daily_ohlc.index <= current_rebalance_date, benchmark_hist_data_cols]
        else:
            all_historical_data_for_strat = price_data_daily_ohlc.loc[price_data_daily_ohlc.index <= current_rebalance_date, universe_tickers]
            benchmark_historical_data_for_strat = price_data_daily_ohlc.loc[price_data_daily_ohlc.index <= current_rebalance_date, [benchmark_ticker]]

        non_universe_tickers = strategy.get_non_universe_data_requirements()
        non_universe_historical_data_for_strat = pd.DataFrame()
        if non_universe_tickers:
            if isinstance(price_data_daily_ohlc.columns, pd.MultiIndex) and 'Ticker' in price_data_daily_ohlc.columns.names:
                non_universe_hist_data_cols = pd.MultiIndex.from_product([non_universe_tickers, list(price_data_daily_ohlc.columns.get_level_values('Field').unique())], names=['Ticker', 'Field'])
                non_universe_hist_data_cols = [col for col in non_universe_hist_data_cols if col in price_data_daily_ohlc.columns]
                non_universe_historical_data_for_strat = price_data_daily_ohlc.loc[price_data_daily_ohlc.index <= current_rebalance_date, non_universe_hist_data_cols]
            else:
                non_universe_historical_data_for_strat = price_data_daily_ohlc.loc[price_data_daily_ohlc.index <= current_rebalance_date, non_universe_tickers]

        import inspect
        sig = inspect.signature(strategy.generate_signals)
        if 'non_universe_historical_data' in sig.parameters:
            current_weights_df = strategy.generate_signals(
                all_historical_data=all_historical_data_for_strat,
                benchmark_historical_data=benchmark_historical_data_for_strat,
                non_universe_historical_data=non_universe_historical_data_for_strat,
                current_date=current_rebalance_date,
                start_date=wfo_start_date,
                end_date=wfo_end_date
            )
        else:
            current_weights_df = strategy.generate_signals(
                all_historical_data=all_historical_data_for_strat,
                benchmark_historical_data=benchmark_historical_data_for_strat,
                current_date=current_rebalance_date,
                start_date=wfo_start_date,
                end_date=wfo_end_date
            )

        if current_weights_df is not None and not current_weights_df.empty:
            if len(current_weights_df) > 0:
                current_weights_series = current_weights_df.iloc[0]
                timing_controller.update_signal_state(current_rebalance_date, current_weights_series)

                try:
                    if isinstance(price_data_daily_ohlc.columns, pd.MultiIndex) and 'Close' in price_data_daily_ohlc.columns.get_level_values(1):
                        current_prices = price_data_daily_ohlc.loc[current_rebalance_date].xs('Close', level='Field')
                    elif not isinstance(price_data_daily_ohlc.columns, pd.MultiIndex):
                        current_prices = price_data_daily_ohlc.loc[current_rebalance_date]
                    else:
                        try:
                            current_prices = price_data_daily_ohlc.loc[current_rebalance_date].xs('Close', level=-1)
                        except:
                            current_prices = price_data_daily_ohlc.loc[current_rebalance_date].iloc[:len(universe_tickers)]

                    universe_prices = current_prices.reindex(universe_tickers).ffill()

                    timing_controller.update_position_state(
                        current_rebalance_date, 
                        current_weights_series, 
                        universe_prices
                    )

                except Exception as e:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Could not update position state for {current_rebalance_date}: {e}")

        all_monthly_weights.append(current_weights_df)

    if not all_monthly_weights:
        if logger.isEnabledFor(logging.WARNING):
            logger.warning(f"No signals generated for scenario {scenario_config['name']}. This might be due to WFO window or other issues.")
        signals = pd.DataFrame(columns=universe_tickers, index=rebalance_dates)
    else:
        signals = pd.concat(all_monthly_weights)
        signals = signals.reindex(rebalance_dates).fillna(0.0)

    return signals

def size_positions(signals, scenario_config, price_data_monthly_closes, price_data_daily_ohlc, universe_tickers, benchmark_ticker):
    sizer_name = scenario_config.get("position_sizer", "equal_weight")
    sizer_func = get_position_sizer(sizer_name)

    sizer_param_mapping = SIZER_PARAM_MAPPING

    filtered_sizer_params = {}
    strategy_params = scenario_config.get("strategy_params", {})

    window_param = None
    target_return_param = None
    max_leverage_param = None

    for key, value in strategy_params.items():
        if key in sizer_param_mapping:
            new_key = sizer_param_mapping[key]
            if new_key == "window":
                window_param = value
            elif new_key == "target_return":
                target_return_param = value
            elif new_key == "max_leverage":
                max_leverage_param = value
            else:
                filtered_sizer_params[new_key] = value

    strategy_monthly_closes = price_data_monthly_closes[universe_tickers]
    benchmark_monthly_closes = price_data_monthly_closes[benchmark_ticker]

    sizer_args = [signals, strategy_monthly_closes, benchmark_monthly_closes]

    if sizer_name == "rolling_downside_volatility":
        if isinstance(price_data_daily_ohlc.columns, pd.MultiIndex) and \
           'Close' in price_data_daily_ohlc.columns.get_level_values(1):
            daily_closes_for_sizer = price_data_daily_ohlc.xs('Close', level='Field', axis=1)[universe_tickers]
        elif not isinstance(price_data_daily_ohlc.columns, pd.MultiIndex):
            daily_closes_for_sizer = price_data_daily_ohlc[universe_tickers]
        else:
            raise ValueError("rolling_downside_volatility sizer: Could not extract daily close prices from price_data_daily_ohlc.")
        sizer_args.append(daily_closes_for_sizer)

    if sizer_name in ["rolling_sharpe", "rolling_sortino", "rolling_beta", "rolling_benchmark_corr", "rolling_downside_volatility"]:
        if window_param is None:
            raise ValueError(f"Sizer '{sizer_name}' requires a 'window' parameter, but it was not found in strategy_params.")
        sizer_args.append(window_param)

    if sizer_name == "rolling_sortino":
        if target_return_param is None:
            sizer_args.append(0.0)
        else:
            sizer_args.append(target_return_param)

    if sizer_name == "rolling_downside_volatility" and max_leverage_param is not None:
        filtered_sizer_params["max_leverage"] = max_leverage_param

    sized_signals = sizer_func(
        *sizer_args,
        **filtered_sizer_params,
    )

    return sized_signals
