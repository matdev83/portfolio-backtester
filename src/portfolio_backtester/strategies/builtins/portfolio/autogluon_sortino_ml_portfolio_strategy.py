from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, cast

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from portfolio_backtester.numba_optimized import rolling_sortino_batch
from portfolio_backtester.strategies._core.base import PortfolioStrategy

logger = logging.getLogger(__name__)


class AutogluonSortinoMlPortfolioStrategy(PortfolioStrategy):
    """Long-only portfolio strategy powered by AutoGluon.

    The model is trained on rolling Sortino, relative returns vs benchmark,
    and rolling correlation pair features. It predicts per-symbol weights
    which are clipped to long-only, normalized to 100% exposure, and then
    optionally scaled by volatility targeting (allowing gross exposure to drift).
    """

    def __init__(self, strategy_config: Dict[str, Any]) -> None:
        super().__init__(strategy_config)
        params = self._get_params_dict()

        defaults = {
            "trade_longs": True,
            "trade_shorts": False,
            "price_column_asset": "Close",
            "price_column_benchmark": "Close",
            "rebalance_frequency": "ME",
            "feature_windows": [21, 42, 63, 126],
            "correlation_window": 21,
            "label_lookback_days": 126,
            "label_horizons_days": [21, 63, 126],
            "label_horizon_weights": [1.0, 1.0, 1.0],
            "training_lookback_days": 504,
            "min_training_dates": 6,
            "min_label_observations": 63,
            "target_return": 0.0,
            "exposure_penalty": 0.0,
            "pretrain_models": False,
            "retrain_interval_years": 5,
            "vol_target_enabled": False,
            "target_vol_annual": 0.15,
            "vol_lookback_days": 63,
            "vol_max_gross_exposure": 1.0,
            "autogluon_presets": "medium_quality",
            "autogluon_time_limit_sec": 60,
            "autogluon_eval_metric": "mean_squared_error",
            "autogluon_num_bag_folds": 0,
            "autogluon_num_stack_levels": 0,
            "autogluon_cache_models": True,
        }
        for key, value in defaults.items():
            params.setdefault(key, value)

        self._predictor_cache: dict[str, Any] = {}
        self._pretrained = False

    @classmethod
    def tunable_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter metadata used by the optimizer."""
        return {
            "rebalance_frequency": {
                "type": "categorical",
                "choices": ["M", "ME", "BM", "BME", "Q", "QE"],
                "default": "ME",
            },
            "feature_windows": {
                "type": "list",
                "default": [21, 42, 63, 126],
            },
            "correlation_window": {"type": "int", "min": 5, "max": 63, "default": 21},
            "label_lookback_days": {"type": "int", "min": 21, "max": 252, "default": 126},
            "label_horizons_days": {"type": "list", "default": [21, 63, 126]},
            "label_horizon_weights": {"type": "list", "default": [1.0, 1.0, 1.0]},
            "training_lookback_days": {"type": "int", "min": 63, "max": 756, "default": 504},
            "min_training_dates": {"type": "int", "min": 1, "max": 36, "default": 6},
            "min_label_observations": {"type": "int", "min": 5, "max": 252, "default": 63},
            "target_return": {"type": "float", "min": -0.05, "max": 0.05, "default": 0.0},
            "exposure_penalty": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.0},
            "pretrain_models": {"type": "bool", "default": False},
            "retrain_interval_years": {"type": "int", "min": 1, "max": 10, "default": 5},
            "vol_target_enabled": {"type": "bool", "default": False},
            "target_vol_annual": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.15},
            "vol_lookback_days": {"type": "int", "min": 10, "max": 252, "default": 63},
            "vol_max_gross_exposure": {"type": "float", "min": 0.0, "max": 2.0, "default": 1.0},
            "price_column_asset": {"type": "str", "default": "Close"},
            "price_column_benchmark": {"type": "str", "default": "Close"},
            "autogluon_presets": {
                "type": "str",
                "choices": [
                    "medium_quality",
                    "good_quality",
                    "high_quality",
                    "best_quality",
                    "interpretable",
                    "optimize_for_deployment",
                ],
                "default": "medium_quality",
            },
            "autogluon_time_limit_sec": {"type": "int", "min": 30, "max": 600, "default": 60},
            "autogluon_eval_metric": {"type": "str", "default": "mean_squared_error"},
            "autogluon_num_bag_folds": {"type": "int", "min": 0, "max": 10, "default": 0},
            "autogluon_num_stack_levels": {"type": "int", "min": 0, "max": 10, "default": 0},
            "autogluon_cache_models": {"type": "bool", "default": True},
            "trade_longs": {"type": "bool", "default": True},
            "trade_shorts": {"type": "bool", "default": False},
        }

    def get_minimum_required_periods(self) -> int:
        """Estimate minimum months required based on daily lookbacks."""
        params = self._get_params_dict()
        feature_windows = self._ensure_int_list(params.get("feature_windows", [126]))
        label_horizons = self._ensure_int_list(params.get("label_horizons_days", []))
        if not label_horizons:
            label_horizons = [int(params.get("label_lookback_days", 126))]
        max_days = max(
            feature_windows
            + [
                int(params.get("label_lookback_days", 126)),
                max(label_horizons),
                int(params.get("training_lookback_days", 504)),
                int(params.get("vol_lookback_days", 63)),
            ]
        )
        return max(1, int(np.ceil(max_days / 21)))

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: Optional[pd.DataFrame] = None,
        current_date: Optional[pd.Timestamp] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        if current_date is None:
            current_date = all_historical_data.index[-1]

        current_date = pd.Timestamp(current_date)
        params = self._get_params_dict()
        self._sync_timing_config(params.get("rebalance_frequency", "ME"))

        if (start_date and current_date < start_date) or (end_date and current_date > end_date):
            return self._empty_weights(all_historical_data, current_date)

        is_sufficient, _ = self.validate_data_sufficiency(
            all_historical_data, benchmark_historical_data, current_date
        )
        if not is_sufficient:
            return self._empty_weights(all_historical_data, current_date)

        original_assets = self._get_original_assets(all_historical_data)
        valid_assets = self.filter_universe_by_data_availability(all_historical_data, current_date)
        if not valid_assets:
            return self._empty_weights(all_historical_data, current_date, original_assets)

        close_df_all = self._extract_close_prices(
            all_historical_data, params.get("price_column_asset", "Close")
        )
        benchmark_close_all = self._extract_benchmark_close(
            benchmark_historical_data, params.get("price_column_benchmark", "Close")
        )
        if benchmark_close_all.empty:
            logger.warning("Benchmark data missing; using universe average close as proxy.")
            if close_df_all.empty:
                return self._empty_weights(all_historical_data, current_date, original_assets)
            benchmark_close_all = close_df_all.mean(axis=1)

        available_assets = [asset for asset in original_assets if asset in close_df_all.columns]
        if not available_assets:
            return self._empty_weights(all_historical_data, current_date, original_assets)

        valid_assets = [asset for asset in valid_assets if asset in close_df_all.columns]
        if not valid_assets:
            return self._empty_weights(all_historical_data, current_date, original_assets)

        close_df_all = close_df_all.loc[:, available_assets]
        returns_df_all = close_df_all.pct_change(fill_method=None).dropna(how="all")

        close_df = close_df_all.loc[close_df_all.index <= current_date, valid_assets]
        benchmark_close = benchmark_close_all.loc[benchmark_close_all.index <= current_date]

        if close_df.empty or benchmark_close.empty:
            return self._empty_weights(all_historical_data, current_date, original_assets)

        returns_df = close_df.pct_change(fill_method=None).dropna(how="all")

        if bool(params.get("pretrain_models", False)) and not self._pretrained:
            self._pretrain_models(close_df_all, benchmark_close_all, returns_df_all)
            self._pretrained = True

        feature_frame = self._build_feature_frame(
            dates=[current_date],
            close_df=close_df,
            benchmark_close=benchmark_close,
            returns_df=returns_df,
        )
        if feature_frame.empty:
            return self._empty_weights(all_historical_data, current_date, original_assets)

        model_date = self._get_model_anchor_date(current_date, pd.DatetimeIndex(close_df_all.index))
        feature_signature = self._get_feature_signature()
        label_signature = self._get_label_signature()
        universe_signature = self._get_universe_signature(close_df_all)
        predictor = self._get_or_train_predictor(
            model_date,
            feature_signature,
            label_signature,
            universe_signature,
            close_df_all,
            benchmark_close_all,
            returns_df_all,
        )
        if predictor is None:
            return self._empty_weights(all_historical_data, current_date, original_assets)

        inference_data = feature_frame.drop(columns=["date", "target_weight"], errors="ignore")
        predictions = predictor.predict(inference_data)

        predicted_weights = pd.Series(
            predictions.to_numpy(), index=inference_data["symbol"].tolist(), dtype=float
        )
        weights = self._post_process_weights(predicted_weights, returns_df, current_date)

        output = pd.DataFrame(0.0, index=[current_date], columns=original_assets)
        output.loc[current_date, weights.index] = weights
        output = self._enforce_trade_direction_constraints(output)
        return output

    def _get_params_dict(self) -> Dict[str, Any]:
        params_any = self.strategy_config.get("strategy_params", self.strategy_config)
        if params_any is None:
            self.strategy_config["strategy_params"] = {}
            params_any = self.strategy_config["strategy_params"]
        if not isinstance(params_any, dict):
            return {}
        return cast(Dict[str, Any], params_any)

    def _sync_timing_config(self, rebalance_frequency: str) -> None:
        if not isinstance(self.config, dict):
            return
        timing_config = self.config.get("timing_config")
        if not isinstance(timing_config, dict) or timing_config.get("rebalance_frequency") != (
            rebalance_frequency
        ):
            self.config["timing_config"] = {
                "mode": "time_based",
                "rebalance_frequency": rebalance_frequency,
            }

    def _get_original_assets(self, all_historical_data: pd.DataFrame) -> List[str]:
        if isinstance(all_historical_data.columns, pd.MultiIndex):
            return list(all_historical_data.columns.get_level_values("Ticker").unique())
        return list(all_historical_data.columns)

    def _extract_close_prices(self, data: pd.DataFrame, price_column: str) -> pd.DataFrame:
        if isinstance(data.columns, pd.MultiIndex):
            if "Field" in data.columns.names:
                extracted = data.xs(price_column, level="Field", axis=1)
                return extracted.to_frame() if isinstance(extracted, pd.Series) else extracted
            return data
        return data

    def _extract_benchmark_close(
        self, benchmark_data: pd.DataFrame, price_column: str
    ) -> pd.Series:
        if benchmark_data.empty:
            return pd.Series(dtype=float)
        if isinstance(benchmark_data.columns, pd.MultiIndex):
            if "Field" in benchmark_data.columns.names:
                bench = benchmark_data.xs(price_column, level="Field", axis=1)
            else:
                bench = benchmark_data
            if isinstance(bench, pd.DataFrame):
                return bench.iloc[:, 0]
            return bench
        if isinstance(benchmark_data, pd.Series):
            return benchmark_data
        if price_column in benchmark_data.columns:
            return benchmark_data[price_column]
        return benchmark_data.iloc[:, 0]

    def _ensure_int_list(self, value: Any) -> List[int]:
        if isinstance(value, list):
            return [int(v) for v in value]
        return [int(value)]

    def _ensure_float_list(self, value: Any) -> List[float]:
        if isinstance(value, list):
            return [float(v) for v in value]
        return [float(value)]

    def _normalize_label_weights(self, horizons: List[int], weights: List[float]) -> List[float]:
        if not horizons:
            return []
        if len(weights) != len(horizons):
            weights = [1.0] * len(horizons)
        cleaned = [max(0.0, float(weight)) for weight in weights]
        total = float(sum(cleaned))
        if total <= 0.0:
            return [1.0 / len(horizons)] * len(horizons)
        return [weight / total for weight in cleaned]

    def _get_training_dates(
        self,
        current_date: pd.Timestamp,
        available_dates: pd.DatetimeIndex,
        training_lookback_days: int,
    ) -> List[pd.Timestamp]:
        training_start = current_date - pd.Timedelta(days=training_lookback_days)

        timing_controller = self.get_timing_controller()
        timing_state = getattr(timing_controller, "timing_state", None)
        scheduled_dates = getattr(timing_state, "scheduled_dates", None)
        if scheduled_dates:
            return sorted(
                [
                    pd.Timestamp(date)
                    for date in scheduled_dates
                    if training_start <= date < current_date and date in available_dates
                ]
            )

        frequency = self._convert_frequency(
            self._get_params_dict().get("rebalance_frequency", "ME")
        )
        rebalance_dates = (
            available_dates.to_series(index=available_dates)
            .resample(frequency)
            .last()
            .dropna()
            .index
        )
        return [
            pd.Timestamp(date) for date in rebalance_dates if training_start <= date < current_date
        ]

    def _convert_frequency(self, frequency: str) -> str:
        freq = str(frequency).upper()
        if freq == "M":
            return "ME"
        if freq == "Q":
            return "QE"
        if freq in {"A", "Y"}:
            return "YE"
        return frequency

    def _build_feature_frame(
        self,
        dates: List[pd.Timestamp],
        close_df: pd.DataFrame,
        benchmark_close: pd.Series,
        returns_df: pd.DataFrame,
    ) -> pd.DataFrame:
        params = self._get_params_dict()
        feature_windows = self._ensure_int_list(params.get("feature_windows", [21, 42, 63, 126]))
        target_return = float(params.get("target_return", 0.0))

        symbols = list(close_df.columns)

        sortino_frames = self._calculate_sortino_frames(returns_df, feature_windows, target_return)
        return_diff_frames = self._calculate_return_diff_frames(
            close_df, benchmark_close, feature_windows
        )

        records: List[Dict[str, Any]] = []
        for date in dates:
            if date not in close_df.index:
                continue
            for symbol in symbols:
                row: Dict[str, Any] = {"date": date, "symbol": symbol}
                for window in feature_windows:
                    row[f"sortino_{window}d"] = float(
                        sortino_frames[window].get(symbol, pd.Series()).get(date, 0.0)
                    )
                    row[f"ret_diff_{window}d"] = float(
                        return_diff_frames[window].get(symbol, pd.Series()).get(date, 0.0)
                    )
                records.append(row)

        if not records:
            return pd.DataFrame()

        feature_frame = pd.DataFrame.from_records(records)
        return feature_frame.fillna(0.0)

    def _calculate_sortino_frames(
        self,
        returns_df: pd.DataFrame,
        windows: List[int],
        target_return: float,
    ) -> Dict[int, pd.DataFrame]:
        if returns_df.empty:
            return {window: pd.DataFrame() for window in windows}

        returns_np = returns_df.fillna(0.0).to_numpy(dtype=np.float64)
        frames: Dict[int, pd.DataFrame] = {}
        for window in windows:
            if returns_np.shape[0] < window:
                frames[window] = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
                continue
            sortino_mat = rolling_sortino_batch(
                returns_np, window, target_return, annualization_factor=1.0
            )
            frames[window] = pd.DataFrame(
                sortino_mat, index=returns_df.index, columns=returns_df.columns
            ).fillna(0.0)
        return frames

    def _calculate_return_diff_frames(
        self,
        close_df: pd.DataFrame,
        benchmark_close: pd.Series,
        windows: List[int],
    ) -> Dict[int, pd.DataFrame]:
        bench = benchmark_close.reindex(close_df.index).ffill()
        frames: Dict[int, pd.DataFrame] = {}
        for window in windows:
            asset_returns = close_df / close_df.shift(window) - 1.0
            bench_returns = bench / bench.shift(window) - 1.0
            frames[window] = asset_returns.sub(bench_returns, axis=0).fillna(0.0)
        return frames

    def _calculate_corr_matrix(
        self, returns_df: pd.DataFrame, date: pd.Timestamp, window: int
    ) -> Optional[pd.DataFrame]:
        if returns_df.empty or date not in returns_df.index:
            return None
        window_returns = returns_df.loc[:date].tail(window)
        if len(window_returns) < window:
            return None
        return window_returns.corr().fillna(0.0)

    def _sanitize_symbol(self, symbol: str) -> str:
        return "".join(ch if ch.isalnum() else "_" for ch in symbol)

    def _build_label_frame(
        self,
        training_dates: List[pd.Timestamp],
        returns_df: pd.DataFrame,
        min_observations: int,
        label_horizons_days: List[int],
        label_horizon_weights: List[float],
        target_return: float,
        exposure_penalty: float,
    ) -> pd.DataFrame:
        horizons = [int(h) for h in label_horizons_days if int(h) > 0]
        if not horizons:
            return pd.DataFrame()
        normalized_weights = self._normalize_label_weights(horizons, label_horizon_weights)

        records: List[Dict[str, Any]] = []
        for date in training_dates:
            combined = pd.Series(0.0, index=returns_df.columns, dtype=float)
            valid_any = False
            for horizon, weight in zip(horizons, normalized_weights):
                if weight <= 0.0:
                    continue
                forward_window = returns_df.loc[returns_df.index > date].head(horizon)
                required_obs = min(min_observations, horizon)
                if len(forward_window) < required_obs:
                    continue
                horizon_weights = self._optimize_sortino_weights(
                    forward_window, target_return=target_return, exposure_penalty=exposure_penalty
                )
                combined = combined.add(horizon_weights * weight, fill_value=0.0)
                valid_any = True

            if not valid_any:
                continue

            combined = combined.fillna(0.0).clip(lower=0.0)
            total = float(combined.sum())
            if total > 0.0:
                combined = combined / total
            for symbol, weight in combined.items():
                records.append({"date": date, "symbol": symbol, "target_weight": float(weight)})

        if not records:
            return pd.DataFrame()
        return pd.DataFrame.from_records(records)

    def _optimize_sortino_weights(
        self,
        returns_window: pd.DataFrame,
        target_return: float,
        exposure_penalty: float,
    ) -> pd.Series:
        returns_np = returns_window.fillna(0.0).to_numpy(dtype=np.float64)
        n_assets = returns_np.shape[1]
        if n_assets == 0:
            return pd.Series(dtype=float)

        initial = np.full(n_assets, 1.0 / n_assets)
        bounds = [(0.0, 1.0) for _ in range(n_assets)]

        def objective(weights: np.ndarray) -> float:
            portfolio_returns = returns_np @ weights
            mean_ret = float(np.mean(portfolio_returns))
            downside = portfolio_returns[portfolio_returns < target_return]
            if downside.size == 0:
                downside_dev = 1e-8
            else:
                downside_dev = float(np.sqrt(np.mean((downside - target_return) ** 2)))
            sortino = (mean_ret - target_return) / (downside_dev + 1e-8)
            return -sortino

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=cast(Any, bounds),
            constraints=cast(Any, constraints),
        )

        if not result.success or result.x is None:
            logger.debug("Sortino optimization failed: %s", result.message if result else "unknown")
            return pd.Series(initial, index=returns_window.columns, dtype=float)

        weights = np.clip(result.x, 0.0, 1.0)
        return pd.Series(weights, index=returns_window.columns, dtype=float)

    def _get_feature_signature(self) -> str:
        params = self._get_params_dict()
        feature_windows = self._ensure_int_list(params.get("feature_windows", [21, 42, 63, 126]))
        target_return = float(params.get("target_return", 0.0))
        payload = f"sortino:{feature_windows}|ret_diff:{feature_windows}|target:{target_return}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]

    def _get_label_signature(self) -> str:
        params = self._get_params_dict()
        horizons = self._ensure_int_list(params.get("label_horizons_days", []))
        if not horizons:
            horizons = [int(params.get("label_lookback_days", 126))]
        weights = self._ensure_float_list(params.get("label_horizon_weights", []))
        normalized = self._normalize_label_weights(horizons, weights)
        payload = (
            f"horizons:{horizons}|weights:{normalized}|minobs:{params.get('min_label_observations', 63)}|"
            f"target:{params.get('target_return', 0.0)}|constraint:sum_eq_1"
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]

    def _get_universe_signature(self, close_df: pd.DataFrame) -> str:
        columns = [str(col) for col in close_df.columns]
        payload = "|".join(sorted(columns))
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]

    def _get_model_cache_key(
        self,
        model_date: pd.Timestamp,
        feature_signature: str,
        label_signature: str,
        universe_signature: str,
    ) -> str:
        params = self._get_params_dict()
        preset = str(params.get("autogluon_presets", "medium_quality"))
        time_limit = str(params.get("autogluon_time_limit_sec", 60))
        retrain_years = str(params.get("retrain_interval_years", 5))
        config_signature = hashlib.sha1(
            f"{preset}|{time_limit}|{retrain_years}".encode("utf-8")
        ).hexdigest()[:6]
        return (
            f"{model_date.strftime('%Y-%m-%d')}_{config_signature}_"
            f"{feature_signature}_{label_signature}_{universe_signature}"
        )

    def _get_model_anchor_date(
        self, current_date: pd.Timestamp, available_dates: pd.DatetimeIndex
    ) -> pd.Timestamp:
        params = self._get_params_dict()
        interval_years = int(params.get("retrain_interval_years", 5))
        if interval_years <= 0 or available_dates.empty:
            return current_date

        start_date = pd.Timestamp(available_dates.min())
        min_history_days = int(params.get("training_lookback_days", 504))
        min_history_idx = min(max(min_history_days, 0), len(available_dates) - 1)
        min_history_date = pd.Timestamp(available_dates[min_history_idx])
        start_year = start_date.year
        block_index = max(0, (current_date.year - start_year) // interval_years)
        block_start_year = start_year + block_index * interval_years
        anchor = pd.Timestamp(year=block_start_year, month=1, day=1)
        if available_dates.tz is not None:
            anchor = anchor.tz_localize(available_dates.tz)

        if anchor < min_history_date:
            anchor = min_history_date

        candidates = available_dates[
            (available_dates >= anchor) & (available_dates <= current_date)
        ]
        if candidates.empty:
            candidates = available_dates[available_dates <= current_date]
        return pd.Timestamp(candidates[0]) if not candidates.empty else current_date

    def _compute_model_dates(self, available_dates: pd.DatetimeIndex) -> List[pd.Timestamp]:
        params = self._get_params_dict()
        interval_years = int(params.get("retrain_interval_years", 5))
        if interval_years <= 0 or available_dates.empty:
            return []

        start_date = pd.Timestamp(available_dates.min())
        min_history_days = int(params.get("training_lookback_days", 504))
        min_history_idx = min(max(min_history_days, 0), len(available_dates) - 1)
        min_history_date = pd.Timestamp(available_dates[min_history_idx])
        end_date = pd.Timestamp(available_dates.max())
        dates: List[pd.Timestamp] = []
        for year in range(start_date.year, end_date.year + 1, interval_years):
            anchor = pd.Timestamp(year=year, month=1, day=1)
            if available_dates.tz is not None:
                anchor = anchor.tz_localize(available_dates.tz)
            if anchor < min_history_date:
                continue
            candidates = available_dates[available_dates >= anchor]
            if candidates.empty:
                continue
            dates.append(pd.Timestamp(candidates[0]))

        unique_dates = sorted(set(dates))
        return unique_dates

    def _build_training_data_for_date(
        self,
        model_date: pd.Timestamp,
        close_df: pd.DataFrame,
        benchmark_close: pd.Series,
        returns_df: pd.DataFrame,
    ) -> pd.DataFrame:
        params = self._get_params_dict()
        close_df = close_df.loc[close_df.index <= model_date]
        benchmark_close = benchmark_close.loc[benchmark_close.index <= model_date]
        returns_df = returns_df.loc[returns_df.index <= model_date]
        training_dates = self._get_training_dates(
            current_date=model_date,
            available_dates=pd.DatetimeIndex(close_df.index),
            training_lookback_days=int(params.get("training_lookback_days", 504)),
        )
        if len(training_dates) < int(params.get("min_training_dates", 6)):
            return pd.DataFrame()

        feature_dates = training_dates + [model_date]
        feature_frame = self._build_feature_frame(
            dates=feature_dates,
            close_df=close_df,
            benchmark_close=benchmark_close,
            returns_df=returns_df,
        )
        if feature_frame.empty:
            return pd.DataFrame()

        labels = self._build_label_frame(
            training_dates=training_dates,
            returns_df=returns_df,
            min_observations=int(params.get("min_label_observations", 63)),
            label_horizons_days=self._ensure_int_list(
                params.get("label_horizons_days", [int(params.get("label_lookback_days", 126))])
            ),
            label_horizon_weights=self._ensure_float_list(params.get("label_horizon_weights", [])),
            target_return=float(params.get("target_return", 0.0)),
            exposure_penalty=float(params.get("exposure_penalty", 0.01)),
        )
        if labels.empty:
            return pd.DataFrame()

        training_data = feature_frame.merge(
            labels, on=["date", "symbol"], how="inner", validate="many_to_one"
        )
        if training_data.empty:
            return pd.DataFrame()

        return training_data.drop(columns=["date"], errors="ignore")

    def _pretrain_models(
        self,
        close_df: pd.DataFrame,
        benchmark_close: pd.Series,
        returns_df: pd.DataFrame,
    ) -> None:
        feature_signature = self._get_feature_signature()
        label_signature = self._get_label_signature()
        universe_signature = self._get_universe_signature(close_df)
        model_dates = self._compute_model_dates(pd.DatetimeIndex(close_df.index))
        for model_date in model_dates:
            self._get_or_train_predictor(
                model_date,
                feature_signature,
                label_signature,
                universe_signature,
                close_df,
                benchmark_close,
                returns_df,
            )

    def _get_or_train_predictor(
        self,
        model_date: pd.Timestamp,
        feature_signature: str,
        label_signature: str,
        universe_signature: str,
        close_df: pd.DataFrame,
        benchmark_close: pd.Series,
        returns_df: pd.DataFrame,
    ):
        params = self._get_params_dict()
        cache_key = self._get_model_cache_key(
            model_date, feature_signature, label_signature, universe_signature
        )
        if cache_key in self._predictor_cache:
            return self._predictor_cache[cache_key]

        model_path = self._get_model_cache_path(cache_key)
        if params.get("autogluon_cache_models", True) and model_path.exists():
            predictor = self._load_predictor(model_path)
            if predictor is not None:
                self._predictor_cache[cache_key] = predictor
                return predictor

        training_data = self._build_training_data_for_date(
            model_date, close_df, benchmark_close, returns_df
        )
        if training_data.empty:
            return None

        try:
            predictor = self._train_predictor(training_data, model_path)
        except ImportError as exc:
            logger.error("AutoGluon is not available: %s", exc)
            return None
        if predictor is not None:
            self._predictor_cache[cache_key] = predictor
        return predictor

    def _get_model_cache_path(self, cache_key: str) -> Path:
        base_dir = Path(__file__).resolve().parents[5] / ".strategy_cache" / "autogluon_models"
        return base_dir / self.__class__.__name__ / cache_key

    def _load_predictor(self, model_path: Path):
        predictor_cls = self._get_predictor_cls()
        try:
            return predictor_cls.load(str(model_path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load AutoGluon model from %s: %s", model_path, exc)
            return None

    def _train_predictor(self, training_data: pd.DataFrame, model_path: Path):
        predictor_cls = self._get_predictor_cls()
        params = self._get_params_dict()

        training_data = training_data.copy()
        training_data["symbol"] = training_data["symbol"].astype("category")
        label_column = "target_weight"

        predictor = predictor_cls(
            label=label_column,
            path=str(model_path),
            problem_type="regression",
            eval_metric=params.get("autogluon_eval_metric", "mean_squared_error"),
        )

        try:
            predictor.fit(
                training_data,
                time_limit=int(params.get("autogluon_time_limit_sec", 60)),
                presets=params.get("autogluon_presets", "medium_quality"),
                num_bag_folds=int(params.get("autogluon_num_bag_folds", 0)),
                num_stack_levels=int(params.get("autogluon_num_stack_levels", 0)),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("AutoGluon fit failed: %s", exc, exc_info=True)
            return None

        return predictor

    def _get_predictor_cls(self):
        try:
            from autogluon.tabular import TabularPredictor
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                "AutoGluon is required for AutogluonSortinoMlPortfolioStrategy. "
                "Install dependencies with: .venv\\Scripts\\python.exe -m pip install -e .[dev]"
            ) from exc
        return TabularPredictor

    def _post_process_weights(
        self,
        weights: pd.Series,
        returns_df: Optional[pd.DataFrame] = None,
        current_date: Optional[pd.Timestamp] = None,
    ) -> pd.Series:
        cleaned = weights.fillna(0.0).clip(lower=0.0)
        if cleaned.empty:
            return cleaned

        total = float(cleaned.sum())
        if total > 0.0:
            cleaned = cleaned / total
        cleaned = self._apply_vol_target(cleaned, returns_df, current_date)
        return cleaned

    def _apply_vol_target(
        self,
        weights: pd.Series,
        returns_df: Optional[pd.DataFrame],
        current_date: Optional[pd.Timestamp],
    ) -> pd.Series:
        params = self._get_params_dict()
        if not bool(params.get("vol_target_enabled", False)):
            return weights

        if returns_df is None or returns_df.empty or current_date is None:
            return weights

        target_vol_annual = float(params.get("target_vol_annual", 0.10))
        vol_lookback_days = int(params.get("vol_lookback_days", 63))
        max_gross = float(params.get("vol_max_gross_exposure", 1.0))
        if target_vol_annual <= 0 or max_gross <= 0:
            return weights

        returns_hist = returns_df.loc[returns_df.index < current_date]
        if returns_hist.empty:
            return weights

        returns_hist = returns_hist.tail(vol_lookback_days)
        returns_hist = returns_hist.reindex(columns=weights.index)
        portfolio_rets = returns_hist.fillna(0.0).mul(weights, axis=1).sum(axis=1)

        vol_annual = self._annualized_vol_from_returns(portfolio_rets)
        if vol_annual is None or vol_annual <= 0:
            return weights

        gross = float(weights.sum())
        if gross <= 0:
            return weights

        scale = target_vol_annual / max(vol_annual, 1e-12)
        max_scale = max_gross / gross
        scale = float(np.clip(scale, 0.0, max_scale))
        if not np.isfinite(scale) or scale <= 0:
            return weights

        return weights * scale

    @staticmethod
    def _annualized_vol_from_returns(returns: pd.Series) -> Optional[float]:
        clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) < 2:
            return None
        vol = float(clean.std(ddof=1))
        if not np.isfinite(vol) or vol <= 0:
            return None
        return vol * float(np.sqrt(252.0))

    def _empty_weights(
        self,
        all_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
        assets: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        if assets is None:
            assets = self._get_original_assets(all_historical_data)
        output = pd.DataFrame(0.0, index=[current_date], columns=list(assets))
        return self._enforce_trade_direction_constraints(output)


__all__ = ["AutogluonSortinoMlPortfolioStrategy"]
