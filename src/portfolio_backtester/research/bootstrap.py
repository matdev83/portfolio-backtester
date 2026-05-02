"""Bootstrap significance helpers for research_validate (post-selection only)."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from portfolio_backtester.research.cost_sensitivity import survival_metric_for_selection
from portfolio_backtester.research.protocol_config import BootstrapConfig
from portfolio_backtester.research.results import (
    SelectedProtocol,
    UnseenValidationResult,
    WFOArchitectureResult,
)

logger = logging.getLogger(__name__)


def block_shuffle_returns_preserving_blocks(
    returns: pd.Series,
    *,
    block_size_days: int,
    rng: np.random.Generator,
) -> pd.Series:
    """Shuffle contiguous return blocks in-place order, preserving length and index."""

    if returns is None or returns.empty:
        return returns.copy() if returns is not None else pd.Series(dtype=float)
    idx = returns.index
    vals = np.asarray(returns.to_numpy(dtype=float), dtype=float)
    n = int(vals.size)
    bs = max(1, int(block_size_days))
    n_blocks = (n + bs - 1) // bs
    blocks: list[np.ndarray] = [vals[i * bs : min((i + 1) * bs, n)] for i in range(n_blocks)]
    order = rng.permutation(n_blocks)
    shuffled = np.concatenate([blocks[int(j)] for j in order])
    return pd.Series(shuffled[:n], index=idx, dtype=float)


def block_shuffle_weights_preserving_blocks(
    weights: pd.DataFrame,
    *,
    block_size_days: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Shuffle contiguous calendar blocks of rows, preserving length and index."""

    if weights is None or weights.empty:
        return weights.copy() if weights is not None else pd.DataFrame()
    idx = weights.index
    vals = np.asarray(weights.to_numpy(dtype=float), dtype=float)
    n_rows, n_cols = vals.shape
    bs = max(1, int(block_size_days))
    n_blocks = (n_rows + bs - 1) // bs
    blocks: list[np.ndarray] = [
        vals[i * bs : min((i + 1) * bs, n_rows), :] for i in range(n_blocks)
    ]
    order = rng.permutation(n_blocks)
    shuffled = np.vstack([blocks[int(j)] for j in order])
    trimmed = shuffled[:n_rows, :]
    return pd.DataFrame(trimmed, index=idx, columns=weights.columns)


def _trade_history_weight_column(trade_history: pd.DataFrame) -> str | None:
    cols = set(trade_history.columns)
    if "weight" in cols:
        return "weight"
    if "position" in cols:
        return "position"
    return None


def weights_daily_from_trade_history(
    trade_history: pd.DataFrame,
    asset_returns: pd.DataFrame,
    *,
    weight_column: str,
) -> pd.DataFrame:
    """Forward-filled daily weights aligned to ``asset_returns`` index/columns."""

    idx = asset_returns.index
    tickers = [str(c) for c in asset_returns.columns]
    th = trade_history.copy()
    th["date"] = pd.to_datetime(th["date"], errors="coerce").dt.normalize()
    th["ticker"] = th["ticker"].astype(str)
    th = th.dropna(subset=["date"])
    wide = th.pivot_table(
        index="date",
        columns="ticker",
        values=weight_column,
        aggfunc="sum",
    )
    wide.columns = [str(c) for c in wide.columns]
    aligned = wide.reindex(columns=tickers, fill_value=0.0)
    daily = aligned.reindex(idx).ffill().fillna(0.0)
    return daily


def portfolio_returns_from_weights(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
) -> pd.Series:
    """Daily portfolio returns via row-wise dot product."""

    w = weights.reindex(index=asset_returns.index).fillna(0.0)
    r = asset_returns.reindex(columns=w.columns, fill_value=0.0)
    combined = np.asarray(w.to_numpy(dtype=float) * r.to_numpy(dtype=float))
    vals = np.asarray(combined.sum(axis=1), dtype=float)
    return pd.Series(vals, index=w.index, dtype=float)


def bootstrap_block_shuffled_positions_p_value(
    *,
    trade_history: pd.DataFrame | None,
    asset_returns: pd.DataFrame | None,
    returns_fallback: pd.Series | None,
    n_samples: int,
    block_size_days: int,
    survival_metric: str,
    observed_value: float,
    metrics_from_returns: Callable[[pd.Series], Mapping[str, float]],
    rng: np.random.Generator,
    collect_samples: list[float] | None = None,
) -> float:
    """One-sided empirical p-value from block-shuffled position paths."""

    if trade_history is None or trade_history.empty:
        logger.warning(
            "block_shuffled_positions bootstrap skipped: missing or empty trade_history "
            "(p_value unavailable)."
        )
        return float("nan")
    if asset_returns is None or asset_returns.empty:
        logger.warning(
            "block_shuffled_positions bootstrap skipped: missing or empty asset_returns "
            "(p_value unavailable)."
        )
        return float("nan")
    if n_samples <= 0:
        return float("nan")
    if math.isnan(float(observed_value)):
        return float("nan")

    wc = _trade_history_weight_column(trade_history)
    if wc is None:
        # Without usable weights, block-shuffling positions is undefined; block-shuffled *returns*
        # on the same unseen window is the documented proxy (path dependence under zero info).
        logger.warning(
            "block_shuffled_positions: trade_history has neither weight nor position column; "
            "falling back to block_shuffled_returns on returns_fallback."
        )
        if returns_fallback is None or returns_fallback.empty:
            return float("nan")
        return bootstrap_block_shuffled_returns_p_value(
            returns=returns_fallback,
            n_samples=n_samples,
            block_size_days=block_size_days,
            survival_metric=survival_metric,
            observed_value=observed_value,
            metrics_from_returns=metrics_from_returns,
            rng=rng,
            collect_samples=collect_samples,
        )

    weights_base = weights_daily_from_trade_history(
        trade_history,
        asset_returns,
        weight_column=wc,
    )
    obs = float(observed_value)
    count = 0
    for _ in range(int(n_samples)):
        w_shuf = block_shuffle_weights_preserving_blocks(
            weights_base,
            block_size_days=block_size_days,
            rng=rng,
        )
        port = portfolio_returns_from_weights(w_shuf, asset_returns)
        m = metrics_from_returns(port)
        raw = m.get(survival_metric)
        if raw is None:
            v = float("nan")
        else:
            try:
                v = float(raw)
            except (TypeError, ValueError):
                v = float("nan")
        if collect_samples is not None and not math.isnan(v):
            collect_samples.append(float(v))
        if not math.isnan(v) and v >= obs:
            count += 1
    return float(count) / float(n_samples)


def bootstrap_random_wfo_architecture_p_value(
    *,
    eligible_scores: Sequence[float],
    selected_score: float,
    n_samples: int,
    rng: np.random.Generator,
    collect_samples: list[float] | None = None,
) -> float:
    """Return empirical p-value: fraction of bootstrap draws >= ``selected_score``."""

    pool = [float(x) for x in eligible_scores if not math.isnan(float(x))]
    if not pool or n_samples <= 0:
        return float("nan")
    arr = np.asarray(pool, dtype=float)
    draws = rng.choice(arr, size=int(n_samples), replace=True)
    if collect_samples is not None:
        collect_samples.extend(float(x) for x in draws.tolist())
    ss = float(selected_score)
    return float(np.mean(draws >= ss))


def bootstrap_block_shuffled_returns_p_value(
    *,
    returns: pd.Series,
    n_samples: int,
    block_size_days: int,
    survival_metric: str,
    observed_value: float,
    metrics_from_returns: Callable[[pd.Series], Mapping[str, float]],
    rng: np.random.Generator,
    collect_samples: list[float] | None = None,
) -> float:
    """Return one-sided empirical p-value for shuffled paths vs observed survival metric."""

    if returns is None or returns.empty or n_samples <= 0:
        return float("nan")
    if math.isnan(float(observed_value)):
        return float("nan")
    obs = float(observed_value)
    count = 0
    for _ in range(int(n_samples)):
        sh = block_shuffle_returns_preserving_blocks(
            returns,
            block_size_days=block_size_days,
            rng=rng,
        )
        m = metrics_from_returns(sh)
        raw = m.get(survival_metric)
        if raw is None:
            v = float("nan")
        else:
            try:
                v = float(raw)
            except (TypeError, ValueError):
                v = float("nan")
        if collect_samples is not None and not math.isnan(v):
            collect_samples.append(float(v))
        if not math.isnan(v) and v >= obs:
            count += 1
    return float(count) / float(n_samples)


def bootstrap_random_strategy_parameters_p_value(
    *,
    selected_unseen_survival_value: float,
    n_samples: int,
    sample_size: int,
    param_space: Mapping[str, list[Any]] | None,
    run_with_params_fn: Callable[[Mapping[str, Any]], Mapping[str, float]] | None,
    survival_metric: str,
    rng: np.random.Generator,
    collect_samples: list[float] | None = None,
) -> float:
    """Return fraction of draws where survival metric >= selected unseen survival value."""

    if param_space is None or len(param_space) == 0:
        return float("nan")
    if run_with_params_fn is None:
        return float("nan")
    if sample_size <= 0 or n_samples <= 0:
        return float("nan")
    if math.isnan(float(selected_unseen_survival_value)):
        return float("nan")
    obs = float(selected_unseen_survival_value)
    count = 0
    keys = list(param_space.keys())
    for _ in range(int(sample_size)):
        sampled: dict[str, Any] = {}
        for k in keys:
            vals = param_space[k]
            sampled[k] = rng.choice(list(vals))
        m = run_with_params_fn(sampled)
        raw = m.get(survival_metric)
        if raw is None:
            v = float("nan")
        else:
            try:
                v = float(raw)
            except (TypeError, ValueError):
                v = float("nan")
        if collect_samples is not None and not math.isnan(v):
            collect_samples.append(float(v))
        if not math.isnan(v) and v >= obs:
            count += 1
    return float(count) / float(sample_size)


def run_research_bootstrap(
    *,
    cfg: BootstrapConfig,
    grid_results: Sequence[WFOArchitectureResult],
    selected: SelectedProtocol,
    unseen_result: UnseenValidationResult | None,
    selection_metric: str,
    metrics_from_returns: Callable[[pd.Series], Mapping[str, float]],
    param_space: Mapping[str, list[Any]] | None = None,
    run_with_params_fn: Callable[[Mapping[str, Any]], Mapping[str, float]] | None = None,
    trade_history: pd.DataFrame | None = None,
    asset_returns: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[float]] | None] | None:
    """Compute bootstrap summaries; returns ``None`` when skipped."""

    if not cfg.enabled or unseen_result is None:
        return None
    run_any = (
        cfg.random_wfo_architecture.enabled
        or cfg.block_shuffled_returns.enabled
        or cfg.block_shuffled_positions.enabled
        or cfg.random_strategy_parameters.enabled
    )
    if not run_any:
        return None

    rng = np.random.default_rng(int(cfg.random_seed))
    survival = survival_metric_for_selection(selection_metric)
    selected_score = float(selected.score)
    persist_samples = cfg.persist_distribution_samples

    rwfo_samples: list[float] | None = (
        [] if persist_samples and cfg.random_wfo_architecture.enabled else None
    )
    br_samples: list[float] | None = (
        [] if persist_samples and cfg.block_shuffled_returns.enabled else None
    )
    bp_samples: list[float] | None = (
        [] if persist_samples and cfg.block_shuffled_positions.enabled else None
    )
    rsp_samples: list[float] | None = (
        [] if persist_samples and cfg.random_strategy_parameters.enabled else None
    )

    th_eff = trade_history
    if th_eff is None:
        th_eff = unseen_result.trade_history

    p_wfo: float | None = None
    if cfg.random_wfo_architecture.enabled:
        pool = [
            float(r.score)
            for r in grid_results
            if r.constraint_passed and not math.isnan(float(r.score))
        ]
        if pool:
            p_wfo = bootstrap_random_wfo_architecture_p_value(
                eligible_scores=pool,
                selected_score=selected_score,
                n_samples=int(cfg.n_samples),
                rng=rng,
                collect_samples=rwfo_samples,
            )

    obs_raw = unseen_result.metrics.get(survival)
    try:
        observed_survival = float(obs_raw) if obs_raw is not None else float("nan")
    except (TypeError, ValueError):
        observed_survival = float("nan")

    p_block: float | None = None
    block_days: int | None = None
    if cfg.block_shuffled_returns.enabled:
        block_days = int(cfg.block_shuffled_returns.block_size_days)
        rets = unseen_result.returns
        if rets is not None and not rets.empty and not math.isnan(observed_survival):
            p_block = bootstrap_block_shuffled_returns_p_value(
                returns=rets,
                n_samples=int(cfg.n_samples),
                block_size_days=block_days,
                survival_metric=survival,
                observed_value=observed_survival,
                metrics_from_returns=metrics_from_returns,
                rng=rng,
                collect_samples=br_samples,
            )

    p_pos: float | None = None
    pos_block_days: int | None = None
    if cfg.block_shuffled_positions.enabled:
        pos_block_days = int(cfg.block_shuffled_positions.block_size_days)
        if math.isnan(observed_survival):
            pass
        elif th_eff is None or th_eff.empty:
            logger.warning(
                "block_shuffled_positions enabled but trade_history is missing or empty; "
                "skipping position bootstrap."
            )
        elif asset_returns is None or asset_returns.empty:
            logger.warning(
                "block_shuffled_positions enabled but asset_returns is missing or empty; "
                "skipping position bootstrap."
            )
        else:
            p_pos = bootstrap_block_shuffled_positions_p_value(
                trade_history=th_eff,
                asset_returns=asset_returns,
                returns_fallback=unseen_result.returns,
                n_samples=int(cfg.n_samples),
                block_size_days=pos_block_days,
                survival_metric=survival,
                observed_value=observed_survival,
                metrics_from_returns=metrics_from_returns,
                rng=rng,
                collect_samples=bp_samples,
            )

    p_rsp: float | None = None
    rsp_sample_size: int | None = None
    if cfg.random_strategy_parameters.enabled:
        rsp_sample_size = int(cfg.random_strategy_parameters.sample_size)
        if (
            param_space is not None
            and len(param_space) > 0
            and run_with_params_fn is not None
            and not math.isnan(observed_survival)
        ):
            p_rsp = bootstrap_random_strategy_parameters_p_value(
                selected_unseen_survival_value=observed_survival,
                n_samples=int(cfg.n_samples),
                sample_size=rsp_sample_size,
                param_space=param_space,
                run_with_params_fn=run_with_params_fn,
                survival_metric=survival,
                rng=rng,
                collect_samples=rsp_samples,
            )

    summary: dict[str, Any] = {
        "enabled": True,
        "n_samples": int(cfg.n_samples),
        "random_seed": int(cfg.random_seed),
        "report_order": "unseen_validation, cost_sensitivity_when_enabled, bootstrap_significance",
        "selected_protocol_primary_score": selected_score,
        "survival_metric": survival,
        "observed_survival_metric_unseen": (
            observed_survival if not math.isnan(observed_survival) else None
        ),
        "random_wfo_architecture": {
            "enabled": bool(cfg.random_wfo_architecture.enabled),
            "p_value": p_wfo,
        },
        "block_shuffled_returns": {
            "enabled": bool(cfg.block_shuffled_returns.enabled),
            "p_value": p_block,
            "block_size_days": block_days,
        },
        "block_shuffled_positions": {
            "enabled": bool(cfg.block_shuffled_positions.enabled),
            "p_value": p_pos,
            "block_size_days": pos_block_days,
        },
        "random_strategy_parameters": {
            "enabled": bool(cfg.random_strategy_parameters.enabled),
            "p_value": p_rsp,
            "sample_size": rsp_sample_size,
        },
    }

    rows: list[dict[str, Any]] = []
    if cfg.random_wfo_architecture.enabled:
        rows.append(
            {
                "test": "random_wfo_architecture",
                "p_value": "" if p_wfo is None or math.isnan(p_wfo) else float(p_wfo),
                "n_samples": int(cfg.n_samples),
                "block_size_days": "",
                "selected_protocol_primary_score": selected_score,
                "survival_metric": "",
                "observed_survival_metric_unseen": "",
            }
        )
    if cfg.block_shuffled_returns.enabled:
        rows.append(
            {
                "test": "block_shuffled_returns",
                "p_value": "" if p_block is None or math.isnan(p_block) else float(p_block),
                "n_samples": int(cfg.n_samples),
                "block_size_days": int(cfg.block_shuffled_returns.block_size_days),
                "selected_protocol_primary_score": "",
                "survival_metric": survival,
                "observed_survival_metric_unseen": (
                    observed_survival if not math.isnan(observed_survival) else ""
                ),
            }
        )
    if cfg.block_shuffled_positions.enabled:
        rows.append(
            {
                "test": "block_shuffled_positions",
                "p_value": "" if p_pos is None or math.isnan(p_pos) else float(p_pos),
                "n_samples": int(cfg.n_samples),
                "block_size_days": int(cfg.block_shuffled_positions.block_size_days),
                "selected_protocol_primary_score": "",
                "survival_metric": survival,
                "observed_survival_metric_unseen": (
                    observed_survival if not math.isnan(observed_survival) else ""
                ),
            }
        )
    if cfg.random_strategy_parameters.enabled:
        rsp_n = (
            int(rsp_sample_size)
            if rsp_sample_size is not None
            else int(cfg.random_strategy_parameters.sample_size)
        )
        rows.append(
            {
                "test": "random_strategy_parameters",
                "p_value": "" if p_rsp is None or math.isnan(p_rsp) else float(p_rsp),
                "n_samples": rsp_n,
                "block_size_days": "",
                "selected_protocol_primary_score": "",
                "survival_metric": survival,
                "observed_survival_metric_unseen": (
                    observed_survival if not math.isnan(observed_survival) else ""
                ),
            }
        )
    distributions: dict[str, list[float]] | None = None
    if persist_samples:
        distributions = {}
        if rwfo_samples is not None and rwfo_samples:
            distributions["random_wfo_architecture"] = rwfo_samples
        if br_samples is not None and br_samples:
            distributions["block_shuffled_returns"] = br_samples
        if bp_samples is not None and bp_samples:
            distributions["block_shuffled_positions"] = bp_samples
        if rsp_samples is not None and rsp_samples:
            distributions["random_strategy_parameters"] = rsp_samples
        if not distributions:
            distributions = {}
    return summary, rows, distributions


def write_bootstrap_distribution_artifacts(
    run_dir: Path | str,
    samples_by_test: Mapping[str, Sequence[float]] | None,
) -> None:
    """Write optional ``bootstrap_distribution_<test>.csv`` files alongside bootstrap summaries."""

    if not samples_by_test:
        return
    root = Path(run_dir)
    for key, seq in samples_by_test.items():
        vals = []
        for x in seq:
            try:
                fx = float(x)
            except (TypeError, ValueError):
                continue
            if not math.isnan(fx):
                vals.append(fx)
        if not vals:
            continue
        df_out = pd.DataFrame({"value": vals})
        df_out.to_csv(root / f"bootstrap_distribution_{key}.csv", index=False)


def write_bootstrap_artifacts(
    run_dir: Path | str,
    summary: Mapping[str, Any],
    csv_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write ``bootstrap_summary.yaml`` and ``bootstrap_significance.csv``."""

    root = Path(run_dir)
    df = pd.DataFrame(list(csv_rows))
    df.to_csv(root / "bootstrap_significance.csv", index=False)
    text = yaml.safe_dump(dict(summary), sort_keys=False, allow_unicode=True)
    (root / "bootstrap_summary.yaml").write_text(text, encoding="utf-8")
