r"""
Walk-forward evaluation of ASRI signal tradeability.

Goals:
- Compare statistically sound rolling signal mappings vs rolling min-max.
- Evaluate whether ASRI adds value beyond price-only baselines (trailing realized vol).
"""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import market_data_multi_provider as mdm
from market_data_multi_provider.indicators.asri_core import (
    AsriConfig,
    AsriRawInputs,
    compute_asri,
)


DEFILLAMA_SYMBOLS = [
    "DEFILLAMA:DEFI_TOTAL_TVL_USD",
    "DEFILLAMA:STABLECOIN_TOTAL_SUPPLY_USD",
    "DEFILLAMA:STABLECOIN_HHI_RAW",
    "DEFILLAMA:STABLECOIN_TOP2_SHARE_PCT",
    "DEFILLAMA:STABLECOIN_SIG_COUNT",
    "DEFILLAMA:STABLECOIN_WEIGHTED_ABS_DEVIATION",
    "DEFILLAMA:PROTOCOL_TOP10_HHI_RAW",
    "DEFILLAMA:PROTOCOL_AUDIT_COVERAGE_PCT",
    "DEFILLAMA:PROTOCOL_MEAN_ABS_CHANGE_1D_PCT",
    "DEFILLAMA:PROTOCOL_LENDING_SHARE_PCT",
    "DEFILLAMA:PROTOCOL_RWA_SHARE_PCT",
    "DEFILLAMA:BRIDGES_ACTIVE_COUNT",
]


def _parse_date(s: str) -> dt.date:
    try:
        return dt.date.fromisoformat(s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date {s!r}; expected YYYY-MM-DD.") from exc


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Walk-forward evaluation of ASRI signal.")
    p.add_argument(
        "--start",
        type=_parse_date,
        default=None,
        help="Optional start date YYYY-MM-DD (default: earliest available).",
    )
    p.add_argument(
        "--end",
        type=_parse_date,
        default=None,
        help="Optional end date YYYY-MM-DD (default: latest available).",
    )
    p.add_argument(
        "--train-days",
        type=int,
        default=730,
        help="Training window length in days (default: 730).",
    )
    p.add_argument(
        "--val-days",
        type=int,
        default=180,
        help="Validation window length in days inside training (default: 180).",
    )
    p.add_argument(
        "--test-days",
        type=int,
        default=90,
        help="Test window length in days per fold (default: 90).",
    )
    p.add_argument(
        "--step-days",
        type=int,
        default=90,
        help="Step size in days between folds (default: 90).",
    )
    p.add_argument(
        "--cost-bps",
        type=float,
        default=10.0,
        help="Transaction cost in bps per 1.0 weight turnover (default: 10).",
    )
    p.add_argument(
        "--rv-window-days",
        type=int,
        default=30,
        help="Trailing realized-vol window in days (default: 30).",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output markdown path for results table.",
    )
    return p


def _ensure_naive_daily_index(idx: pd.Index) -> pd.DatetimeIndex:
    di = pd.to_datetime(idx, errors="coerce", utc=True)
    if isinstance(di, pd.DatetimeIndex) and di.tz is not None:
        di = di.tz_localize(None)
    return pd.DatetimeIndex(di).normalize()


def _extract_series(df: pd.DataFrame | None, *, name: str) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="float64", name=name)
    col = None
    for cand in ("Close", "close", "Value", "value", "Adj_close", "adj_close"):
        if cand in df.columns:
            col = cand
            break
    if col is None:
        col = df.columns[0]
    s = pd.to_numeric(df[col], errors="coerce")
    s.index = _ensure_naive_daily_index(s.index)
    s.name = name
    return s.sort_index()


def _annualized_metrics(r: pd.Series) -> dict[str, float]:
    r = pd.to_numeric(r, errors="coerce").dropna()
    if r.empty:
        return {"cagr": float("nan"), "vol": float("nan"), "sharpe": float("nan"), "max_dd": float("nan")}
    eq = (1.0 + r).cumprod()
    dd = eq / eq.cummax() - 1.0
    cagr = float(eq.iloc[-1] ** (365.0 / len(eq)) - 1.0)
    vol = float(r.std() * np.sqrt(365.0))
    sharpe = float((r.mean() / r.std()) * np.sqrt(365.0)) if r.std() > 0 else float("nan")
    max_dd = float(dd.min())
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "max_dd": max_dd}


def _portfolio_returns(
    *,
    returns: pd.Series,
    weights: pd.Series,
    cost_bps: float,
) -> pd.Series:
    """Compute daily portfolio returns with 1-day execution lag and turnover costs."""
    r = pd.to_numeric(returns, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    df = pd.concat({"r": r, "w": w}, axis=1).dropna()
    if df.empty:
        return pd.Series(dtype="float64")
    cost = float(cost_bps) / 10_000.0
    w_prev = df["w"].shift(1)
    turn_prev = df["w"].diff().abs().shift(1).fillna(0.0)
    out = (w_prev * df["r"] - cost * turn_prev).dropna()
    out.name = "portfolio_return"
    return out


def _weight_vol_target(rv_trailing_pct: pd.Series, *, target_vol_pct: float) -> pd.Series:
    rv = pd.to_numeric(rv_trailing_pct, errors="coerce")
    w = (float(target_vol_pct) / rv).clip(lower=0.0, upper=1.0)
    return w.rename("w_vol")


def _weight_threshold(signal_0_100: pd.Series, *, threshold: float) -> pd.Series:
    sig = pd.to_numeric(signal_0_100, errors="coerce")
    w = (sig < float(threshold)).astype("float64")
    return w.rename("w_thr")


def _weight_logistic(signal_0_100: pd.Series, *, center: float, slope: float) -> pd.Series:
    sig = pd.to_numeric(signal_0_100, errors="coerce")
    x = float(slope) * (sig - float(center))
    w = 1.0 / (1.0 + np.exp(x.clip(-50, 50)))
    return w.rename("w_logistic")


def _walk_forward_optimize(
    *,
    base: pd.DataFrame,
    weight_factory,
    param_grid: list[dict[str, float]],
    train_days: int,
    val_days: int,
    test_days: int,
    step_days: int,
    cost_bps: float,
) -> tuple[pd.Series, dict[str, float]]:
    """Walk-forward optimization using time-ordered validation inside each fold."""
    df = base.dropna().copy()
    dates = df.index
    min_required = int(train_days + val_days + 5)
    if len(dates) < min_required:
        return pd.Series(dtype="float64"), {"folds": 0.0}

    oos_returns: list[pd.Series] = []
    folds = 0
    i = int(train_days + val_days)
    while i < len(dates):
        tr_start = dates[i - (train_days + val_days)]
        val_start = dates[i - val_days]
        te_start = dates[i]
        te_end = dates[min(i + test_days - 1, len(dates) - 1)]

        train = df.loc[tr_start: dates[i - 1]].copy()
        val = df.loc[val_start: dates[i - 1]].copy()
        test = df.loc[te_start:te_end].copy()

        best_params: dict[str, float] | None = None
        best_sharpe = -1e9
        for params in param_grid:
            w = weight_factory(df, train, params)
            r_full = _portfolio_returns(returns=df["ret"], weights=w, cost_bps=cost_bps)
            r_val = r_full.reindex(val.index).dropna()
            m = _annualized_metrics(r_val)
            if np.isfinite(m["sharpe"]) and m["sharpe"] > best_sharpe:
                best_sharpe = float(m["sharpe"])
                best_params = params

        if best_params is None:
            i += int(step_days)
            continue

        w_full = weight_factory(df, train, best_params)
        r_full = _portfolio_returns(returns=df["ret"], weights=w_full, cost_bps=cost_bps)
        r_te = r_full.reindex(test.index).dropna()
        if not r_te.empty:
            oos_returns.append(r_te)
            folds += 1

        i += int(step_days)

    out = pd.concat(oos_returns).sort_index() if oos_returns else pd.Series(dtype="float64")
    return out, {"folds": float(folds)}


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    # Load canonical series via MDMP
    syms = [
        "AMEX:SPY",
        "CRYPTO:BTCUSD",
        "FRED:DGS10",
        "FRED:T10Y2Y",
        "CBOE:VIX",
        *DEFILLAMA_SYMBOLS,
    ]
    frames = mdm.fetch_many(syms)
    s = {sym: _extract_series(frames.get(sym), name=sym) for sym in syms}

    inputs = AsriRawInputs(
        stablecoin_total_supply_usd=s["DEFILLAMA:STABLECOIN_TOTAL_SUPPLY_USD"],
        stablecoin_hhi_raw=s["DEFILLAMA:STABLECOIN_HHI_RAW"],
        stablecoin_weighted_abs_deviation=s["DEFILLAMA:STABLECOIN_WEIGHTED_ABS_DEVIATION"],
        stablecoin_top2_share_pct=s["DEFILLAMA:STABLECOIN_TOP2_SHARE_PCT"],
        stablecoin_sig_count=s["DEFILLAMA:STABLECOIN_SIG_COUNT"],
        defi_total_tvl_usd=s["DEFILLAMA:DEFI_TOTAL_TVL_USD"],
        protocol_top10_hhi_raw=s["DEFILLAMA:PROTOCOL_TOP10_HHI_RAW"],
        protocol_audit_coverage_pct=s["DEFILLAMA:PROTOCOL_AUDIT_COVERAGE_PCT"],
        protocol_mean_abs_change_1d_pct=s["DEFILLAMA:PROTOCOL_MEAN_ABS_CHANGE_1D_PCT"],
        protocol_lending_share_pct=s["DEFILLAMA:PROTOCOL_LENDING_SHARE_PCT"],
        protocol_rwa_share_pct=s["DEFILLAMA:PROTOCOL_RWA_SHARE_PCT"],
        bridges_active_count=s["DEFILLAMA:BRIDGES_ACTIVE_COUNT"],
        us10y_yield_pct=s["FRED:DGS10"],
        vix=s["CBOE:VIX"],
        yield_spread_10y2y_pct=s["FRED:T10Y2Y"],
        btc_close=s["CRYPTO:BTCUSD"],
        spy_close=s["AMEX:SPY"],
    )

    btc = pd.to_numeric(s["CRYPTO:BTCUSD"], errors="coerce").sort_index().ffill()
    logret = np.log(btc).diff()
    rv_trailing = logret.rolling(int(args.rv_window_days)).std() * np.sqrt(365.0) * 100.0
    ret = btc.pct_change()

    idx = ret.index
    if args.start is not None:
        idx = idx[idx >= pd.Timestamp(args.start)]
    if args.end is not None:
        idx = idx[idx <= pd.Timestamp(args.end)]

    base_cfg = AsriConfig()
    first_asri = compute_asri(inputs, cfg=base_cfg).frame["asri"].first_valid_index()
    if first_asri is not None:
        idx = idx[idx >= pd.Timestamp(first_asri)]

    cfg_variants: list[tuple[str, AsriConfig]] = [
        ("scr_expanding", base_cfg),
        ("scr_roll120", replace(base_cfg, scr_tvl_drawdown_window_days=120)),
    ]
    signal_methods = ["ecdf_midrank", "robust_z_cdf", "minmax"]

    base = pd.DataFrame(
        {
            "ret": ret.reindex(idx),
            "rv_tr": rv_trailing.reindex(idx),
        }
    ).dropna()
    if base.empty:
        raise SystemExit("No BTC returns available in requested date range.")

    vt_grid = [{"target_vol_pct": v} for v in [40.0, 60.0, 80.0]]

    def vt_factory(full: pd.DataFrame, _train: pd.DataFrame, params: dict[str, float]) -> pd.Series:
        return _weight_vol_target(full["rv_tr"], target_vol_pct=float(params["target_vol_pct"]))

    vt_oos, vt_meta = _walk_forward_optimize(
        base=base,
        weight_factory=vt_factory,
        param_grid=vt_grid,
        train_days=int(args.train_days),
        val_days=int(args.val_days),
        test_days=int(args.test_days),
        step_days=int(args.step_days),
        cost_bps=float(args.cost_bps),
    )

    oos_start = base.index[min(int(args.train_days + args.val_days), len(base.index) - 1)]
    hodl_full = _portfolio_returns(
        returns=base["ret"], weights=pd.Series(1.0, index=base.index), cost_bps=0.0
    )
    hodl = hodl_full.loc[hodl_full.index >= oos_start]

    rows: list[dict[str, object]] = []
    rows.append({"variant": "baseline", "signal": "none", "strategy": "HODL", **_annualized_metrics(hodl), "folds": np.nan})
    rows.append({"variant": "baseline", "signal": "none", "strategy": "VOL_TARGET", **_annualized_metrics(vt_oos), "folds": int(vt_meta.get("folds", 0.0))})

    logistic_grid = [
        {"q_center": q, "slope": k}
        for q in [0.60, 0.70, 0.80, 0.90]
        for k in [0.05, 0.10, 0.20, 0.30]
    ]
    thr_grid = [{"q_thr": q} for q in [0.60, 0.70, 0.80, 0.90, 0.95]]
    combo_grid = [
        {"q_center": q, "slope": k, "target_vol_pct": tv}
        for q in [0.60, 0.70, 0.80, 0.90]
        for k in [0.05, 0.10, 0.20, 0.30]
        for tv in [40.0, 60.0, 80.0]
    ]

    for vname, cfg0 in cfg_variants:
        for method in signal_methods:
            cfg = replace(cfg0, norm_rolling_method=method)
            frame = compute_asri(inputs, cfg=cfg).frame
            sig = pd.to_numeric(frame["asri_norm_roll"], errors="coerce").reindex(base.index)
            df = base.copy()
            df["sig"] = sig
            df = df.dropna()
            if df.empty:
                continue

            def thr_factory(full: pd.DataFrame, train: pd.DataFrame, params: dict[str, float]) -> pd.Series:
                q = float(params["q_thr"])
                thr = float(train["sig"].quantile(q))
                return _weight_threshold(full["sig"], threshold=thr)

            thr_oos, thr_meta = _walk_forward_optimize(
                base=df,
                weight_factory=thr_factory,
                param_grid=thr_grid,
                train_days=int(args.train_days),
                val_days=int(args.val_days),
                test_days=int(args.test_days),
                step_days=int(args.step_days),
                cost_bps=float(args.cost_bps),
            )
            rows.append(
                {
                    "variant": vname,
                    "signal": method,
                    "strategy": "ASRI_THR",
                    **_annualized_metrics(thr_oos),
                    "folds": int(thr_meta.get("folds", 0.0)),
                }
            )

            def log_factory(full: pd.DataFrame, train: pd.DataFrame, params: dict[str, float]) -> pd.Series:
                q = float(params["q_center"])
                center = float(train["sig"].quantile(q))
                return _weight_logistic(full["sig"], center=center, slope=float(params["slope"]))

            log_oos, log_meta = _walk_forward_optimize(
                base=df,
                weight_factory=log_factory,
                param_grid=logistic_grid,
                train_days=int(args.train_days),
                val_days=int(args.val_days),
                test_days=int(args.test_days),
                step_days=int(args.step_days),
                cost_bps=float(args.cost_bps),
            )
            rows.append(
                {
                    "variant": vname,
                    "signal": method,
                    "strategy": "ASRI_LOG",
                    **_annualized_metrics(log_oos),
                    "folds": int(log_meta.get("folds", 0.0)),
                }
            )

            def combo_factory(full: pd.DataFrame, train: pd.DataFrame, params: dict[str, float]) -> pd.Series:
                q = float(params["q_center"])
                center = float(train["sig"].quantile(q))
                w_log = _weight_logistic(full["sig"], center=center, slope=float(params["slope"]))
                w_vol = _weight_vol_target(full["rv_tr"], target_vol_pct=float(params["target_vol_pct"]))
                return (w_log * w_vol).clip(0.0, 1.0)

            combo_oos, combo_meta = _walk_forward_optimize(
                base=df,
                weight_factory=combo_factory,
                param_grid=combo_grid,
                train_days=int(args.train_days),
                val_days=int(args.val_days),
                test_days=int(args.test_days),
                step_days=int(args.step_days),
                cost_bps=float(args.cost_bps),
            )
            rows.append(
                {
                    "variant": vname,
                    "signal": method,
                    "strategy": "ASRI_LOG_VOL",
                    **_annualized_metrics(combo_oos),
                    "folds": int(combo_meta.get("folds", 0.0)),
                }
            )

    res_df = pd.DataFrame(rows).sort_values(["sharpe", "cagr"], ascending=False)
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        out_path = repo_root / "data" / "processed" / "asri_strategy_reports" / "asri_signal_wfo.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(res_df.round(4).to_markdown(index=False), encoding="utf-8")
    print(f"[OK] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
