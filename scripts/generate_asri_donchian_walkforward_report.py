r"""
Generate a walk-forward report for BTC Donchian channel strategies with ASRI overlays.

Donchian rules (close-based, long-only):
- Enter when close breaks above prior N-day high (default: 20)
- Exit when close breaks below prior M-day low (default: 10)

Variants compared:
1) Donchian + ASRI risk-off filter (ASRI_THR gating)
2) Donchian + ASRI risk-off + ASRI position sizing (logistic)
3) Donchian + ASRI position sizing only (no ASRI exits)
4) Pure Donchian
Plus baselines:
- HODL
- ASRI_THR simple (no Donchian)

All strategies are evaluated via walk-forward optimization:
- Hyperparams are selected on a validation window inside each fold
- Weights are applied with a 1-day lag (signals at t affect returns t+1)
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

import market_data_multi_provider as mdm
from market_data_multi_provider.indicators.asri_core import AsriConfig
from portfolio_backtester.strategies.builtins.signal.donchian_asri_signal_strategy import (
    donchian_position_close_breakout,
)


def _parse_date(s: str) -> dt.date:
    try:
        return dt.date.fromisoformat(s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date {s!r}; expected YYYY-MM-DD.") from exc


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate Donchian+ASRI walk-forward report.")
    p.add_argument("--start", type=_parse_date, default=None, help="Optional start date.")
    p.add_argument("--end", type=_parse_date, default=None, help="Optional end date.")
    p.add_argument("--entry-lookback", type=int, default=20, help="Donchian entry lookback (default: 20).")
    p.add_argument("--exit-lookback", type=int, default=10, help="Donchian exit lookback (default: 10).")
    p.add_argument("--train-days", type=int, default=730, help="Train window length in days (default: 730).")
    p.add_argument("--val-days", type=int, default=180, help="Validation window length in days (default: 180).")
    p.add_argument("--test-days", type=int, default=90, help="Test window length in days per fold (default: 90).")
    p.add_argument("--step-days", type=int, default=90, help="Step between folds (default: 90).")
    p.add_argument("--cost-bps", type=float, default=10.0, help="Turnover cost in bps (default: 10).")
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output markdown path (default: data/processed/asri_strategy_reports/...).",
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


def _total_return_after_fees(r: pd.Series) -> float:
    r = pd.to_numeric(r, errors="coerce").dropna()
    if r.empty:
        return float("nan")
    return float((1.0 + r).prod() - 1.0)


def _portfolio_frame(
    *,
    returns: pd.Series,
    weights: pd.Series,
    cost_bps: float,
) -> pd.DataFrame:
    """Daily portfolio returns with 1-day lag execution + turnover costs."""
    r = pd.to_numeric(returns, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    df = pd.concat({"r": r, "w": w}, axis=1).dropna()
    if df.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([]))
    cost = float(cost_bps) / 10_000.0
    w_prev = df["w"].shift(1)
    turn_prev = df["w"].diff().abs().shift(1).fillna(0.0)
    port = (w_prev * df["r"] - cost * turn_prev).dropna()
    out = pd.DataFrame(
        {"port_ret": port, "w_prev": w_prev.reindex(port.index), "turn_prev": turn_prev.reindex(port.index)}
    )
    out.index.name = "Date"
    return out


def _logistic_weight(sig_0_100: pd.Series, *, center: float, slope: float) -> pd.Series:
    sig = pd.to_numeric(sig_0_100, errors="coerce")
    x = float(slope) * (sig - float(center))
    w = 1.0 / (1.0 + np.exp(x.clip(-50, 50)))
    return w.clip(0.0, 1.0)


def _donchian_pos_filtered(
    *,
    breakout: pd.Series,
    stop: pd.Series,
    risk_on: pd.Series,
) -> pd.Series:
    """Donchian state machine with risk-on entry filter + risk-off exit override."""
    idx = breakout.index
    ron = risk_on.reindex(idx).fillna(False).astype(bool)
    entry = breakout.fillna(False) & ron
    exit_ = stop.fillna(False) | (~ron)

    pos = pd.Series(0.0, index=idx, name="donchian_pos")
    state = 0.0
    for i, dt in enumerate(idx):
        if i == 0:
            pos.iloc[i] = 0.0
            continue
        if state <= 0.0 and bool(entry.loc[dt]):
            state = 1.0
        elif state > 0.0 and bool(exit_.loc[dt]):
            state = 0.0
        pos.iloc[i] = state
    return pos


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
) -> tuple[pd.Series, list[dict[str, object]]]:
    """Walk-forward optimization with explicit train/val/test windows."""
    df = base.dropna().copy()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    dates = df.index
    min_required = int(train_days + val_days + test_days + 5)
    if len(dates) < min_required:
        return pd.Series(dtype="float64"), []

    records: list[dict[str, object]] = []
    oos_returns: list[pd.Series] = []

    i = int(train_days + val_days)
    while i < len(dates):
        tr_start_i = i - int(train_days + val_days)
        tr_end_i = i - int(val_days) - 1
        val_start_i = i - int(val_days)
        val_end_i = i - 1
        te_start_i = i
        te_end_i = min(i + int(test_days) - 1, len(dates) - 1)

        train = df.iloc[tr_start_i : tr_end_i + 1]
        train_val = df.iloc[tr_start_i : val_end_i + 1]
        val = df.iloc[val_start_i : val_end_i + 1]
        test = df.iloc[te_start_i : te_end_i + 1]

        # Pick best params on validation Sharpe.
        best_params: dict[str, float] | None = None
        best_sharpe = -1e9
        for params in param_grid:
            w = weight_factory(df, train, params)
            pf = _portfolio_frame(returns=df["ret"], weights=w, cost_bps=cost_bps)
            r_val = pf["port_ret"].reindex(val.index).dropna()
            m = _annualized_metrics(r_val)
            if np.isfinite(m["sharpe"]) and m["sharpe"] > best_sharpe:
                best_sharpe = float(m["sharpe"])
                best_params = params

        if best_params is None:
            i += int(step_days)
            continue

        # Refit using train+val for the test window.
        w_full = weight_factory(df, train_val, best_params)
        pf_full = _portfolio_frame(returns=df["ret"], weights=w_full, cost_bps=cost_bps)
        r_te = pf_full["port_ret"].reindex(test.index).dropna()
        if not r_te.empty:
            oos_returns.append(r_te)
            records.append(
                {
                    "train_start": train.index.min().date(),
                    "train_end": train.index.max().date(),
                    "val_start": val.index.min().date(),
                    "val_end": val.index.max().date(),
                    "test_start": test.index.min().date(),
                    "test_end": test.index.max().date(),
                    "params": dict(best_params),
                    "val_sharpe": float(best_sharpe),
                }
            )

        i += int(step_days)

    out = pd.concat(oos_returns).sort_index() if oos_returns else pd.Series(dtype="float64")
    return out, records


def _plot_curves(curves: dict[str, pd.Series], *, out_path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not curves:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, r in curves.items():
        r = pd.to_numeric(r, errors="coerce").dropna()
        if r.empty:
            continue
        eq = (1.0 + r).cumprod()
        ax.plot(eq.index, eq, linewidth=1.7, label=name)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_drawdowns(curves: dict[str, pd.Series], *, out_path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not curves:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, r in curves.items():
        r = pd.to_numeric(r, errors="coerce").dropna()
        if r.empty:
            continue
        eq = (1.0 + r).cumprod()
        dd = eq / eq.cummax() - 1.0
        ax.plot(dd.index, dd, linewidth=1.4, label=name)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    # Compute ASRI with the best-performing signal form (paper SCR + rolling ECDF midrank).
    cfg = AsriConfig(norm_rolling_method="ecdf_midrank", scr_tvl_drawdown_window_days=0)
    asri_res = mdm.compute_asri_crypto(end_date=args.end, cfg=cfg, write_local=False)
    frame = asri_res.frame
    sig = pd.to_numeric(frame["asri_norm_roll"], errors="coerce")

    close_df = mdm.fetch_one("CRYPTO:BTCUSD", end=args.end)
    close = pd.to_numeric(_extract_series(close_df, name="CRYPTO:BTCUSD"), errors="coerce").sort_index().ffill()

    # Align to ASRI availability window
    first_asri = frame["asri"].first_valid_index()
    if first_asri is not None:
        close = close.loc[close.index >= pd.Timestamp(first_asri)]
        sig = sig.loc[sig.index >= pd.Timestamp(first_asri)]

    if args.start is not None:
        close = close.loc[close.index >= pd.Timestamp(args.start)]
        sig = sig.loc[sig.index >= pd.Timestamp(args.start)]
    if args.end is not None:
        close = close.loc[close.index <= pd.Timestamp(args.end)]
        sig = sig.loc[sig.index <= pd.Timestamp(args.end)]

    close = close.reindex(sig.index.union(close.index)).sort_index().ffill()
    sig = sig.reindex(close.index)
    ret = close.pct_change()

    # Donchian precomputations (close-based).
    e = int(args.entry_lookback)
    x = int(args.exit_lookback)
    high_n = close.shift(1).rolling(e, min_periods=e).max()
    low_n = close.shift(1).rolling(x, min_periods=x).min()
    breakout = (close > high_n) & close.notna()
    stop = (close < low_n) & close.notna()
    pos_donch = donchian_position_close_breakout(close, entry_lookback=e, exit_lookback=x)

    base = pd.DataFrame(
        {"close": close, "ret": ret, "sig": sig, "pos_donch": pos_donch, "breakout": breakout, "stop": stop}
    ).dropna(subset=["ret", "sig"])
    if base.empty:
        raise SystemExit("No overlapping BTC returns and ASRI signal window.")

    data_start_date = base.index.min().date()
    data_end_date = base.index.max().date()

    # Output paths
    end_date = base.index.max().date()
    stem = f"asri_donchian_wfo_{end_date.isoformat()}"
    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else (repo_root / "data" / "processed" / "asri_strategy_reports" / f"{stem}.md")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assets_dir = out_path.parent / f"{stem}_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparameter grids
    q_thr_grid = [{"q_thr": q} for q in [0.60, 0.70, 0.80, 0.90, 0.95]]
    log_grid = [{"q_center": q, "slope": k} for q in [0.60, 0.70, 0.80, 0.90] for k in [0.05, 0.10, 0.20, 0.30]]
    filt_log_grid = [{"q_thr": qt["q_thr"], "q_center": q, "slope": k} for qt in q_thr_grid for q in [0.60, 0.70, 0.80, 0.90] for k in [0.05, 0.10, 0.20, 0.30]]

    # Strategy definitions (weight factories)
    def w_hodl(full: pd.DataFrame, _fit: pd.DataFrame, _p: dict[str, float]) -> pd.Series:
        return pd.Series(1.0, index=full.index)

    def w_asri_thr(full: pd.DataFrame, fit: pd.DataFrame, p: dict[str, float]) -> pd.Series:
        thr = float(fit["sig"].quantile(float(p["q_thr"])))
        return (full["sig"] < thr).astype("float64")

    def w_donchian(full: pd.DataFrame, _fit: pd.DataFrame, _p: dict[str, float]) -> pd.Series:
        return full["pos_donch"].astype("float64")

    def w_donchian_asri_filter(full: pd.DataFrame, fit: pd.DataFrame, p: dict[str, float]) -> pd.Series:
        thr = float(fit["sig"].quantile(float(p["q_thr"])))
        risk_on = full["sig"] < thr
        pos = _donchian_pos_filtered(breakout=full["breakout"], stop=full["stop"], risk_on=risk_on)
        return pos.astype("float64")

    def w_donchian_asri_filter_and_size(full: pd.DataFrame, fit: pd.DataFrame, p: dict[str, float]) -> pd.Series:
        thr = float(fit["sig"].quantile(float(p["q_thr"])))
        center = float(fit["sig"].quantile(float(p["q_center"])))
        slope = float(p["slope"])
        risk_on = full["sig"] < thr
        pos = _donchian_pos_filtered(breakout=full["breakout"], stop=full["stop"], risk_on=risk_on)
        size = _logistic_weight(full["sig"], center=center, slope=slope)
        return (pos * size).clip(0.0, 1.0).astype("float64")

    def w_donchian_asri_size_only(full: pd.DataFrame, fit: pd.DataFrame, p: dict[str, float]) -> pd.Series:
        center = float(fit["sig"].quantile(float(p["q_center"])))
        slope = float(p["slope"])
        size = _logistic_weight(full["sig"], center=center, slope=slope)
        return (full["pos_donch"] * size).clip(0.0, 1.0).astype("float64")

    strategies: list[tuple[str, list[dict[str, float]], object]] = [
        ("HODL", [{}], w_hodl),
        ("ASRI_THR", q_thr_grid, w_asri_thr),
        ("DONCHIAN", [{}], w_donchian),
        ("DONCHIAN_ASRI_FILTER", q_thr_grid, w_donchian_asri_filter),
        ("DONCHIAN_ASRI_FILTER_SIZE", filt_log_grid, w_donchian_asri_filter_and_size),
        ("DONCHIAN_ASRI_SIZE_ONLY", log_grid, w_donchian_asri_size_only),
    ]

    results: list[dict[str, object]] = []
    oos_curves: dict[str, pd.Series] = {}
    params_summary: dict[str, list[dict[str, object]]] = {}

    for name, grid, wf in strategies:
        r_oos, recs = _walk_forward_optimize(
            base=base[["ret", "sig", "pos_donch", "breakout", "stop"]],
            weight_factory=wf,
            param_grid=grid,
            train_days=int(args.train_days),
            val_days=int(args.val_days),
            test_days=int(args.test_days),
            step_days=int(args.step_days),
            cost_bps=float(args.cost_bps),
        )
        oos_curves[name] = r_oos
        params_summary[name] = recs
        m = _annualized_metrics(r_oos)
        tr = _total_return_after_fees(r_oos)
        results.append(
            {
                "strategy": name,
                "total_return": tr,
                "cagr": m["cagr"],
                "vol": m["vol"],
                "sharpe": m["sharpe"],
                "max_dd": m["max_dd"],
                "folds": int(len(recs)),
            }
        )

    res_df = pd.DataFrame(results).sort_values(["sharpe", "cagr"], ascending=False)

    # Extract OOS date range from records (all strategies use same base data)
    oos_start_date = None
    oos_end_date = None
    for recs in params_summary.values():
        if recs:
            oos_start_date = min(r["test_start"] for r in recs)
            oos_end_date = max(r["test_end"] for r in recs)
            break

    # Create display copy with formatted total_return as percentage
    res_df_display = res_df.copy()
    numeric_cols = ["cagr", "vol", "sharpe", "max_dd"]
    for col in numeric_cols:
        if col in res_df_display.columns:
            res_df_display[col] = res_df_display[col].round(4)
    res_df_display["total_return"] = res_df_display["total_return"].apply(
        lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
    )

    # Charts
    curves_path = assets_dir / "equity_curves.png"
    _plot_curves(oos_curves, out_path=curves_path, title="BTC — Donchian + ASRI variants (walk-forward OOS)")
    dd_path = assets_dir / "drawdowns.png"
    _plot_drawdowns(oos_curves, out_path=dd_path, title="BTC — Drawdowns (walk-forward OOS)")

    # Parameter summaries (counts)
    param_lines: list[str] = []
    for strat, recs in params_summary.items():
        if not recs:
            continue
        pstr = [str(r.get("params", {})) for r in recs]
        vc = pd.Series(pstr).value_counts().head(8)
        param_lines.append(f"### {strat}\n\n{vc.to_frame('count').to_markdown()}\n")

    rel_curves = f"{assets_dir.name}/equity_curves.png"
    rel_dd = f"{assets_dir.name}/drawdowns.png"

    md = f"""# BTC Donchian + ASRI — Walk-forward strategy comparison

**As-of (data end):** {end_date.isoformat()}

## Setup

- **BTC universe:** `CRYPTO:BTCUSD` (close-to-close returns)
- **Data period:** {data_start_date.isoformat()} to {data_end_date.isoformat()}
- **Donchian rules:** entry `{e}d` breakout over prior high; exit `{x}d` break below prior low
- **ASRI signal:** rolling ECDF mid-rank (`AsriConfig.norm_rolling_method = 'ecdf_midrank'`)
- **Transaction costs:** {float(args.cost_bps):.1f} bps per 1.0 turnover
- **Walk-forward:** train/val/test/step = {int(args.train_days)}/{int(args.val_days)}/{int(args.test_days)}/{int(args.step_days)} days
- **Execution:** weights decided at close \(t\), applied to return \(t+1\)
{f"- **OOS period:** {oos_start_date.isoformat()} to {oos_end_date.isoformat()}" if oos_start_date and oos_end_date else ""}

## Performance (walk-forward OOS)

{res_df_display.to_markdown(index=False)}

## Equity curves

![Equity curves]({rel_curves})

## Drawdowns

![Drawdowns]({rel_dd})

## Hyperparameters selected (top counts)

{chr(10).join(param_lines) if param_lines else 'N/A'}
"""

    out_path.write_text(md, encoding="utf-8")
    print(f"[OK] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
