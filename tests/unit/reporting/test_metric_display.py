"""Tests for Rich-table metric label mapping."""

from portfolio_backtester.reporting.metric_display import (
    SHARPE_PATH_LEGACY_CAGR_VOL,
    metric_display_label,
    resolve_metrics_display_profile,
)


def test_resolve_metrics_display_profile_defaults_legacy():
    assert resolve_metrics_display_profile({}) == "legacy"
    assert resolve_metrics_display_profile({"metrics_display_profile": "platform_standard"}) == (
        "platform_standard"
    )


def test_metric_display_label_platform_tail_and_psr():
    assert metric_display_label("Tail Ratio", "platform_standard") == "Tail Ratio (pctile)"
    assert metric_display_label("Deflated Sharpe", "platform_standard") == "PSR (trial-adjusted)"


def test_metric_display_sharpe_legacy_path_label():
    assert (
        metric_display_label(
            "Sharpe",
            "platform_standard",
            sharpe_path=SHARPE_PATH_LEGACY_CAGR_VOL,
        )
        == "Sharpe (CAGR/vol)"
    )
