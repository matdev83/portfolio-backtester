"""Display labels for performance metrics (Rich tables), keyed by canonical metric names."""

from __future__ import annotations

from typing import Any, Literal, Mapping

MetricsDisplayProfile = Literal["legacy", "platform_standard", "verbose"]

SHARPE_PATH_EXCESS = "excess"
SHARPE_PATH_LEGACY_CAGR_VOL = "legacy_cagr_vol"


def resolve_metrics_display_profile(
    global_config: Mapping[str, Any],
    scenario: Any | None = None,
) -> MetricsDisplayProfile:
    """Resolve metrics_display_profile from scenario extras then GLOBAL_CONFIG."""
    raw: Any = None
    if scenario is not None:
        extras = None
        if isinstance(scenario, dict):
            extras = scenario.get("extras")
        else:
            extras = getattr(scenario, "extras", None)
        if isinstance(extras, Mapping):
            raw = extras.get("metrics_display_profile")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raw = global_config.get("metrics_display_profile", "legacy")
    s = str(raw).strip().lower().replace("-", "_")
    if s in {"platform", "platform_standard", "standard"}:
        return "platform_standard"
    if s == "verbose":
        return "verbose"
    return "legacy"


def metric_display_label(
    canonical_key: str,
    profile: MetricsDisplayProfile,
    *,
    sharpe_path: str | None = None,
) -> str:
    """Map canonical Series index name to human-facing row label for Rich tables."""
    if profile == "legacy":
        if canonical_key == "ADF Statistic (equity)":
            return "ADF Statistic"
        if canonical_key == "ADF p-value (equity)":
            return "ADF p-value"
        return canonical_key

    # platform_standard and verbose share labels (verbose may add rows elsewhere).
    if canonical_key == "Sharpe":
        if sharpe_path == SHARPE_PATH_LEGACY_CAGR_VOL:
            return "Sharpe (CAGR/vol)"
        return "Sharpe"

    static = {
        "Tail Ratio": "Tail Ratio (pctile)",
        "ADF Statistic (equity)": "ADF Statistic (equity)",
        "ADF p-value (equity)": "ADF p-value (equity)",
        "ADF Statistic (returns)": "ADF Statistic (returns)",
        "ADF p-value (returns)": "ADF p-value (returns)",
        "Tail Ratio (mean)": "Gain/Loss Mean Ratio",
        "Deflated Sharpe": "PSR (trial-adjusted)",
        "Avg DD Duration": "Avg DD Duration (bars)",
        "Avg Recovery Time": "Avg Recovery Time (bars)",
        "Max DD Recovery Time (days)": "Max DD Recovery Time (calendar days)",
        "Max DD Recovery Time (bars)": "Max DD Recovery Time (bars)",
    }
    if canonical_key in static:
        return static[canonical_key]
    return canonical_key


def formatting_rules_for_label(display_label: str) -> tuple[set[str], set[str]]:
    """Return (percentage_metric_names, high_precision_metric_names) for a display label."""
    percentage_metrics = {
        "Total Return",
        "Ann. Return",
        "Time in Market %",
        "Avg Gross Exposure",
        "Avg Net Exposure",
        "Max Gross Exposure",
        "Avg Long Exposure",
        "Avg Short Exposure",
    }
    high_precision = {
        "ADF p-value",
        "ADF p-value (equity)",
        "ADF p-value (returns)",
    }
    if display_label in percentage_metrics:
        return {display_label}, set()
    if display_label in high_precision:
        return set(), {display_label}
    return set(), set()


def merge_format_sets(labels: set[str]) -> tuple[set[str], set[str]]:
    """Union formatting rule sets for multiple display labels."""
    pct: set[str] = set()
    hp: set[str] = set()
    for lb in labels:
        p, h = formatting_rules_for_label(lb)
        pct |= p
        hp |= h
    return pct, hp
