import pandas as pd
import pytest

from portfolio_backtester.universe_data import spy_holdings


@pytest.mark.unit
@pytest.mark.fast
def test_get_spy_holdings_fast_lookup_exact_and_non_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify fast-path date resolution and exact/non-exact semantics.

    This test patches the MDMP builder history to a tiny deterministic frame so we
    can validate that:
    - exact=True requires an exact match
    - exact=False uses the most recent previous date (never future)
    """

    # Build a tiny history similar to MDMP's schema
    hist = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2020-01-02"),
                pd.Timestamp("2020-01-02"),
                pd.Timestamp("2020-01-06"),
            ],
            "ticker": ["A", "B", "A"],
            "weight_pct": [0.6, 0.4, 1.0],
        }
    )

    class _FakeBuilder:
        _HISTORY_DF = hist

        @staticmethod
        def _ensure_history_loaded() -> None:
            return None

    def _fake_get_holdings_history() -> pd.DataFrame:
        return _FakeBuilder._HISTORY_DF.copy()

    # Patch import inside the module's helper by swapping the module attribute via sys.modules
    import types
    import sys

    fake_mod = types.ModuleType("market_data_multi_provider.sp500")
    fake_mod.builder = _FakeBuilder  # type: ignore[attr-defined]
    fake_mod.get_holdings_history = _fake_get_holdings_history  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "market_data_multi_provider.sp500", fake_mod)

    # Ensure we do not call the slower mdmp_get_holdings fallback during this test
    monkeypatch.setattr(
        spy_holdings,
        "mdmp_get_holdings",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("mdmp_get_holdings should not be called")
        ),
    )

    spy_holdings.reset_history_cache()

    # exact=True: present date works
    df_exact = spy_holdings.get_spy_holdings("2020-01-02", exact=True)
    assert list(df_exact["ticker"]) == ["A", "B"]

    # exact=True: missing date fails
    with pytest.raises(ValueError):
        spy_holdings.get_spy_holdings("2020-01-03", exact=True)

    # exact=False: missing date uses previous (2020-01-02), not future (2020-01-06)
    df_prev = spy_holdings.get_spy_holdings("2020-01-03", exact=False)
    assert set(df_prev["ticker"]) == {"A", "B"}


@pytest.mark.unit
@pytest.mark.fast
def test_get_spy_holdings_fast_lookup_out_of_range_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dates earlier than the earliest holdings date should fail quickly for exact=False."""

    hist = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-10")],
            "ticker": ["A"],
            "weight_pct": [1.0],
        }
    )

    class _FakeBuilder:
        _HISTORY_DF = hist

        @staticmethod
        def _ensure_history_loaded() -> None:
            return None

    def _fake_get_holdings_history() -> pd.DataFrame:
        return _FakeBuilder._HISTORY_DF.copy()

    import types
    import sys

    fake_mod = types.ModuleType("market_data_multi_provider.sp500")
    fake_mod.builder = _FakeBuilder  # type: ignore[attr-defined]
    fake_mod.get_holdings_history = _fake_get_holdings_history  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "market_data_multi_provider.sp500", fake_mod)

    monkeypatch.setattr(
        spy_holdings,
        "mdmp_get_holdings",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("mdmp_get_holdings should not be called")
        ),
    )

    spy_holdings.reset_history_cache()

    with pytest.raises(ValueError):
        spy_holdings.get_spy_holdings("2020-01-01", exact=False)
