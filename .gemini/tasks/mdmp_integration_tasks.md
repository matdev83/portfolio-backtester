# MDMP Integration Tasks

## Status: ✅ COMPLETE (Sprints 1-4 Done)

## Quick Reference

| Sprint | Tasks | Est. Time | Priority | Status |
|--------|-------|-----------|----------|--------|
| 1 | Foundation Setup | 2-3 hours | 🔴 High | ✅ Done |
| 2 | Symbol Registry | 3-4 hours | 🔴 High | ✅ Done |
| 3 | Integration Testing | 2-3 hours | 🔴 High | ✅ Done |
| 4 | Configuration & Rollout | 1-2 hours | 🟡 Medium | ✅ Done |
| 5 | Cleanup (Optional) | 2-3 hours | 🟢 Low | ⬜ Deferred |

---

## Sprint 1: Foundation Setup ✅ COMPLETE

### 1.1 Add Dependency

- [x] Edit `pyproject.toml` to add `market-data-multi-provider` as local dependency
- [x] Run `pip install -e .` to reinstall with new dependency
- [x] Verify import works: `from market_data_multi_provider import MarketDataClient`

### 1.2 Create Symbol Mapper

- [x] Create `src/portfolio_backtester/data_sources/symbol_mapper.py`
- [x] Implement `to_canonical_id(ticker) -> str`
- [x] Implement `from_canonical_id(canonical_id) -> str`
- [x] Add caching with `@lru_cache`
- [x] Run QA checks (`ruff`, `mypy`)

### 1.3 Create MDMP Data Source Adapter

- [x] Create `src/portfolio_backtester/data_sources/mdmp_data_source.py`
- [x] Implement `MarketDataMultiProviderDataSource` class
- [x] Implement `get_data(tickers, start_date, end_date)` method
- [x] Handle MultiIndex column format conversion
- [x] Run QA checks

### 1.4 Register in Factory

- [x] Edit `src/portfolio_backtester/interfaces/data_source_interface.py`
- [x] Add import for `MarketDataMultiProviderDataSource`
- [x] Add `"mdmp"` and `"market-data-multi-provider"` to data source map
- [x] Add factory case for creating MDMP source
- [x] Run QA checks

---

## Sprint 2: Symbol Registry Updates ✅ COMPLETE

### 2.1 Inventory Symbols

- [x] Extract unique symbols from `config/universes/sp500_top50.txt`
- [x] Extract unique symbols from `config/universes/dow_jones.txt`
- [x] Extract unique symbols from `config/universes/nasdaq_top20.txt`
- [x] List hardcoded symbols (SPY, QQQ, GLD, TLT, VTI, UVXY, etc.)
- [x] Create master symbol list (deduplicated) - saved to `.gemini/symbol_inventory.md`

### 2.2 Add Missing Symbols to MDMP

- [x] Edit `market-data-multi-provider/src/market_data_multi_provider/resources/symbols.json`
- [x] Add GLD entry (stooq → yfinance → tradingview)
- [x] Add TLT entry (stooq → yfinance → tradingview)
- [x] Add VTI entry (stooq → yfinance → tradingview)
- [x] Add UVXY entry (yfinance → tradingview)
- [x] Add all S&P 500 top 50 stocks (50 symbols)
- [x] Add remaining Dow Jones stocks (9 additional)
- [x] Add remaining NASDAQ top 20 stocks (3 additional)
- [x] Fix duplicate MULTPL:SP500DY entry

### 2.3 Validate Symbol Resolution

- [x] Test symbol resolution for each added symbol
- [x] Verify provider fallback chain works
- [x] Check alias resolution (e.g., AAPL resolves correctly)

**Total symbols in registry: 112 (up from ~47)**

---

## Sprint 3: Integration Testing ✅ COMPLETE

### 3.1 Create Integration Tests

- [x] Create `tests/integration/test_mdmp_data_source.py`
- [x] Test single ticker fetch
- [x] Test multi-ticker fetch
- [x] Test symbol mapping roundtrip
- [x] Test error handling for invalid symbols

### 3.2 Compatibility Testing

- [x] Create comparison test between HybridDataSource and MDMP
- [x] Verify output structure matches
- [x] Verify data values are within tolerance
- [x] Test with all universe types (fixed, named, method)

### 3.3 Full Test Suite

- [x] Run `pytest tests/` with MDMP as data source
- [x] Verify no test failures (12/12 passed)
- [ ] Check test coverage (optional)

---

## Sprint 4: Configuration & Rollout ✅ COMPLETE

### 4.1 Update Default Configuration

- [x] Edit `config/parameters.yaml`
- [x] Document `data_source` options in comments
- [x] Keep `"hybrid"` as default for backward compatibility
- [x] Added instructions to switch to `"mdmp"`

### 4.2 Update Documentation

- [x] Configuration comments document all options
- [ ] Update README.md with new data source info (optional)
- [ ] Add migration notes for existing users (optional)

### 4.3 Validation Run

- [x] Factory creates MDMP source correctly
- [x] Symbol resolution works for all registered symbols
- [ ] Run a full backtest with MDMP (optional - requires network)

---

## Sprint 5: Cleanup (Optional)

### 5.1 Deprecation Warnings

- [ ] Add deprecation warnings to `StooqDataSource`
- [ ] Add deprecation warnings to `YFinanceDataSource`
- [ ] Add deprecation warnings to `HybridDataSource`

### 5.2 Future Removal (Defer)

- [ ] Plan removal timeline
- [ ] Update tests to not require legacy sources
- [ ] Remove legacy source files

---

## Notes

### Provider Priority Order (for common stocks)

1. **stooq** - Best historical depth, reliable
2. **yfinance** - Good coverage, no auth needed
3. **tradingview** - Requires login, last resort

### Symbol Format Examples

| Local | Canonical |
|-------|-----------|
| `SPY` | `AMEX:SPY` |
| `AAPL` | `NASDAQ:AAPL` |
| `^GSPC` | `SP:SPX` |
| `^VIX` | `CBOE:VIX` |
| `JNJ` | `NYSE:JNJ` |

### Rollback Procedure

1. Set `data_source: "hybrid"` in `config/parameters.yaml`
2. No code changes needed - factory handles it
