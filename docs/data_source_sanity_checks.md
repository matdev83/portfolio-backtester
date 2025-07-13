# Data Source Sanity Checks

This document describes the comprehensive sanity check system implemented to ensure the portfolio backtester has access to up-to-date, high-quality market data.

## Overview

The sanity check system verifies that:
1. **Data Sources are Functional**: Both Stooq and yfinance can fetch data
2. **Data is Recent**: Historical data includes the most recent trading days
3. **Data Quality is Good**: Prices are reasonable and data is properly formatted
4. **Fail-tolerance Works**: The hybrid data source properly falls back between sources

## Test Files

### `tests/data_sources/test_data_sanity_check.py`

Comprehensive standalone sanity checks:

- **`test_hybrid_data_source_aapl_recent_data`**: Main sanity check that verifies AAPL data availability
- **`test_individual_data_sources_aapl`**: Tests each data source individually 
- **`test_market_calendar_logic`**: Validates trading day calculation logic
- **`test_data_lag_tolerance`**: Ensures reasonable data lag expectations

### `tests/data_sources/test_hybrid_data_source.py`

Includes a basic sanity check as part of the regular hybrid data source test suite:

- **`test_hybrid_data_source_basic_sanity_check`**: Lightweight real-data validation

## Key Features

### Market Calendar Awareness

The system correctly identifies the last trading day by accounting for:

- **Weekends**: Automatically skips Saturday and Sunday
- **US Market Holidays**: Includes major holidays like:
  - New Year's Day
  - Independence Day  
  - Christmas Day
  - And other major market holidays
- **Data Provider Lag**: Allows for up to 5 business days of lag

### Example Scenarios

```
Today: Saturday, July 13, 2025
Expected Last Trading Day: Friday, July 11, 2025 ✓

Today: Wednesday, July 4, 2025 (Independence Day)  
Expected Last Trading Day: Tuesday, July 3, 2025 ✓

Today: Thursday, January 2, 2025 (Day after New Year's)
Expected Last Trading Day: Thursday, January 2, 2025 ✓
```

### Data Quality Validation

The sanity checks verify:

1. **Data Availability**: AAPL.csv file is created with recent data
2. **Date Consistency**: Last row date matches expected trading day (±5 business days)
3. **Price Reasonableness**: AAPL price between $50-$1000 (sanity bounds)
4. **Data Completeness**: No excessive NaN values
5. **Format Consistency**: Proper MultiIndex structure maintained

### Fail-tolerance Verification

Tests confirm:

- **Primary Source Success**: Stooq works as primary source
- **Fallback Functionality**: yfinance works as backup
- **Error Handling**: Graceful handling of network issues
- **Reporting Accuracy**: Failure reports correctly track source usage

## Running the Tests

### All Sanity Checks (Offline Only)
```bash
python -m pytest tests/data_sources/test_data_sanity_check.py -v
```

### Network-Dependent Tests
```bash
python -m pytest tests/data_sources/test_data_sanity_check.py -v -m network
```

### Specific AAPL Test
```bash
python -m pytest tests/data_sources/test_data_sanity_check.py::TestDataSanityCheck::test_hybrid_data_source_aapl_recent_data -v -s -m network
```

### Hybrid Data Source with Sanity Check
```bash
python -m pytest tests/data_sources/test_hybrid_data_source.py -v -m network
```

## Expected Output

### Successful Test Output
```
============================================================
SANITY CHECK: Testing hybrid data source for recent AAPL data
============================================================
Expected last trading day: 2025-07-11
Fetching AAPL data from 2025-06-13 to 2025-07-13
  Downloading AAPL… ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
  Fetching from stooq... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
Last available data date: 2025-07-11
Last date in AAPL.csv file: 2025-07-11  
Last AAPL close price: $211.16
Data source failure report: {'stooq_failures': [], 'yfinance_failures': [], 'total_failures': [], 'stooq_only_count': 0, 'yfinance_only_count': 0, 'complete_failure_count': 0}
✅ AAPL sanity check PASSED
```

## Integration with CI/CD

### Environment Variables

- **`SKIP_NETWORK_TESTS=true`**: Disables network-dependent tests in CI environments
- Tests automatically skip if network connectivity issues occur

### Pytest Markers

- **`@pytest.mark.network`**: Marks tests requiring network access
- Tests are deselected by default unless explicitly run with `-m network`

## Troubleshooting

### Common Issues

1. **"Data too old" Error**: 
   - Check if markets are closed for extended periods (holidays)
   - Verify data provider is operational
   - Consider adjusting `max_acceptable_lag` for holiday periods

2. **"No data fetched" Error**:
   - Check network connectivity
   - Verify both Stooq and yfinance are accessible
   - Check for API rate limiting

3. **"Price seems unreasonable" Error**:
   - Update price bounds if AAPL has split or moved significantly
   - Check for data quality issues from providers

### Manual Verification

To manually verify data availability:

```python
from src.portfolio_backtester.data_sources.hybrid_data_source import HybridDataSource
from datetime import datetime, timedelta

ds = HybridDataSource(prefer_stooq=True)
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')

data = ds.get_data(['AAPL'], start_date, end_date)
print(f"Last AAPL data: {data[('AAPL', 'Close')].dropna().tail(1)}")
print(f"Failure report: {ds.get_failure_report()}")
```

## Maintenance

### Holiday Calendar Updates

The holiday list in `_get_expected_last_trading_day()` should be updated annually:

```python
us_holidays_2025_2026 = [
    pd.Timestamp('2025-01-01'),  # New Year's Day
    pd.Timestamp('2025-01-20'),  # Martin Luther King Jr. Day
    # ... add new holidays
]
```

### Price Bounds Updates

If AAPL undergoes stock splits or significant price changes, update validation bounds:

```python
self.assertGreater(last_price, 50, "AAPL price seems too low")     # Update lower bound
self.assertLess(last_price, 1000, "AAPL price seems too high")     # Update upper bound
```

This sanity check system ensures reliable, up-to-date data access for the portfolio backtester while providing comprehensive diagnostics when issues occur. 