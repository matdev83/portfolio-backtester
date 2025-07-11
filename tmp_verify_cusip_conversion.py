import logging
import pandas as pd
import re
from pathlib import Path
import sys

# Add the src directory to the Python path to allow importing modules
script_dir = Path(__file__).parent
src_path = script_dir / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Temporarily set up basic logging for this script
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the functions from spy_holdings.py
try:
    from portfolio_backtester.universe_data.spy_holdings import _cusip_to_ticker, _process_holding_item, _load_cusip_mappings, _CUSIP_TICKER_CACHE
except ImportError as e:
    logger.error(f"Failed to import functions from spy_holdings.py. Make sure the 'src' directory is in your Python path. Error: {e}")
    sys.exit(1)

# Ensure CUSIP mappings are loaded for the test
_load_cusip_mappings()
logger.info(f"CUSIP_TICKER_CACHE after loading: {_CUSIP_TICKER_CACHE}")

# Mock classes to simulate EdgarTools objects for _process_holding_item
class MockInvestmentOrSecurity:
    def __init__(self, ticker=None, cusip=None, name=None, pct_value=0.0, balance=0.0, value_usd=0.0, asset_category='EC'):
        self.ticker = ticker
        self.cusip = cusip
        self.name = name
        self.pct_value = pct_value
        self.balance = balance
        self.value_usd = value_usd
        self.asset_category = asset_category

    def model_dump():
        return {
            "ticker": self.ticker,
            "cusip": self.cusip,
            "name": self.name,
            "pct_value": self.pct_value,
            "balance": self.balance,
            "value_usd": self.value_usd,
            "asset_category": self.asset_category
        }

def test_cusip_conversion():
    logger.info("--- Testing _cusip_to_ticker function directly ---")

    test_cases_cusip_to_ticker = {
        "037833100": "AAPL",  # Known CUSIP
        "594918104": "MSFT",  # Known CUSIP
        "02079K305": "GOOGL", # Known CUSIP
        "123456789": "123456789", # Unknown CUSIP
        "NOTACUSIP": "NOTACUSIP", # Not a CUSIP format
        "AAPL": "AAPL",       # Already a ticker
        "037833100.0": "037833100.0", # CUSIP with suffix, should not match
        "03783310": "03783310", # Too short
        "037833100A": "037833100A", # Too long
        None: None # Test None input
    }

    for cusip_input, expected_output in test_cases_cusip_to_ticker.items():
        result = _cusip_to_ticker(cusip_input)
        logger.info(f"Input: '{cusip_input}' -> Expected: '{expected_output}', Got: '{result}'")
        assert result == expected_output, f"Failed for input '{cusip_input}': Expected '{expected_output}', Got '{result}'"

    logger.info("--- _cusip_to_ticker tests passed ---\n")

    logger.info("--- Testing _process_holding_item function ---")

    test_date = pd.Timestamp('2023-01-15')

    # Test cases for _process_holding_item
    test_holding_items = [
        # Case 1: New format, CUSIP maps to ticker
        MockInvestmentOrSecurity(cusip="037833100", ticker="AAPL_RAW", name="Apple Inc."),
        # Case 2: New format, CUSIP does not map, raw_ticker exists
        MockInvestmentOrSecurity(cusip="999999999", ticker="UNKNOWN_RAW", name="Unknown Co."),
        # Case 3: New format, CUSIP does not map, no raw_ticker
        MockInvestmentOrSecurity(cusip="888888888", name="Another Unknown Co."),
        # Case 4: New format, no CUSIP, raw_ticker exists
        MockInvestmentOrSecurity(ticker="GOOGL", name="Google Inc."),
        # Case 5: New format, no CUSIP, no raw_ticker, name exists
        MockInvestmentOrSecurity(name="Microsoft Corp."),
        # Case 6: Old format (dict), CUSIP maps to ticker
        {"security_type": "common stock", "identifier": "594918104", "pct_value": 10.0, "shares": 100, "value": 1000},
        # Case 7: Old format (dict), CUSIP does not map
        {"security_type": "common stock", "identifier": "777777777", "pct_value": 5.0, "shares": 50, "value": 500},
        # Case 8: Old format (dict), identifier is already a ticker
        {"security_type": "common stock", "identifier": "TSLA", "pct_value": 2.0, "shares": 20, "value": 200},
        # Case 9: Old format (dict), identifier is a ticker with special chars
        {"security_type": "common stock", "identifier": "BRK.B", "pct_value": 1.5, "shares": 15, "value": 150},
        # Case 10: New format, ticker with special characters
        MockInvestmentOrSecurity(ticker="MSFT*", name="Microsoft Corp."),
        # Case 11: New format, CUSIP maps, raw_ticker with special characters
        MockInvestmentOrSecurity(cusip="037833100", ticker="AAPL-INC", name="Apple Inc."),
        # Case 12: New format, asset_category not 'EC'
        MockInvestmentOrSecurity(cusip="037833100", ticker="AAPL", asset_category='BD'),
    ]

    expected_results = [
        ("AAPL", "CUSIP 037833100 resolved to ticker AAPL"),
        ("UNKNOWNRAW", "CUSIP 999999999 not resolved, falling back to raw_ticker UNKNOWN_RAW"),
        ("888888888", "CUSIP 888888888 not resolved and no raw_ticker, using original CUSIP."),
        ("GOOGL", "No CUSIP, using raw_ticker GOOGL"),
        ("MICROSOFTCORP", "No ticker/CUSIP, falling back to name-based heuristic: MICROSOFTCORP"),
        ("MSFT", "ticker_to_use after _cusip_to_ticker=MSFT"),
        ("777777777", "ticker_to_use after _cusip_to_ticker=777777777"),
        ("TSLA", "ticker_to_use after _cusip_to_ticker=TSLA"),
        ("BRKB", "ticker_to_use after _cusip_to_ticker=BRK.B"),
        ("MSFT", "ticker=MSFT"), # From MSFT*
        ("AAPL", "CUSIP 037833100 resolved to ticker AAPL"), # CUSIP takes precedence, then cleaned
        (None, "No valid item processed for date 2023-01-15"), # BD asset category
    ]

    for i, item in enumerate(test_holding_items):
        logger.info(f"\n--- Processing test case {i+1} ---")
        result = _process_holding_item(item, test_date)
        logger.info(f"Result for test case {i+1}: {result}")

        # Basic assertion: check if the ticker in the result matches expectation
        if result:
            assert result[1] == expected_results[i][0], f"Test case {i+1} failed: Expected ticker '{expected_results[i][0]}', Got '{result[1]}'"
        else:
            assert expected_results[i][0] is None, f"Test case {i+1} failed: Expected None, Got '{result}'"

    logger.info("--- _process_holding_item tests passed ---\n")

    logger.info("All CUSIP conversion tests completed successfully!")

if __name__ == "__main__":
    test_cusip_conversion()
