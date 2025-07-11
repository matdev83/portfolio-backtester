import pandas as pd
import datetime as dt
import logging
from pathlib import Path
import sys

# Add the src directory to the Python path
script_dir = Path(__file__).parent
src_path = script_dir / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Set up basic logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from portfolio_backtester.universe_data.spy_holdings import get_top_weight_sp500_components
except ImportError as e:
    logger.error(f"Failed to import get_top_weight_sp500_components. Error: {e}")
    sys.exit(1)

def get_last_n_business_days(end_date: pd.Timestamp, n: int) -> list[pd.Timestamp]:
    """Returns the last N business days leading up to and including end_date."""
    business_days = pd.bdate_range(end=end_date, periods=n)
    return sorted(business_days.tolist(), reverse=True)

def main():
    logger.info("Demonstrating top 5 S&P holdings for the last 10 available dates.")

    # Use a date within the Kaggle data range to avoid long downloads
    # KAGGLE_SP500_END_DATE is 2024-10-30
    end_date_for_demo = pd.Timestamp('2024-10-30')
    num_dates = 10
    top_n = 5

    dates_to_check = get_last_n_business_days(end_date_for_demo, num_dates)

    logger.info(f"Checking top {top_n} holdings for {num_dates} dates ending {end_date_for_demo.strftime('%Y-%m-%d')}:")

    for date in dates_to_check:
        try:
            components = get_top_weight_sp500_components(date, top_n=top_n)
            logger.info(f"--- Date: {date.strftime('%Y-%m-%d')} ---")
            if components:
                for ticker, weight in components:
                    logger.info(f"  Ticker: {ticker}, Weight: {weight:.2f}%")
            else:
                logger.info("  No components found for this date.")
        except ValueError as e:
            logger.warning(f"Could not retrieve components for {date.strftime('%Y-%m-%d')}: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred for date {date.strftime('%Y-%m-%d')}: {e}")

    logger.info("\nDemonstration complete.")

if __name__ == "__main__":
    main()
