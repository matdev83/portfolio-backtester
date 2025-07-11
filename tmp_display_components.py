import pandas as pd
import logging
import datetime as dt
from pathlib import Path

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import spy_holdings functions AFTER setting up basicConfig for logging
from portfolio_backtester.universe_data.spy_holdings import get_top_weight_sp500_components, get_spy_holdings, _HISTORY_DF

def display_top_components_for_recent_dates(num_dates: int = 10, top_n_components: int = 5):
    logger.info(f"Displaying top {top_n_components} S&P 500 components for the {num_dates} most recent dates.")

    # Determine the path to spy_holdings_full.parquet
    script_dir = Path(__file__).parent
    spy_holdings_full_path = script_dir / "data" / "spy_holdings_full.parquet"

    if not spy_holdings_full_path.exists():
        logger.error(f"spy_holdings_full.parquet not found at {spy_holdings_full_path}. Please run spy_holdings.py first to generate the full history.")
        return

    # Read all unique dates from the aggregated history file
    try:
        full_history_dates = pd.read_parquet(spy_holdings_full_path, columns=['date'])['date'].dt.normalize().unique()
        full_history_dates = pd.Series(full_history_dates).sort_values().unique() # Sort and ensure uniqueness
    except Exception as e:
        logger.error(f"Failed to read dates from spy_holdings_full.parquet: {e}")
        return

    if not full_history_dates.size:
        logger.error("No dates found in spy_holdings_full.parquet. Cannot display components.")
        return

    recent_dates = pd.Series(full_history_dates).tail(num_dates).tolist() # Get the last num_dates
    recent_dates.reverse() # Reverse to get most recent first

    logger.info(f"Most recent {num_dates} dates: {[d.strftime('%Y-%m-%d') for d in recent_dates]}")

    for date in recent_dates:
        try:
            # get_top_weight_sp500_components will handle loading _HISTORY_DF internally
            top_components = get_top_weight_sp500_components(date, top_n=top_n_components, exact=True)
            logger.info(f"\n--- Top {top_n_components} components for {date:%Y-%m-%d} ---")
            if top_components:
                for ticker, weight in top_components:
                    logger.info(f"  Ticker: {ticker}, Weight: {weight:.4f}%")
            else:
                logger.info("  No components found for this date.")
        except ValueError as e:
            logger.warning(f"Could not retrieve top components for {date:%Y-%m-%d}: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred for date {date:%Y-%m-%d}: {e}")

if __name__ == "__main__":
    display_top_components_for_recent_dates()