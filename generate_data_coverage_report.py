import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_report(start_year: int, end_year: int, output_path: Path):
    """
    Generates a data coverage report for SPY holdings on the last business day of each quarter.

    Args:
        start_year (int): The starting year for the report.
        end_year (int): The ending year for the report.
        output_path (Path): The path to save the CSV report.
    """
    holdings_file = Path("data/spy_holdings_full.parquet")

    if not holdings_file.exists():
        logger.error(f"Holdings file not found: {holdings_file}. Please ensure it has been generated.")
        return

    logger.info(f"Loading holdings data from {holdings_file}...")
    df = pd.read_parquet(holdings_file)
    df['date'] = pd.to_datetime(df['date'])

    # Filter for the specified date range
    start_date = pd.Timestamp(f"{start_year}-01-01")
    end_date = pd.Timestamp(f"{end_year}-12-31")
    df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()

    if df_filtered.empty:
        logger.warning(f"No data found for the period {start_year}-{end_year}.")
        return

    # Get all unique dates in the filtered DataFrame
    available_dates = pd.Series(df_filtered['date'].dt.normalize().unique()).sort_values()

    # Identify the last business day of each quarter within the range
    quarter_ends = []
    current_year = start_year
    while current_year <= end_year:
        for month in [3, 6, 9, 12]:
            quarter_end_candidate = pd.Timestamp(f"{current_year}-{month}-01") + pd.offsets.QuarterEnd(0)
            # Ensure it's a business day
            if quarter_end_candidate.weekday() >= 5: # Saturday or Sunday
                quarter_end_candidate = pd.Timestamp(quarter_end_candidate) - pd.tseries.offsets.BDay(1)
            
            # Find the closest available date on or before the quarter end
            closest_date = None
            # Filter available_dates using boolean indexing on the Series
            dates_on_or_before = available_dates[available_dates <= quarter_end_candidate]
            if not dates_on_or_before.empty:
                closest_date = dates_on_or_before.max()
            
            if closest_date and closest_date >= start_date and closest_date <= end_date:
                quarter_ends.append(closest_date)
        current_year += 1
    
    # Remove duplicates and sort
    quarter_ends = sorted(list(set(quarter_ends)))

    if not quarter_ends:
        logger.warning(f"No quarter-end dates found with available data in the period {start_year}-{end_year}.")
        return

    report_lines = []
    for q_date in quarter_ends:
        holdings_on_date = df_filtered[df_filtered['date'] == q_date]
        if not holdings_on_date.empty:
            num_tickers = len(holdings_on_date)
            # Format: Ticker:Weight%
            pairs = [f"{row['ticker']}:{row['weight_pct']:.2f}%" for _, row in holdings_on_date.iterrows()]
            report_lines.append(f"{q_date.strftime('%Y-%m-%d')};{num_tickers} tickers;{';'.join(pairs)}")
        else:
            report_lines.append(f"{q_date.strftime('%Y-%m-%d')};0 tickers;No data")
            logger.warning(f"No holdings data found for {q_date.strftime('%Y-%m-%d')}")

    # Save the report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for line in report_lines:
            f.write(line + '\n')
    logger.info(f"Data coverage report saved to {output_path}")

if __name__ == "__main__":
    # Generate report for 2004-2015
    generate_report(2004, 2015, Path("data/data_coverage_results.csv"))
