# S&P 500 Universe Data Management

This document outlines the process for managing the historical S&P 500 index constituents and their weights within this project. The goal is to maintain a comprehensive and up-to-date dataset, prioritizing high-granularity data (daily) and ensuring data integrity.

## Data Sources and Prioritization

The S&P 500 universe data is compiled from multiple sources, with a clear prioritization to ensure the highest quality and granularity:

1.  **Kaggle Dataset (Primary Historical Source):** Provides daily S&P 500 constituent weights from 2009-01-30 to 2024-10-30. This dataset is considered the "frozen" and authoritative source for its covered date range.
2.  **SSGA Daily Basket XLSX:** Provides daily S&P 500 ETF (SPY) holdings from approximately 2015-03-16 to present (with a 1-day lag).
3.  **SEC N-PORT-P XML (Monthly):** Provides monthly holdings data from 2019-present.
4.  **SEC N-Q HTML (Quarterly):** Provides quarterly holdings data from 2004-2018.

The system is designed to automatically prioritize the Kaggle data for its covered period. Any data fetched from SSGA or SEC sources for dates within the Kaggle dataset's range will be ignored to prevent overwriting or degrading the high-quality Kaggle data.

## Location of Scripts

All scripts related to S&P 500 universe data management are located in:
`src/portfolio_backtester/universe_data/`

*   `spy_holdings.py`: The main script for downloading and aggregating S&P 500 holdings data from SSGA and SEC. It now intelligently integrates with the Kaggle data.
*   `load_kaggle_sp500_data.py`: A utility script used for the initial loading and conversion of the Kaggle S&P 500 historical CSV data into a Parquet file, which is then consumed by `spy_holdings.py`.

## Initial Setup: Loading Kaggle Data

Before performing any regular updates, you must first load the Kaggle S&P 500 historical data. This is a one-time setup process.

1.  **Download the Kaggle Dataset:**
    *   Go to the Kaggle dataset page: `https://www.kaggle.com/datasets/akshaytolwani/s-and-p-500-constituents-with-weights-2009-2024`
    *   Manually download the `sp500_historical.csv` file using the "Download" button.
    *   Place the downloaded `sp500_historical.csv` file into the `data/` directory of your project.

2.  **Process the Kaggle Data:**
    *   Ensure your Python virtual environment is activated.
    *   Run the `load_kaggle_sp500_data.py` script to convert the CSV into a Parquet file, which is the preferred format for the project:

        ```bash
        python src/portfolio_backtester/universe_data/load_kaggle_sp500_data.py
        ```
    *   This will create `data/kaggle_sp500_weights/sp500_historical.parquet`. This Parquet file will serve as the primary historical source for the S&P 500 universe.

## Regular Updates: Additive Data Collection

The `spy_holdings.py` script is designed to perform additive updates to your S&P 500 universe data. It will fetch the latest available data from SSGA and SEC, and intelligently merge it with your existing dataset, ensuring the Kaggle data's integrity.

### How it Works

*   When `spy_holdings.py` is run, it first checks for the presence of `data/spy_holdings_full.parquet`.
*   It then loads the Kaggle data (`data/kaggle_sp500_weights/sp500_historical.parquet`) and prioritizes it for its covered date range (2009-01-30 to 2024-10-30).
*   It fetches new data from SSGA and SEC for dates *after* the latest data available in your combined dataset (or after the Kaggle data's end date if no newer data exists).
*   The newly fetched data is then appended to the existing dataset, creating `data/spy_holdings_full.parquet`.

### Performing an Update

To update the S&P 500 universe data, run the `spy_holdings.py` script. You can specify the start and end dates for the data collection.

```bash
# Example: Update data from the earliest SEC N-Q (2004-01-01) to today
# This will fetch any missing data and append it.
python src/portfolio_backtester/universe_data/spy_holdings.py --start 2004-01-01 --end $(date +%Y-%m-%d) --out spy_holdings_full.parquet

# On Windows, you might need to use a different way to get today's date:
# For example, using PowerShell:
# python src/portfolio_backtester/universe_data/spy_holdings.py --start 2004-01-01 --end (Get-Date -Format "yyyy-MM-dd") --out spy_holdings_full.parquet
```

**Important Notes:**

*   The `--out` parameter specifies the output filename for the aggregated history. It is recommended to always use `spy_holdings_full.parquet` to maintain a single, comprehensive history file.
*   The script will automatically handle caching of intermediate SSGA and SEC data in the `cache/` directory.
*   The `--update` flag is implicitly handled by the script's logic when `spy_holdings_full.parquet` already exists.
*   The `--rebuild` flag (e.g., `python src/portfolio_backtester/universe_data/spy_holdings.py --rebuild --out spy_holdings_full.parquet`) will force a rebuild of `spy_holdings_full.parquet` starting from the Kaggle data, then fetching all subsequent data. Use this if you suspect data corruption or want to re-aggregate everything.

## Data Integrity and Freezing

The Kaggle S&P 500 data (from 2009-01-30 to 2024-10-30) is considered "frozen." This means:

*   Any data fetched from SSGA or SEC for dates within this range will be discarded.
*   The `spy_holdings_full.parquet` file will always prioritize the Kaggle data for this period.
*   This ensures that the most reliable and granular historical data is preserved and used by the backtesting system.

By following these steps, you can ensure a robust and up-to-date S&P 500 universe dataset for your backtesting needs.
