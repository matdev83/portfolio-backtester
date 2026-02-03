import yfinance as yf
import pandas as pd

def check_vix(ticker):
    print(f"Checking ticker: {ticker}")
    try:
        data = yf.download(ticker, start="2018-01-01", end="2023-01-31")
        if data.empty:
            print(f"No data found for {ticker}")
        else:
            print(f"Successfully fetched {len(data)} rows for {ticker}")
            print(data.tail())
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")

if __name__ == "__main__":
    check_vix("^VIX")
    check_vix("INDEXCBOE:VIX")
