import pandas as pd

file_path = 'C:/Users/Mateusz/source/repos/portfolio-backtester/data/kaggle_sp500_weights/sp500_historical.parquet'

df = pd.read_parquet(file_path, columns=['date'])
min_date = df['date'].min()
max_date = df['date'].max()

print(f"Kaggle S&P 500 data date range: {min_date} to {max_date}")