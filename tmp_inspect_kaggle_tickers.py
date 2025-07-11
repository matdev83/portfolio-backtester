import pandas as pd

file_path = 'C:/Users/Mateusz/source/repos/portfolio-backtester/data/kaggle_sp500_weights/sp500_historical.parquet'

# Read a larger portion of the file to inspect ticker and name columns
df = pd.read_parquet(file_path)

print('--- Kaggle Data Sample ---')
print(df[['ticker', 'name']].head(20))
print('\n--- Unique Ticker Lengths in Kaggle Data ---')
print(df['ticker'].apply(len).value_counts())
print('\n--- Unique Ticker isalnum() status in Kaggle Data ---')
print(df['ticker'].apply(lambda x: x.isalnum()).value_counts())
