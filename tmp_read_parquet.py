import pandas as pd

file_path = 'C:/Users/Mateusz/source/repos/portfolio-backtester/data/spy_holdings_full.parquet'
df = pd.read_parquet(file_path)
print('--- Head ---')
print(df.head())
print('\n--- Info ---')
print(df.info())