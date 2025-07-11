import pandas as pd

file_path = 'C:/Users/Mateusz/source/repos/portfolio-backtester/cache/sec_holdings/0000884394_2004-01-01_2025-06-30.parquet'
df = pd.read_parquet(file_path)
print('--- Head ---')
print(df.head())
print('\n--- Info ---')
print(df.info())