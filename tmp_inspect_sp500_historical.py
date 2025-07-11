import pandas as pd

file_path = 'C:/Users/Mateusz/source/repos/portfolio-backtester/data/sp500_historical.csv'

# Read only a small portion of the file to infer schema and see head
df_head = pd.read_csv(file_path, nrows=5)
print('--- Head ---')
print(df_head)

# Read a larger chunk to get better dtype inference, but still not the whole file
# This is useful for very large files where nrows=5 might not capture all column types
# For now, we'll stick to head for quick inspection.

# To get column names and dtypes without loading all data, we can read the header
# and then infer types from a sample if needed, but head() usually gives enough.
print('\n--- Info (from head) ---')
print(df_head.info())