import pandas as pd
import os

# Define paths
csv_file_path = 'C:/Users/Mateusz/source/repos/portfolio-backtester/data/sp500_historical.csv'
output_dir = 'C:/Users/Mateusz/source/repos/portfolio-backtester/data/kaggle_sp500_weights'
output_parquet_path = os.path.join(output_dir, 'sp500_historical.parquet')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

print(f"Loading data from: {csv_file_path}")

# Load the CSV file
df = pd.read_csv(csv_file_path)

print("Converting 'date' column to datetime...")
# Convert 'date' column to datetime objects
df['date'] = pd.to_datetime(df['date'])

print(f"Saving processed data to: {output_parquet_path}")
# Save the DataFrame to a Parquet file
df.to_parquet(output_parquet_path, index=False)

print("Data loading complete.")
print("\n--- Info of saved Parquet file ---")
# Verify the saved file by reading its info
loaded_df = pd.read_parquet(output_parquet_path)
print(loaded_df.info())