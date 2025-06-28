import pandas as pd

# Path to your historical data file
input_csv = "historical_data/NIFTYBEES_historical_data.csv"
output_csv = "historical_data/NIFTYBEES_historical_data_cleaned.csv"

# Load the data
df = pd.read_csv(input_csv)

# Remove rows where volume is zero or missing
df_clean = df[df['volume'] > 0].copy()

# Optionally, drop rows with any missing OHLCV data
df_clean.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)

# Save the cleaned data
df_clean.to_csv(output_csv, index=False)

print(f"Cleaned data saved to {output_csv}. Original rows: {len(df)}, Cleaned rows: {len(df_clean)}")