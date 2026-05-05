import pandas as pd
import numpy as np

np.random.seed(42)

# Create 2 years of daily data
dates = pd.date_range(start="2024-01-01", end="2025-12-31", freq="D")

df = pd.DataFrame({"date": dates})

# Basic time features
df["day_of_week"] = df["date"].dt.day_name()
df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)
df["month"] = df["date"].dt.month   

# Sales pattern
base_sales = 1200

# Weekend boost
weekend_boost = df["is_weekend"] * 500

# Monthly seasonality (more realistic trend)
monthly_seasonality = df["month"].map({
    1: -100, 2: -80, 3: 50, 4: 100,
    5: 150, 6: 200, 7: 250, 8: 220,
    9: 100, 10: 80, 11: 150, 12: 300
})

# Yearly growth trend (improves realism)
df["time_index"] = np.arange(len(df))
trend = df["time_index"] * 0.5   # gradual increase over time

# Random noise
noise = np.random.normal(0, 100, len(df))

# Final sales calculation
df["sales"] = base_sales + weekend_boost + monthly_seasonality + trend + noise

# Ensure minimum sales
df["sales"] = df["sales"].apply(lambda x: max(500, int(x)))

# Sort (important for time-series)
df = df.sort_values("date")

# Save dataset (same filename for your dashboard)
df.to_csv("data/restaurant_sales.csv", index=False)

# Output check
print("Dataset Preview:")
print(df.head())
print(df.tail())

print("\nDataset Shape:", df.shape)
print("Date Range:", df["date"].min(), "to", df["date"].max())