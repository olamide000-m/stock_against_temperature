# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 11:28:08 2025

@author: DELL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# input the dataset from desktop
stock_data = pd.read_csv("HistoricalData_1742632061390.csv")
weather_data = pd.read_csv("NYC_Central_Park_weather_1869-2022.csv")


# Convert date columns to datetime format
stock_data["Date"] = pd.to_datetime(stock_data["Date"])
weather_data["DATE"] = pd.to_datetime(weather_data["DATE"])



# Rename weather date column for merging
weather_data.rename(columns={"DATE": "Date"}, inplace=True)

# Merge datasets on Date
merged_df = pd.merge(stock_data, weather_data, on="Date", how="inner")

# Convert stock price columns from string to float
price_columns = ["Close/Last", "Open", "High", "Low"]
for col in price_columns:
    merged_df[col] = merged_df[col].replace({'\$': ''}, regex=True).astype(float)

# Convert Volume to integer
merged_df["Volume"] = merged_df["Volume"].astype(int)

# --- Visualization ---
sns.set(style="darkgrid")
fig, axes = plt.subplots(3, 1, figsize=(12, 12))

# Stock Price Trend
axes[0].plot(merged_df["Date"], merged_df["Close/Last"], color="blue", label="Stock Price")
axes[0].set_title("Stock Closing Price Over Time")
axes[0].set_ylabel("Close/Last Price ($)")
axes[0].tick_params(axis='x', rotation=45)
axes[0].legend()  # Uncommented to display the legend

# Show the plot
plt.show()

# Temperature Trends
axes[1].plot(merged_df["Date"], merged_df["TMIN"], color="red", label="Min Temperature (°F)")
axes[1].plot(merged_df["Date"], merged_df["TMAX"], color="orange", label="Max Temperature (°F)")
axes[1].set_title("Daily Temperature Over Time")
axes[1].set_ylabel("Temperature (°F)")
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend()
plt.show()
# Scatter Plot: Stock Price vs Temperature
sns.scatterplot(data=merged_df, x="TMAX", y="Close/Last", ax=axes[2], color="purple", alpha=0.6)
axes[2].set_title("Stock Price vs. Max Temperature")
axes[2].set_xlabel("Max Temperature (°F)")
axes[2].set_ylabel("Stock Closing Price ($)")

plt.tight_layout()
plt.show()

# --- Regression Analysis ---
X = merged_df[["TMIN", "TMAX", "PRCP", "SNOW"]]
y = merged_df["Close/Last"]

# Add constant term
X = sm.add_constant(X)

# Run OLS regression
model = sm.OLS(y, X).fit()

# Print results
print(model.summary())
















