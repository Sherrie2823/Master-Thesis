import yfinance as yf
import pandas as pd

# Test basic functionality
print("Testing yfinance...")
data = yf.download('JPM', start='2024-01-01', end='2024-01-31')
print("✅ yfinance working!")
print(f"Downloaded {len(data)} days of JPM data")
print(data.head())