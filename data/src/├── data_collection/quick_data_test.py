"""
Quick Data Test - Check if banking data was collected successfully
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os


def test_banking_data():
    print("🔍 Quick Banking Data Test")
    print("=" * 40)

    # Banking stocks to test
    banks = ['JPM', 'BAC', 'WFC', 'C', 'GS']

    try:
        # Download data
        print("📊 Downloading sample banking data...")
        data = yf.download(banks, start='2015-01-01', end='2024-12-31', progress=False)

        # Check data structure
        print(f"✅ Data downloaded successfully!")
        print(f"📈 Data shape: {data.shape}")
        print(f"📅 Date range: {data.index[0]} to {data.index[-1]}")

        # Get adjusted close prices
        if isinstance(data.columns, pd.MultiIndex):
            adj_close = data.xs('Adj Close', axis=1, level=0)
        else:
            adj_close = data['Adj Close'] if 'Adj Close' in data.columns else data

        print(f"💰 Price data shape: {adj_close.shape}")
        print(f"🏦 Banks in dataset: {list(adj_close.columns)}")

        # Calculate basic stats
        returns = adj_close.pct_change().dropna()
        annual_returns = returns.mean() * 252 * 100
        annual_vol = returns.std() * np.sqrt(252) * 100

        print("\n📊 PERFORMANCE SUMMARY:")
        print("-" * 40)
        for bank in adj_close.columns:
            print(f"{bank}: {annual_returns[bank]:.1f}% return, {annual_vol[bank]:.1f}% volatility")

        # Save a sample to check
        sample_file = '../../../real_data/sample_banking_data.csv'
        os.makedirs('../../../real_data', exist_ok=True)
        adj_close.tail(10).to_csv(sample_file)
        print(f"\n💾 Sample data saved to: {sample_file}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = test_banking_data()
    if success:
        print("\n🎉 Data collection is working perfectly!")
        print("✅ Ready to proceed with full analysis!")
    else:
        print("\n⚠️  Need to troubleshoot data collection")