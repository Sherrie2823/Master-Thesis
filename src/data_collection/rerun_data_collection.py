"""
Re-run Data Collection with Fixed Paths
Ensures all data is saved to the correct location
"""

import yfinance as yf
import pandas as pd
import numpy as np
from fredapi import Fred
import os
from datetime import datetime


def create_directories():
    """Create necessary directories."""
    project_root = "/Users/sherrie/PycharmProjects/PythonProject/AI-Enhanced-Portfolio-Optimization"
    data_dir = os.path.join(project_root, "data")
    results_dir = os.path.join(project_root, "results")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"📁 Project root: {project_root}")
    print(f"📁 Data directory: {data_dir}")
    print(f"📁 Results directory: {results_dir}")

    return data_dir, results_dir


def collect_banking_data(data_dir):
    """Collect banking sector data."""
    print("\n🏦 Collecting Banking Data...")

    banking_stocks = ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'COF',
                      'BK', 'STT', 'AXP', 'SCHW', 'CB']

    try:
        # Download data
        print("📊 Downloading banking stock data...")
        data = yf.download(banking_stocks, start='2015-01-01', end='2024-12-31', progress=True)

        # Process data
        if isinstance(data.columns, pd.MultiIndex):
            prices = data.xs('Close', axis=1, level=0)
        else:
            prices = data

        # Calculate returns
        returns = prices.pct_change().dropna()

        # Calculate correlation
        correlation = returns.corr()

        # Save files
        prices.to_csv(os.path.join(data_dir, 'banking_prices.csv'))
        returns.to_csv(os.path.join(data_dir, 'banking_returns.csv'))
        correlation.to_csv(os.path.join(data_dir, 'banking_correlation.csv'))

        print(f"✅ Banking data saved:")
        print(f"   📊 Prices: {prices.shape}")
        print(f"   📊 Returns: {returns.shape}")
        print(f"   📊 Correlation: {correlation.shape}")

        return True

    except Exception as e:
        print(f"❌ Error collecting banking data: {e}")
        return False


def collect_fred_data(data_dir):
    """Collect FRED economic data."""
    print("\n📈 Collecting FRED Economic Data...")

    try:
        # Initialize FRED with your API key
        fred = Fred(api_key='c9619c84919b1fb2be8d5a5dd96cd73c')

        # Key economic indicators
        indicators = {
            'fed_funds_rate': 'FEDFUNDS',
            'treasury_10y': 'DGS10',
            'treasury_2y': 'DGS2',
            'unemployment_rate': 'UNRATE',
            'cpi_inflation': 'CPIAUCSL',
            'vix': 'VIXCLS',
            'term_spread': 'T10Y2Y'
        }

        economic_data = pd.DataFrame()

        for name, series_id in indicators.items():
            try:
                print(f"   📊 {name}...")
                data = fred.get_series(series_id, start='2015-01-01', end='2024-12-31')
                economic_data[name] = data
                print(f"   ✅ {name}: {len(data)} observations")
            except Exception as e:
                print(f"   ❌ {name}: {e}")

        # Fill missing values and resample to daily
        economic_data = economic_data.resample('D').ffill()

        # Save data
        economic_data.to_csv(os.path.join(data_dir, 'fred_economic_data.csv'))

        print(f"✅ FRED data saved: {economic_data.shape}")
        return True

    except Exception as e:
        print(f"❌ Error collecting FRED data: {e}")
        return False


def collect_treasury_data(data_dir):
    """Collect Treasury data using FRED."""
    print("\n🏛️  Collecting Treasury Data...")

    try:
        fred = Fred(api_key='c9619c84919b1fb2be8d5a5dd96cd73c')

        # Treasury yields
        treasury_series = {
            '3_month': 'DGS3MO',
            '1_year': 'DGS1',
            '2_year': 'DGS2',
            '5_year': 'DGS5',
            '10_year': 'DGS10',
            '30_year': 'DGS30'
        }

        treasury_data = pd.DataFrame()

        for name, series_id in treasury_series.items():
            try:
                print(f"   📊 {name}...")
                data = fred.get_series(series_id, start='2015-01-01', end='2024-12-31')
                treasury_data[name] = data
                print(f"   ✅ {name}: {len(data)} observations")
            except Exception as e:
                print(f"   ❌ {name}: {e}")

        # Calculate derived metrics
        if '10_year' in treasury_data.columns and '2_year' in treasury_data.columns:
            treasury_data['yield_slope_10y2y'] = treasury_data['10_year'] - treasury_data['2_year']

        if '3_month' in treasury_data.columns:
            treasury_data['risk_free_rate'] = treasury_data['3_month']

        # Fill missing values and resample to daily
        treasury_data = treasury_data.resample('D').ffill()

        # Save data
        treasury_data.to_csv(os.path.join(data_dir, 'treasury_complete.csv'))

        print(f"✅ Treasury data saved: {treasury_data.shape}")
        return True

    except Exception as e:
        print(f"❌ Error collecting Treasury data: {e}")
        return False


def verify_data_files(data_dir):
    """Verify all data files are created correctly."""
    print("\n🔍 Verifying Data Files...")

    expected_files = [
        'banking_prices.csv',
        'banking_returns.csv',
        'banking_correlation.csv',
        'fred_economic_data.csv',
        'treasury_complete.csv'
    ]

    success_count = 0

    for filename in expected_files:
        filepath = os.path.join(data_dir, filename)

        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, index_col=0)
                size_mb = os.path.getsize(filepath) / (1024 ** 2)
                print(f"✅ {filename}: {df.shape} ({size_mb:.1f} MB)")
                success_count += 1
            except Exception as e:
                print(f"❌ {filename}: Error reading - {e}")
        else:
            print(f"❌ {filename}: File not found")

    print(f"\n📊 Summary: {success_count}/{len(expected_files)} files successfully created")

    if success_count == len(expected_files):
        print("🎉 All data files created successfully!")
        return True
    else:
        print("⚠️  Some data files missing or corrupted")
        return False


def main():
    """Main function to re-run all data collection."""
    print("🚀 AI-Enhanced Portfolio Optimization - Data Collection")
    print("=" * 70)
    print("Re-running data collection with fixed paths...")

    # Create directories
    data_dir, results_dir = create_directories()

    # Collect all data
    banking_success = collect_banking_data(data_dir)
    fred_success = collect_fred_data(data_dir)
    treasury_success = collect_treasury_data(data_dir)

    # Verify results
    all_success = verify_data_files(data_dir)

    print("\n" + "=" * 70)
    if all_success:
        print("✅ DATA COLLECTION COMPLETED SUCCESSFULLY!")
        print("🎯 Ready to run data exploration!")
        print(f"📁 All files saved to: {data_dir}")
    else:
        print("⚠️  DATA COLLECTION PARTIALLY COMPLETED")
        print("🔄 Some files may need manual checking")

    return all_success


if __name__ == "__main__":
    success = main()