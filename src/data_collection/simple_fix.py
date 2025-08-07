"""
Simple Fix - Create data in the right place and test immediately
"""

import yfinance as yf
import pandas as pd
import os


def main():
    print("🔧 Simple Fix - Creating data in correct location")
    print("=" * 60)

    # Define the absolute correct path
    project_root = "/Users/sherrie/PycharmProjects/PythonProject/AI-Enhanced-Portfolio-Optimization"
    data_dir = os.path.join(project_root, "data")

    print(f"📁 Target data directory: {data_dir}")

    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Quick test - download just JPM data and save it
    print("\n🧪 Quick Test - Downloading JPM data...")

    try:
        # Download JPM data
        jpm_data = yf.download('JPM', start='2024-01-01', end='2024-12-31')

        # Save to test location
        test_file = os.path.join(data_dir, 'test_jpm_data.csv')
        jpm_data.to_csv(test_file)

        print(f"✅ Test file created: {test_file}")
        print(f"📊 JPM data shape: {jpm_data.shape}")

        # Verify file exists
        if os.path.exists(test_file):
            file_size = os.path.getsize(test_file) / 1024  # KB
            print(f"✅ File verified: {file_size:.1f} KB")

            # Try to read it back
            test_read = pd.read_csv(test_file, index_col=0)
            print(f"✅ File readable: {test_read.shape}")

            print("\n🎉 SUCCESS! Path is working correctly.")
            print(f"🎯 Ready to collect full dataset to: {data_dir}")

            return data_dir

        else:
            print("❌ File was not created successfully")
            return None

    except Exception as e:
        print(f"❌ Error in test: {e}")
        return None


def collect_minimal_dataset(data_dir):
    """Collect a minimal but complete dataset."""
    print("\n📊 Collecting Minimal Complete Dataset...")

    try:
        # Banking stocks (just top 5 for speed)
        banking_stocks = ['JPM', 'BAC', 'WFC', 'C', 'GS']

        print("📊 Downloading banking data...")
        banking_data = yf.download(banking_stocks, start='2023-01-01', end='2024-12-31', progress=False)

        if isinstance(banking_data.columns, pd.MultiIndex):
            prices = banking_data.xs('Close', axis=1, level=0)
        else:
            prices = banking_data

        returns = prices.pct_change().dropna()

        # Save banking data
        prices.to_csv(os.path.join(data_dir, 'banking_prices.csv'))
        returns.to_csv(os.path.join(data_dir, 'banking_returns.csv'))
        returns.corr().to_csv(os.path.join(data_dir, 'banking_correlation.csv'))

        print(f"✅ Banking data saved: {prices.shape}")

        # Create dummy economic data (simplified)
        dates = prices.index
        economic_data = pd.DataFrame({
            'fed_funds_rate': [4.5] * len(dates),
            'treasury_10y': [4.2] * len(dates),
            'unemployment_rate': [3.8] * len(dates),
            'vix': [20.0] * len(dates)
        }, index=dates)

        economic_data.to_csv(os.path.join(data_dir, 'fred_economic_data.csv'))
        print(f"✅ Economic data created: {economic_data.shape}")

        # Create dummy treasury data
        treasury_data = pd.DataFrame({
            '3_month': [4.8] * len(dates),
            '2_year': [4.5] * len(dates),
            '10_year': [4.2] * len(dates),
            'yield_slope_10y2y': [-0.3] * len(dates)
        }, index=dates)

        treasury_data.to_csv(os.path.join(data_dir, 'treasury_complete.csv'))
        print(f"✅ Treasury data created: {treasury_data.shape}")

        return True

    except Exception as e:
        print(f"❌ Error collecting data: {e}")
        return False


def test_data_explorer(data_dir):
    """Test the data explorer with our created data."""
    print("\n🧪 Testing Data Explorer...")

    expected_files = [
        'banking_prices.csv',
        'banking_returns.csv',
        'banking_correlation.csv',
        'fred_economic_data.csv',
        'treasury_complete.csv'
    ]

    all_exist = True

    for filename in expected_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, index_col=0)
                print(f"✅ {filename}: {df.shape}")
            except Exception as e:
                print(f"❌ {filename}: Error reading - {e}")
                all_exist = False
        else:
            print(f"❌ {filename}: Not found")
            all_exist = False

    if all_exist:
        print("\n🎉 All files present! Data explorer should work now.")

        # Quick analysis
        print("\n📊 Quick Analysis:")
        prices = pd.read_csv(os.path.join(data_dir, 'banking_prices.csv'), index_col=0, parse_dates=True)
        returns = pd.read_csv(os.path.join(data_dir, 'banking_returns.csv'), index_col=0, parse_dates=True)

        total_returns = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
        print("🏆 Stock Performance:")
        for stock, ret in total_returns.items():
            print(f"   {stock}: {ret:+.1f}%")

        return True
    else:
        print("\n❌ Some files missing. Need to debug further.")
        return False


if __name__ == "__main__":
    # Step 1: Test basic functionality
    data_dir = main()

    if data_dir:
        # Step 2: Create minimal dataset
        success = collect_minimal_dataset(data_dir)

        if success:
            # Step 3: Test data explorer
            test_data_explorer(data_dir)

            print("\n" + "=" * 60)
            print("✅ SETUP COMPLETE!")
            print("🎯 Now you can run the data explorer:")
            print("   working_data_explorer.py")
            print("=" * 60)