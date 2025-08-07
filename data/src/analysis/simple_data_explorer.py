"""
Simple Data Explorer for AI-Enhanced Portfolio Optimization
Author: MSc Banking and Digital Finance Student
Date: July 2025

Simplified version that focuses on loading and exploring the collected data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


def find_project_root():
    """Find the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up directories until we find the data folder
    search_dir = current_dir
    for _ in range(5):  # Search up to 5 levels
        data_path = os.path.join(search_dir, 'data')
        if os.path.exists(data_path):
            return search_dir
        search_dir = os.path.dirname(search_dir)

    # If not found, assume we're in the project root
    return os.getcwd()


def load_data():
    """Load all available datasets."""
    project_root = find_project_root()
    data_dir = os.path.join(project_root, 'data')

    print("🚀 Simple Data Explorer")
    print("=" * 50)
    print(f"📁 Project root: {project_root}")
    print(f"📁 Data directory: {data_dir}")

    # Check if data directory exists
    if not os.path.exists(data_dir):
        print("❌ Data directory not found!")
        return None

    # List all files in data directory
    data_files = os.listdir(data_dir)
    print(f"\n📄 Available data files:")
    for file in sorted(data_files):
        file_path = os.path.join(data_dir, file)
        size_mb = os.path.getsize(file_path) / (1024 ** 2)
        print(f"   {file} ({size_mb:.1f} MB)")

    # Load datasets
    datasets = {}

    # Banking data
    try:
        banking_prices_path = os.path.join(data_dir, 'banking_prices.csv')
        if os.path.exists(banking_prices_path):
            datasets['banking_prices'] = pd.read_csv(banking_prices_path, index_col=0, parse_dates=True)
            print(f"✅ Loaded banking prices: {datasets['banking_prices'].shape}")
        else:
            print("⚠️  Banking prices file not found")
    except Exception as e:
        print(f"❌ Error loading banking prices: {e}")

    try:
        banking_returns_path = os.path.join(data_dir, 'banking_returns.csv')
        if os.path.exists(banking_returns_path):
            datasets['banking_returns'] = pd.read_csv(banking_returns_path, index_col=0, parse_dates=True)
            print(f"✅ Loaded banking returns: {datasets['banking_returns'].shape}")
        else:
            print("⚠️  Banking returns file not found")
    except Exception as e:
        print(f"❌ Error loading banking returns: {e}")

    # Economic data
    try:
        fred_path = os.path.join(data_dir, 'fred_economic_data.csv')
        if os.path.exists(fred_path):
            datasets['fred_data'] = pd.read_csv(fred_path, index_col=0, parse_dates=True)
            print(f"✅ Loaded FRED economic data: {datasets['fred_data'].shape}")
        else:
            print("⚠️  FRED economic data file not found")
    except Exception as e:
        print(f"❌ Error loading FRED data: {e}")

    # Treasury data
    try:
        treasury_path = os.path.join(data_dir, 'treasury_complete.csv')
        if os.path.exists(treasury_path):
            datasets['treasury_data'] = pd.read_csv(treasury_path, index_col=0, parse_dates=True)
            print(f"✅ Loaded Treasury data: {datasets['treasury_data'].shape}")
        else:
            print("⚠️  Treasury data file not found")
    except Exception as e:
        print(f"❌ Error loading Treasury data: {e}")

    return datasets


def analyze_banking_data(datasets):
    """Analyze banking sector data."""
    if 'banking_prices' not in datasets or 'banking_returns' not in datasets:
        print("❌ Banking data not available for analysis")
        return

    print("\n" + "=" * 50)
    print("🏦 BANKING SECTOR ANALYSIS")
    print("=" * 50)

    prices = datasets['banking_prices']
    returns = datasets['banking_returns']

    print(f"\n📊 Data Overview:")
    print(f"   Banks: {len(prices.columns)}")
    print(f"   Date Range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"   Trading Days: {len(prices)}")

    # Performance analysis
    latest_prices = prices.iloc[-1]
    first_prices = prices.iloc[0]
    total_returns = (latest_prices / first_prices - 1) * 100

    print(f"\n🏆 TOP PERFORMERS (Total Return):")
    for i, (stock, ret) in enumerate(total_returns.sort_values(ascending=False).head(5).items(), 1):
        print(f"   {i}. {stock}: {ret:+.1f}%")

    # Risk analysis
    annual_vol = returns.std() * np.sqrt(252) * 100
    annual_return = returns.mean() * 252 * 100
    sharpe_ratio = annual_return / annual_vol

    print(f"\n📊 Risk-Return Profile:")
    print(f"   Average Annual Return: {annual_return.mean():.1f}%")
    print(f"   Average Annual Volatility: {annual_vol.mean():.1f}%")
    print(f"   Average Sharpe Ratio: {sharpe_ratio.mean():.3f}")

    # Best risk-adjusted performers
    print(f"\n🎯 BEST RISK-ADJUSTED PERFORMERS (Sharpe Ratio):")
    for i, (stock, sharpe) in enumerate(sharpe_ratio.sort_values(ascending=False).head(3).items(), 1):
        ret = annual_return[stock]
        vol = annual_vol[stock]
        print(f"   {i}. {stock}: {sharpe:.3f} ({ret:.1f}% return, {vol:.1f}% volatility)")


def analyze_economic_data(datasets):
    """Analyze economic indicators."""
    if 'fred_data' not in datasets:
        print("❌ Economic data not available for analysis")
        return

    print("\n" + "=" * 50)
    print("📈 ECONOMIC INDICATORS ANALYSIS")
    print("=" * 50)

    fred_data = datasets['fred_data']

    print(f"\n📊 Data Overview:")
    print(f"   Indicators: {len(fred_data.columns)}")
    print(f"   Date Range: {fred_data.index[0].date()} to {fred_data.index[-1].date()}")
    print(f"   Observations: {len(fred_data)}")

    # Key indicators
    key_indicators = [
        'fed_funds_rate', 'treasury_10y', 'unemployment_rate',
        'cpi_inflation', 'vix'
    ]

    latest_data = fred_data.iloc[-1]

    print(f"\n📊 Latest Economic Conditions:")
    for indicator in key_indicators:
        if indicator in fred_data.columns:
            value = latest_data[indicator]
            if not pd.isna(value):
                print(f"   {indicator.replace('_', ' ').title()}: {value:.2f}")

    # Data completeness
    print(f"\n📊 Data Completeness:")
    for col in fred_data.columns[:10]:  # Show first 10 indicators
        completeness = (1 - fred_data[col].isnull().sum() / len(fred_data)) * 100
        print(f"   {col}: {completeness:.1f}%")


def analyze_treasury_data(datasets):
    """Analyze Treasury yield curve data."""
    if 'treasury_data' not in datasets:
        print("❌ Treasury data not available for analysis")
        return

    print("\n" + "=" * 50)
    print("🏛️  TREASURY YIELD CURVE ANALYSIS")
    print("=" * 50)

    treasury_data = datasets['treasury_data']

    print(f"\n📊 Data Overview:")
    print(f"   Metrics: {len(treasury_data.columns)}")
    print(f"   Date Range: {treasury_data.index[0].date()} to {treasury_data.index[-1].date()}")
    print(f"   Observations: {len(treasury_data)}")

    # Find yield curve rates
    yield_columns = [col for col in treasury_data.columns
                     if any(term in col for term in ['month', 'year'])
                     and 'slope' not in col and 'change' not in col
                     and 'volatility' not in col and 'momentum' not in col]

    if yield_columns:
        latest_yields = treasury_data[yield_columns].iloc[-1]

        print(f"\n📊 Current Yield Curve:")
        for col in sorted(yield_columns):
            if not pd.isna(latest_yields[col]):
                maturity = col.replace('_', '-').upper()
                rate = latest_yields[col]
                print(f"   {maturity}: {rate:.2f}%")

    # Yield curve shape
    if 'yield_slope_10y2y' in treasury_data.columns:
        latest_slope = treasury_data['yield_slope_10y2y'].iloc[-1]
        if not pd.isna(latest_slope):
            print(f"\n📊 Yield Curve Shape:")
            print(f"   10Y-2Y Slope: {latest_slope:.2f}%")

            if latest_slope > 1:
                shape = "Steep (Normal)"
            elif latest_slope > 0:
                shape = "Moderate (Normal)"
            elif latest_slope > -0.5:
                shape = "Flat"
            else:
                shape = "Inverted (Recession Signal)"

            print(f"   Interpretation: {shape}")


def create_simple_visualizations(datasets):
    """Create basic visualizations."""
    print("\n📊 Creating visualizations...")

    # Create results directory
    project_root = find_project_root()
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)

    try:
        # Set up the plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('AI-Enhanced Portfolio Optimization - Data Overview', fontsize=16, fontweight='bold')

        # 1. Banking sector price evolution
        if 'banking_prices' in datasets:
            prices = datasets['banking_prices']
            # Normalize to 100 for comparison
            normalized_prices = (prices / prices.iloc[0] * 100)

            for col in normalized_prices.columns:
                axes[0, 0].plot(normalized_prices.index, normalized_prices[col],
                                label=col, alpha=0.7, linewidth=1)

            axes[0, 0].set_title('Banking Sector Stock Performance (Normalized to 100)')
            axes[0, 0].set_ylabel('Normalized Price')
            axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Banking sector returns distribution
        if 'banking_returns' in datasets:
            returns = datasets['banking_returns']
            # Plot histogram of average daily returns
            avg_returns = returns.mean() * 252 * 100  # Annualized
            axes[0, 1].bar(range(len(avg_returns)), avg_returns.sort_values(ascending=False))
            axes[0, 1].set_title('Annual Returns by Bank')
            axes[0, 1].set_ylabel('Annual Return (%)')
            axes[0, 1].set_xticks(range(len(avg_returns)))
            axes[0, 1].set_xticklabels(avg_returns.sort_values(ascending=False).index,
                                       rotation=45, fontsize=8)
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Key economic indicators
        if 'fred_data' in datasets:
            fred_data = datasets['fred_data']
            key_indicators = ['fed_funds_rate', 'treasury_10y', 'unemployment_rate']

            for indicator in key_indicators:
                if indicator in fred_data.columns:
                    data = fred_data[indicator].dropna()
                    if len(data) > 0:
                        axes[1, 0].plot(data.index, data,
                                        label=indicator.replace('_', ' ').title(), linewidth=2)

            axes[1, 0].set_title('Key Economic Indicators')
            axes[1, 0].set_ylabel('Rate (%)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # 4. Yield curve current vs historical
        if 'treasury_data' in datasets:
            treasury_data = datasets['treasury_data']

            # Find yield curve columns
            yield_cols = ['3_month', '1_year', '2_year', '5_year', '10_year', '30_year']
            available_yields = [col for col in yield_cols if col in treasury_data.columns]

            if available_yields:
                # Current yield curve
                current_yields = treasury_data[available_yields].iloc[-1]
                maturities = [col.replace('_', ' ').title() for col in available_yields]

                axes[1, 1].plot(range(len(current_yields)), current_yields,
                                'o-', linewidth=2, markersize=6, label='Current')

                # Historical average
                avg_yields = treasury_data[available_yields].mean()
                axes[1, 1].plot(range(len(avg_yields)), avg_yields,
                                's--', linewidth=2, markersize=6, label='10Y Average', alpha=0.7)

                axes[1, 1].set_title('Treasury Yield Curve')
                axes[1, 1].set_ylabel('Yield (%)')
                axes[1, 1].set_xticks(range(len(maturities)))
                axes[1, 1].set_xticklabels(maturities, rotation=45, fontsize=8)
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the plot
        plot_file = os.path.join(results_dir, 'data_overview.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📁 Visualization saved: {plot_file}")

        plt.show()

    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")


def generate_summary_report(datasets):
    """Generate a summary report."""
    project_root = find_project_root()
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)

    report_file = os.path.join(results_dir, 'data_summary_report.txt')

    with open(report_file, 'w') as f:
        f.write("=== AI-ENHANCED PORTFOLIO OPTIMIZATION ===\n")
        f.write("=== DATA SUMMARY REPORT ===\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("AVAILABLE DATASETS:\n")
        f.write("-" * 40 + "\n")

        for name, df in datasets.items():
            f.write(f"\n{name.upper()}:\n")
            f.write(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
            f.write(f"  Date Range: {df.index[0].date()} to {df.index[-1].date()}\n")
            f.write(f"  Memory Usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.1f} MB\n")

            # Data completeness
            completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            f.write(f"  Data Completeness: {completeness:.1f}%\n")

        f.write(f"\nTOTAL DATASETS: {len(datasets)}\n")
        f.write(f"PROJECT STATUS: Data collection complete ✅\n")
        f.write(f"NEXT STEPS: Begin model development\n")

    print(f"📁 Summary report saved: {report_file}")


def main():
    """Main function to run simple data exploration."""
    # Load data
    datasets = load_data()

    if not datasets:
        print("❌ No datasets loaded. Please check your data files.")
        return

    print(f"\n✅ Successfully loaded {len(datasets)} datasets")

    # Analyze each dataset
    analyze_banking_data(datasets)
    analyze_economic_data(datasets)
    analyze_treasury_data(datasets)

    # Create visualizations
    create_simple_visualizations(datasets)

    # Generate report
    generate_summary_report(datasets)

    print("\n" + "=" * 60)
    print("✅ DATA EXPLORATION COMPLETED!")
    print("=" * 60)
    print("🎯 Ready for next phase: Model development")

    return datasets


if __name__ == "__main__":
    exploration_results = main()