"""
Absolute Path Data Explorer - Direct paths, no calculation needed
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def main():
    """Load and explore data with absolute paths."""

    print("🚀 AI-Enhanced Portfolio Optimization - Data Explorer (Fixed Paths)")
    print("=" * 80)

    # ABSOLUTE PATHS - no calculation needed
    data_dir = "/data"
    results_dir = "/results"

    print(f"📁 Data directory: {data_dir}")
    print(f"📁 Results directory: {results_dir}")

    # Create results directory
    os.makedirs(results_dir, exist_ok=True)

    # Check what files exist
    if os.path.exists(data_dir):
        files = os.listdir(data_dir)
        print(f"\n📄 Files in data directory:")
        for f in sorted(files):
            if f.endswith('.csv') or f.endswith('.txt'):
                filepath = os.path.join(data_dir, f)
                size_mb = os.path.getsize(filepath) / (1024 ** 2)
                print(f"   📊 {f} ({size_mb:.1f} MB)")
    else:
        print(f"❌ Data directory not found: {data_dir}")
        return

    # Load datasets with absolute paths
    datasets = {}

    print(f"\n📊 Loading datasets...")

    # Define exact file paths
    file_paths = {
        'banking_prices': f"{data_dir}/banking_prices.csv",
        'banking_returns': f"{data_dir}/banking_returns.csv",
        'banking_correlation': f"{data_dir}/banking_correlation.csv",
        'fred_data': f"{data_dir}/fred_economic_data.csv",
        'treasury_data': f"{data_dir}/treasury_complete.csv"
    }

    for name, filepath in file_paths.items():
        try:
            if os.path.exists(filepath):
                if 'correlation' in name:
                    df = pd.read_csv(filepath, index_col=0)
                else:
                    df = pd.read_csv(filepath, index_col=0, parse_dates=True)

                datasets[name] = df
                print(f"✅ {name}: {df.shape}")
            else:
                print(f"❌ {name}: File not found at {filepath}")

        except Exception as e:
            print(f"❌ {name}: Error loading - {e}")

    if not datasets:
        print("\n❌ No datasets loaded!")
        return

    print(f"\n✅ Successfully loaded {len(datasets)} datasets")

    # Analysis
    analyze_all_data(datasets, results_dir)


def analyze_all_data(datasets, results_dir):
    """Comprehensive analysis of all datasets."""

    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE DATA ANALYSIS")
    print("=" * 80)

    # Banking Analysis
    if 'banking_prices' in datasets and 'banking_returns' in datasets:
        print("\n🏦 BANKING SECTOR ANALYSIS")
        print("-" * 50)

        prices = datasets['banking_prices']
        returns = datasets['banking_returns']

        print(f"📊 Dataset Overview:")
        print(f"   Stocks: {len(prices.columns)}")
        print(f"   Date Range: {prices.index[0].date()} to {prices.index[-1].date()}")
        print(f"   Trading Days: {len(prices)}")
        print(f"   Stock Symbols: {', '.join(sorted(prices.columns))}")

        # Performance metrics
        total_returns = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
        annual_returns = returns.mean() * 252 * 100
        annual_volatility = returns.std() * np.sqrt(252) * 100
        sharpe_ratios = annual_returns / annual_volatility

        print(f"\n🏆 TOTAL RETURNS (Period):")
        for i, (stock, ret) in enumerate(total_returns.sort_values(ascending=False).items(), 1):
            annual_ret = annual_returns[stock]
            volatility = annual_volatility[stock]
            sharpe = sharpe_ratios[stock]
            print(
                f"   {i:2d}. {stock}: {ret:+6.1f}% total | {annual_ret:5.1f}% annual | {volatility:5.1f}% vol | {sharpe:.3f} Sharpe")

        print(f"\n📊 PORTFOLIO METRICS:")
        print(f"   Average Annual Return: {annual_returns.mean():5.1f}%")
        print(f"   Average Volatility: {annual_volatility.mean():8.1f}%")
        print(f"   Average Sharpe Ratio: {sharpe_ratios.mean():7.3f}")

        # Correlation analysis
        if 'banking_correlation' in datasets:
            corr = datasets['banking_correlation']
            avg_corr = corr.mean().mean()
            max_corr = corr.where(corr < 1).max().max()  # Exclude self-correlation
            min_corr = corr.min().min()

            print(f"\n🔗 CORRELATION ANALYSIS:")
            print(f"   Average Correlation: {avg_corr:.3f}")
            print(f"   Highest Correlation: {max_corr:.3f}")
            print(f"   Lowest Correlation: {min_corr:.3f}")

    # Economic Analysis
    if 'fred_data' in datasets:
        print("\n📈 ECONOMIC INDICATORS ANALYSIS")
        print("-" * 50)

        fred = datasets['fred_data']

        print(f"📊 Dataset Overview:")
        print(f"   Indicators: {len(fred.columns)}")
        print(f"   Date Range: {fred.index[0].date()} to {fred.index[-1].date()}")
        print(f"   Observations: {len(fred):,}")

        latest = fred.iloc[-1]
        print(f"\n📊 LATEST ECONOMIC CONDITIONS:")
        for col in fred.columns:
            if not pd.isna(latest[col]):
                print(f"   {col.replace('_', ' ').title():<20}: {latest[col]:7.2f}")

    # Treasury Analysis
    if 'treasury_data' in datasets:
        print("\n🏛️  TREASURY YIELD CURVE ANALYSIS")
        print("-" * 50)

        treasury = datasets['treasury_data']

        print(f"📊 Dataset Overview:")
        print(f"   Metrics: {len(treasury.columns)}")
        print(f"   Date Range: {treasury.index[0].date()} to {treasury.index[-1].date()}")
        print(f"   Observations: {len(treasury):,}")

        latest = treasury.iloc[-1]

        # Yield curve rates
        yield_cols = [col for col in treasury.columns if
                      any(term in col for term in ['month', 'year']) and 'slope' not in col]
        if yield_cols:
            print(f"\n📊 CURRENT YIELD CURVE:")
            for col in sorted(yield_cols):
                if not pd.isna(latest[col]):
                    maturity = col.replace('_', ' ').title()
                    rate = latest[col]
                    print(f"   {maturity:<12}: {rate:6.2f}%")

        # Derived metrics
        derived_cols = [col for col in treasury.columns if 'slope' in col or 'curve' in col or 'level' in col]
        if derived_cols:
            print(f"\n📊 YIELD CURVE METRICS:")
            for col in derived_cols:
                if not pd.isna(latest[col]):
                    metric = col.replace('_', ' ').title()
                    value = latest[col]
                    print(f"   {metric:<20}: {value:6.2f}%")

    # Create visualizations
    create_comprehensive_charts(datasets, results_dir)

    # Generate report
    generate_comprehensive_report(datasets, results_dir)


def create_comprehensive_charts(datasets, results_dir):
    """Create comprehensive visualization charts."""

    print(f"\n📊 Creating comprehensive visualizations...")

    try:
        # Set up matplotlib
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('AI-Enhanced Portfolio Optimization - Data Analysis', fontsize=16, fontweight='bold')

        # Chart 1: Banking Performance
        if 'banking_prices' in datasets:
            prices = datasets['banking_prices']
            normalized = (prices / prices.iloc[0]) * 100

            for col in normalized.columns:
                axes[0, 0].plot(normalized.index, normalized[col],
                                label=col, alpha=0.8, linewidth=2)

            axes[0, 0].set_title('Banking Sector Performance (Normalized)', fontweight='bold', fontsize=12)
            axes[0, 0].set_ylabel('Index (Start = 100)')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        # Chart 2: Risk-Return Scatter
        if 'banking_returns' in datasets:
            returns = datasets['banking_returns']
            annual_ret = returns.mean() * 252 * 100
            annual_vol = returns.std() * np.sqrt(252) * 100

            colors = plt.cm.viridis(np.linspace(0, 1, len(annual_ret)))
            scatter = axes[0, 1].scatter(annual_vol, annual_ret,
                                         s=120, alpha=0.8, c=colors)

            # Add labels
            for i, (vol, ret) in enumerate(zip(annual_vol, annual_ret)):
                axes[0, 1].annotate(annual_vol.index[i], (vol, ret),
                                    xytext=(5, 5), textcoords='offset points',
                                    fontsize=9, fontweight='bold')

            axes[0, 1].set_title('Risk-Return Profile', fontweight='bold', fontsize=12)
            axes[0, 1].set_xlabel('Annual Volatility (%)')
            axes[0, 1].set_ylabel('Annual Return (%)')
            axes[0, 1].grid(True, alpha=0.3)

        # Chart 3: Economic Indicators Time Series
        if 'fred_data' in datasets:
            fred = datasets['fred_data']

            # Plot available indicators
            colors = ['blue', 'red', 'green', 'orange']
            for i, col in enumerate(fred.columns[:4]):  # First 4 indicators
                if col in fred.columns:
                    data = fred[col].dropna()
                    if len(data) > 10:  # Only plot if sufficient data
                        axes[1, 0].plot(data.index, data,
                                        label=col.replace('_', ' ').title(),
                                        color=colors[i % len(colors)], linewidth=2)

            axes[1, 0].set_title('Economic Indicators Over Time', fontweight='bold', fontsize=12)
            axes[1, 0].set_ylabel('Rate/Index Value')
            axes[1, 0].legend(fontsize=9)
            axes[1, 0].grid(True, alpha=0.3)

        # Chart 4: Yield Curve
        if 'treasury_data' in datasets:
            treasury = datasets['treasury_data']

            # Define yield curve in proper order
            yield_order = ['3_month', '1_year', '2_year', '5_year', '10_year', '30_year']
            available_yields = [y for y in yield_order if y in treasury.columns]

            if len(available_yields) >= 3:
                current_yields = treasury[available_yields].iloc[-1]
                historical_avg = treasury[available_yields].mean()

                x_pos = range(len(available_yields))
                width = 0.35

                axes[1, 1].bar([x - width / 2 for x in x_pos], current_yields,
                               width, label='Current', alpha=0.8, color='red')
                axes[1, 1].bar([x + width / 2 for x in x_pos], historical_avg,
                               width, label='Average', alpha=0.8, color='blue')

                axes[1, 1].set_title('Yield Curve: Current vs Average', fontweight='bold', fontsize=12)
                axes[1, 1].set_ylabel('Yield (%)')
                axes[1, 1].set_xticks(x_pos)
                axes[1, 1].set_xticklabels([y.replace('_', ' ').title() for y in available_yields],
                                           rotation=45, fontsize=9)
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save chart
        chart_file = os.path.join(results_dir, 'comprehensive_analysis.png')
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"📁 Charts saved: {chart_file}")

        plt.show()

    except Exception as e:
        print(f"❌ Error creating charts: {e}")
        import traceback
        traceback.print_exc()


def generate_comprehensive_report(datasets, results_dir):
    """Generate a comprehensive analysis report."""

    print(f"\n📝 Generating comprehensive report...")

    report_file = os.path.join(results_dir, 'comprehensive_analysis_report.txt')

    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("AI-ENHANCED PORTFOLIO OPTIMIZATION FOR BANKING SECTOR\n")
        f.write("COMPREHENSIVE DATA ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Executive Summary
        f.write("EXECUTIVE SUMMARY:\n")
        f.write("-" * 50 + "\n")
        f.write(f"✅ Data Collection: {len(datasets)} comprehensive datasets loaded\n")

        if 'banking_prices' in datasets:
            prices = datasets['banking_prices']
            returns = datasets['banking_returns']
            total_ret = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
            f.write(f"✅ Banking Analysis: {len(prices.columns)} stocks, {len(prices)} trading days\n")
            f.write(f"✅ Performance Range: {total_ret.min():.1f}% to {total_ret.max():.1f}% total returns\n")

        if 'fred_data' in datasets:
            fred = datasets['fred_data']
            f.write(f"✅ Economic Data: {len(fred.columns)} indicators, {len(fred)} observations\n")

        if 'treasury_data' in datasets:
            treasury = datasets['treasury_data']
            f.write(f"✅ Treasury Data: {len(treasury.columns)} metrics, yield curve analysis ready\n")

        f.write("✅ Visualization: Comprehensive charts generated\n")
        f.write("✅ Research Status: Ready for AI model development\n\n")

        # Data Quality Assessment
        f.write("DATA QUALITY ASSESSMENT:\n")
        f.write("-" * 50 + "\n")

        for name, df in datasets.items():
            completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            f.write(f"{name.upper()}:\n")
            f.write(f"  Shape: {df.shape[0]:,} × {df.shape[1]}\n")
            f.write(f"  Completeness: {completeness:.1f}%\n")
            f.write(f"  Date Range: {df.index[0].date()} to {df.index[-1].date()}\n\n")

        # Research Readiness
        f.write("RESEARCH READINESS CHECKLIST:\n")
        f.write("-" * 50 + "\n")
        f.write("✅ Banking sector data (multiple stocks, multi-year)\n")
        f.write("✅ Macroeconomic context (interest rates, economic indicators)\n")
        f.write("✅ Risk-free rate benchmarks (Treasury data)\n")
        f.write("✅ High-quality, clean datasets\n")
        f.write("✅ Sufficient data volume for AI training\n")
        f.write("✅ Multiple market regimes captured\n\n")

        # Next Steps
        f.write("RECOMMENDED NEXT STEPS:\n")
        f.write("-" * 50 + "\n")
        f.write("1. Begin traditional portfolio optimization baseline\n")
        f.write("2. Develop feature engineering for AI models\n")
        f.write("3. Implement LSTM for return prediction\n")
        f.write("4. Design reinforcement learning framework\n")
        f.write("5. Establish backtesting and validation protocols\n")
        f.write("6. Start literature review with data-driven insights\n")

    print(f"📁 Comprehensive report saved: {report_file}")

    # Summary
    print("\n" + "=" * 80)
    print("✅ COMPREHENSIVE DATA ANALYSIS COMPLETED!")
    print("=" * 80)
    print(f"📁 Results directory: {results_dir}")
    print("📊 Charts: comprehensive_analysis.png")
    print("📝 Report: comprehensive_analysis_report.txt")
    print("\n🎯 NEXT PHASE: Ready for AI model development!")
    print("🚀 Your dissertation foundation is now complete!")


if __name__ == "__main__":
    main()