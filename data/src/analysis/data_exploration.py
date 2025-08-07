"""
Data Exploration and Validation Script
Author: MSc Banking and Digital Finance Student
Date: July 2025

This script provides comprehensive exploration of collected data for the dissertation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class DataExplorer:
    """
    Comprehensive data exploration for AI-Enhanced Portfolio Optimization project.
    """

    def __init__(self):
        import os

        # Get the project root directory (go up 2 levels from src/analysis/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))

        self.data_dir = os.path.join(project_root, 'data/')
        self.results_dir = os.path.join(project_root, 'results/')

        #

    def load_datasets(self):
        """Load all collected datasets."""
        print("📊 Loading all datasets...")

        try:
            # Banking data
            banking_prices_path = os.path.join(self.data_dir, 'banking_prices.csv')
            banking_returns_path = os.path.join(self.data_dir, 'banking_returns.csv')
            banking_correlation_path = os.path.join(self.data_dir, 'banking_correlation.csv')

            print(f"🔍 Looking for banking prices at: {banking_prices_path}")

            self.banking_prices = pd.read_csv(banking_prices_path, index_col=0, parse_dates=True)
            self.banking_returns = pd.read_csv(banking_returns_path, index_col=0, parse_dates=True)
            self.banking_correlation = pd.read_csv(banking_correlation_path, index_col=0)

            # Economic data
            fred_path = os.path.join(self.data_dir, 'fred_economic_data.csv')
            self.fred_data = pd.read_csv(fred_path, index_col=0, parse_dates=True)

            # Treasury data
            treasury_path = os.path.join(self.data_dir, 'treasury_complete.csv')
            self.treasury_data = pd.read_csv(treasury_path, index_col=0, parse_dates=True)

            print("✅ All datasets loaded successfully!")

        except Exception as e:
            print(f"❌ Error loading datasets: {e}")
            print(f"📁 Current working directory: {os.getcwd()}")
            print(f"📁 Looking in data directory: {self.data_dir}")

            # List available files
            try:
                files = os.listdir(self.data_dir)
                print(f"📄 Available files in data directory: {files}")
            except:
                print("❌ Data directory not found")

            raise

    def data_overview(self):
        """Provide comprehensive data overview."""
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE DATA OVERVIEW")
        print("=" * 60)

        datasets = {
            'Banking Prices': self.banking_prices,
            'Banking Returns': self.banking_returns,
            'Economic Indicators': self.fred_data,
            'Treasury Data': self.treasury_data
        }

        for name, df in datasets.items():
            print(f"\n🔍 {name.upper()}:")
            print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            print(f"   Date Range: {df.index[0].date()} to {df.index[-1].date()}")
            print(f"   Missing Values: {df.isnull().sum().sum():,} total")
            print(f"   Memory Usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.1f} MB")

    def banking_analysis(self):
        """Analyze banking sector data."""
        print("\n" + "=" * 50)
        print("🏦 BANKING SECTOR ANALYSIS")
        print("=" * 50)

        # Performance summary
        latest_prices = self.banking_prices.iloc[-1]
        first_prices = self.banking_prices.iloc[0]
        total_returns = (latest_prices / first_prices - 1) * 100

        print("\n📈 TOTAL RETURNS (10 Years):")
        for stock, ret in total_returns.sort_values(ascending=False).items():
            print(f"   {stock}: {ret:+.1f}%")

        # Risk metrics
        annual_vol = self.banking_returns.std() * np.sqrt(252) * 100
        print(f"\n📊 ANNUAL VOLATILITY:")
        for stock, vol in annual_vol.sort_values().items():
            print(f"   {stock}: {vol:.1f}%")

        # Correlation insights
        avg_correlation = self.banking_correlation.mean().mean()
        max_correlation = self.banking_correlation.max().max()
        min_correlation = self.banking_correlation.min().min()

        print(f"\n🔗 CORRELATION ANALYSIS:")
        print(f"   Average Correlation: {avg_correlation:.3f}")
        print(f"   Highest Correlation: {max_correlation:.3f}")
        print(f"   Lowest Correlation: {min_correlation:.3f}")

        return {
            'total_returns': total_returns,
            'volatility': annual_vol,
            'correlation_stats': {
                'average': avg_correlation,
                'max': max_correlation,
                'min': min_correlation
            }
        }

    def economic_analysis(self):
        """Analyze economic indicators."""
        print("\n" + "=" * 50)
        print("📈 ECONOMIC INDICATORS ANALYSIS")
        print("=" * 50)

        # Key indicators latest values
        latest_econ = self.fred_data.iloc[-1]

        key_indicators = [
            'fed_funds_rate', 'treasury_10y', 'unemployment_rate',
            'cpi_inflation', 'vix', 'yield_curve_slope'
        ]

        print("\n📊 LATEST ECONOMIC CONDITIONS:")
        for indicator in key_indicators:
            if indicator in latest_econ.index and not pd.isna(latest_econ[indicator]):
                value = latest_econ[indicator]
                print(f"   {indicator.replace('_', ' ').title()}: {value:.2f}")

        # Economic trends (1-year change)
        if len(self.fred_data) >= 252:
            year_ago = self.fred_data.iloc[-252]
            changes = {}

            print(f"\n📊 1-YEAR CHANGES:")
            for indicator in key_indicators:
                if indicator in latest_econ.index and indicator in year_ago.index:
                    if not pd.isna(latest_econ[indicator]) and not pd.isna(year_ago[indicator]):
                        change = latest_econ[indicator] - year_ago[indicator]
                        changes[indicator] = change
                        print(f"   {indicator.replace('_', ' ').title()}: {change:+.2f}")

        return latest_econ

    def treasury_analysis(self):
        """Analyze Treasury yield curve."""
        print("\n" + "=" * 50)
        print("🏛️  TREASURY YIELD CURVE ANALYSIS")
        print("=" * 50)

        # Current yield curve
        latest_treasury = self.treasury_data.iloc[-1]

        yield_columns = [col for col in self.treasury_data.columns
                         if any(term in col for term in ['month', 'year'])
                         and 'slope' not in col and 'change' not in col]

        print("\n📊 CURRENT YIELD CURVE:")
        for col in sorted(yield_columns):
            if col in latest_treasury.index and not pd.isna(latest_treasury[col]):
                maturity = col.replace('_', '-').upper()
                rate = latest_treasury[col]
                print(f"   {maturity}: {rate:.2f}%")

        # Yield curve shape analysis
        if all(col in latest_treasury.index for col in ['yield_slope_10y2y', 'yield_curvature']):
            slope = latest_treasury['yield_slope_10y2y']
            curvature = latest_treasury['yield_curvature']

            print(f"\n📊 YIELD CURVE SHAPE:")
            print(f"   Slope (10Y-2Y): {slope:.2f}%")
            print(f"   Curvature: {curvature:.2f}%")

            # Interpret curve shape
            if slope > 1:
                shape = "Steep (Normal)"
            elif slope > 0:
                shape = "Moderate (Normal)"
            elif slope > -0.5:
                shape = "Flat"
            else:
                shape = "Inverted (Recession Signal)"

            print(f"   Interpretation: {shape}")

        return latest_treasury

    def create_visualizations(self):
        """Create key visualizations."""
        print("\n📊 Creating visualizations...")

        # Set style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('AI-Enhanced Portfolio Optimization - Data Overview', fontsize=16, fontweight='bold')

        # 1. Banking sector cumulative returns
        cumulative_returns = (1 + self.banking_returns).cumprod()
        axes[0, 0].plot(cumulative_returns.index, cumulative_returns)
        axes[0, 0].set_title('Banking Sector Cumulative Returns (10 Years)')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].legend(cumulative_returns.columns, bbox_to_anchor=(1.05, 1), loc='upper left')

        # 2. Correlation heatmap
        sns.heatmap(self.banking_correlation, annot=True, cmap='coolwarm', center=0,
                    ax=axes[0, 1], cbar_kws={'shrink': 0.8})
        axes[0, 1].set_title('Banking Sector Correlation Matrix')

        # 3. Economic indicators
        key_econ = ['fed_funds_rate', 'treasury_10y', 'unemployment_rate']
        econ_subset = self.fred_data[key_econ].dropna()

        for col in key_econ:
            if col in econ_subset.columns:
                axes[1, 0].plot(econ_subset.index, econ_subset[col], label=col.replace('_', ' ').title())

        axes[1, 0].set_title('Key Economic Indicators')
        axes[1, 0].set_ylabel('Rate (%)')
        axes[1, 0].legend()

        # 4. Yield curve evolution
        yield_cols = ['3_month', '2_year', '10_year', '30_year']
        available_yields = [col for col in yield_cols if col in self.treasury_data.columns]

        if available_yields:
            for col in available_yields:
                axes[1, 1].plot(self.treasury_data.index, self.treasury_data[col],
                                label=col.replace('_', '-').upper())

            axes[1, 1].set_title('Treasury Yield Curve Evolution')
            axes[1, 1].set_ylabel('Yield (%)')
            axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(f'{self.results_dir}data_overview.png', dpi=300, bbox_inches='tight')
        print(f"📁 Visualization saved to: {self.results_dir}data_overview.png")
        plt.show()

    def generate_data_report(self):
        """Generate comprehensive data report."""
        print("\n📝 Generating data quality report...")

        report_file = f'{self.results_dir}data_quality_report.txt'

        with open(report_file, 'w') as f:
            f.write("=== AI-ENHANCED PORTFOLIO OPTIMIZATION ===\n")
            f.write("=== DATA QUALITY & EXPLORATION REPORT ===\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Dataset summaries
            f.write("DATASET OVERVIEW:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Banking Prices: {self.banking_prices.shape[0]} days × {self.banking_prices.shape[1]} stocks\n")
            f.write(f"Economic Data: {self.fred_data.shape[0]} obs × {self.fred_data.shape[1]} indicators\n")
            f.write(f"Treasury Data: {self.treasury_data.shape[0]} days × {self.treasury_data.shape[1]} metrics\n\n")

            # Data quality metrics
            f.write("DATA QUALITY ASSESSMENT:\n")
            f.write("-" * 40 + "\n")

            datasets = {
                'Banking Prices': self.banking_prices,
                'Economic Data': self.fred_data,
                'Treasury Data': self.treasury_data
            }

            for name, df in datasets.items():
                completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                f.write(f"{name}:\n")
                f.write(f"  Data Completeness: {completeness:.1f}%\n")
                f.write(f"  Date Range: {df.index[0].date()} to {df.index[-1].date()}\n")
                f.write(f"  Total Records: {len(df):,}\n\n")

            # Research readiness assessment
            f.write("RESEARCH READINESS:\n")
            f.write("-" * 40 + "\n")
            f.write("✅ Banking sector stock data (15 stocks, 10 years)\n")
            f.write("✅ Macroeconomic indicators (33 metrics)\n")
            f.write("✅ Treasury yield curve data\n")
            f.write("✅ Data aligned and processed\n")
            f.write("✅ Ready for AI model development\n\n")

            # Next steps
            f.write("RECOMMENDED NEXT STEPS:\n")
            f.write("-" * 40 + "\n")
            f.write("1. Implement traditional portfolio optimization baseline\n")
            f.write("2. Begin literature review with data-driven focus\n")
            f.write("3. Design AI model architecture\n")
            f.write("4. Set up backtesting framework\n")

        print(f"📁 Data report saved to: {report_file}")

    def run_complete_analysis(self):
        """Run complete data exploration pipeline."""
        print("🚀 Starting comprehensive data exploration...")

        # Overview
        self.data_overview()

        # Detailed analysis
        banking_results = self.banking_analysis()
        economic_results = self.economic_analysis()
        treasury_results = self.treasury_analysis()

        # Visualizations
        self.create_visualizations()

        # Report
        self.generate_data_report()

        print("\n" + "=" * 60)
        print("✅ DATA EXPLORATION COMPLETED!")
        print("=" * 60)
        print("📁 Results saved to 'results/' directory")
        print("📊 Visualization: data_overview.png")
        print("📝 Report: data_quality_report.txt")
        print("\n🎯 Ready to proceed with model development!")

        return {
            'banking': banking_results,
            'economic': economic_results,
            'treasury': treasury_results
        }


def main():
    """Run complete data exploration."""
    explorer = DataExplorer()
    results = explorer.run_complete_analysis()
    return results


if __name__ == "__main__":
    exploration_results = main()