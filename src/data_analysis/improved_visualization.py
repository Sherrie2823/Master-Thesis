"""
Improved Data Visualization - Separate, Clear Charts
Creates individual, well-spaced charts for better readability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings

warnings.filterwarnings('ignore')


class ImprovedDataVisualizer:
    """Create clear, separated visualizations for 10-year data."""

    def __init__(self):
        self.data_dir = "/data"
        self.results_dir = "/results"

        # Load data
        self.datasets = self.load_data()

    def load_data(self):
        """Load 10-year datasets."""
        print("📊 Loading 10-year datasets for improved visualization...")

        datasets = {}

        files_to_load = {
            'banking_prices': 'banking_prices_10y.csv',
            'banking_returns': 'banking_returns_10y.csv',
            'banking_correlation': 'banking_correlation_10y.csv',
            'economic_data': 'fred_economic_data_10y.csv'
        }

        for name, filename in files_to_load.items():
            filepath = os.path.join(self.data_dir, filename)

            try:
                if os.path.exists(filepath):
                    if 'correlation' in name:
                        df = pd.read_csv(filepath, index_col=0)
                    else:
                        df = pd.read_csv(filepath, index_col=0, parse_dates=True)

                    datasets[name] = df
                    print(f"✅ {name}: {df.shape}")
                else:
                    print(f"⚠️  {name}: File not found")
            except Exception as e:
                print(f"❌ {name}: Error - {e}")

        return datasets

    def create_performance_chart(self):
        """Create a clean banking performance chart."""
        if 'banking_prices' not in self.datasets:
            return

        print("📈 Creating banking performance chart...")

        prices = self.datasets['banking_prices']
        normalized = (prices / prices.iloc[0]) * 100

        # Create figure with proper size
        plt.figure(figsize=(16, 10))

        # Define colors for better distinction
        colors = plt.cm.tab20(np.linspace(0, 1, len(normalized.columns)))

        # Plot each stock
        for i, col in enumerate(normalized.columns):
            plt.plot(normalized.index, normalized[col],
                     label=col, linewidth=2.5, alpha=0.8, color=colors[i])

        # Formatting
        plt.title('Banking Sector Performance - 10 Year Analysis\n(Normalized to 100 at Start)',
                  fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Normalized Index (Start = 100)', fontsize=14)

        # Legend
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                   fontsize=11, frameon=True, fancybox=True, shadow=True)

        # Grid
        plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

        # Add major events annotations
        self.add_clean_annotations(plt.gca())

        # Tight layout
        plt.tight_layout()

        # Save
        chart_file = os.path.join(self.results_dir, 'banking_performance_clean.png')
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"📁 Performance chart saved: {chart_file}")

        plt.show()

    def create_risk_return_chart(self):
        """Create a clean risk-return scatter plot."""
        if 'banking_returns' not in self.datasets:
            return

        print("📊 Creating risk-return analysis chart...")

        returns = self.datasets['banking_returns']
        annual_ret = returns.mean() * 252 * 100
        annual_vol = returns.std() * np.sqrt(252) * 100

        # Create figure
        plt.figure(figsize=(12, 8))

        # Create scatter plot with different colors for each bank
        colors = plt.cm.viridis(np.linspace(0, 1, len(annual_ret)))
        scatter = plt.scatter(annual_vol, annual_ret, s=200, alpha=0.8, c=colors,
                              edgecolors='black', linewidth=2)

        # Add bank labels with better positioning
        for i, (vol, ret) in enumerate(zip(annual_vol, annual_ret)):
            plt.annotate(annual_vol.index[i], (vol, ret),
                         xytext=(8, 8), textcoords='offset points',
                         fontsize=11, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Formatting
        plt.title('Banking Sector Risk-Return Profile\n(10-Year Analysis)',
                  fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Annual Volatility (%)', fontsize=14)
        plt.ylabel('Annual Return (%)', fontsize=14)

        # Grid
        plt.grid(True, alpha=0.3)

        # Add efficient frontier reference line (approximate)
        vol_range = np.linspace(annual_vol.min(), annual_vol.max(), 100)
        efficient_line = 0.5 * vol_range  # Simple approximation
        plt.plot(vol_range, efficient_line, '--', color='red', alpha=0.5,
                 label='Reference Line', linewidth=2)

        plt.legend()
        plt.tight_layout()

        # Save
        chart_file = os.path.join(self.results_dir, 'risk_return_analysis.png')
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"📁 Risk-return chart saved: {chart_file}")

        plt.show()

    def create_economic_indicators_chart(self):
        """Create clean economic indicators chart."""
        if 'economic_data' not in self.datasets:
            return

        print("📈 Creating economic indicators chart...")

        econ = self.datasets['economic_data']

        # Create subplots for different indicator groups
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Economic Environment - 10 Year Evolution', fontsize=18, fontweight='bold')

        # 1. Interest Rates
        ax1 = axes[0, 0]
        interest_rates = ['fed_funds_rate', 'treasury_10y', 'treasury_2y']
        colors = ['red', 'blue', 'green']

        for rate, color in zip(interest_rates, colors):
            if rate in econ.columns:
                data = econ[rate].dropna()
                ax1.plot(data.index, data, label=rate.replace('_', ' ').title(),
                         color=color, linewidth=2.5)

        ax1.set_title('Interest Rates', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Rate (%)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Economic Growth
        ax2 = axes[0, 1]
        if 'unemployment_rate' in econ.columns:
            unemployment = econ['unemployment_rate'].dropna()
            ax2.plot(unemployment.index, unemployment, color='orange', linewidth=2.5)
            ax2.set_title('Unemployment Rate', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Rate (%)', fontsize=12)
            ax2.grid(True, alpha=0.3)

        # 3. Market Volatility
        ax3 = axes[1, 0]
        if 'vix' in econ.columns:
            vix = econ['vix'].dropna()
            ax3.plot(vix.index, vix, color='purple', linewidth=2)
            ax3.axhline(y=20, color='orange', linestyle='--', alpha=0.7, label='Elevated (20)')
            ax3.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='High Stress (30)')
            ax3.fill_between(vix.index, vix, 30, where=(vix > 30),
                             alpha=0.3, color='red', label='Crisis Periods')

            ax3.set_title('Market Volatility (VIX)', fontsize=14, fontweight='bold')
            ax3.set_ylabel('VIX Level', fontsize=12)
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 4. Yield Curve Slope
        ax4 = axes[1, 1]
        if 'term_spread_10y2y' in econ.columns:
            spread = econ['term_spread_10y2y'].dropna()
            ax4.plot(spread.index, spread, color='brown', linewidth=2.5)
            ax4.axhline(y=0, color='red', linestyle='-', alpha=0.7, label='Inversion Line')
            ax4.fill_between(spread.index, spread, 0, where=(spread < 0),
                             alpha=0.3, color='red', label='Inverted Periods')

            ax4.set_title('Yield Curve Slope (10Y-2Y)', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Spread (%)', fontsize=12)
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        chart_file = os.path.join(self.results_dir, 'economic_indicators_clean.png')
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"📁 Economic indicators chart saved: {chart_file}")

        plt.show()

    def create_correlation_heatmap(self):
        """Create a clean correlation heatmap."""
        if 'banking_correlation' not in self.datasets:
            return

        print("🔗 Creating correlation analysis chart...")

        corr = self.datasets['banking_correlation']

        # Create figure
        plt.figure(figsize=(12, 10))

        # Create heatmap with better formatting
        mask = np.triu(np.ones_like(corr, dtype=bool))  # Show only lower triangle

        sns.heatmap(corr, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                    square=True, fmt='.2f', cbar_kws={'shrink': 0.8},
                    annot_kws={'size': 10})

        plt.title('Banking Sector Correlation Matrix\n(10-Year Daily Returns)',
                  fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()

        # Save
        chart_file = os.path.join(self.results_dir, 'correlation_heatmap_clean.png')
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"📁 Correlation heatmap saved: {chart_file}")

        plt.show()

    def create_returns_distribution_chart(self):
        """Create a clean returns distribution chart."""
        if 'banking_prices' not in self.datasets:
            return

        print("📊 Creating returns distribution chart...")

        prices = self.datasets['banking_prices']
        total_returns = (prices.iloc[-1] / prices.iloc[0] - 1) * 100

        # Create figure
        plt.figure(figsize=(14, 8))

        # Sort returns for better visualization
        sorted_returns = total_returns.sort_values(ascending=False)

        # Create bar chart with gradient colors
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_returns)))
        bars = plt.bar(range(len(sorted_returns)), sorted_returns,
                       alpha=0.8, color=colors, edgecolor='black', linewidth=1)

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, sorted_returns)):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f'{value:.0f}%', ha='center', va='bottom',
                     fontweight='bold', fontsize=11)

        # Formatting
        plt.title('10-Year Total Returns by Bank\n(2015-2024)',
                  fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Total Return (%)', fontsize=14)
        plt.xlabel('Banking Institutions', fontsize=14)

        # X-axis labels
        plt.xticks(range(len(sorted_returns)), sorted_returns.index,
                   rotation=45, ha='right', fontsize=12)

        # Grid
        plt.grid(True, alpha=0.3, axis='y')

        # Add average line
        avg_return = sorted_returns.mean()
        plt.axhline(y=avg_return, color='red', linestyle='--',
                    label=f'Average: {avg_return:.1f}%', linewidth=2)
        plt.legend()

        plt.tight_layout()

        # Save
        chart_file = os.path.join(self.results_dir, 'returns_distribution_clean.png')
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"📁 Returns distribution chart saved: {chart_file}")

        plt.show()

    def add_clean_annotations(self, ax):
        """Add clean event annotations."""
        events = [
            ('2020-03-01', 'COVID-19\nCrash'),
            ('2020-11-01', 'Election\n& Vaccine'),
            ('2022-03-01', 'Fed Rate\nHikes Begin'),
            ('2023-03-01', 'Banking Crisis\n(SVB/Credit Suisse)')
        ]

        try:
            for date_str, event in events:
                event_date = pd.to_datetime(date_str)
                ax.axvline(x=event_date, color='red', linestyle='--', alpha=0.7, linewidth=2)
                ax.text(event_date, ax.get_ylim()[1] * 0.9, event,
                        rotation=0, fontsize=9, ha='center', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        except:
            pass

    def create_all_charts(self):
        """Create all improved charts separately."""
        print("🎨 Creating improved, separated visualizations...")
        print("=" * 60)

        # Create each chart separately
        self.create_performance_chart()
        self.create_risk_return_chart()
        self.create_economic_indicators_chart()
        self.create_correlation_heatmap()
        self.create_returns_distribution_chart()

        print("\n" + "=" * 60)
        print("✅ ALL IMPROVED CHARTS CREATED!")
        print("=" * 60)
        print("📁 Charts saved to results/ directory:")
        print("   📈 banking_performance_clean.png")
        print("   📊 risk_return_analysis.png")
        print("   📉 economic_indicators_clean.png")
        print("   🔗 correlation_heatmap_clean.png")
        print("   📊 returns_distribution_clean.png")
        print("\n🎯 Each chart is now clear and readable!")


def main():
    """Main function to create improved visualizations."""
    visualizer = ImprovedDataVisualizer()

    if not visualizer.datasets:
        print("❌ No datasets loaded. Please run data collection first.")
        return False

    visualizer.create_all_charts()
    return True


if __name__ == "__main__":
    success = main()