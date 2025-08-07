"""
Rolling Efficient Frontier Analysis for Banking Sector Portfolio Optimization
MSc Banking and Digital Finance - AI-Enhanced Portfolio Optimization Project

This module generates efficient frontier visualizations that evolve over time,
demonstrating how optimization opportunities change as additional market information
becomes available through the expanding window methodology.

Features:
- Multiple efficient frontiers across different analysis periods
- Temporal evolution visualization maintaining rolling framework integrity
- Banking sector specific analysis with regulatory constraints
- Comprehensive visualization suite for dissertation inclusion

Author: Sherrie
Date: July 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from scipy.optimize import minimize
import warnings
from typing import Tuple, List, Dict, Optional

warnings.filterwarnings('ignore')


class RollingEfficientFrontierAnalyzer:
    """
    Generate efficient frontier analysis that evolves throughout the rolling optimization period
    """

    def __init__(self, data_path: str = "data/raw", output_path: str = "results/charts"):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Portfolio parameters (matching your implementation)
        self.risk_free_rate = 0.02
        self.min_weight = 0.01
        self.max_weight = 0.20
        self.min_history = 252

        # Data storage
        self.returns = None
        self.analysis_periods = []

        print("📈 Rolling Efficient Frontier Analyzer Initialized")
        print(f"Output directory: {self.output_path}")

        self.load_data()
        self.setup_analysis_periods()

    def load_data(self):
        """Load stock returns data with validation"""
        try:
            csv_files = list(self.data_path.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError("No CSV files found in data directory")

            df = pd.read_csv(csv_files[0], index_col=0, parse_dates=True)

            if df.shape[1] < 2:
                raise ValueError("Insufficient number of assets in dataset")

            self.returns = df.pct_change().dropna().sort_index()

            print(f"✅ Loaded returns data: {self.returns.shape}")
            print(f"Assets: {list(self.returns.columns)}")
            print(f"Date range: {self.returns.index[0].date()} to {self.returns.index[-1].date()}")

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise

    def setup_analysis_periods(self):
        """Setup representative analysis periods for efficient frontier evolution"""
        if self.returns is None:
            return

        # Define key analysis periods that represent different market conditions
        total_length = len(self.returns)

        # Start after minimum history requirement
        start_idx = self.min_history

        # Define periods: Early, Mid-expansion, Pre-crisis, Crisis, Recovery, Recent
        period_definitions = [
            ("Early Period", start_idx, start_idx + 252),  # First year after min history
            ("Mid Expansion", start_idx + 378, start_idx + 630),  # 1.5-2.5 years
            ("Pre-Crisis", start_idx + 756, start_idx + 1008),  # 3-4 years
            ("Crisis Period", start_idx + 1134, start_idx + 1386),  # 4.5-5.5 years
            ("Recovery Phase", start_idx + 1512, start_idx + 1764),  # 6-7 years
            ("Recent Period", max(start_idx + 1890, total_length - 252), total_length - 1)  # Last year
        ]

        self.analysis_periods = []
        for name, start, end in period_definitions:
            if start < total_length and end <= total_length:
                self.analysis_periods.append({
                    'name': name,
                    'start_idx': start,
                    'end_idx': end,
                    'start_date': self.returns.index[start],
                    'end_date': self.returns.index[min(end, total_length - 1)],
                    'data_points': end - start
                })

        print(f"📅 Analysis periods defined: {len(self.analysis_periods)} periods")
        for period in self.analysis_periods:
            print(f"   {period['name']}: {period['start_date'].date()} to {period['end_date'].date()}")

    def calculate_portfolio_metrics(self, weights: np.ndarray,
                                    mean_returns: np.ndarray,
                                    cov_matrix: np.ndarray) -> Tuple[float, float, float]:
        """Calculate portfolio performance metrics"""
        portfolio_return = np.sum(weights * mean_returns)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
        return portfolio_return, portfolio_vol, sharpe_ratio

    def generate_efficient_frontier(self, returns_data: pd.DataFrame, n_points: int = 50) -> Tuple[
        List[float], List[float]]:
        """Generate efficient frontier for given returns data"""
        mean_returns = returns_data.mean() * 252  # Annualized
        cov_matrix = returns_data.cov() * 252  # Annualized
        n_assets = len(mean_returns)

        # Portfolio volatility function
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))

        # Efficient return function
        def efficient_return(target_return):
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.sum(x * mean_returns) - target_return}
            ]
            bounds = tuple((self.min_weight, self.max_weight) for _ in range(n_assets))
            x0 = np.array([1 / n_assets] * n_assets)

            result = minimize(portfolio_volatility, x0, method='SLSQP',
                              bounds=bounds, constraints=constraints)
            return result.x if result.success else None

        # Find minimum and maximum feasible returns
        min_vol_weights = self.find_min_volatility_portfolio(returns_data)
        max_sharpe_weights = self.find_max_sharpe_portfolio(returns_data)

        if min_vol_weights is None or max_sharpe_weights is None:
            return [], []

        min_ret = np.sum(min_vol_weights * mean_returns)
        max_ret = np.sum(max_sharpe_weights * mean_returns) * 1.2  # Extend slightly

        # Generate efficient frontier points
        target_returns = np.linspace(min_ret, max_ret, n_points)
        frontier_volatilities = []
        frontier_returns = []

        for target_ret in target_returns:
            weights = efficient_return(target_ret)
            if weights is not None:
                ret, vol, _ = self.calculate_portfolio_metrics(weights, mean_returns, cov_matrix)
                frontier_returns.append(ret)
                frontier_volatilities.append(vol)

        return frontier_volatilities, frontier_returns

    def find_min_volatility_portfolio(self, returns_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Find minimum volatility portfolio"""
        cov_matrix = returns_data.cov() * 252
        n_assets = len(returns_data.columns)

        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n_assets))
        x0 = np.array([1 / n_assets] * n_assets)

        result = minimize(portfolio_volatility, x0, method='SLSQP',
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else None

    def find_max_sharpe_portfolio(self, returns_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Find maximum Sharpe ratio portfolio"""
        mean_returns = returns_data.mean() * 252
        cov_matrix = returns_data.cov() * 252
        n_assets = len(returns_data.columns)

        def negative_sharpe(weights):
            ret, vol, sharpe = self.calculate_portfolio_metrics(weights, mean_returns, cov_matrix)
            return -sharpe

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n_assets))
        x0 = np.array([1 / n_assets] * n_assets)

        result = minimize(negative_sharpe, x0, method='SLSQP',
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else None

    def create_evolution_visualization(self):
        """Create visualization showing efficient frontier evolution over time"""
        if not self.analysis_periods:
            print("❌ No analysis periods available")
            return

        # Create figure with subplots
        n_periods = len(self.analysis_periods)
        n_cols = 3
        n_rows = (n_periods + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        fig.suptitle('Efficient Frontier Evolution Through Rolling Analysis Periods\n' +
                     'Banking Sector Portfolio Optimization (Expanding Window Framework)',
                     fontsize=16, fontweight='bold', y=0.98)

        # Flatten axes for easier indexing
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()

        colors = plt.cm.viridis(np.linspace(0, 1, n_periods))

        for i, period in enumerate(self.analysis_periods):
            ax = axes[i]

            # Get data up to end of period (expanding window)
            period_data = self.returns.iloc[:period['end_idx']]

            # Generate efficient frontier
            frontier_vols, frontier_rets = self.generate_efficient_frontier(period_data)

            if frontier_vols and frontier_rets:
                # Plot efficient frontier
                ax.plot(frontier_vols, frontier_rets, color=colors[i], linewidth=3,
                        label=f'Efficient Frontier', alpha=0.8)

                # Find and plot optimal portfolios
                min_vol_weights = self.find_min_volatility_portfolio(period_data)
                max_sharpe_weights = self.find_max_sharpe_portfolio(period_data)

                if min_vol_weights is not None and max_sharpe_weights is not None:
                    mean_returns = period_data.mean() * 252
                    cov_matrix = period_data.cov() * 252

                    # Min volatility point
                    min_ret, min_vol, min_sharpe = self.calculate_portfolio_metrics(
                        min_vol_weights, mean_returns, cov_matrix)
                    ax.scatter(min_vol, min_ret, color='blue', s=100, marker='o',
                               label=f'Min Vol (SR: {min_sharpe:.3f})', zorder=5)

                    # Max Sharpe point
                    max_ret, max_vol, max_sharpe = self.calculate_portfolio_metrics(
                        max_sharpe_weights, mean_returns, cov_matrix)
                    ax.scatter(max_vol, max_ret, color='red', s=100, marker='*',
                               label=f'Max Sharpe (SR: {max_sharpe:.3f})', zorder=5)

                    # Capital Allocation Line
                    cal_vols = np.linspace(0, max(frontier_vols) * 1.1, 50)
                    cal_rets = self.risk_free_rate + (max_ret - self.risk_free_rate) / max_vol * cal_vols
                    ax.plot(cal_vols, cal_rets, '--', color='gold', alpha=0.6, linewidth=2)

                # Risk-free rate
                ax.scatter(0, self.risk_free_rate, color='black', marker='D', s=60, zorder=5)

            # Formatting
            ax.set_title(
                f'{period["name"]}\n{period["start_date"].strftime("%Y-%m")} to {period["end_date"].strftime("%Y-%m")}\n'
                f'Data Points: {period["data_points"]:,}', fontweight='bold', fontsize=11)
            ax.set_xlabel('Annual Volatility', fontweight='bold')
            ax.set_ylabel('Expected Annual Return', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc='upper left')

            # Format as percentages
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

        # Hide unused subplots
        for i in range(n_periods, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        # Save plot
        filename = self.output_path / f"efficient_frontier_evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📈 Efficient frontier evolution saved: {filename}")

        plt.show()

    def create_comparative_overlay(self):
        """Create single plot with multiple efficient frontiers overlaid"""
        if not self.analysis_periods:
            print("❌ No analysis periods available")
            return

        fig, ax = plt.subplots(figsize=(14, 10))

        colors = plt.cm.plasma(np.linspace(0, 1, len(self.analysis_periods)))

        # Store data for summary statistics
        all_sharpe_ratios = []
        all_volatilities = []
        all_returns = []

        for i, period in enumerate(self.analysis_periods):
            # Get expanding window data
            period_data = self.returns.iloc[:period['end_idx']]

            # Generate efficient frontier
            frontier_vols, frontier_rets = self.generate_efficient_frontier(period_data)

            if frontier_vols and frontier_rets:
                # Plot efficient frontier
                ax.plot(frontier_vols, frontier_rets, color=colors[i], linewidth=2.5,
                        label=f'{period["name"]} (n={period["data_points"]:,})', alpha=0.8)

                # Calculate optimal portfolios for statistics
                max_sharpe_weights = self.find_max_sharpe_portfolio(period_data)
                if max_sharpe_weights is not None:
                    mean_returns = period_data.mean() * 252
                    cov_matrix = period_data.cov() * 252
                    ret, vol, sharpe = self.calculate_portfolio_metrics(
                        max_sharpe_weights, mean_returns, cov_matrix)
                    all_sharpe_ratios.append(sharpe)
                    all_volatilities.append(vol)
                    all_returns.append(ret)

        # Add risk-free rate
        ax.scatter(0, self.risk_free_rate, color='black', marker='D', s=100,
                   label='Risk-Free Rate (2%)', zorder=10, edgecolors='white', linewidth=2)

        # Formatting
        ax.set_title('Efficient Frontier Evolution: Banking Sector Portfolio Optimization\n' +
                     'Expanding Window Analysis - Temporal Development of Optimization Opportunities',
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Annual Volatility (Risk)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Expected Annual Return', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='upper left', framealpha=0.9)

        # Format as percentages
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

        # Add summary statistics box
        if all_sharpe_ratios:
            textstr = f'''Evolution Summary:
• Analysis Periods: {len(self.analysis_periods)}
• Sharpe Ratio Range: {min(all_sharpe_ratios):.3f} - {max(all_sharpe_ratios):.3f}
• Volatility Range: {min(all_volatilities):.1%} - {max(all_volatilities):.1%}
• Return Range: {min(all_returns):.1%} - {max(all_returns):.1%}
• Data Window: Expanding from {self.min_history} to {len(self.returns):,} observations'''

            props = dict(boxstyle='round,pad=0.6', facecolor='lightcyan', alpha=0.9, edgecolor='navy')
            ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=11,
                    verticalalignment='bottom', horizontalalignment='right', bbox=props)

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)

        # Save plot
        filename = self.output_path / f"efficient_frontier_overlay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📈 Efficient frontier overlay saved: {filename}")

        plt.show()

    def create_frontier_metrics_analysis(self):
        """Create analysis of how frontier characteristics evolve"""
        if not self.analysis_periods:
            print("❌ No analysis periods available")
            return

        # Collect metrics for each period
        period_names = []
        max_sharpe_ratios = []
        min_volatilities = []
        max_returns = []
        data_points = []

        for period in self.analysis_periods:
            period_data = self.returns.iloc[:period['end_idx']]

            # Find optimal portfolios
            min_vol_weights = self.find_min_volatility_portfolio(period_data)
            max_sharpe_weights = self.find_max_sharpe_portfolio(period_data)

            if min_vol_weights is not None and max_sharpe_weights is not None:
                mean_returns = period_data.mean() * 252
                cov_matrix = period_data.cov() * 252

                # Min volatility metrics
                _, min_vol, _ = self.calculate_portfolio_metrics(min_vol_weights, mean_returns, cov_matrix)

                # Max Sharpe metrics
                max_ret, _, max_sharpe = self.calculate_portfolio_metrics(max_sharpe_weights, mean_returns, cov_matrix)

                period_names.append(period['name'])
                max_sharpe_ratios.append(max_sharpe)
                min_volatilities.append(min_vol)
                max_returns.append(max_ret)
                data_points.append(period['data_points'])

        # Create metrics visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Efficient Frontier Characteristics Evolution\n' +
                     'How Optimization Opportunities Change with Expanding Information',
                     fontsize=14, fontweight='bold', y=0.98)

        x_pos = np.arange(len(period_names))
        colors = plt.cm.viridis(np.linspace(0, 1, len(period_names)))

        # 1. Maximum Sharpe Ratio Evolution
        bars1 = ax1.bar(x_pos, max_sharpe_ratios, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_title('Maximum Sharpe Ratio Evolution', fontweight='bold')
        ax1.set_ylabel('Sharpe Ratio', fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(period_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')

        for bar, value in zip(bars1, max_sharpe_ratios):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

        # 2. Minimum Volatility Evolution
        bars2 = ax2.bar(x_pos, [v * 100 for v in min_volatilities], color=colors, alpha=0.8, edgecolor='black')
        ax2.set_title('Minimum Portfolio Volatility Evolution', fontweight='bold')
        ax2.set_ylabel('Minimum Volatility (%)', fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(period_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')

        for bar, value in zip(bars2, min_volatilities):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     f'{value:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=10)

        # 3. Maximum Expected Return Evolution
        bars3 = ax3.bar(x_pos, [r * 100 for r in max_returns], color=colors, alpha=0.8, edgecolor='black')
        ax3.set_title('Maximum Expected Return Evolution', fontweight='bold')
        ax3.set_ylabel('Expected Return (%)', fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(period_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')

        for bar, value in zip(bars3, max_returns):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     f'{value:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=10)

        # 4. Data Points Evolution
        bars4 = ax4.bar(x_pos, data_points, color=colors, alpha=0.8, edgecolor='black')
        ax4.set_title('Expanding Window Size Evolution', fontweight='bold')
        ax4.set_ylabel('Number of Observations', fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(period_names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')

        for bar, value in zip(bars4, data_points):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                     f'{value:,}', ha='center', va='bottom', fontweight='bold', fontsize=10)

        plt.tight_layout()

        # Save plot
        filename = self.output_path / f"frontier_metrics_evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Frontier metrics evolution saved: {filename}")

        plt.show()

    def generate_all_efficient_frontier_analysis(self):
        """Generate complete efficient frontier analysis suite"""
        print("📈 Generating Complete Efficient Frontier Analysis")
        print("=" * 65)

        if self.returns is None:
            print("❌ No data available for analysis")
            return

        print("1. Creating efficient frontier evolution visualization...")
        self.create_evolution_visualization()

        print("\n2. Creating comparative overlay visualization...")
        self.create_comparative_overlay()

        print("\n3. Creating frontier metrics analysis...")
        self.create_frontier_metrics_analysis()

        print(f"\n✅ Complete efficient frontier analysis generated!")
        print(f"📁 All plots saved to: {self.output_path}")
        print("\n🎯 Analysis Insights:")
        print("   • Demonstrates evolution of optimization opportunities")
        print("   • Maintains consistency with expanding window methodology")
        print("   • Provides theoretical foundation for empirical results")
        print("   • Shows impact of additional information on frontier characteristics")


def main():
    """Main execution function"""
    print("🏦 Rolling Efficient Frontier Analysis Generator")
    print("MSc Banking and Digital Finance - Dissertation Research")
    print("=" * 65)

    try:
        # Initialize analyzer
        analyzer = RollingEfficientFrontierAnalyzer(
            data_path="../../../data/raw",
            output_path="../../../results/charts"
        )

        # Generate complete analysis
        analyzer.generate_all_efficient_frontier_analysis()

        print("\n🎉 Efficient frontier analysis completed successfully!")
        print("Ready for dissertation inclusion and academic presentation.")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()