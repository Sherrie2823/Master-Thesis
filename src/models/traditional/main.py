#!/usr/bin/env python3
"""
Main execution script for AI-Enhanced Portfolio Optimization Project
MSc Banking and Digital Finance

This script serves as the primary entry point for running rolling portfolio optimization
with traditional methods. It implements expanding window analysis with monthly rebalancing
to provide academically rigorous baseline performance metrics.

Features:
- Rolling optimization with expanding windows
- Monthly portfolio rebalancing
- Transaction cost modeling
- Comprehensive performance tracking
- Multiple traditional optimization methods

Usage:
    python main.py

Author: Banking & Digital Finance Student
Date: Jul 2025
"""

import sys
from pathlib import Path
import numpy as np
import warnings
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.append(str(PROJECT_ROOT))

# Import project modules
try:
    from src.models.traditional.config import (
        RollingOptimizationConfig,
        BacktestConfig,
        PlotConfig,
        PerformanceMetrics,
        setup_logging,
        validate_data_files,
        get_rolling_config
    )
    from src.models.traditional.complete_traditional_methods import RollingPortfolioOptimizer
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all required files are in the project directory")
    sys.exit(1)


class PortfolioOptimizationManager:
    """
    Main manager class for coordinating rolling portfolio optimization analysis
    """

    def __init__(self):
        self.config = get_rolling_config()
        self.logger = setup_logging()
        self.optimizer = None
        self.results = None

        print("🏦 AI-Enhanced Portfolio Optimization - Rolling Framework")
        print("=" * 70)
        print("MSc Banking and Digital Finance Dissertation Project")
        print("=" * 70)

    def validate_environment(self):
        """Validate that the environment is ready for optimization"""
        print("\n🔍 Environment Validation")
        print("-" * 40)

        # Check data files
        if not validate_data_files():
            print("❌ Data validation failed")
            return False

        # Check required packages
        required_packages = [
            'pandas', 'numpy', 'matplotlib', 'seaborn',
            'scipy', 'pathlib', 'json'
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            print(f"❌ Missing packages: {', '.join(missing_packages)}")
            print(f"Install with: pip install {' '.join(missing_packages)}")
            return False

        print("✅ Environment validation completed successfully")
        return True

    def initialize_optimizer(self):
        """Initialize the rolling portfolio optimizer"""
        print("\n🚀 Initializing Rolling Portfolio Optimizer")
        print("-" * 50)

        try:
            self.optimizer = RollingPortfolioOptimizer(
                data_path=str(PROJECT_ROOT / "data" / "raw"),
                rebalance_freq=self.config['rebalance_frequency'],
                min_history=self.config['min_history'],
                transaction_cost=self.config['transaction_cost'],
                risk_free_rate=self.config['risk_free_rate']
            )

            if self.optimizer.returns is None:
                print("❌ Failed to load data")
                return False

            print("✅ Optimizer initialized successfully")
            self.logger.info("Rolling portfolio optimizer initialized")
            return True

        except Exception as e:
            print(f"❌ Error initializing optimizer: {e}")
            self.logger.error(f"Optimizer initialization failed: {e}")
            return False

    def display_configuration(self):
        """Display optimization configuration details"""
        print("\n📊 Optimization Configuration")
        print("-" * 40)
        print(f"Rebalancing frequency: {self.config['rebalance_frequency']} trading days")
        print(f"Minimum history required: {self.config['min_history']} trading days")
        print(f"Transaction cost: {self.config['transaction_cost']:.2%} per rebalancing")
        print(f"Risk-free rate: {self.config['risk_free_rate']:.1%} annual")
        print(f"Weight constraints: {self.config['min_weight']:.1%} - {self.config['max_weight']:.1%}")

        if self.optimizer:
            print(f"\nData characteristics:")
            print(f"Number of assets: {self.optimizer.returns.shape[1]}")
            print(f"Time period: {self.optimizer.returns.index[0].date()} to {self.optimizer.returns.index[-1].date()}")
            print(f"Total observations: {len(self.optimizer.returns)}")
            print(f"Rebalancing periods: {len(self.optimizer.rebalance_dates)}")

    def run_optimization(self):
        """Execute the rolling portfolio optimization"""
        print("\n🔄 Running Rolling Portfolio Optimization")
        print("-" * 50)

        try:
            # Run the optimization
            self.results = self.optimizer.run_rolling_optimization()

            if not self.results:
                print("❌ Optimization failed to generate results")
                return False

            print("✅ Rolling optimization completed successfully")
            self.logger.info("Rolling optimization completed")
            return True

        except Exception as e:
            print(f"❌ Error during optimization: {e}")
            self.logger.error(f"Optimization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def analyze_performance(self):
        """Analyze and display performance results"""
        print("\n📈 Performance Analysis")
        print("-" * 30)

        try:
            # Calculate performance metrics
            performance_df = self.optimizer.calculate_performance_metrics()

            print("\nPerformance Summary:")
            print("=" * 80)
            print(performance_df.to_string(index=False))

            # Identify best performing strategy
            sharpe_ratios = []
            for _, row in performance_df.iterrows():
                try:
                    sharpe_str = row['Sharpe Ratio'].rstrip('%')
                    sharpe_ratios.append(float(sharpe_str))
                except:
                    sharpe_ratios.append(0.0)

            if sharpe_ratios:
                best_idx = np.argmax(sharpe_ratios)
                best_strategy = performance_df.iloc[best_idx]['Strategy']
                best_sharpe = sharpe_ratios[best_idx]

                print(f"\n🏆 Best Risk-Adjusted Performance:")
                print(f"Strategy: {best_strategy}")
                print(f"Sharpe Ratio: {best_sharpe:.3f}")

            self.logger.info("Performance analysis completed")
            return performance_df

        except Exception as e:
            print(f"❌ Error in performance analysis: {e}")
            self.logger.error(f"Performance analysis failed: {e}")
            return None

    def generate_visualizations(self):
        """Generate performance visualizations"""
        print("\n📊 Generating Visualizations")
        print("-" * 35)

        try:
            self.optimizer.plot_portfolio_performance()
            print("✅ Performance plots generated successfully")
            self.logger.info("Visualizations generated")
            return True

        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")
            self.logger.error(f"Visualization generation failed: {e}")
            return False

    def save_results(self):
        """Save optimization results to files"""
        print("\n💾 Saving Results")
        print("-" * 20)

        try:
            performance_df = self.optimizer.save_results()
            print("✅ Results saved successfully")
            self.logger.info("Results saved to files")
            return True

        except Exception as e:
            print(f"❌ Error saving results: {e}")
            self.logger.error(f"Result saving failed: {e}")
            return False

    def generate_summary_report(self):
        """Generate executive summary of optimization results"""
        print("\n📋 Executive Summary")
        print("=" * 50)

        if not self.results:
            print("❌ No results available for summary")
            return

        # Count successful optimizations
        successful_methods = sum(1 for method, data in self.results.items()
                                 if data.get('weights_history'))

        total_rebalancing_periods = len(self.optimizer.rebalance_dates)

        print(f"Optimization Framework: Rolling with Expanding Windows")
        print(f"Analysis Period: {self.optimizer.returns.index[0].date()} to {self.optimizer.returns.index[-1].date()}")
        print(f"Total Assets: {self.optimizer.returns.shape[1]} banking sector stocks")
        print(f"Rebalancing Periods: {total_rebalancing_periods}")
        print(f"Successful Methods: {successful_methods}/5 traditional approaches")

        # Calculate aggregate statistics
        total_transaction_costs = {}
        final_values = {}

        for method_name, results in self.results.items():
            if results.get('transaction_costs'):
                total_transaction_costs[method_name] = sum(results['transaction_costs'])
            if results.get('portfolio_values'):
                final_values[method_name] = results['portfolio_values'][-1]

        if total_transaction_costs:
            avg_transaction_cost = np.mean(list(total_transaction_costs.values()))
            print(f"Average Transaction Costs: ${avg_transaction_cost:.2f}")

        if final_values:
            avg_final_value = np.mean(list(final_values.values()))
            print(f"Average Final Portfolio Value: ${avg_final_value:.2f}")

        print(f"\n🎯 Key Achievements:")
        print(f"   ✅ Eliminated look-ahead bias through expanding windows")
        print(f"   ✅ Implemented realistic transaction cost modeling")
        print(f"   ✅ Established robust baseline for AI method comparison")
        print(f"   ✅ Generated comprehensive performance metrics")

        print(f"\n📈 Next Steps:")
        print(f"   🔄 Implement LSTM-based return prediction models")
        print(f"   🔄 Develop reinforcement learning optimization")
        print(f"   🔄 Create ensemble AI methods")
        print(f"   🔄 Conduct comprehensive comparative analysis")

        self.logger.info("Executive summary generated")

    def run_complete_analysis(self):
        """Execute the complete rolling optimization analysis workflow"""
        start_time = datetime.now()

        # Step 1: Validate environment
        if not self.validate_environment():
            return False

        # Step 2: Initialize optimizer
        if not self.initialize_optimizer():
            return False

        # Step 3: Display configuration
        self.display_configuration()

        # Step 4: Run optimization
        if not self.run_optimization():
            return False

        # Step 5: Analyze performance
        performance_df = self.analyze_performance()
        if performance_df is None:
            return False

        # Step 6: Generate visualizations
        self.generate_visualizations()

        # Step 7: Save results
        self.save_results()

        # Step 8: Generate summary
        self.generate_summary_report()

        # Calculate execution time
        end_time = datetime.now()
        execution_time = end_time - start_time

        print(f"\n✅ Complete Analysis Finished Successfully!")
        print(f"Execution time: {execution_time}")
        print(f"Results saved to: {PROJECT_ROOT / 'results'}")

        self.logger.info(f"Complete analysis finished in {execution_time}")
        return True


def main():
    """Main execution function"""
    try:
        # Initialize the optimization manager
        manager = PortfolioOptimizationManager()

        # Run the complete analysis
        success = manager.run_complete_analysis()

        if success:
            print("\n🎉 Traditional Portfolio Optimization Baseline Established!")
            print("Ready for AI model development and comparison.")
            return 0
        else:
            print("\n💡 Analysis completed with issues. Please review logs for details.")
            return 1

    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)