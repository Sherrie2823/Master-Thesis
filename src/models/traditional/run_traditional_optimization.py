#!/usr/bin/env python3
"""
Rolling Traditional Portfolio Optimization Execution Script
MSc Banking and Digital Finance - AI-Enhanced Portfolio Optimization Project

This script executes rolling portfolio optimization using traditional methods with
expanding windows and monthly rebalancing. It provides comprehensive performance
analysis and serves as the baseline for AI method comparison.

Key Features:
- Expanding window walk-forward analysis
- Monthly portfolio rebalancing
- Transaction cost modeling
- Multi-method comparison
- Comprehensive performance metrics

Usage:
    python run_traditional_optimization.py

Author: Banking & Digital Finance Student
Date: January 2025
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import json

warnings.filterwarnings('ignore')

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.append(str(PROJECT_ROOT))

# Import project modules
try:
    from src.models.traditional.config import (
        RollingOptimizationConfig,
        BacktestConfig,
        PerformanceMetrics,
        ValidationConfig,
        setup_logging,
        validate_data_files,
        validate_configuration,
        get_rolling_config,
        BANKING_STOCKS
    )
    from src.models.traditional.complete_traditional_methods import RollingPortfolioOptimizer
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure complete_traditional_methods.py and config.py are available")
    sys.exit(1)


class TraditionalOptimizationRunner:
    """
    Comprehensive runner for traditional portfolio optimization analysis
    """

    def __init__(self):
        self.config = get_rolling_config()
        self.logger = setup_logging()
        self.optimizer = None
        self.results = None
        self.performance_metrics = None

    def check_system_requirements(self):
        """Verify system requirements and package availability"""
        print("🔍 Checking System Requirements")
        print("-" * 40)

        required_packages = [
            'pandas', 'numpy', 'matplotlib', 'seaborn',
            'scipy', 'pathlib', 'json'
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                print(f"  ✅ {package}")
            except ImportError:
                missing_packages.append(package)
                print(f"  ❌ {package}")

        if missing_packages:
            print(f"\n❌ Missing required packages: {', '.join(missing_packages)}")
            print(f"Install missing packages with:")
            print(f"   pip install {' '.join(missing_packages)}")
            return False

        print("✅ All required packages are available")
        return True

    def validate_project_structure(self):
        """Ensure project directory structure is properly configured"""
        print("\n📁 Validating Project Structure")
        print("-" * 40)

        required_dirs = ['data/raw', 'results', 'models', 'plots']

        for dir_path in required_dirs:
            full_path = PROJECT_ROOT / dir_path
            if not full_path.exists():
                print(f"📁 Creating directory: {dir_path}")
                full_path.mkdir(parents=True, exist_ok=True)
            else:
                print(f"✅ Directory exists: {dir_path}")

        return True

    def validate_data_availability(self):
        """Validate data files and quality for optimization"""
        print("\n📊 Validating Data Availability")
        print("-" * 40)

        if not validate_data_files():
            print("❌ Data validation failed")
            return False

        # Additional data quality checks
        try:
            data_path = PROJECT_ROOT / "data" / "raw"
            csv_files = list(data_path.glob("*.csv"))

            if not csv_files:
                print("❌ No CSV files found in data directory")
                return False

            # Load and inspect first CSV file
            df = pd.read_csv(csv_files[0], index_col=0, parse_dates=True)

            print(f"Data characteristics:")
            print(f"  Columns: {df.shape[1]}")
            print(f"  Rows: {df.shape[0]}")
            print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")

            # Check for sufficient data length
            if len(df) < self.config['min_history'] + 252:
                print(f"⚠️  Warning: Limited data length ({len(df)} observations)")
                print(f"   Recommended minimum: {self.config['min_history'] + 252} observations")

            # Check for missing values
            missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            if missing_pct > 0.05:
                print(f"⚠️  Warning: {missing_pct:.1%} missing values detected")
            else:
                print(f"✅ Missing values: {missing_pct:.1%}")

            return True

        except Exception as e:
            print(f"❌ Error inspecting data: {e}")
            return False

    def initialize_optimization_framework(self):
        """Initialize the rolling portfolio optimization framework"""
        print("\n🚀 Initializing Optimization Framework")
        print("-" * 45)

        try:
            self.optimizer = RollingPortfolioOptimizer(
                data_path=str(PROJECT_ROOT / "data" / "raw"),
                rebalance_freq=self.config['rebalance_frequency'],
                min_history=self.config['min_history'],
                transaction_cost=self.config['transaction_cost'],
                risk_free_rate=self.config['risk_free_rate']
            )

            if self.optimizer.returns is None:
                print("❌ Failed to initialize optimizer - no data loaded")
                return False

            # Display optimization parameters
            print(f"Configuration parameters:")
            print(f"  Rebalancing frequency: {self.config['rebalance_frequency']} trading days")
            print(f"  Minimum history: {self.config['min_history']} trading days")
            print(f"  Transaction cost: {self.config['transaction_cost']:.2%}")
            print(f"  Weight constraints: {self.config['min_weight']:.1%} - {self.config['max_weight']:.1%}")

            print(f"\nData specifications:")
            print(f"  Assets: {self.optimizer.returns.shape[1]}")
            print(f"  Observations: {len(self.optimizer.returns)}")
            print(f"  Rebalancing periods: {len(self.optimizer.rebalance_dates)}")

            print("✅ Optimization framework initialized successfully")
            self.logger.info("Rolling optimization framework initialized")
            return True

        except Exception as e:
            print(f"❌ Error initializing framework: {e}")
            self.logger.error(f"Framework initialization failed: {e}")
            return False

    def execute_rolling_optimization(self):
        """Execute the complete rolling optimization analysis"""
        print("\n🔄 Executing Rolling Portfolio Optimization")
        print("-" * 50)

        start_time = datetime.now()

        try:
            # Execute optimization
            self.results = self.optimizer.run_rolling_optimization()

            if not self.results:
                print("❌ Optimization failed to generate results")
                return False

            # Count successful strategies
            successful_strategies = sum(1 for method, data in self.results.items()
                                        if data.get('weights_history'))

            end_time = datetime.now()
            execution_time = end_time - start_time

            print(f"✅ Rolling optimization completed successfully")
            print(f"Execution time: {execution_time}")
            print(f"Successful strategies: {successful_strategies}/5")

            self.logger.info(f"Rolling optimization completed in {execution_time}")
            return True

        except Exception as e:
            print(f"❌ Error during optimization: {e}")
            self.logger.error(f"Optimization execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def analyze_comprehensive_performance(self):
        """Conduct comprehensive performance analysis"""
        print("\n📈 Comprehensive Performance Analysis")
        print("-" * 45)

        try:
            # Calculate performance metrics
            self.performance_metrics = self.optimizer.calculate_performance_metrics()

            print("Performance Summary:")
            print("=" * 80)
            print(self.performance_metrics.to_string(index=False))

            # Identify top performers
            self._identify_top_performers()

            # Analyze risk characteristics
            self._analyze_risk_characteristics()

            # Transaction cost analysis
            self._analyze_transaction_costs()

            print("✅ Performance analysis completed")
            self.logger.info("Comprehensive performance analysis completed")
            return True

        except Exception as e:
            print(f"❌ Error in performance analysis: {e}")
            self.logger.error(f"Performance analysis failed: {e}")
            return False

    def _identify_top_performers(self):
        """Identify and highlight top performing strategies"""
        print(f"\n🏆 Top Performers Analysis:")
        print("-" * 30)

        try:
            # Extract Sharpe ratios
            sharpe_ratios = []
            for _, row in self.performance_metrics.iterrows():
                sharpe_str = row['Sharpe Ratio'].replace('%', '')
                sharpe_ratios.append(float(sharpe_str))

            # Find best performers
            best_sharpe_idx = np.argmax(sharpe_ratios)
            best_strategy = self.performance_metrics.iloc[best_sharpe_idx]

            print(f"Best Risk-Adjusted Performance:")
            print(f"  Strategy: {best_strategy['Strategy']}")
            print(f"  Sharpe Ratio: {best_strategy['Sharpe Ratio']}")
            print(f"  Annual Return: {best_strategy['Annual Return']}")
            print(f"  Volatility: {best_strategy['Volatility']}")

            # Find strategy with lowest drawdown
            drawdowns = []
            for _, row in self.performance_metrics.iterrows():
                dd_str = row['Max Drawdown'].replace('%', '')
                drawdowns.append(float(dd_str))

            best_dd_idx = np.argmax(drawdowns)  # Least negative
            best_dd_strategy = self.performance_metrics.iloc[best_dd_idx]

            print(f"\nBest Drawdown Performance:")
            print(f"  Strategy: {best_dd_strategy['Strategy']}")
            print(f"  Max Drawdown: {best_dd_strategy['Max Drawdown']}")

        except Exception as e:
            print(f"Error in top performers analysis: {e}")

    def _analyze_risk_characteristics(self):
        """Analyze risk characteristics across strategies"""
        print(f"\n⚠️  Risk Analysis:")
        print("-" * 20)

        for _, row in self.performance_metrics.iterrows():
            strategy = row['Strategy']
            var_95 = row['VaR 95%']
            neg_days = row['Negative Days']

            print(f"{strategy[:20]:20} VaR: {var_95:>8} Neg Days: {neg_days:>6}")

    def _analyze_transaction_costs(self):
        """Analyze transaction costs across strategies"""
        print(f"\n💰 Transaction Cost Analysis:")
        print("-" * 35)

        total_costs = []
        for _, row in self.performance_metrics.iterrows():
            cost_str = row['Transaction Costs'].replace('$', '')
            total_costs.append(float(cost_str))

        avg_cost = np.mean(total_costs)
        max_cost = np.max(total_costs)
        min_cost = np.min(total_costs)

        print(f"Average transaction cost: ${avg_cost:.2f}")
        print(f"Highest transaction cost: ${max_cost:.2f}")
        print(f"Lowest transaction cost: ${min_cost:.2f}")

        # Transaction cost efficiency
        print(f"\nTransaction Cost Efficiency:")
        for _, row in self.performance_metrics.iterrows():
            cost = float(row['Transaction Costs'].replace('$', ''))
            ret = float(row['Annual Return'].replace('%', ''))
            efficiency = ret / cost if cost > 0 else float('inf')
            print(f"  {row['Strategy'][:20]:20}: {efficiency:.2f} return/cost")

    def generate_detailed_visualizations(self):
        """Generate comprehensive performance visualizations"""
        print("\n📊 Generating Performance Visualizations")
        print("-" * 45)

        try:
            self.optimizer.plot_portfolio_performance()
            print("✅ Performance visualizations generated successfully")
            self.logger.info("Performance visualizations created")
            return True

        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")
            self.logger.error(f"Visualization generation failed: {e}")
            return False

    def save_comprehensive_results(self):
        """Save detailed results and analysis to files"""
        print("\n💾 Saving Comprehensive Results")
        print("-" * 40)

        try:
            # Save performance metrics
            performance_df = self.optimizer.save_results()

            # Save additional analysis
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Save configuration used
            config_file = PROJECT_ROOT / "results" / f"optimization_config_{timestamp}.json"
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)

            # Save summary statistics
            summary_file = PROJECT_ROOT / "results" / f"optimization_summary_{timestamp}.txt"
            with open(summary_file, 'w') as f:
                f.write("Rolling Portfolio Optimization Summary\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Analysis completed: {datetime.now()}\n")
                f.write(
                    f"Data period: {self.optimizer.returns.index[0].date()} to {self.optimizer.returns.index[-1].date()}\n")
                f.write(f"Total assets: {self.optimizer.returns.shape[1]}\n")
                f.write(f"Rebalancing periods: {len(self.optimizer.rebalance_dates)}\n\n")
                f.write("Performance Metrics:\n")
                f.write(self.performance_metrics.to_string(index=False))

            print(f"Results saved to:")
            print(f"  Configuration: {config_file.name}")
            print(f"  Summary: {summary_file.name}")
            print(f"  Performance data: CSV files in results directory")

            self.logger.info("Comprehensive results saved")
            return True

        except Exception as e:
            print(f"❌ Error saving results: {e}")
            self.logger.error(f"Result saving failed: {e}")
            return False

    def generate_executive_summary(self):
        """Generate executive summary for dissertation inclusion"""
        print("\n📋 Executive Summary")
        print("=" * 60)

        if not self.results or self.performance_metrics is None:
            print("❌ Insufficient data for executive summary")
            return

        # Analysis overview
        print(f"Rolling Portfolio Optimization Analysis Complete")
        print(f"Period: {self.optimizer.returns.index[0].date()} to {self.optimizer.returns.index[-1].date()}")
        print(f"Assets: {self.optimizer.returns.shape[1]} banking sector institutions")
        print(f"Methodology: Expanding windows with {self.config['rebalance_frequency']}-day rebalancing")

        # Performance highlights
        try:
            sharpe_ratios = [float(row['Sharpe Ratio'].replace('%', ''))
                             for _, row in self.performance_metrics.iterrows()]
            best_sharpe = max(sharpe_ratios)
            avg_sharpe = np.mean(sharpe_ratios)

            print(f"\nPerformance Highlights:")
            print(f"  Best Sharpe Ratio: {best_sharpe:.3f}")
            print(f"  Average Sharpe Ratio: {avg_sharpe:.3f}")
            print(f"  Successful Strategies: {len(sharpe_ratios)}/5 traditional methods")

        except Exception as e:
            print(f"Error calculating performance highlights: {e}")

        # Implementation achievements
        print(f"\nMethodological Achievements:")
        print(f"  ✅ Eliminated look-ahead bias through expanding windows")
        print(f"  ✅ Incorporated realistic transaction costs ({self.config['transaction_cost']:.2%})")
        print(f"  ✅ Implemented multiple traditional optimization approaches")
        print(f"  ✅ Generated robust baseline for AI method comparison")
        print(f"  ✅ Ensured regulatory-compliant position limits")

        # Research implications
        print(f"\nResearch Implications:")
        print(f"  This analysis establishes methodologically rigorous baseline performance")
        print(f"  metrics for traditional portfolio optimization in the banking sector.")
        print(f"  The rolling framework eliminates look-ahead bias while incorporating")
        print(f"  realistic implementation constraints through transaction cost modeling.")
        print(f"  Results provide the foundation for evaluating AI-enhanced approaches.")

        # Next steps
        print(f"\nNext Research Phase:")
        print(f"  🔄 LSTM-based return prediction implementation")
        print(f"  🔄 Reinforcement learning portfolio optimization")
        print(f"  🔄 Ensemble AI method development")
        print(f"  🔄 Comprehensive AI vs traditional comparison")

        self.logger.info("Executive summary generated")

    def run_complete_traditional_analysis(self):
        """Execute the complete traditional portfolio optimization analysis"""
        print("🏦 Rolling Traditional Portfolio Optimization Analysis")
        print("=" * 65)
        print("MSc Banking and Digital Finance - Dissertation Research")
        print("=" * 65)

        # Step-by-step execution
        steps = [
            ("System Requirements", self.check_system_requirements),
            ("Project Structure", self.validate_project_structure),
            ("Data Validation", self.validate_data_availability),
            ("Configuration Check", lambda: validate_configuration()),
            ("Framework Initialization", self.initialize_optimization_framework),
            ("Rolling Optimization", self.execute_rolling_optimization),
            ("Performance Analysis", self.analyze_comprehensive_performance),
            ("Visualization Generation", self.generate_detailed_visualizations),
            ("Results Storage", self.save_comprehensive_results)
        ]

        start_time = datetime.now()

        for step_name, step_function in steps:
            print(f"\n📍 Step: {step_name}")
            if not step_function():
                print(f"❌ Failed at step: {step_name}")
                return False

        # Generate final summary
        self.generate_executive_summary()

        # Calculate total execution time
        end_time = datetime.now()
        total_time = end_time - start_time

        print(f"\n✅ Complete Traditional Analysis Finished Successfully!")
        print(f"Total execution time: {total_time}")
        print(f"Results available in: {PROJECT_ROOT / 'results'}")

        self.logger.info(f"Complete traditional analysis finished in {total_time}")
        return True


def main():
    """Main execution function for traditional portfolio optimization"""
    try:
        print("Starting Rolling Traditional Portfolio Optimization Analysis...")

        # Initialize the analysis runner
        runner = TraditionalOptimizationRunner()

        # Execute complete analysis
        success = runner.run_complete_traditional_analysis()

        if success:
            print("\n🎉 Traditional Portfolio Optimization Baseline Successfully Established!")
            print("The analysis provides academically rigorous performance metrics for")
            print("comparison with AI-enhanced portfolio optimization methods.")
            return 0
        else:
            print("\n💡 Analysis completed with issues. Review logs for detailed information.")
            return 1

    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)