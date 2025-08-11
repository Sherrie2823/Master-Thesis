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


# ⬇️ main.py 位于 .../src/models/traditional/
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]         # -> .../Sherry-s-Master-Thesis
sys.path.append(str(PROJECT_ROOT))     # 让 import 按仓库根解析
DATA_DIR = PROJECT_ROOT / "data" / "raw"

import numpy as np
import warnings
from datetime import datetime
import pandas as pd
from real_data.config import PLOTS_PATH
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.append(str(PROJECT_ROOT))

# Import project modules
try:
    from real_data.config import (
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

# === Significance tests helpers (only used in main.py) ===

def _newey_west_t(x: np.ndarray, lags: int | None = None):
    """NW(HAC) t-stat for mean(x)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    T = len(x)
    if T < 10:
        return np.nan, np.nan
    mu = x.mean()
    e  = x - mu
    if lags is None:
        lags = int(round(T ** (1/3)))
        lags = max(1, min(lags, T-1))
    gamma0 = (e @ e) / T
    var = gamma0
    for L in range(1, lags + 1):
        w = 1 - L/(lags+1)
        cov = (e[L:] @ e[:-L]) / T
        var += 2 * w * cov
    se = (var / T) ** 0.5
    tval = mu / se if se > 0 else np.nan
    return mu, tval

def _block_bootstrap_sharpe_diff_p(r_a: np.ndarray, r_b: np.ndarray,
                                   B: int = 2000, block: int = 10, seed: int = 42):
    """p-value for Sharpe(a)-Sharpe(b) via circular block bootstrap."""
    rng = np.random.default_rng(seed)
    r_a = np.asarray(r_a, float); r_b = np.asarray(r_b, float)
    T = min(len(r_a), len(r_b))
    r_a, r_b = r_a[:T], r_b[:T]

    def sharpe(x):
        s = x.std(ddof=1)
        return x.mean()/s if s > 0 else 0.0

    obs = sharpe(r_a) - sharpe(r_b)

    idx = np.arange(T)
    def resample_take():
        k = int(np.ceil(T / block))
        starts = rng.integers(0, T, size=k)
        take = np.concatenate([ (idx[s:(s+block)] % T) for s in starts ])[:T]
        return take

    ge = 0
    for _ in range(B):
        take = resample_take()
        val = sharpe(r_a[take]) - sharpe(r_b[take])
        if abs(val) >= abs(obs):
            ge += 1
    pval = (ge + 1) / (B + 1)
    return obs, pval

def _capm_alpha(r_s: np.ndarray, r_b: np.ndarray, rf_daily: float = 0.0):
    """CAPM alpha (annualized), NW t(alpha), beta, R^2."""
    r_s = np.asarray(r_s, float); r_b = np.asarray(r_b, float)
    T = min(len(r_s), len(r_b))
    r_s, r_b = r_s[:T], r_b[:T]
    xs = r_b - rf_daily
    ys = r_s - rf_daily
    X = np.vstack([np.ones(T), xs]).T
    beta = np.linalg.lstsq(X, ys, rcond=None)[0]
    a, b = beta[0], beta[1]
    resid = ys - (a + b*xs)
    # 用 NW 对 alpha 做 t（近似：对 resid+a 的均值做 HAC）
    _, t_alpha = _newey_west_t(resid + a)
    alpha_ann = a * 252.0
    ss_tot = np.sum((ys - ys.mean())**2)
    ss_res = np.sum(resid**2)
    R2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return alpha_ann, t_alpha, b, R2

def build_significance_table(optimizer, rf_annual: float = 0.02, benchmark: str = "equal_weight"):
    """Assemble significance table from optimizer.portfolio_results."""
    out = []
    rf_daily = rf_annual/252.0

    # 选择基准（没有的话就用第一个策略兜底）
    if benchmark not in optimizer.portfolio_results:
        benchmark = next(iter(optimizer.portfolio_results.keys()))
    r_bench = np.array(optimizer.portfolio_results[benchmark]['returns_history'], float)

    for name, res in optimizer.portfolio_results.items():
        r = np.array(res['returns_history'], float)
        if len(r) < 30:  # 太短跳过
            continue

        # 1) 均值超额收益 (对无风险)
        mu_ex, t_ex = _newey_west_t(r - rf_daily)

        # 2) Sharpe 与 Sharpe 差
        mu_r, _ = _newey_west_t(r)
        vol = np.std(r, ddof=1)
        sharpe = mu_r/vol if vol > 0 else np.nan
        sdiff, p_sdiff = _block_bootstrap_sharpe_diff_p(r, r_bench)

        # 3) CAPM alpha / beta / R2（用基准作“市场”）
        alpha_ann, t_alpha, beta, R2 = _capm_alpha(r, r_bench, rf_daily)

        out.append({
            "Strategy": name.replace('_',' ').title(),
            "Obs": len(r),
            "Mean Excess (bp/day)": mu_ex*1e4,
            "t(Mean Excess)": t_ex,
            "Sharpe (daily)": sharpe,
            "SharpeDiff vs Bench": sdiff,
            "p(SharpeDiff)": p_sdiff,
            "Alpha (annual %)": alpha_ann*100,
            "t(Alpha)": t_alpha,
            "Beta": beta,
            "R2": R2,
            "Bench": benchmark.replace('_',' ').title()
        })
    df = pd.DataFrame(out)
    return df.sort_values("Sharpe (daily)", ascending=False)




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
            data_dir = PROJECT_ROOT / "data" / "raw"
        # >>> Debug: 看看路径是否存在，以及里面有哪些 CSV
            print(f"[DEBUG] data_dir = {DATA_DIR} | exists={DATA_DIR.exists()}")
            print(f"[DEBUG] files = {[p.name for p in DATA_DIR.glob('*.csv')]}")

            self.optimizer = RollingPortfolioOptimizer(
                data_path=DATA_DIR,
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
            try:
                import pandas as pd, glob, os
                # 取最新的概率矩阵（你 test_xgb_weights.py 保存的文件）
                prob_files = sorted(glob.glob(str(PROJECT_ROOT / "results" / "xgb_probs_5D_*.csv")))
                if not prob_files:
                    print("⚠️ 未找到 xgb_probs_5D_*.csv，跳过 XGB 概率策略")
                    return True
                probs_path = prob_files[-1]
                probs_df = pd.read_csv(probs_path, index_col=0, parse_dates=True)
                print(f"📥 Loaded probs: {os.path.basename(probs_path)}, shape={probs_df.shape}")

                # 跑两种权重规则，名字会体现在绩效表里
                self.optimizer.backtest_xgb_prob_strategy(
                    probs_df, method='normalize', hold=15, tc=0.001, max_w=0.20, name='xgb_norm_5D_hold15'
                )
                self.optimizer.backtest_xgb_prob_strategy(
                    probs_df, method='topn', top_n=5, hold=15, tc=0.001, max_w=0.20, name='xgb_top5_5D_hold15'
                )
            except Exception as e:
                print(f"❌ XGB 概率策略接入失败: {e}")
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
            
            # === Significance tests ===
            sig_df = build_significance_table(
                self.optimizer,
                rf_annual=RollingOptimizationConfig.RISK_FREE_RATE,  # 和你的配置一致
                benchmark="equal_weight"  # 或 "markowitz_min_vol"
            )
            print("\nSignificance tests (HAC & bootstrap):")
            print(sig_df.to_string(index=False))

            # 保存到 results/
            from datetime import datetime
            results_dir = PROJECT_ROOT / "results"
            results_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            sig_path = results_dir / f"significance_tests_{ts}.csv"
            sig_df.to_csv(sig_path, index=False)
            print(f"\n✅ Significance table saved -> {sig_path}")

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
            import matplotlib.pyplot as plt
            from pathlib import Path
            from datetime import datetime

            Path(PLOTS_PATH).mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fig = plt.gcf()
            fig.savefig(Path(PLOTS_PATH) / f"performance_overview_{ts}.png", dpi=300, bbox_inches="tight")
            fig.savefig(Path(PLOTS_PATH) / f"performance_overview_{ts}.pdf", bbox_inches="tight")
            plt.close(fig)

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