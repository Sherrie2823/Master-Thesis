#!/usr/bin/env python3
"""
Main execution script for AI-Enhanced Portfolio Optimization Project
MSc Banking and Digital Finance

Extended version: Runs Traditional, RF, and XGB strategies together.
Generates performance comparison tables and plots.

Author: Banking & Digital Finance Student
Date: Aug 2025
"""

import sys
from pathlib import Path
import numpy as np
import warnings
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# ======================
# Paths
# ======================
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]  # 根据你的项目层级；必要时改成 parents[2]/[1]
sys.path.append(str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# ======================
# Project imports
# ======================
from real_data.config import PLOTS_PATH
from real_data.config import (
    RollingOptimizationConfig,
    setup_logging,
    validate_data_files,
    get_rolling_config
)
from src.models.traditional.complete_traditional_methods import RollingPortfolioOptimizer

# Suppress warnings
warnings.filterwarnings('ignore')

# ======================
# Helper functions
# ======================
def _newey_west_t(x: np.ndarray, lags: int | None = None):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    T = len(x)
    if T < 10:
        return np.nan, np.nan
    mu = x.mean()
    e = x - mu
    if lags is None:
        lags = int(round(T ** (1 / 3)))
        lags = max(1, min(lags, T - 1))
    gamma0 = (e @ e) / T
    var = gamma0
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        cov = (e[L:] @ e[:-L]) / T
        var += 2 * w * cov
    se = (var / T) ** 0.5
    tval = mu / se if se > 0 else np.nan
    return mu, tval


def _block_bootstrap_sharpe_diff_p(r_a: np.ndarray, r_b: np.ndarray,
                                   B: int = 2000, block: int = 10, seed: int = 42):
    rng = np.random.default_rng(seed)
    r_a = np.asarray(r_a, float)
    r_b = np.asarray(r_b, float)
    T = min(len(r_a), len(r_b))
    r_a, r_b = r_a[:T], r_b[:T]

    def sharpe(x):
        s = x.std(ddof=1)
        return x.mean() / s if s > 0 else 0.0

    obs = sharpe(r_a) - sharpe(r_b)
    idx = np.arange(T)

    def resample_take():
        k = int(np.ceil(T / block))
        starts = rng.integers(0, T, size=k)
        take = np.concatenate([(idx[s:(s + block)] % T) for s in starts])[:T]
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
    r_s = np.asarray(r_s, float)
    r_b = np.asarray(r_b, float)
    T = min(len(r_s), len(r_b))
    r_s, r_b = r_s[:T], r_b[:T]
    xs = r_b - rf_daily
    ys = r_s - rf_daily
    X = np.vstack([np.ones(T), xs]).T
    beta = np.linalg.lstsq(X, ys, rcond=None)[0]
    a, b = beta[0], beta[1]
    resid = ys - (a + b * xs)
    _, t_alpha = _newey_west_t(resid + a)
    alpha_ann = a * 252.0
    ss_tot = np.sum((ys - ys.mean()) ** 2)
    ss_res = np.sum(resid ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return alpha_ann, t_alpha, b, R2


def build_significance_table(optimizer, rf_annual: float = 0.02, benchmark: str = "equal_weight"):
    out = []
    rf_daily = rf_annual / 252.0
    if benchmark not in optimizer.portfolio_results:
        benchmark = next(iter(optimizer.portfolio_results.keys()))
    r_bench = np.array(optimizer.portfolio_results[benchmark]['returns_history'], float)

    for name, res in optimizer.portfolio_results.items():
        r = np.array(res['returns_history'], float)
        if len(r) < 30:
            continue
        mu_ex, t_ex = _newey_west_t(r - rf_daily)
        mu_r, _ = _newey_west_t(r)
        vol = np.std(r, ddof=1)
        sharpe = mu_r / vol if vol > 0 else np.nan
        sdiff, p_sdiff = _block_bootstrap_sharpe_diff_p(r, r_bench)
        alpha_ann, t_alpha, beta, R2 = _capm_alpha(r, r_bench, rf_daily)
        out.append({
            "Strategy": name,
            "Obs": len(r),
            "Mean Excess (bp/day)": mu_ex * 1e4,
            "t(Mean Excess)": t_ex,
            "Sharpe (daily)": sharpe,
            "SharpeDiff vs Bench": sdiff,
            "p(SharpeDiff)": p_sdiff,
            "Alpha (annual %)": alpha_ann * 100,
            "t(Alpha)": t_alpha,
            "Beta": beta,
            "R2": R2,
            "Bench": benchmark
        })
    return pd.DataFrame(out).sort_values("Sharpe (daily)", ascending=False)

# ======================
# Plotting Suite
# ======================
class PlotSuite:
    def __init__(self, optimizer, outdir: Path):
        self.opt = optimizer
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    @staticmethod
    def _compute_drawdown(equity: pd.Series):
        running_max = equity.cummax()
        dd = equity / running_max - 1.0
        return dd

    @staticmethod
    def _ann_metrics(returns: pd.Series, rf=0.0):
        if returns is None or len(returns) < 2:
            return np.nan, np.nan, np.nan
        mu = (1 + returns).prod() ** (252 / len(returns)) - 1
        vol = returns.std(ddof=1) * np.sqrt(252)
        sharpe = (mu - rf) / vol if vol and vol > 0 else np.nan
        return mu, vol, sharpe

    def _collect(self):
        bag = {}
        for name, res in self.opt.portfolio_results.items():
            # dates
            dates = None
            for key in ['dates', 'index', 'timestamps']:
                if key in res and res[key] is not None and len(res[key]) > 0:
                    dates = pd.to_datetime(res[key])
                    break

            # returns
            r = None
            if 'returns_history' in res and res['returns_history'] is not None:
                r = pd.Series(res['returns_history'])
                if dates is not None and len(dates) >= len(r):
                    r.index = dates[-len(r):]

            # equity
            v = None
            if 'value_history' in res and res['value_history'] is not None:
                v = pd.Series(res['value_history'])
                if dates is not None and len(dates) >= len(v):
                    v.index = dates[-len(v):]
            elif r is not None:
                v = (1 + r).cumprod()

            # transaction costs
            tc = None
            if 'tc_history' in res and res['tc_history'] is not None:
                tc = pd.Series(res['tc_history'])
                if dates is not None and len(dates) >= len(tc):
                    tc.index = dates[-len(tc):]

            # turnover
            tovr = None
            if 'turnover_history' in res and res['turnover_history'] is not None:
                tovr = pd.Series(res['turnover_history'])
                if dates is not None and len(dates) >= len(tovr):
                    tovr.index = dates[-len(tovr):]

            # weights
            weights_df = None
            w = res.get('weights_history', None)
            if w is not None and len(w) > 0:
                try:
                    if isinstance(w, pd.DataFrame):
                        weights_df = w.copy()
                        if dates is not None and len(dates) >= len(weights_df):
                            weights_df.index = dates[-len(weights_df):]
                    else:
                        weights_df = pd.DataFrame(w)
                        if dates is not None and len(dates) >= len(weights_df):
                            weights_df.index = dates[-len(weights_df):]
                except Exception:
                    weights_df = None

            bag[name] = {'returns': r, 'equity': v, 'tc': tc, 'turnover': tovr, 'weights': weights_df}
        return bag

    def plot_equity_curves(self):
        data = self._collect()
        ok = False
        plt.figure(figsize=(10, 6))
        for name, d in data.items():
            if d['equity'] is not None and len(d['equity']) > 1:
                d['equity'].plot(label=name)
                ok = True
        if not ok:
            plt.close(); return
        plt.title('Portfolio Value Over Time')
        plt.ylabel('Net Asset Value')
        plt.xlabel('Date')
        plt.legend()
        plt.tight_layout()
        fp = self.outdir / f"equity_curves_{self.ts}.png"
        plt.savefig(fp, dpi=300)
        plt.close()

    def plot_rolling_sharpe(self, window=63, rf_daily=0.0):
        data = self._collect()
        ok = False
        plt.figure(figsize=(10, 6))
        for name, d in data.items():
            r = d['returns']
            if r is not None and len(r) > window:
                roll_mu = r.rolling(window).mean()
                roll_vol = r.rolling(window).std()
                rs = (roll_mu - rf_daily) / (roll_vol + 1e-12)
                rs.plot(label=name)
                ok = True
        if not ok:
            plt.close(); return
        plt.title(f'Rolling Sharpe (window={window})')
        plt.ylabel('Sharpe')
        plt.xlabel('Date')
        plt.legend()
        plt.tight_layout()
        fp = self.outdir / f"rolling_sharpe_{self.ts}.png"
        plt.savefig(fp, dpi=300)
        plt.close()

    def plot_drawdowns(self):
        data = self._collect()
        ok = False
        plt.figure(figsize=(10, 6))
        for name, d in data.items():
            eq = d['equity']
            if eq is not None and len(eq) > 1:
                dd = self._compute_drawdown(eq)
                dd.plot(label=name)
                ok = True
        if not ok:
            plt.close(); return
        plt.title('Drawdown (Underwater) Analysis')
        plt.ylabel('Drawdown')
        plt.xlabel('Date')
        plt.legend()
        plt.tight_layout()
        fp = self.outdir / f"drawdown_{self.ts}.png"
        plt.savefig(fp, dpi=300)
        plt.close()

    def plot_total_tc(self):
        data = self._collect()
        names, totals = [], []
        for name, d in data.items():
            tc = d['tc']
            if tc is not None and len(tc) > 0:
                names.append(name)
                totals.append(np.nansum(tc.values))
        if not names:
            return
        plt.figure(figsize=(8, 5))
        pd.Series(totals, index=names).sort_values(ascending=False).plot(kind='bar')
        plt.title('Total Transaction Costs by Strategy')
        plt.ylabel('Total TC (currency units)')
        plt.xlabel('Strategy')
        plt.tight_layout()
        fp = self.outdir / f"total_tc_{self.ts}.png"
        plt.savefig(fp, dpi=300)
        plt.close()

    def plot_ret_vol_scatter(self, rf_annual=0.02):
        data = self._collect()
        rows = []
        for name, d in data.items():
            r = d['returns']
            if r is not None and len(r) > 10:
                mu, vol, sr = self._ann_metrics(r, rf=rf_annual)
                rows.append((name, mu, vol, sr))
        if not rows:
            return
        df = pd.DataFrame(rows, columns=['Strategy', 'AnnualReturn', 'AnnualVol', 'Sharpe'])
        plt.figure(figsize=(7, 6))
        plt.scatter(df['AnnualVol'], df['AnnualReturn'])
        for _, row in df.iterrows():
            plt.annotate(row['Strategy'], (row['AnnualVol'], row['AnnualReturn']),
                         xytext=(3, 3), textcoords='offset points')
        plt.xlabel('Annualized Volatility')
        plt.ylabel('Annualized Return')
        plt.title('Return vs Volatility')
        plt.tight_layout()
        fp = self.outdir / f"ret_vol_scatter_{self.ts}.png"
        plt.savefig(fp, dpi=300)
        plt.close()

    def plot_return_distribution(self, bins=50):
        data = self._collect()
        ok = False
        plt.figure(figsize=(10, 6))
        for name, d in data.items():
            r = d['returns']
            if r is not None and len(r) > 10:
                plt.hist(r.dropna().values, bins=bins, alpha=0.4, density=True, label=name)
                ok = True
        if not ok:
            plt.close(); return
        plt.title('Daily Return Distribution')
        plt.xlabel('Daily Return')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        fp = self.outdir / f"return_dist_{self.ts}.png"
        plt.savefig(fp, dpi=300)
        plt.close()

    def plot_turnover(self):
        data = self._collect()
        ok = False
        plt.figure(figsize=(10, 5))
        for name, d in data.items():
            tovr = d['turnover']
            if tovr is not None and len(tovr) > 1:
                tovr.plot(label=name, alpha=0.85)
                ok = True
        if not ok:
            plt.close(); return
        plt.title('Portfolio Turnover Over Time')
        plt.ylabel('Turnover')
        plt.xlabel('Date')
        plt.legend()
        plt.tight_layout()
        fp = self.outdir / f"turnover_{self.ts}.png"
        plt.savefig(fp, dpi=300)
        plt.close()

    def plot_weights_heatmap(self):
        data = self._collect()
        for name, d in data.items():
            wdf = d['weights']
            if wdf is not None and isinstance(wdf, pd.DataFrame) and wdf.shape[0] > 1:
                avg_w = wdf.mean(axis=0).sort_values(ascending=False)
                plt.figure(figsize=(max(6, 0.4 * len(avg_w)), 4))
                plt.imshow(avg_w.values.reshape(1, -1), aspect='auto')
                plt.yticks([0], ['avg weight'])
                plt.xticks(range(len(avg_w)), avg_w.index, rotation=60, ha='right')
                plt.colorbar(label='Weight')
                plt.title(f'Average Weights Heatmap ({name})')
                plt.tight_layout()
                fp = self.outdir / f"avg_weights_heatmap_{name}_{self.ts}.png"
                plt.savefig(fp, dpi=300)
                plt.close()
                break  # 只画一个策略，避免产生过多图片

    def run_all(self, rf_annual=0.02):
        self.plot_equity_curves()
        self.plot_rolling_sharpe(window=63, rf_daily=rf_annual / 252.0)
        self.plot_drawdowns()
        self.plot_total_tc()
        # Extras
        self.plot_ret_vol_scatter(rf_annual=rf_annual)
        self.plot_return_distribution()
        self.plot_turnover()
        self.plot_weights_heatmap()

# ======================
# Main Manager
# ======================
class PortfolioOptimizationManager:
    def __init__(self):
        self.config = get_rolling_config()
        self.logger = setup_logging()
        self.optimizer = None
        self.results = None
        print("🏦 AI-Enhanced Portfolio Optimization - Rolling Framework")

    def validate_environment(self):
        if not validate_data_files():
            print("❌ Data validation failed")
            return False
        print("✅ Environment validation completed successfully")
        return True

    def initialize_optimizer(self):
        try:
            self.optimizer = RollingPortfolioOptimizer(
                data_path=DATA_DIR,
                rebalance_freq=self.config['rebalance_frequency'],
                min_history=self.config['min_history'],
                transaction_cost=self.config['transaction_cost'],
                risk_free_rate=self.config['risk_free_rate']
            )
            return True
        except Exception as e:
            print(f"❌ Error initializing optimizer: {e}")
            return False

    def run_optimization(self):
        print("\n🔄 Running Rolling Portfolio Optimization (Traditional + AI)")
        try:
            self.results = self.optimizer.run_rolling_optimization()
            if not self.results:
                return False

            import glob
            # XGB 概率文件
            prob_files_xgb = sorted(glob.glob(str(PROJECT_ROOT / "results" / "xgb_probs_5D_*.csv")))
            if prob_files_xgb:
                probs_xgb = pd.read_csv(prob_files_xgb[-1], index_col=0, parse_dates=True)
                self.optimizer.backtest_xgb_prob_strategy(
                    probs_xgb, method='normalize', hold=15, tc=0.001, max_w=0.20, name='XGB_norm'
                )
                self.optimizer.backtest_xgb_prob_strategy(
                    probs_xgb, method='topn', top_n=5, hold=15, tc=0.001, max_w=0.20, name='XGB_top5'
                )

            # RF 概率文件（复用同一回测方法）
            prob_files_rf = sorted(glob.glob(str(PROJECT_ROOT / "results" / "rf_probs_5D_*.csv")))
            if prob_files_rf:
                probs_rf = pd.read_csv(prob_files_rf[-1], index_col=0, parse_dates=True)
                self.optimizer.backtest_xgb_prob_strategy(
                    probs_rf, method='normalize', hold=15, tc=0.001, max_w=0.20, name='RF_norm'
                )
                self.optimizer.backtest_xgb_prob_strategy(
                    probs_rf, method='topn', top_n=5, hold=15, tc=0.001, max_w=0.20, name='RF_top5'
                )

            return True
        except Exception as e:
            print(f"❌ Error during optimization: {e}")
            return False

    def analyze_performance(self):
        print("\n📈 Performance Analysis")
        try:
            performance_df = self.optimizer.calculate_performance_metrics()

            # 保存显著性检验
            sig_df = build_significance_table(
                self.optimizer, rf_annual=RollingOptimizationConfig.RISK_FREE_RATE
            )
            results_dir = PROJECT_ROOT / "results"
            results_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            sig_path = results_dir / f"significance_tests_{ts}.csv"
            sig_df.to_csv(sig_path, index=False)

            # Sharpe 对比图（兼容字符串/数值）
            comp_df = performance_df.copy()
            if "Sharpe Ratio" in comp_df.columns:
                try:
                    comp_df["Sharpe_val"] = (
                        comp_df["Sharpe Ratio"].astype(str).str.rstrip('%').astype(float)
                    )
                except Exception:
                    # 回退：如果 Sharpe 本身就是数值
                    comp_df["Sharpe_val"] = pd.to_numeric(comp_df["Sharpe Ratio"], errors='coerce')
            elif "Sharpe" in comp_df.columns:
                comp_df["Sharpe_val"] = pd.to_numeric(comp_df["Sharpe"], errors='coerce')
            else:
                # 最后回退：尝试用 returns 现算（如果可得）
                comp_df["Sharpe_val"] = np.nan

            plt.figure(figsize=(10, 6))
            comp_df.plot(x="Strategy", y=["Sharpe_val"], kind="bar", legend=False, ax=plt.gca())
            plt.ylabel("Sharpe Ratio")
            plt.title("Sharpe Ratio Comparison - Traditional vs RF vs XGB")
            plt.tight_layout()
            cmp_path = results_dir / f"strategy_comparison_{ts}.png"
            plt.savefig(cmp_path, dpi=300)
            plt.close()

            # 追加生成全套图表（优先保存到 PLOTS_PATH，失败则落到 results_dir）
            try:
                plot_outdir = PLOTS_PATH if PLOTS_PATH else results_dir
            except Exception:
                plot_outdir = results_dir

            plotter = PlotSuite(self.optimizer, plot_outdir)
            plotter.run_all(rf_annual=RollingOptimizationConfig.RISK_FREE_RATE)
            print(f"🖼️ Extra plots saved to: {plot_outdir}")

            # 也顺手保存一份 performance_df
            performance_df.to_csv(results_dir / f"performance_metrics_{ts}.csv", index=False)

            return True
        except Exception as e:
            print(f"❌ Error in performance analysis: {e}")
            return False

    def run_complete_analysis(self):
        if not self.validate_environment():
            return False
        if not self.initialize_optimizer():
            return False
        if not self.run_optimization():
            return False
        if not self.analyze_performance():
            return False
        print("\n✅ Complete Analysis Finished Successfully!")
        return True


def main():
    manager = PortfolioOptimizationManager()
    if manager.run_complete_analysis():
        print("\n🎉 All strategies analyzed successfully!")
        return 0
    else:
        print("\n💡 Analysis completed with issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
