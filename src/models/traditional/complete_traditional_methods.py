"""
Rolling Traditional Portfolio Optimization Implementation
MSc Banking and Digital Finance - AI-Enhanced Portfolio Optimization Project

Implements traditional portfolio optimization methods with rolling rebalancing:
1. Markowitz Mean-Variance Optimization
2. Black-Litterman Model
3. Risk Parity Approaches

Key Features:
- Expanding window walk-forward analysis
- Monthly rebalancing (21 trading days)
- Transaction cost modeling
- No look-ahead bias
- Comprehensive performance tracking

Author: Sherrie
Date: Jul 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from scipy.optimize import minimize
import warnings
from typing import Dict, Tuple, List, Optional

warnings.filterwarnings('ignore')


class RollingPortfolioOptimizer:
    """
    Rolling portfolio optimization with expanding windows and monthly rebalancing
    """

    def __init__(self,
                 data_path: str = "data/raw",
                 rebalance_freq: int = 21,  # Monthly rebalancing
                 min_history: int = 252,  # Minimum 1 year of data for first optimization
                 transaction_cost: float = 0.001,  # 0.1% transaction cost
                 risk_free_rate: float = 0.02):

        self.data_path = Path(data_path)
        self.rebalance_freq = rebalance_freq
        self.min_history = min_history
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate

        # Data storage
        self.returns = None
        self.prices = None
        self.rebalance_dates = []

        # Results storage
        self.portfolio_results = {}
        self.performance_history = {}

        # Load data
        self.load_data()
        self.setup_rebalancing_schedule()

    def load_data(self):
        """Load stock prices or returns with robust file selection"""
        try:
            data_dir = Path(self.data_path)
            if not data_dir.exists():
                raise FileNotFoundError(f"Data dir not found: {data_dir}")

            # 1) 先找价格文件，再找收益文件
            prices_fp = next((p for p in data_dir.glob("banking_prices_10y*.csv")), None)
            returns_fp = next((p for p in data_dir.glob("banking_returns_10y*.csv")), None)

            source = None
            prices = None

            if prices_fp and prices_fp.exists():
                source = prices_fp.name
                df = pd.read_csv(prices_fp, index_col=0, parse_dates=True)
                df = df.sort_index()
                df = df.apply(pd.to_numeric, errors="coerce")
                prices = df
                returns = df.pct_change().dropna(how="all")
            elif returns_fp and returns_fp.exists():
                source = returns_fp.name
                df = pd.read_csv(returns_fp, index_col=0, parse_dates=True)
                df = df.sort_index()
                returns = df.apply(pd.to_numeric, errors="coerce")
            else:
                raise FileNotFoundError(
                    "Neither banking_prices_10y*.csv nor banking_returns_10y*.csv found "
                    f"in {data_dir}"
                )

            # 2) 基础清洗
            # 丢掉全为空的列
            returns = returns.dropna(axis=1, how="all")
            # 行内全部为空则去掉
            returns = returns.dropna(how="all")
            # 前后再补一次排序保证稳定
            returns = returns.sort_index()

            # 3) 简单数据质量提示（可选）
            extreme = (returns.abs() > 0.5).sum().sum()
            if extreme > 0:
                print(f"⚠️Warning: {extreme} extreme returns (>50%) detected")

            # 4) 长度校验
            if len(returns) < (self.min_history + 252):  # 至少训练1年 + 一些OOS
                raise ValueError(
                    f"Insufficient data: {len(returns)} observations, need at least "
                    f"{self.min_history + 252}"
                )

            # 5) 赋值保存
            self.prices = prices
            self.returns = returns

            print(f"✅ Loaded file: {source}")
            print(f"📈 Returns shape: {self.returns.shape}")
            print(f"Date range: {self.returns.index[0].date()} → {self.returns.index[-1].date()}")

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise

    def _validate_data(self):
        """Validate data quality and completeness"""
        # Check for missing values
        missing_pct = self.returns.isnull().sum().sum() / (self.returns.shape[0] * self.returns.shape[1])
        if missing_pct > 0.05:
            print(f"⚠️ Warning: {missing_pct:.1%} missing values detected")

        # Check for extreme returns
        extreme_returns = (self.returns.abs() > 0.5).sum().sum()
        if extreme_returns > 0:
            print(f"⚠️ Warning: {extreme_returns} extreme returns (>50%) detected")

        # Check for sufficient data length
        if len(self.returns) < self.min_history + 252:  # Need extra data for out-of-sample
            raise ValueError(
                f"Insufficient data: {len(self.returns)} observations, need at least {self.min_history + 252}")

    def setup_rebalancing_schedule(self):
        """Setup rebalancing dates using expanding windows"""
        if self.returns is None:
            return

        start_date = self.returns.index[self.min_history]
        end_date = self.returns.index[-1]

        current_date = start_date
        self.rebalance_dates = []

        while current_date <= end_date:
            if current_date in self.returns.index:
                self.rebalance_dates.append(current_date)

            # Find next rebalancing date
            next_idx = self.returns.index.get_loc(current_date) + self.rebalance_freq
            if next_idx < len(self.returns.index):
                current_date = self.returns.index[next_idx]
            else:
                break

        print(f"📅 Rebalancing schedule: {len(self.rebalance_dates)} rebalancing dates")
        print(f"First rebalance: {self.rebalance_dates[0].date()}")
        print(f"Last rebalance: {self.rebalance_dates[-1].date()}")

    def calculate_portfolio_metrics(self, weights: np.ndarray,
                                    returns_data: pd.DataFrame,
                                    cov_matrix: np.ndarray) -> Tuple[float, float, float]:
        """Calculate portfolio performance metrics for given weights"""
        mean_returns = returns_data.mean() * 252  # Annualized

        portfolio_return = np.sum(weights * mean_returns)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol

        return portfolio_return, portfolio_vol, sharpe_ratio

    def optimize_markowitz_max_sharpe(self, returns_data: pd.DataFrame) -> Dict:
        """Markowitz Maximum Sharpe Ratio optimization"""
        mean_returns = returns_data.mean() * 252
        cov_matrix = returns_data.cov() * 252

        n_assets = len(mean_returns)
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0.01, 0.20) for _ in range(n_assets))
        x0 = np.array([1 / n_assets] * n_assets)

        def negative_sharpe(weights):
            portfolio_return = np.sum(weights * mean_returns)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            return -(portfolio_return - self.risk_free_rate) / portfolio_vol

        result = minimize(negative_sharpe, x0, method='SLSQP',
                          bounds=bounds, constraints=constraints)

        if result.success:
            weights = result.x
            performance = self.calculate_portfolio_metrics(weights, returns_data, cov_matrix)
            return {
                'weights': weights,
                'performance': performance,
                'method': 'Markowitz Maximum Sharpe',
                'success': True
            }
        else:
            return {'success': False, 'error': 'Optimization failed'}

    def optimize_markowitz_min_vol(self, returns_data: pd.DataFrame) -> Dict:
        """Markowitz Minimum Volatility optimization"""
        cov_matrix = returns_data.cov() * 252

        n_assets = len(returns_data.columns)
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0.01, 0.20) for _ in range(n_assets))
        x0 = np.array([1 / n_assets] * n_assets)

        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))

        result = minimize(portfolio_volatility, x0, method='SLSQP',
                          bounds=bounds, constraints=constraints)

        if result.success:
            weights = result.x
            performance = self.calculate_portfolio_metrics(weights, returns_data, cov_matrix)
            return {
                'weights': weights,
                'performance': performance,
                'method': 'Markowitz Minimum Volatility',
                'success': True
            }
        else:
            return {'success': False, 'error': 'Optimization failed'}

    def optimize_black_litterman(self, returns_data: pd.DataFrame) -> Dict:
        """Black-Litterman optimization with momentum-based views"""
        try:
            n_assets = len(returns_data.columns)
            cov_matrix = returns_data.cov() * 252

            # Market equilibrium (equal weights as prior)
            w_market = np.array([1 / n_assets] * n_assets)
            risk_aversion = 3.0
            pi = risk_aversion * np.dot(cov_matrix, w_market)

            # Generate momentum-based views
            if len(returns_data) >= 126:  # Need at least 6 months for momentum
                recent_period = min(63, len(returns_data) // 4)  # 3 months or 1/4 of data
                recent_returns = returns_data.tail(recent_period).mean() * 252
                long_term_returns = returns_data.mean() * 252

                views = []
                view_matrix = []
                view_uncertainty = []

                for i, asset in enumerate(returns_data.columns):
                    momentum = recent_returns[asset] - long_term_returns[asset]
                    if abs(momentum) > 0.02:  # 2% threshold
                        views.append(recent_returns[asset])
                        pick_vector = np.zeros(n_assets)
                        pick_vector[i] = 1.0
                        view_matrix.append(pick_vector)
                        uncertainty = 0.05 / (1 + abs(momentum))
                        view_uncertainty.append(uncertainty)

                # If no strong views, create relative view
                if len(views) == 0:
                    returns_diff = recent_returns - long_term_returns
                    best_idx = returns_diff.idxmax()
                    worst_idx = returns_diff.idxmin()

                    best_asset_idx = list(returns_data.columns).index(best_idx)
                    worst_asset_idx = list(returns_data.columns).index(worst_idx)

                    relative_view = returns_diff[best_idx] - returns_diff[worst_idx]
                    views.append(relative_view)

                    pick_vector = np.zeros(n_assets)
                    pick_vector[best_asset_idx] = 1.0
                    pick_vector[worst_asset_idx] = -1.0
                    view_matrix.append(pick_vector)
                    view_uncertainty.append(0.1)

                # Black-Litterman calculations
                Q = np.array(views)
                P = np.array(view_matrix)
                Omega = np.diag(view_uncertainty)
                tau = 0.025

                tau_sigma_inv = np.linalg.inv(tau * cov_matrix)
                p_omega_inv_p = np.dot(P.T, np.dot(np.linalg.inv(Omega), P))

                bl_precision = tau_sigma_inv + p_omega_inv_p
                bl_mu = np.dot(np.linalg.inv(bl_precision),
                               np.dot(tau_sigma_inv, pi) + np.dot(P.T, np.dot(np.linalg.inv(Omega), Q)))
                bl_sigma = np.linalg.inv(bl_precision)

                # Optimize using Black-Litterman inputs
                def bl_negative_sharpe(weights):
                    portfolio_return = np.sum(weights * bl_mu)
                    portfolio_vol = np.sqrt(np.dot(weights, np.dot(bl_sigma, weights)))
                    return -(portfolio_return - self.risk_free_rate) / portfolio_vol

                constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
                bounds = tuple((0.01, 0.20) for _ in range(n_assets))

                result = minimize(bl_negative_sharpe, w_market, method='SLSQP',
                                  bounds=bounds, constraints=constraints)

                if result.success:
                    weights = result.x
                    # Calculate performance using original data for consistency
                    performance = self.calculate_portfolio_metrics(weights, returns_data, cov_matrix)
                    return {
                        'weights': weights,
                        'performance': performance,
                        'method': 'Black-Litterman',
                        'views_used': len(views),
                        'success': True
                    }

            # Fallback to equal weights if BL fails
            weights = w_market
            performance = self.calculate_portfolio_metrics(weights, returns_data, cov_matrix)
            return {
                'weights': weights,
                'performance': performance,
                'method': 'Black-Litterman (Equal Weight Fallback)',
                'views_used': 0,
                'success': True
            }

        except Exception as e:
            # Fallback to equal weights
            n_assets = len(returns_data.columns)
            weights = np.array([1 / n_assets] * n_assets)
            cov_matrix = returns_data.cov() * 252
            performance = self.calculate_portfolio_metrics(weights, returns_data, cov_matrix)
            return {
                'weights': weights,
                'performance': performance,
                'method': 'Black-Litterman (Error Fallback)',
                'views_used': 0,
                'success': True,
                'error': str(e)
            }

    def optimize_risk_parity(self, returns_data: pd.DataFrame) -> Dict:
        """Equal Risk Contribution optimization"""
        cov_matrix = returns_data.cov() * 252
        n_assets = len(returns_data.columns)

        def risk_budget_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            target_contrib = portfolio_vol / len(weights)
            return np.sum((contrib - target_contrib) ** 2)

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0.01, 0.20) for _ in range(n_assets))
        x0 = np.array([1 / n_assets] * n_assets)

        result = minimize(risk_budget_objective, x0, method='SLSQP',
                          bounds=bounds, constraints=constraints)

        if result.success:
            weights = result.x
            performance = self.calculate_portfolio_metrics(weights, returns_data, cov_matrix)
            return {
                'weights': weights,
                'performance': performance,
                'method': 'Risk Parity',
                'success': True
            }
        else:
            # Fallback to equal weights
            weights = x0
            performance = self.calculate_portfolio_metrics(weights, returns_data, cov_matrix)
            return {
                'weights': weights,
                'performance': performance,
                'method': 'Risk Parity (Equal Weight Fallback)',
                'success': True
            }

    def optimize_equal_weight(self, returns_data: pd.DataFrame) -> Dict:
        """Equal Weight portfolio (naive diversification)"""
        n_assets = len(returns_data.columns)
        weights = np.array([1 / n_assets] * n_assets)
        cov_matrix = returns_data.cov() * 252
        performance = self.calculate_portfolio_metrics(weights, returns_data, cov_matrix)

        return {
            'weights': weights,
            'performance': performance,
            'method': 'Equal Weight',
            'success': True
        }

    def calculate_transaction_costs(self, new_weights: np.ndarray,
                                    old_weights: np.ndarray,
                                    portfolio_value: float) -> float:
        """Calculate transaction costs for rebalancing"""
        weight_changes = np.abs(new_weights - old_weights)
        total_turnover = np.sum(weight_changes)
        return total_turnover * self.transaction_cost * portfolio_value

    def run_rolling_optimization(self):
        """Execute rolling optimization with expanding windows"""
        if not self.rebalance_dates:
            print("❌ No rebalancing dates available")
            return

        print("\n🔄 Starting Rolling Portfolio Optimization...")
        print(f"Rebalancing frequency: {self.rebalance_freq} days")
        print(f"Transaction cost: {self.transaction_cost:.1%}")
        print(f"Number of rebalancing periods: {len(self.rebalance_dates)}")

        # Initialize optimization methods
        optimization_methods = {
            'markowitz_max_sharpe': self.optimize_markowitz_max_sharpe,
            'markowitz_min_vol': self.optimize_markowitz_min_vol,
            'black_litterman': self.optimize_black_litterman,
            'risk_parity': self.optimize_risk_parity,
            'equal_weight': self.optimize_equal_weight
        }

        # Initialize results storage
        for method_name in optimization_methods.keys():
            self.portfolio_results[method_name] = {
                'weights_history': [],
                'returns_history': [],
                'portfolio_values': [100.0],  # Start with $100
                'transaction_costs': [],
                'rebalance_dates': [],
                'optimization_results': []
            }

        # Rolling optimization loop
        for i, rebalance_date in enumerate(self.rebalance_dates):
            print(f"\n📊 Rebalancing {i + 1}/{len(self.rebalance_dates)}: {rebalance_date.date()}")

            # Get expanding window data (all data up to rebalance date)
            historical_data = self.returns.loc[:rebalance_date].iloc[:-1]  # Exclude current date

            if len(historical_data) < self.min_history:
                continue

            # Run optimization for each method
            for method_name, optimize_func in optimization_methods.items():
                try:
                    # Optimize portfolio
                    opt_result = optimize_func(historical_data)

                    if opt_result.get('success', False):
                        new_weights = opt_result['weights']

                        # Calculate transaction costs if not first period
                        transaction_cost = 0.0
                        if self.portfolio_results[method_name]['weights_history']:
                            old_weights = self.portfolio_results[method_name]['weights_history'][-1]
                            current_value = self.portfolio_results[method_name]['portfolio_values'][-1]
                            transaction_cost = self.calculate_transaction_costs(
                                new_weights, old_weights, current_value)

                        # Store results
                        self.portfolio_results[method_name]['weights_history'].append(new_weights)
                        self.portfolio_results[method_name]['transaction_costs'].append(transaction_cost)
                        self.portfolio_results[method_name]['rebalance_dates'].append(rebalance_date)
                        self.portfolio_results[method_name]['optimization_results'].append(opt_result)

                        print(f"✅ {method_name}: Sharpe {opt_result['performance'][2]:.3f}")
                    else:
                        print(f"❌ {method_name}: Optimization failed")

                except Exception as e:
                    print(f"❌ {method_name}: Error - {e}")

        # Calculate portfolio performance between rebalancing dates
        self._calculate_portfolio_performance()

        print("\n✅ Rolling optimization completed!")
        return self.portfolio_results
    
    def backtest_xgb_prob_strategy(self, probs_df,
                               method='topn',        # 'topn' 或 'normalize'
                               top_n=5,              # method='topn' 时生效
                               min_w=0.00, max_w=0.20,
                               hold=15,              # 持有期（天）
                               tc=0.001,             # 交易成本（单边）
                               t_plus_one=True,      # T+1 生效
                               name='xgb_prob'):
        """
        用外部概率矩阵进行回测，生成组合净值并写入 self.portfolio_results[name]
        probs_df: 行=日期(index), 列=股票，值=上涨概率(0~1)
        """
        assert self.returns is not None, "returns 尚未加载"
        # 1) 对齐日期 & 资产
        assets = [c for c in self.returns.columns if c in probs_df.columns]
        if len(assets) == 0:
            raise ValueError("probs_df 与 returns 无共同资产列")
        idx = self.returns.index.intersection(probs_df.index)
        idx = idx.sort_values()
        rets = self.returns.loc[idx, assets]
        probs = probs_df.loc[idx, assets]

        # 2) 概率 -> 当期权重 的规则
        def prob_to_weights(row):
            s = row.dropna()
            if s.empty:
                return pd.Series(0.0, index=assets)
            if method == 'normalize':
                w = s.clip(lower=0)
                w = w / w.sum() if w.sum() > 0 else pd.Series(1/len(s), index=s.index)
            else:  # 'topn'
                k = min(top_n, len(s))
                sel = s.nlargest(k)
                w = sel / sel.sum() if sel.sum() > 0 else pd.Series(1/k, index=sel.index)
            # 约束 + 重新归一
            w = w.clip(lower=min_w, upper=max_w)
            w = w / w.sum() if w.sum() > 0 else pd.Series(1/len(w), index=w.index)
            # 展开到全资产
            return w.reindex(assets).fillna(0.0)
    
        # 3) 回测主循环（支持 T+1 + 持有期 + 成本）
        pv = 100.0
        values = [pv]
        returns_hist = []
        txn_costs = []
        rebalance_dates = []

        cur_w = pd.Series(0.0, index=assets)
        pending_w = cur_w.copy()
        last_reb_i = -10**9
        
        for i, dt in enumerate(idx):
            # 到期重平衡：生成“新权重”
            if (i == 0) or (i - last_reb_i) >= hold:
                new_w = prob_to_weights(probs.loc[dt])
                if t_plus_one:
                        # 当天仍用旧权重，交易排到“今日收盘后”
                    pending_w = new_w
                else:
                        # 当天切换，先扣成本再计算今日收益
                    turnover = (new_w - cur_w).abs().sum()
                    cost = turnover * tc * pv
                    pv -= cost
                    txn_costs.append(cost)
                    rebalance_dates.append(dt)
                    cur_w = new_w
                last_reb_i = i

            
            # 计算当日收益（用当前权重）
            day_ret = float((cur_w * rets.loc[dt]).sum())
            pv *= (1.0 + day_ret)
            returns_hist.append(day_ret)
            values.append(pv)
        
            # 若 T+1：在“收盘后”切换权重并扣成本（影响次日起的净值）
            if t_plus_one and (i == last_reb_i):
                turnover = (pending_w - cur_w).abs().sum()
                cost = turnover * tc * pv
                pv -= cost
                txn_costs.append(cost)
                rebalance_dates.append(dt)
                cur_w = pending_w

            # 4) 写入到统一结果容器
        self.portfolio_results[name] = {
            'weights_history': [],              # 可选：如需记录可在重平衡日 append(cur_w.values)
            'returns_history': returns_hist,    # 日度序列
            'portfolio_values': values,         # 与 returns_hist 对齐（长度= returns+1）
            'transaction_costs': txn_costs,     # 只在重平衡日记录
            'rebalance_dates': rebalance_dates, # 重平衡日期列表
            'optimization_results': []
        }

        # 5) 辅助打印
        ann = (values[-1]/values[0])**(252/len(returns_hist)) - 1
        vol = np.std(returns_hist) * np.sqrt(252)
        sharpe = (ann - self.risk_free_rate) / vol if vol > 0 else 0.0
        from datetime import datetime as _dt
        print(f"\n== XGB Prob Strategy [{name}] ==")
        print(f"Period: {idx[0].date()} ~ {idx[-1].date()} | Days={len(idx)}")
        print(f"Ann: {ann:.2%}, Vol: {vol:.2%}, Sharpe: {sharpe:.2f}, "
            f"Rebals: {len(rebalance_dates)}, TC sum: ${sum(txn_costs):.2f}")
        return self.portfolio_results[name]
        

    def _calculate_portfolio_performance(self):
        """Calculate portfolio performance between rebalancing dates"""
        for method_name, results in self.portfolio_results.items():
            if not results['weights_history']:
                continue

            portfolio_returns = []
            current_value = 100.0  # Start with $100

            for i, rebalance_date in enumerate(results['rebalance_dates']):
                weights = results['weights_history'][i]
                transaction_cost = results['transaction_costs'][i]

                # Apply transaction cost
                current_value -= transaction_cost

                # Calculate returns until next rebalancing date
                if i < len(results['rebalance_dates']) - 1:
                    next_rebalance = results['rebalance_dates'][i + 1]
                    period_returns = self.returns.loc[rebalance_date:next_rebalance].iloc[1:]
                else:
                    # Last period - calculate to end of data
                    period_returns = self.returns.loc[rebalance_date:].iloc[1:]

                # Calculate portfolio returns for this period
                for _, day_returns in period_returns.iterrows():
                    portfolio_return = np.sum(weights * day_returns.values)
                    portfolio_returns.append(portfolio_return)
                    current_value *= (1 + portfolio_return)
                    results['portfolio_values'].append(current_value)

            results['returns_history'] = portfolio_returns

    def calculate_performance_metrics(self) -> pd.DataFrame:
        """Calculate comprehensive performance metrics for all strategies"""
        performance_summary = []

        for method_name, results in self.portfolio_results.items():
            if not results['returns_history']:
                continue

            returns = np.array(results['returns_history'])
            portfolio_values = np.array(results['portfolio_values'])

            # Basic metrics
            total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
            annual_return = (portfolio_values[-1] / portfolio_values[0]) ** (252 / len(returns)) - 1
            volatility = np.std(returns) * np.sqrt(252)

            # Risk metrics
            sharpe_ratio = (annual_return - self.risk_free_rate) / volatility
            max_drawdown = self._calculate_max_drawdown(portfolio_values)
            var_95 = np.percentile(returns, 5)

            # Additional metrics
            negative_days = np.sum(returns < 0) / len(returns)
            total_transaction_costs = np.sum(results['transaction_costs'])

            performance_summary.append({
                'Strategy': method_name.replace('_', ' ').title(),
                'Total Return': f"{total_return:.2%}",
                'Annual Return': f"{annual_return:.2%}",
                'Volatility': f"{volatility:.2%}",
                'Sharpe Ratio': f"{sharpe_ratio:.3f}",
                'Max Drawdown': f"{max_drawdown:.2%}",
                'VaR 95%': f"{var_95:.2%}",
                'Negative Days': f"{negative_days:.1%}",
                'Transaction Costs': f"${total_transaction_costs:.2f}"
            })

        return pd.DataFrame(performance_summary)

    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        return np.min(drawdown)

    def plot_portfolio_performance(self):
        """Plot portfolio performance over time"""
        plt.figure(figsize=(15, 10))

        # Create subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Portfolio values over time
        for method_name, results in self.portfolio_results.items():
            if results['portfolio_values']:
                dates = pd.date_range(
                    start=self.returns.index[self.min_history],
                    periods=len(results['portfolio_values']),
                    freq='D'
                )
                ax1.plot(dates, results['portfolio_values'],
                         label=method_name.replace('_', ' ').title(), linewidth=2)

        ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Performance comparison
        perf_df = self.calculate_performance_metrics()
        sharpe_ratios = [float(sr.rstrip('%')) for sr in perf_df['Sharpe Ratio']]
        ax2.bar(range(len(perf_df)), sharpe_ratios, color='steelblue', alpha=0.7)
        ax2.set_title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_xticks(range(len(perf_df)))
        ax2.set_xticklabels(perf_df['Strategy'], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)

        # Drawdown analysis
        for method_name, results in self.portfolio_results.items():
            if results['portfolio_values']:
                values = np.array(results['portfolio_values'])
                peak = np.maximum.accumulate(values)
                drawdown = (values - peak) / peak
                dates = pd.date_range(
                    start=self.returns.index[self.min_history],
                    periods=len(values),
                    freq='D'
                )
                ax3.fill_between(dates, drawdown, 0, alpha=0.3,
                                 label=method_name.replace('_', ' ').title())

        ax3.set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Drawdown (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Transaction costs
        methods = []
        costs = []
        for method_name, results in self.portfolio_results.items():
            if results['transaction_costs']:
                methods.append(method_name.replace('_', ' ').title())
                costs.append(sum(results['transaction_costs']))

        ax4.bar(range(len(methods)), costs, color='coral', alpha=0.7)
        ax4.set_title('Total Transaction Costs', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Transaction Costs ($)')
        ax4.set_xticks(range(len(methods)))
        ax4.set_xticklabels(methods, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def save_results(self, output_dir: str = "results"):
        """Save comprehensive results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save performance metrics
        perf_df = self.calculate_performance_metrics()
        perf_file = output_path / f"rolling_performance_metrics_{timestamp}.csv"
        perf_df.to_csv(perf_file, index=False)

        # Save detailed results
        results_file = output_path / f"rolling_optimization_results_{timestamp}.json"

        # Convert results to JSON-serializable format
        json_results = {}
        for method_name, results in self.portfolio_results.items():
            json_results[method_name] = {
                'total_periods': len(results['weights_history']),
                'total_transaction_costs': float(np.sum(results['transaction_costs'])) if results[
                    'transaction_costs'] else 0.0,
                'final_portfolio_value': float(results['portfolio_values'][-1]) if results[
                    'portfolio_values'] else 100.0,
                'rebalancing_dates': [date.isoformat() for date in results['rebalance_dates']]
            }

        # Add configuration parameters
        json_results['configuration'] = {
            'rebalance_frequency': self.rebalance_freq,
            'minimum_history': self.min_history,
            'transaction_cost': self.transaction_cost,
            'risk_free_rate': self.risk_free_rate,
            'data_period': {
                'start': self.returns.index[0].isoformat(),
                'end': self.returns.index[-1].isoformat()
            }
        }

        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"\n💾 Results saved:")
        print(f"   Performance metrics: {perf_file}")
        print(f"   Detailed results: {results_file}")

        return perf_df


def main():
    """Main execution function"""
    print("🏦 Rolling Traditional Portfolio Optimization")
    print("=" * 60)
    print("Features: Expanding windows | Monthly rebalancing | Transaction costs")
    print("=" * 60)

    try:
        # Initialize optimizer
        optimizer = RollingPortfolioOptimizer(
            data_path="../../../data/raw",
            rebalance_freq=21,  # Monthly
            min_history=252,  # 1 year minimum
            transaction_cost=0.001,  # 0.1%
            risk_free_rate=0.02
        )

        # Run rolling optimization
        results = optimizer.run_rolling_optimization()

        if not results:
            print("❌ No optimization results generated")
            return None

        # Calculate and display performance metrics
        print("\n📊 PERFORMANCE SUMMARY")
        print("=" * 60)
        perf_df = optimizer.calculate_performance_metrics()
        print(perf_df.to_string(index=False))

        # Generate plots
        print("\n📈 Generating performance plots...")
        optimizer.plot_portfolio_performance()

        # Save results
        optimizer.save_results()

        print("\n✅ Rolling optimization completed successfully!")
        print("🎯 Key features implemented:")
        print("   • Expanding window analysis (no look-ahead bias)")
        print("   • Monthly portfolio rebalancing")
        print("   • Transaction cost modeling")
        print("   • Comprehensive performance tracking")
        print("   • Multiple optimization methods")

        return optimizer

    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    optimizer = main()