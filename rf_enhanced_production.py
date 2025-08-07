# simple_rf_optimizer.py - 极简稳定版

"""
极简稳定的RF增强优化器
- 避免复杂的数值优化
- 使用简单但稳定的权重分配
- 专注于RF信号的有效利用
"""
import os
import pandas as pd
import numpy as np
import joblib
from skops.io import load
from pathlib import Path
from datetime import datetime
import warnings
from typing import Dict  # 添加类型提示导入
from scipy import stats
import scipy.stats as stats

# 导入基础优化器
from complete_traditional_methods import RollingPortfolioOptimizer

warnings.filterwarnings('ignore')


class SimpleRFOptimizer(RollingPortfolioOptimizer):
    """极简RF增强优化器"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rf_models = {}
        self.features_data = None
        self.signal_history = {}

        print("🚀 启动极简RF优化器...")
        self._load_rf_models()
        self._load_features_data()

    def _load_rf_models(self):
        """加载RF模型"""

        print(list(Path(".").rglob("rf_v4_complete_*.skops")))

        model_files = list(Path(".").rglob("rf_v4_complete_*.skops"))

        asset_names = set(col.split('_')[0] for col in self.returns.columns)
        
        self.asset_names = asset_names

        for model_file in model_files:
            ticker = model_file.stem.split('_')[-1]
            if ticker in asset_names:
                self.rf_models[ticker] = load(model_file)
                print(f"{ticker} 加载成功！")
            else:
                print(f"资产名 {ticker} 不在 returns.columns")
                
        print(f"当前工作目录: {Path.cwd()}")
        print(f"✅ 加载了 {len(self.rf_models)} 个RF模型")


    def _load_features_data(self):
        """加载特征数据"""
        features_files = list(Path(".").glob("banking_features_ai.csv"))
        if features_files:
            self.features_data = pd.read_csv(features_files[0], index_col=0, parse_dates=True)
            print(f"✅ 加载特征数据: {self.features_data.shape}")


    def _simple_rf_prediction(self, ticker: str, trade_date: pd.Timestamp) -> float:
        if ticker not in self.rf_models or self.features_data is None:
            if ticker not in self.rf_models:
                print(f"{ticker}: 无RF模型")
            if self.features_data is None:
                print("特征数据未加载")
            print(f"{ticker}: 无模型或特征数据")
            return 0.0

        ticker_features = [col for col in self.features_data.columns if col.startswith(f'{ticker}_')]
        if not ticker_features:
            print(f"{ticker}: 无匹配特征")
            return 0.0

        available_dates = self.features_data.index[self.features_data.index <= trade_date]
        if len(available_dates) == 0:
            print(f"{ticker}: 无可用日期")
            return 0.0

        actual_date = available_dates[-1]
        feature_vector = self.features_data.loc[actual_date, ticker_features].values
        feature_vector = np.nan_to_num(feature_vector, nan=0)

        # 强制统一到50维
        if len(feature_vector) < 15:
            feature_vector = np.pad(feature_vector, (0, 15 - len(feature_vector)))
        elif len(feature_vector) > 15:
            feature_vector = feature_vector[:15]
        feature_vector = feature_vector.reshape(1, -1)

        model_data = self.rf_models[ticker]
        model = model_data['model']

        # --- debug
        # print(f"[{trade_date.date()}] {ticker} features: {feature_vector[0][:5]} ... (共{len(feature_vector[0])}维, 非零={np.count_nonzero(feature_vector)})")

        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(feature_vector)[0]
            print(f"[{trade_date.date()}] {ticker} predict_proba: {prob}")
            if len(prob) > 1:
                signal = prob[1] - 0.5
            else:
                signal = prob[0] - 0.5
        else:
            pred = model.predict(feature_vector)[0]
            print(f"[{trade_date.date()}] {ticker} predict: {pred}")
            signal = (pred - 0.5) if pred <= 1 else 0

        signal = np.clip(signal, -0.2, 0.2)
        print(f"[{trade_date.date()}] {ticker} signal: {signal:.4f}")
        return signal

    def generate_rf_signals(self, trade_date: pd.Timestamp) -> pd.Series:
        print(f"🎯 生成RF信号: {trade_date.date()}")

        signals = {}
        active_count = 0

        asset_names = set(col.split('_')[0] for col in self.returns.columns)
        for ticker in asset_names:
            signal = self._simple_rf_prediction(ticker, trade_date)
            signals[ticker] = signal

        if abs(signal) > 0.01:
            active_count += 1

        print(f"  ✅ {active_count}/{len(signals)} 个活跃信号")

        # 存储
        self.signal_history[trade_date] = {
            'signals': signals.copy(),
            'active_count': active_count
        }

        return pd.Series(signals, index=sorted(asset_names))


    def optimize_simple_rf_enhanced(self, returns_data: pd.DataFrame,
                                    trade_date: pd.Timestamp) -> Dict:
        """简单的RF增强优化"""
        print(f"🔧 简单RF增强优化: {trade_date.date()}")

        # 获取RF信号
        rf_signals = self.generate_rf_signals(trade_date)

        # 🔧 简单方法：基于信号强度分配权重
        n_assets = len(self.asset_names)

        # 基础等权重
        base_weights = np.ones(n_assets) / n_assets
        print(f"n_assets: {n_assets}, 基础权重: {base_weights}")

        # RF信号调整
        signal_values = rf_signals.values

        # 将信号转换为权重调整
        # 正信号增加权重，负信号减少权重
        weight_adjustments = signal_values * 0.1  # 10%的最大调整

        # 新权重 = 基础权重 + 调整
        new_weights = base_weights + weight_adjustments

        # 确保权重为正
        new_weights = np.maximum(new_weights, 0.02)  # 最小2%

        # 确保权重不超过25%
        new_weights = np.minimum(new_weights, 0.25)

        # 标准化使得和为1
        new_weights = new_weights / new_weights.sum()

        # 简单的性能估计
        # 使用历史收益率估计
        hist_returns = returns_data.mean() * 252
        portfolio_return = np.sum(new_weights * hist_returns)

        # 简单的风险估计
        returns_cov = returns_data.cov() * 252
        portfolio_vol = np.sqrt(np.dot(new_weights, np.dot(returns_cov.values, new_weights)))

        # Sharpe比率
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0

        active_signals = len([s for s in signal_values if abs(s) > 0.01])

        print(f"  ✅ 简单优化成功: Sharpe={sharpe:.3f}, 活跃信号={active_signals}")

        return {
            'weights': new_weights,
            'performance': (portfolio_return, portfolio_vol, sharpe),
            'method': 'Simple RF Enhanced',
            'rf_signals_used': active_signals,
            'success': True
        }

    def optimize_rf_momentum(self, returns_data: pd.DataFrame,
                             trade_date: pd.Timestamp) -> Dict:
        """RF + 动量组合策略"""
        print(f"🔧 RF+动量策略: {trade_date.date()}")

        # RF信号
        rf_signals = self.generate_rf_signals(trade_date)

        # 动量信号（3个月）
        momentum_window = min(21, len(returns_data))  # 3个月或可用数据
        momentum_returns = returns_data.tail(momentum_window).mean()

        # 标准化动量信号
        momentum_signals = (momentum_returns - momentum_returns.mean()) / momentum_returns.std()
        momentum_signals = momentum_signals.fillna(0)

        # 组合信号 (50% RF + 50% 动量)
        combined_signals = 0.5 * rf_signals + 0.5 * momentum_signals

        # 基于组合信号分配权重
        n_assets = len(returns_data.columns)

        # 将信号转换为权重得分
        signal_scores = combined_signals.values

        # 将负信号设为0（只做多）
        positive_scores = np.maximum(signal_scores, 0)

        if positive_scores.sum() > 0:
            # 基于正信号分配权重
            weights = positive_scores / positive_scores.sum()

            # 限制单个权重
            weights = np.minimum(weights, 0.25)  # 最大25%
            weights = np.maximum(weights, 0.02)  # 最小2%

            # 重新标准化
            weights = weights / weights.sum()
        else:
            # 如果没有正信号，等权重
            weights = np.ones(n_assets) / n_assets

        # 性能估计
        hist_returns = returns_data.mean() * 252
        portfolio_return = np.sum(weights * hist_returns)

        returns_cov = returns_data.cov() * 252
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(returns_cov.values, weights)))

        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0

        active_signals = len([s for s in rf_signals.values if abs(s) > 0.01])

        print(f"  ✅ RF+动量成功: Sharpe={sharpe:.3f}, RF信号={active_signals}")

        return {
            'weights': weights,
            'performance': (portfolio_return, portfolio_vol, sharpe),
            'method': 'RF + Momentum',
            'rf_signals_used': active_signals,
            'success': True
        }


    def run_simple_optimization(self, max_periods: int = 10):
        """运行简单优化"""
        print("🚀 开始简单RF优化...")

        if not self.rebalance_dates:
            print("❌ 没有重新平衡日期")
            return

        # 测试期数
        test_dates = self.rebalance_dates[:max_periods]
        print(f"📅 测试 {len(test_dates)} 个重新平衡期")

        # 简化的方法集合
        methods = {
            'equal_weight': self.optimize_equal_weight,
            'simple_rf_enhanced': self.optimize_simple_rf_enhanced,
            'rf_momentum': self.optimize_rf_momentum
        }

        # 初始化结果存储
        for method_name in methods.keys():
            self.portfolio_results[method_name] = {
                'weights_history': [],
                'returns_history': [],
                'portfolio_values': [100.0],
                'transaction_costs': [],
                'rebalance_dates': [],
                'optimization_results': []
            }

        successful_periods = 0

        # 执行优化
        for i, rebalance_date in enumerate(test_dates):
            print(f"\n📊 期间 {i + 1}/{len(test_dates)}: {rebalance_date.date()}")

            # 获取历史数据
            historical_data = self.returns.loc[:rebalance_date].iloc[:-1]

            if len(historical_data) < self.min_history:
                print(f"  ⚠️ 历史数据不足")
                continue

            period_success = 0

            # 测试每种方法
            for method_name, optimize_func in methods.items():
                if method_name == 'equal_weight':
                    result = optimize_func(historical_data)
                else:
                    result = optimize_func(historical_data, rebalance_date)

                if result.get('success', False):
                    weights = result['weights']

                    # 存储结果
                    self.portfolio_results[method_name]['weights_history'].append(weights)
                    self.portfolio_results[method_name]['rebalance_dates'].append(rebalance_date)
                    self.portfolio_results[method_name]['optimization_results'].append(result)

                    # 显示结果
                    perf = result['performance']
                    signals = result.get('rf_signals_used', 0)

                    if signals > 0:
                        print(f"  ✅ {method_name}: Sharpe={perf[2]:.3f}, 信号={signals}")
                    else:
                        print(f"  ✅ {method_name}: Sharpe={perf[2]:.3f}")

                    period_success += 1
                else:
                    print(f"  ❌ {method_name}: 失败")


            if period_success > 0:
                successful_periods += 1

        print(f"\n🎉 简单优化完成: {successful_periods}/{len(test_dates)} 期成功")

        # 计算和导出性能
        self._calculate_simple_performance()
        self._export_simple_results()

        return self.portfolio_results

    def _calculate_simple_performance(self):
        """简单性能计算"""
        print("📊 计算性能...")

        for method_name, results in self.portfolio_results.items():
            if not results.get('weights_history'):
                continue

            portfolio_values = [100.0]
            weights_history = results['weights_history']
            rebalance_dates = results['rebalance_dates']

            for i in range(len(rebalance_dates)):
                current_weights = weights_history[i]

                # 计算到下一个重新平衡日期的收益
                if i < len(rebalance_dates) - 1:
                    start_date = rebalance_dates[i]
                    end_date = rebalance_dates[i + 1]
                else:
                    start_date = rebalance_dates[i]
                    end_date = self.returns.index[-1]

                period_returns = self.returns.loc[start_date:end_date]

                if len(period_returns) > 1:
                    # 计算每日投资组合收益
                    daily_portfolio_returns = period_returns.iloc[1:] @ current_weights

                    # 更新投资组合价值
                    for daily_return in daily_portfolio_returns:
                        if not np.isnan(daily_return):
                            portfolio_values.append(portfolio_values[-1] * (1 + daily_return))

            results['portfolio_values'] = portfolio_values

            if len(portfolio_values) > 1:
                total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
                print(f"  ✅ {method_name}: 总收益 = {total_return:.1f}%")

    def _export_simple_results(self):
        """导出简单结果"""
        print("📁 导出结果...")

        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 性能对比
        performance_data = []

        for method_name, results in self.portfolio_results.items():
            if results.get('portfolio_values') and len(results['portfolio_values']) > 1:
                values = results['portfolio_values']
                total_return = (values[-1] / values[0] - 1)

                # 计算其他指标
                returns = []
                for i in range(1, len(values)):
                    ret = values[i] / values[i - 1] - 1
                    returns.append(ret)

                if returns:
                    annual_return = np.mean(returns) * 252
                    volatility = np.std(returns) * np.sqrt(252)
                    sharpe = (annual_return - self.risk_free_rate) / volatility if volatility > 0 else 0
                    max_dd = self._calculate_max_drawdown(values)
                    var95 = -np.percentile(returns, 5)
                    worst = [r for r in returns if r <= np.percentile(returns, 5)]
                    cvar95 = -np.mean(worst) if worst else 0
                    calmar = annual_return / max_dd if max_dd > 0 else 0
                    
                else:
                    annual_return = volatility = sharpe = max_dd = var95 = cvar95 = calmar = 0

                performance_data.append({
                    'Method': method_name,
                    'Total_Return': total_return,
                    'Annual_Return': annual_return,
                    'Volatility': volatility,
                    'Sharpe_Ratio': sharpe,
                    'Max_Drawdown': max_dd,
                    'VaR95%': var95,
                    'CVaR95%': cvar95,
                    'Calmar_Ratio': calmar
                })

        if performance_data:
            perf_df = pd.DataFrame(performance_data)
            perf_file = results_dir / f"simple_rf_performance_{timestamp}.csv"
            perf_df.to_csv(perf_file, index=False)
            print(f"✅ 性能结果: {perf_file}")

            # 显示结果
            print("\n📊 最终性能对比:")
            for _, row in perf_df.iterrows():
                print((
                    f"  {row['Method']}: "
                    f"总收益={row['Total_Return']:.1%}, "
                    f"Annual={row['Annual_Return']:.2%}, "
                    f"Vol={row['Volatility']:.2%}, "
                    f"Sharpe={row['Sharpe_Ratio']:.3f}, "
                    f"VaR95%={row['VaR95%']:.2%}, "
                    f"CVaR95%={row['CVaR95%']:.2%}, "
                    f"Calmar={row['Calmar_Ratio']:.3f}"
                ))

            # 提取三种策略的日度收益（用净值差分得到）
            eq_nv   = np.array(self.portfolio_results['equal_weight']['portfolio_values'])
            rf_nv   = np.array(self.portfolio_results['simple_rf_enhanced']['portfolio_values'])
            mom_nv  = np.array(self.portfolio_results['rf_momentum']['portfolio_values'])
            eq_r    = np.diff(eq_nv)
            rf_r    = np.diff(rf_nv)
            mom_r   = np.diff(mom_nv)
            # 配对 t-test
            t1, p1   = stats.ttest_rel(rf_r,  eq_r)
            t2, p2   = stats.ttest_rel(mom_r, eq_r)
            print("\n—— 显著性检验 ——")
            print(f"RF vs equal_weight: t={t1:.2f}, p={p1:.3f}")
            print(f"RF_momentum vs equal_weight: t={t2:.2f}, p={p2:.3f}")

            
    def _calculate_max_drawdown(self, values):
        """计算最大回撤"""
        if len(values) < 2:
            return 0

        peak = values[0]
        max_dd = 0

        for value in values[1:]:
            if value > peak:
                peak = value
            else:
                dd = (peak - value) / peak
                max_dd = max(max_dd, dd)

        return max_dd


def run_simple_rf_optimization():
    """运行简单RF优化"""
    print("🚀 启动简单RF优化...")

    optimizer = SimpleRFOptimizer(
        data_path="./",
        rebalance_freq=63,  # 季度
        min_history=252,
        transaction_cost=0.001
    )

    # 运行优化
    results = optimizer.run_simple_optimization(max_periods=15)

    print("\n🎉 简单RF优化完成!")
    print("✅ 避免了复杂的数值优化")
    print("✅ 使用稳定的权重分配方法")
    print("✅ RF信号正常工作")

    return optimizer


if __name__ == "__main__":
    optimizer = run_simple_rf_optimization()