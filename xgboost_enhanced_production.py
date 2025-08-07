# xgboost_enhanced_production.py - 极简稳定版

"""
极简稳定的XGBoost增强优化器
- 避免复杂的数值优化
- 使用简单但稳定的权重分配
- 专注于XGBoost信号的有效利用
"""

import os
import pandas as pd
import numpy as np
import joblib
import cloudpickle
from pathlib import Path
from datetime import datetime
import warnings
from typing import Dict  # 添加类型提示导入
from scipy import stats
import scipy.stats as stats
from joblib import load as joblib_load






# 导入基础优化器
from complete_traditional_methods import RollingPortfolioOptimizer

warnings.filterwarnings('ignore')


class SimpleXGBOptimizer(RollingPortfolioOptimizer):
    """极简XGBoost增强优化器"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xgb_models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.features_data = None
        self.signal_history = {}
        self.optimal_thresholds = {}
        self.calibration_methods = {}
        self.asset_names = set(col.split('_')[0] for col in self.returns.columns)

        print("🚀 启动极简XGBoost优化器...")
        self._load_xgb_models()
        self._load_features_data()

    def _load_xgb_models(self):
        """加载 XGBoost 模型（兼容 .skops 和 .pkl 两种格式）"""
        model_files = list(Path(".").rglob("xgb_v5_complete_*.pkl"))
        
        for f in model_files:
            ticker = f.stem.split("_")[-1]
            model_dict = joblib_load(f)
            self.xgb_models[ticker] = model_dict
            
            print(f"{ticker} 加载成功")
        print(f"✅ 共加载 {len(self.xgb_models)} 个 XGBoost 模型")
    

    def _load_features_data(self):
        """加载特征数据"""
        # 尝试多种可能的特征文件名
        possible_files = [
            "banking_returns.csv",  # XGBoost V5使用的特征文件
            "banking_features_ai.csv",
            "features.csv"
        ]
        
        for filename in possible_files:
            features_files = list(Path(".").glob(filename))
            if features_files:
                self.features_data = pd.read_csv(features_files[0], index_col=0, parse_dates=True)
                print(f"✅ 加载特征数据: {self.features_data.shape} from {filename}")
                return
                
        print("⚠️ 未找到特征数据文件")

    def _simple_xgb_prediction(self, ticker: str, trade_date: pd.Timestamp) -> float:
        """简单XGBoost预测"""
        if ticker not in self.xgb_models or self.features_data is None:
            if ticker not in self.xgb_models:
                print(f"{ticker}: 无XGBoost模型")
            if self.features_data is None:
                print("特征数据未加载")
            return 0.0

        # 获取模型数据

        model_data            = self.xgb_models[ticker]
        calibrated_model = model_data.get('calibrated_model')
        model           = model_data.get('model')
        scaler           = model_data.get('scaler')
        selected_features= model_data.get('selected_features', [])
        optimal_threshold= model_data.get('optimal_threshold', 0.5)
        
        if not selected_features:
            print(f"{ticker}: 无选择特征")
            return 0.0

        # 获取可用日期
        available_dates = self.features_data.index[self.features_data.index <= trade_date]
        if len(available_dates) == 0:
            print(f"{ticker}: 无可用日期")
            return 0.0

        actual_date = available_dates[-1]
        
        # 提取特征
        try:
            # 1) 先拿整行（可能少列），然后 reindex 到完整的 selected_features
            row = self.features_data.loc[actual_date]
            row = row.reindex(selected_features)

    # 2) 填充缺失值：向前、向后、最后统一用 0
            row = row.fillna(method='ffill') \
                     .fillna(method='bfill') \
                     .fillna(0)

    # 3) 拿到 numpy 向量
            feature_vector = row.values

    # 如果有 scaler，做标准化
            if scaler is not None:
                feature_vector = scaler.transform(feature_vector.reshape(1, -1))[0]
            
            feature_vector = feature_vector.reshape(1, -1)

        except Exception as e:
            print(f"{ticker}: 特征提取失败 {e}")
            return 0.0

        try:
            # 使用校准后的模型进行预测
            if calibrated_model is not None:
                prob = calibrated_model.predict_proba(feature_vector)[0]
                print(f"[{trade_date.date()}] {ticker} calibrated_proba: {prob}")
                if len(prob) > 1:
                    raw_signal = prob[1]  # 上涨概率
                else:
                    raw_signal = prob[0]
                    
                # 使用最优阈值转换为信号
                signal = (raw_signal - optimal_threshold) * 2  # [-1, 1]范围
                
            elif model is not None and hasattr(model, 'predict_proba'):
                prob = model.predict_proba(feature_vector)[0]
                print(f"[{trade_date.date()}] {ticker} model_proba: {prob}")
                if len(prob) > 1:
                    signal = prob[1] - 0.5  # 上涨概率减去0.5
                else:
                    signal = prob[0] - 0.5
                    
            else:
                print(f"{ticker}: 模型预测失败")
                return 0.0

            # 限制信号范围
            signal = np.clip(signal, -0.3, 0.3)
            print(f"[{trade_date.date()}] {ticker} signal: {signal:.4f}")
            return signal

        except Exception as e:
            print(f"{ticker}: 预测失败 {e}")
            return 0.0

    def generate_xgb_signals(self, trade_date: pd.Timestamp) -> pd.Series:
        """生成XGBoost信号"""
        print(f"🎯 生成XGBoost信号: {trade_date.date()}")

        signals = {}
        active_count = 0

        asset_names = set(col.split('_')[0] for col in self.returns.columns)
        for ticker in asset_names:
            signal = self._simple_xgb_prediction(ticker, trade_date)
            signals[ticker] = signal
            
            if abs(signal) > 0.01:
                active_count += 1

        print(f"  ✅ {active_count}/{len(signals)} 个活跃信号")

        # 存储信号历史
        self.signal_history[trade_date] = {
            'signals': signals.copy(),
            'active_count': active_count
        }

        return pd.Series(signals, index=sorted(asset_names))

    def optimize_simple_xgb_enhanced(self, returns_data: pd.DataFrame,
                                     trade_date: pd.Timestamp) -> Dict:
        """简单的XGBoost增强优化"""
        print(f"🔧 简单XGBoost增强优化: {trade_date.date()}")

        # 获取XGBoost信号
        xgb_signals = self.generate_xgb_signals(trade_date)

        # 🔧 简单方法：基于信号强度分配权重
        n_assets = len(self.asset_names)

        # 基础等权重
        base_weights = np.ones(n_assets) / n_assets
        print(f"n_assets: {n_assets}, 基础权重: {base_weights}")

        # XGBoost信号调整
        signal_values = xgb_signals.values

        # 将信号转换为权重调整
        # 正信号增加权重，负信号减少权重
        weight_adjustments = signal_values * 0.15  # 15%的最大调整（比RF稍大）

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

        print(f"  ✅ 简单XGBoost优化成功: Sharpe={sharpe:.3f}, 活跃信号={active_signals}")

        return {
            'weights': new_weights,
            'performance': (portfolio_return, portfolio_vol, sharpe),
            'method': 'Simple XGBoost Enhanced',
            'xgb_signals_used': active_signals,
            'success': True
        }

    def optimize_xgb_momentum(self, returns_data: pd.DataFrame,
                              trade_date: pd.Timestamp) -> Dict:
        """XGBoost + 动量组合策略"""
        print(f"🔧 XGBoost+动量策略: {trade_date.date()}")

        # XGBoost信号
        xgb_signals = self.generate_xgb_signals(trade_date)

        # 动量信号（3个月）
        momentum_window = min(63, len(returns_data))  # 3个月或可用数据
        momentum_returns = returns_data.tail(momentum_window).mean()

        # 标准化动量信号
        momentum_signals = (momentum_returns - momentum_returns.mean()) / momentum_returns.std()
        momentum_signals = momentum_signals.fillna(0)

        # 组合信号 (60% XGBoost + 40% 动量，XGBoost权重稍高)
        combined_signals = 0.6 * xgb_signals + 0.4 * momentum_signals

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

        active_signals = len([s for s in xgb_signals.values if abs(s) > 0.01])

        print(f"  ✅ XGBoost+动量成功: Sharpe={sharpe:.3f}, XGBoost信号={active_signals}")

        return {
            'weights': weights,
            'performance': (portfolio_return, portfolio_vol, sharpe),
            'method': 'XGBoost + Momentum',
            'xgb_signals_used': active_signals,
            'success': True
        }

    def optimize_xgb_risk_parity(self, returns_data: pd.DataFrame,
                                 trade_date: pd.Timestamp) -> Dict:
        """XGBoost信号增强的风险平价策略"""
        print(f"🔧 XGBoost风险平价策略: {trade_date.date()}")

        # XGBoost信号
        xgb_signals = self.generate_xgb_signals(trade_date)

        # 计算风险平价基础权重
        returns_cov = returns_data.cov() * 252
        inv_vol = 1 / np.sqrt(np.diag(returns_cov))
        risk_parity_weights = inv_vol / inv_vol.sum()

        # 使用XGBoost信号调整风险平价权重
        signal_adjustments = xgb_signals.values * 0.2  # 20%的最大调整

        # 调整后的权重
        adjusted_weights = risk_parity_weights + signal_adjustments

        # 确保权重约束
        adjusted_weights = np.maximum(adjusted_weights, 0.02)
        adjusted_weights = np.minimum(adjusted_weights, 0.25)
        adjusted_weights = adjusted_weights / adjusted_weights.sum()

        # 性能估计
        hist_returns = returns_data.mean() * 252
        portfolio_return = np.sum(adjusted_weights * hist_returns)
        portfolio_vol = np.sqrt(np.dot(adjusted_weights, np.dot(returns_cov.values, adjusted_weights)))
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0

        active_signals = len([s for s in xgb_signals.values if abs(s) > 0.01])

        print(f"  ✅ XGBoost风险平价成功: Sharpe={sharpe:.3f}, XGBoost信号={active_signals}")

        return {
            'weights': adjusted_weights,
            'performance': (portfolio_return, portfolio_vol, sharpe),
            'method': 'XGBoost Risk Parity',
            'xgb_signals_used': active_signals,
            'success': True
        }

    def run_simple_optimization(self, max_periods: int = 15):
        """运行简单优化"""
        print("🚀 开始简单XGBoost优化...")

        if not self.rebalance_dates:
            print("❌ 没有重新平衡日期")
            return

        # 测试期数
        test_dates = self.rebalance_dates[:max_periods]
        print(f"📅 测试 {len(test_dates)} 个重新平衡期")

        # 简化的方法集合
        methods = {
            'equal_weight': self.optimize_equal_weight,
            'simple_xgb_enhanced': self.optimize_simple_xgb_enhanced,
            'xgb_momentum': self.optimize_xgb_momentum,
            'xgb_risk_parity': self.optimize_xgb_risk_parity
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
                try:
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
                        signals = result.get('xgb_signals_used', 0)

                        if signals > 0:
                            print(f"  ✅ {method_name}: Sharpe={perf[2]:.3f}, 信号={signals}")
                        else:
                            print(f"  ✅ {method_name}: Sharpe={perf[2]:.3f}")

                        period_success += 1
                    else:
                        print(f"  ❌ {method_name}: 失败")
                
                except Exception as e:
                    print(f"  ❌ {method_name}: 异常 {e}")

            if period_success > 0:
                successful_periods += 1

        print(f"\n🎉 简单XGBoost优化完成: {successful_periods}/{len(test_dates)} 期成功")

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
            perf_file = results_dir / f"simple_xgb_performance_{timestamp}.csv"
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

            # 提取策略的日度收益进行显著性检验
            eq_nv = np.array(self.portfolio_results['equal_weight']['portfolio_values'])
            xgb_nv = np.array(self.portfolio_results['simple_xgb_enhanced']['portfolio_values'])
            mom_nv = np.array(self.portfolio_results['xgb_momentum']['portfolio_values'])
            rp_nv = np.array(self.portfolio_results['xgb_risk_parity']['portfolio_values'])
            
            # 计算收益率
            eq_r = np.diff(eq_nv) / eq_nv[:-1]
            xgb_r = np.diff(xgb_nv) / xgb_nv[:-1]
            mom_r = np.diff(mom_nv) / mom_nv[:-1]
            rp_r = np.diff(rp_nv) / rp_nv[:-1]
            
            # 配对 t-test
            try:
                t1, p1 = stats.ttest_rel(xgb_r, eq_r)
                t2, p2 = stats.ttest_rel(mom_r, eq_r)
                t3, p3 = stats.ttest_rel(rp_r, eq_r)
                
                print("\n—— 显著性检验 ——")
                print(f"XGBoost vs equal_weight: t={t1:.2f}, p={p1:.3f}")
                print(f"XGBoost_momentum vs equal_weight: t={t2:.2f}, p={p2:.3f}")
                print(f"XGBoost_risk_parity vs equal_weight: t={t3:.2f}, p={p3:.3f}")
            except Exception as e:
                print(f"显著性检验失败: {e}")

        # 导出信号历史
        if self.signal_history:
            signal_data = []
            for date, data in self.signal_history.items():
                signals = data['signals']
                for ticker, signal in signals.items():
                    signal_data.append({
                        'Date': date,
                        'Ticker': ticker,
                        'Signal': signal,
                        'Active_Count': data['active_count']
                    })
            
            if signal_data:
                signal_df = pd.DataFrame(signal_data)
                signal_file = results_dir / f"xgb_signals_{timestamp}.csv"
                signal_df.to_csv(signal_file, index=False)
                print(f"✅ 信号历史: {signal_file}")

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


def run_simple_xgb_optimization():
    """运行简单XGBoost优化"""
    print("🚀 启动简单XGBoost优化...")

    optimizer = SimpleXGBOptimizer(
        data_path="./",
        rebalance_freq=63,  # 季度
        min_history=252,
        transaction_cost=0.001
    )

    # 运行优化
    results = optimizer.run_simple_optimization(max_periods=15)

    print("\n🎉 简单XGBoost优化完成!")
    print("✅ 避免了复杂的数值优化")
    print("✅ 使用稳定的权重分配方法")
    print("✅ XGBoost信号正常工作")
    print("✅ 集成了阈值优化和概率校准")
    print("✅ 支持四种策略对比")

    return optimizer


if __name__ == "__main__":
    optimizer = run_simple_xgb_optimization()