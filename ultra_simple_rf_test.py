# ultra_simple_rf_test.py - 极简测试版

"""
极简测试版 - 专注于验证RF信号是否有效
先确保基本功能正常，再考虑复杂的优化
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime


def load_data():
    """加载基础数据"""
    print("📊 加载数据...")

    # 加载收益率数据
    returns = pd.read_csv("banking_returns.csv", index_col=0, parse_dates=True)
    print(f"收益率数据: {returns.shape}")

    # 加载特征数据
    features_files = list(Path(".").glob("*features*.csv"))
    if features_files:
        features = pd.read_csv(features_files[0], index_col=0, parse_dates=True)
        print(f"特征数据: {features.shape}")
    else:
        features = None
        print("⚠️ 没有找到特征数据")

    # 加载RF模型
    rf_models = {}
    model_files = list(Path(".").rglob("rf_v4_complete_*.pkl"))

    for model_file in model_files:
        ticker = model_file.stem.split('_')[-1]
        if ticker in returns.columns:
            try:
                rf_models[ticker] = joblib.load(model_file)
            except:
                pass

    print(f"RF模型: {len(rf_models)} 个")

    return returns, features, rf_models


def simple_rf_predict(ticker, features, rf_models, date):
    """简单RF预测"""
    try:
        if ticker not in rf_models or features is None:
            return 0.0

        # 获取特征
        ticker_features = [col for col in features.columns if col.startswith(f'{ticker}_')]
        if not ticker_features:
            return 0.0

        # 找最近日期
        available_dates = features.index[features.index <= date]
        if len(available_dates) == 0:
            return 0.0

        actual_date = available_dates[-1]
        feature_vector = features.loc[actual_date, ticker_features].values

        # 处理NaN和维度
        feature_vector = np.nan_to_num(feature_vector, nan=0)

        # 统一到50维
        if len(feature_vector) < 50:
            feature_vector = np.pad(feature_vector, (0, 50 - len(feature_vector)))
        elif len(feature_vector) > 50:
            feature_vector = feature_vector[:50]

        feature_vector = feature_vector.reshape(1, -1)

        # 预测
        model = rf_models[ticker]['model']

        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(feature_vector)[0]
            if len(prob) > 1:
                signal = prob[1] - 0.5
            else:
                signal = prob[0] - 0.5
        else:
            pred = model.predict(feature_vector)[0]
            signal = (pred - 0.5) if pred <= 1 else 0

        # 限制信号范围
        signal = np.clip(signal, -0.1, 0.1)
        return signal

    except Exception as e:
        return 0.0


def test_rf_signals(returns, features, rf_models):
    """测试RF信号质量"""
    print("\n🧪 测试RF信号质量...")

    # 选择一个测试日期
    test_date = pd.Timestamp('2020-01-01')

    signals = {}
    for ticker in returns.columns:
        signal = simple_rf_predict(ticker, features, rf_models, test_date)
        signals[ticker] = signal

    print(f"测试日期: {test_date.date()}")
    print(f"信号统计:")
    signal_values = list(signals.values())
    print(f"  非零信号: {len([s for s in signal_values if abs(s) > 0.001])}/{len(signal_values)}")
    print(f"  信号范围: [{min(signal_values):.4f}, {max(signal_values):.4f}]")
    print(f"  信号标准差: {np.std(signal_values):.4f}")

    return signals


def simple_backtest(returns, features, rf_models):
    """极简回测"""
    print("\n⏰ 开始极简回测...")

    # 选择回测期间（最近1年）
    end_date = returns.index[-1]
    start_date = end_date - pd.DateOffset(years=1)

    backtest_returns = returns.loc[start_date:end_date]
    print(f"回测期间: {start_date.date()} 到 {end_date.date()}")
    print(f"回测天数: {len(backtest_returns)}")

    # 策略对比
    strategies = {
        'equal_weight': None,  # 等权重基准
        'rf_enhanced': None  # RF增强
    }

    results = {}

    for strategy_name in strategies:
        print(f"\n测试策略: {strategy_name}")

        portfolio_values = [100.0]  # 初始价值

        # 按月重新平衡（简化）
        rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='M')
        rebalance_dates = [d for d in rebalance_dates if d in backtest_returns.index]

        print(f"重新平衡次数: {len(rebalance_dates)}")

        current_weights = None

        for i, reb_date in enumerate(rebalance_dates):
            print(f"  重新平衡 {i + 1}/{len(rebalance_dates)}: {reb_date.date()}")

            # 确定权重
            if strategy_name == 'equal_weight':
                # 等权重
                weights = np.ones(len(returns.columns)) / len(returns.columns)
            else:
                # RF增强
                rf_signals = {}
                for ticker in returns.columns:
                    signal = simple_rf_predict(ticker, features, rf_models, reb_date)
                    rf_signals[ticker] = signal

                # 简单的权重调整
                base_weights = np.ones(len(returns.columns)) / len(returns.columns)
                signal_values = np.array([rf_signals[ticker] for ticker in returns.columns])

                # 权重 = 基础权重 + 信号调整
                weights = base_weights + signal_values * 0.05  # 5%最大调整
                weights = np.maximum(weights, 0.01)  # 最小1%
                weights = weights / weights.sum()  # 标准化

            current_weights = weights

            # 计算到下一个重新平衡日期的表现
            if i < len(rebalance_dates) - 1:
                next_reb_date = rebalance_dates[i + 1]
            else:
                next_reb_date = end_date

            # 获取期间收益
            period_data = backtest_returns.loc[reb_date:next_reb_date]

            if len(period_data) > 1:
                # 跳过重新平衡日，从下一天开始
                period_returns = period_data.iloc[1:]

                if len(period_returns) > 0:
                    # 计算每日组合收益
                    daily_portfolio_returns = period_returns @ current_weights

                    # 更新组合价值
                    current_value = portfolio_values[-1]

                    for daily_ret in daily_portfolio_returns:
                        if pd.notna(daily_ret) and abs(daily_ret) < 0.2:  # 过滤极端值
                            current_value *= (1 + daily_ret)
                            portfolio_values.append(current_value)

                    print(f"    期间收益: {period_returns.shape[0]} 天, 价值: {portfolio_values[-1]:.2f}")

        # 存储结果
        results[strategy_name] = {
            'portfolio_values': portfolio_values,
            'final_value': portfolio_values[-1] if portfolio_values else 100.0
        }

    return results


def analyze_results(results):
    """分析结果"""
    print("\n📊 结果分析:")

    for strategy_name, data in results.items():
        portfolio_values = data['portfolio_values']

        if len(portfolio_values) > 1:
            # 计算基本指标
            initial_value = portfolio_values[0]
            final_value = portfolio_values[-1]
            total_return = (final_value / initial_value - 1) * 100

            # 计算日收益率
            daily_returns = []
            for i in range(1, len(portfolio_values)):
                daily_ret = portfolio_values[i] / portfolio_values[i - 1] - 1
                if abs(daily_ret) < 0.1:  # 过滤极端值
                    daily_returns.append(daily_ret)

            if len(daily_returns) > 10:
                annual_return = np.mean(daily_returns) * 252
                annual_vol = np.std(daily_returns) * np.sqrt(252)
                sharpe = (annual_return - 0.02) / annual_vol if annual_vol > 0 else 0
            else:
                annual_return = annual_vol = sharpe = 0

            print(f"\n{strategy_name}:")
            print(f"  总收益: {total_return:.1f}%")
            print(f"  年化收益: {annual_return:.1%}")
            print(f"  年化波动: {annual_vol:.1%}")
            print(f"  Sharpe: {sharpe:.3f}")
            print(f"  数据点: {len(portfolio_values)}")
            print(f"  最终价值: {final_value:.2f}")
        else:
            print(f"\n{strategy_name}: 无有效数据")


def main():
    """主函数"""
    print("🚀 极简RF测试开始...")

    try:
        # 1. 加载数据
        returns, features, rf_models = load_data()

        if len(rf_models) == 0:
            print("❌ 没有RF模型，无法继续")
            return

        # 2. 测试RF信号
        signals = test_rf_signals(returns, features, rf_models)

        # 3. 简单回测
        results = simple_backtest(returns, features, rf_models)

        # 4. 分析结果
        analyze_results(results)

        print("\n🎉 极简测试完成!")

        # 5. 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存最后的组合价值
        summary = []
        for strategy_name, data in results.items():
            if data['portfolio_values']:
                initial = data['portfolio_values'][0]
                final = data['portfolio_values'][-1]
                total_return = (final / initial - 1)

                summary.append({
                    'Strategy': strategy_name,
                    'Initial_Value': initial,
                    'Final_Value': final,
                    'Total_Return': total_return,
                    'Data_Points': len(data['portfolio_values'])
                })

        if summary:
            df = pd.DataFrame(summary)
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)

            output_file = results_dir / f"ultra_simple_test_{timestamp}.csv"
            df.to_csv(output_file, index=False)
            print(f"\n💾 结果已保存: {output_file}")

            print("\n📋 最终对比:")
            for _, row in df.iterrows():
                print(f"  {row['Strategy']}: {row['Total_Return']:.1%} ({row['Data_Points']} 个数据点)")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()