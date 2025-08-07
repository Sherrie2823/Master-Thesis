import pandas as pd
import numpy as np
import warnings
import os
print("当前工作目录是：", os.getcwd())

warnings.filterwarnings('ignore')


def create_banking_features():
    """银行业AI模型特征工程 - 根目录版本"""

    print("=" * 70)
    print("🏦 银行业AI增强投资组合优化 - 特征工程 🚀")
    print("=" * 70)

    # ==================== 1. 数据加载 ====================
    print("📊 1. 加载银行数据...")

    # 直接从根目录加载
    prices = pd.read_csv(r'C:\Users\Philips Deng\Documents\GitHub\AI-Enhanced-Portfolio-Optimization\banking_prices.csv', index_col='Date', parse_dates=True)
    returns = pd.read_csv(r'C:\Users\Philips Deng\Documents\GitHub\AI-Enhanced-Portfolio-Optimization\banking_returns.csv', index_col='Date', parse_dates=True)
    volume = pd.read_csv(r'C:\Users\Philips Deng\Documents\GitHub\AI-Enhanced-Portfolio-Optimization\src\models\ai_models\banking_volume.csv', index_col='Date', parse_dates=True)

    print(f"   ✅ 价格数据: {prices.shape}")
    print(f"   ✅ 收益数据: {returns.shape}")
    print(f"   ✅ 成交量数据: {volume.shape}")
    print(f"   ✅ 时间范围: {prices.index[0]} 到 {prices.index[-1]}")

    # 银行股票列表
    banking_stocks = ['AXP', 'BAC', 'BK', 'C', 'CB', 'COF', 'GS',
                      'JPM', 'MS', 'PNC', 'SCHW', 'STT', 'TFC', 'USB', 'WFC']
    print(f"   ✅ 银行股票: {len(banking_stocks)} 只")

    # ==================== 2. 技术指标特征 ====================
    print("\n🔧 2. 创建技术指标特征...")

    features = pd.DataFrame(index=prices.index)

    for i, stock in enumerate(banking_stocks):
        print(f"   处理 {stock} ({i + 1}/{len(banking_stocks)})...")

        # === 价格特征 ===
        # 移动平均线
        features[f'{stock}_SMA_5'] = prices[stock].rolling(5).mean()
        features[f'{stock}_SMA_10'] = prices[stock].rolling(10).mean()
        features[f'{stock}_SMA_20'] = prices[stock].rolling(20).mean()
        features[f'{stock}_SMA_50'] = prices[stock].rolling(50).mean()

        # 价格相对位置
        features[f'{stock}_Price_vs_SMA20'] = prices[stock] / features[f'{stock}_SMA_20']
        features[f'{stock}_Price_vs_SMA50'] = prices[stock] / features[f'{stock}_SMA_50']

        # === 动量特征 ===
        features[f'{stock}_Momentum_1D'] = prices[stock].pct_change(1)
        features[f'{stock}_Momentum_5D'] = prices[stock].pct_change(5)
        features[f'{stock}_Momentum_10D'] = prices[stock].pct_change(10)
        features[f'{stock}_Momentum_20D'] = prices[stock].pct_change(20)

        # === 波动率特征 ===
        features[f'{stock}_Vol_5D'] = returns[stock].rolling(5).std() * np.sqrt(252)
        features[f'{stock}_Vol_10D'] = returns[stock].rolling(10).std() * np.sqrt(252)
        features[f'{stock}_Vol_20D'] = returns[stock].rolling(20).std() * np.sqrt(252)
        features[f'{stock}_Vol_60D'] = returns[stock].rolling(60).std() * np.sqrt(252)

        # === RSI指标 ===
        delta = prices[stock].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features[f'{stock}_RSI'] = 100 - (100 / (1 + rs))

        # === MACD指标 ===
        ema_12 = prices[stock].ewm(span=12).mean()
        ema_26 = prices[stock].ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        features[f'{stock}_MACD'] = macd
        features[f'{stock}_MACD_Signal'] = macd_signal
        features[f'{stock}_MACD_Histogram'] = macd - macd_signal

        # === 布林带 ===
        sma_20 = features[f'{stock}_SMA_20']
        std_20 = prices[stock].rolling(20).std()
        bb_upper = sma_20 + (std_20 * 2)
        bb_lower = sma_20 - (std_20 * 2)
        features[f'{stock}_BB_Position'] = (prices[stock] - bb_lower) / (bb_upper - bb_lower)
        features[f'{stock}_BB_Width'] = (bb_upper - bb_lower) / sma_20

        # === 成交量特征 ===
        features[f'{stock}_Volume_SMA_10'] = volume[stock].rolling(10).mean()
        features[f'{stock}_Volume_SMA_20'] = volume[stock].rolling(20).mean()
        features[f'{stock}_Volume_Ratio'] = volume[stock] / features[f'{stock}_Volume_SMA_20']

        # === 价格通道 ===
        high_20 = prices[stock].rolling(20).max()
        low_20 = prices[stock].rolling(20).min()
        features[f'{stock}_Price_Channel'] = (prices[stock] - low_20) / (high_20 - low_20)

        # === 相对强度 ===
        features[f'{stock}_Relative_Strength'] = prices[stock] / prices[stock].rolling(252).mean()

    print(f"   ✅ 技术指标特征完成: {len([c for c in features.columns if any(s in c for s in banking_stocks)])} 个")

    # ==================== 3. 市场特征 ====================
    print("\n📈 3. 创建市场特征...")

    # 整体市场特征
    market_return = returns.mean(axis=1)
    features['Market_Return'] = market_return
    features['Market_Vol'] = returns.std(axis=1) * np.sqrt(252)
    features['Market_Momentum_5D'] = market_return.rolling(5).sum()
    features['Market_Momentum_20D'] = market_return.rolling(20).sum()

    # 大型银行 vs 其他银行
    large_banks = ['JPM', 'BAC', 'WFC', 'C']  # 四大银行
    investment_banks = ['GS', 'MS']  # 投资银行
    other_banks = [s for s in banking_stocks if s not in large_banks + investment_banks]

    large_return = returns[large_banks].mean(axis=1)
    investment_return = returns[investment_banks].mean(axis=1)
    other_return = returns[other_banks].mean(axis=1)

    features['Large_Banks_Return'] = large_return
    features['Investment_Banks_Return'] = investment_return
    features['Other_Banks_Return'] = other_return
    features['Large_vs_Other_Spread'] = large_return - other_return
    features['Investment_vs_Commercial_Spread'] = investment_return - large_return

    print(
        f"   ✅ 市场特征完成: {len([c for c in features.columns if 'Market' in c or 'Banks' in c or 'Spread' in c])} 个")

    # ==================== 4. 相对特征 ====================
    print("\n🔄 4. 创建相对特征...")

    for stock in banking_stocks:
        # 相对市场表现
        features[f'{stock}_Relative_Return'] = returns[stock] - market_return
        features[f'{stock}_Relative_5D'] = (returns[stock].rolling(5).mean() -
                                            market_return.rolling(5).mean())
        features[f'{stock}_Relative_20D'] = (returns[stock].rolling(20).mean() -
                                             market_return.rolling(20).mean())

        # 滚动相关性
        features[f'{stock}_Corr_30D'] = returns[stock].rolling(30).corr(market_return)
        features[f'{stock}_Corr_60D'] = returns[stock].rolling(60).corr(market_return)

        # 滚动Beta
        rolling_cov_30 = returns[stock].rolling(30).cov(market_return)
        rolling_var_30 = market_return.rolling(30).var()
        features[f'{stock}_Beta_30D'] = rolling_cov_30 / rolling_var_30

        rolling_cov_60 = returns[stock].rolling(60).cov(market_return)
        rolling_var_60 = market_return.rolling(60).var()
        features[f'{stock}_Beta_60D'] = rolling_cov_60 / rolling_var_60

        # 排名特征
        rolling_returns_5 = returns.rolling(5).sum()
        rolling_returns_20 = returns.rolling(20).sum()
        features[f'{stock}_Rank_5D'] = rolling_returns_5.rank(axis=1, pct=True)[stock]
        features[f'{stock}_Rank_20D'] = rolling_returns_20.rank(axis=1, pct=True)[stock]

    print(f"   ✅ 相对特征完成")

    # ==================== 5. 创建目标变量 ====================
    print("\n🎯 5. 创建目标变量...")

    targets = pd.DataFrame(index=returns.index)

    for stock in banking_stocks:
        # 未来1日收益率
        targets[f'{stock}_Return_1D'] = returns[stock].shift(-1)
        # 未来5日累计收益率
        targets[f'{stock}_Return_5D'] = returns[stock].rolling(5).sum().shift(-5)

        # 收益方向（分类标签）
        targets[f'{stock}_Direction_1D'] = (targets[f'{stock}_Return_1D'] > 0).astype(int)
        targets[f'{stock}_Direction_5D'] = (targets[f'{stock}_Return_5D'] > 0).astype(int)

        # 相对市场表现
        market_1d = market_return.shift(-1)
        market_5d = market_return.rolling(5).sum().shift(-5)
        targets[f'{stock}_Outperform_1D'] = (targets[f'{stock}_Return_1D'] > market_1d).astype(int)
        targets[f'{stock}_Outperform_5D'] = (targets[f'{stock}_Return_5D'] > market_5d).astype(int)

    print(f"   ✅ 目标变量完成: {targets.shape[1]} 个")

    # ==================== 6. 数据清理和验证 ====================
    print("\n🧹 6. 数据清理...")

    print(f"   原始特征数量: {features.shape[1]}")
    print(f"   原始样本数量: {features.shape[0]}")

    # 替换无穷大值
    features = features.replace([np.inf, -np.inf], np.nan)
    targets = targets.replace([np.inf, -np.inf], np.nan)

    # 删除缺失值过多的特征（保留70%以上数据的特征）
    missing_threshold = 0.3
    feature_missing = features.isnull().mean()
    valid_features = feature_missing[feature_missing < missing_threshold].index
    features_filtered = features[valid_features]

    print(f"   过滤后特征数量: {features_filtered.shape[1]}")
    print(f"   删除了 {features.shape[1] - features_filtered.shape[1]} 个缺失值过多的特征")

    # 对齐特征和目标
    common_dates = features_filtered.dropna().index.intersection(targets.dropna().index)
    features_final = features_filtered.loc[common_dates]
    targets_final = targets.loc[common_dates]

    print(f"   最终样本数量: {features_final.shape[0]}")
    print(f"   时间范围: {features_final.index[0]} 到 {features_final.index[-1]}")

    # ==================== 7. 保存数据 ====================
    print("\n💾 7. 保存特征数据...")

    features_final.to_csv('banking_features_ai.csv')
    targets_final.to_csv('banking_targets_ai.csv')

    # 保存特征描述
    feature_info = pd.DataFrame({
        'Feature': features_final.columns,
        'Type': ['Technical' if any(stock in feat for stock in banking_stocks)
                 else 'Market' if 'Market' in feat or 'Banks' in feat
        else 'Other' for feat in features_final.columns]
    })
    feature_info.to_csv('feature_descriptions.csv', index=False)

    print("   ✅ 文件保存完成:")
    print("   📄 banking_features_ai.csv - 特征数据")
    print("   📄 banking_targets_ai.csv - 目标变量")
    print("   📄 feature_descriptions.csv - 特征描述")

    # ==================== 8. 结果总结 ====================
    print("\n" + "=" * 70)
    print("🎉 特征工程完成！")
    print("=" * 70)
    print(f"✅ 最终特征数量: {features_final.shape[1]}")
    print(f"✅ 样本数量: {features_final.shape[0]}")
    print(f"✅ 目标变量数量: {targets_final.shape[1]}")
    print(f"✅ 银行股票: {len(banking_stocks)} 只")
    print(f"✅ 数据完整性: {(1 - features_final.isnull().mean().mean()) * 100:.1f}%")
    print(f"✅ 时间跨度: {(features_final.index[-1] - features_final.index[0]).days} 天")

    # 特征类型统计
    technical_features = len([f for f in features_final.columns if any(s in f for s in banking_stocks)])
    market_features = len([f for f in features_final.columns if 'Market' in f or 'Banks' in f])
    other_features = features_final.shape[1] - technical_features - market_features

    print(f"\n📊 特征类型分布:")
    print(f"   🔧 技术指标特征: {technical_features}")
    print(f"   📈 市场特征: {market_features}")
    print(f"   🔄 其他特征: {other_features}")

    print(f"\n🚀 准备开始AI模型训练！")
    print("   下一步: Random Forest + XGBoost + LSTM")

    return features_final, targets_final


if __name__ == "__main__":
    # 运行特征工程
    X, y = create_banking_features()

    if X is not None and y is not None:
        print(f"\n✅ 特征工程成功完成！")
        print(f"特征矩阵 X: {X.shape}")
        print(f"目标矩阵 y: {y.shape}")
    else:
        print(f"\n❌ 特征工程失败")