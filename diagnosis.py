# step1_diagnosis.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import glob

print("🔍 开始诊断问题...")

# 1. 检查文件存在
print("\n1. 检查文件存在性:")
files_to_check = [
    "banking_features_ai.csv",
    "*.pkl"  # RF模型文件
]

features_file = "banking_features_ai.csv"
if Path(features_file).exists():
    print(f"✅ 特征文件存在: {features_file}")
    features_data = pd.read_csv(features_file, index_col=0, parse_dates=True)
    print(f"   - 特征数据形状: {features_data.shape}")
    print(f"   - 日期范围: {features_data.index[0]} 到 {features_data.index[-1]}")

    # 检查每个股票的特征数量
    tickers = ['AXP', 'BAC', 'BK', 'C', 'CB', 'COF', 'GS', 'JPM', 'MS', 'PNC', 'SCHW', 'STT', 'TFC', 'USB', 'WFC']
    for ticker in tickers:
        ticker_features = [col for col in features_data.columns if col.startswith(f'{ticker}_')]
        print(f"   - {ticker}: {len(ticker_features)} 个特征")
else:
    print(f"❌ 特征文件不存在: {features_file}")

# 2. 检查模型文件
print("\n2. 检查模型文件:")
model_files = glob.glob("rf_v4_complete_*.pkl")
print(f"找到 {len(model_files)} 个模型文件:")

for model_file in model_files[:3]:  # 只检查前3个
    ticker = Path(model_file).stem.split('_')[-1]
    print(f"\n检查模型: {ticker}")

    try:
        model_data = joblib.load(model_file)
        print(f"  ✅ 可以加载")

        if 'model' in model_data:
            model = model_data['model']
            print(f"  - 模型类型: {type(model).__name__}")

            if hasattr(model, 'n_features_in_'):
                print(f"  - 期望特征数: {model.n_features_in_}")

        if 'scaler' in model_data:
            scaler = model_data['scaler']
            has_scale = hasattr(scaler, 'scale_') and scaler.scale_ is not None
            print(f"  - 缩放器状态: {'已训练' if has_scale else '未训练'}")

    except Exception as e:
        print(f"  ❌ 加载失败: {e}")

# 3. 简单测试预测
print("\n3. 测试模型预测:")
if len(model_files) > 0 and Path(features_file).exists():
    try:
        # 选择第一个模型测试
        test_model_file = model_files[0]
        test_ticker = Path(test_model_file).stem.split('_')[-1]

        model_data = joblib.load(test_model_file)
        model = model_data['model']

        # 获取对应特征
        ticker_features = [col for col in features_data.columns if col.startswith(f'{test_ticker}_')]

        if len(ticker_features) > 0:
            # 使用最后一个日期的数据测试
            test_date = features_data.index[-1]
            test_features = features_data.loc[test_date, ticker_features].values.reshape(1, -1)

            # 处理NaN
            if np.any(np.isnan(test_features)):
                print(f"  ⚠️ 发现NaN值，填充为0")
                test_features = np.nan_to_num(test_features, nan=0)

            print(f"  测试特征形状: {test_features.shape}")

            # 尝试预测
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(test_features)
                print(f"  ✅ 预测成功: {pred}")
            else:
                pred = model.predict(test_features)
                print(f"  ✅ 预测成功: {pred}")

        else:
            print(f"  ❌ 没有找到 {test_ticker} 的特征")

    except Exception as e:
        print(f"  ❌ 预测测试失败: {e}")

print("\n🏁 诊断完成!")
print("\n下一步:")
print("1. 如果模型文件正常，运行 step2_fix_prediction.py")
print("2. 如果模型文件有问题，可能需要重新训练")