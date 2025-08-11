import pandas as pd
from pathlib import Path
from datetime import datetime
from skops.io import load as skload
import numpy as np

print("[RF_EXPORT] 🚀 启动 RF 概率导出 ...")

# 1. 加载特征数据
features_path = Path("banking_features_ai.csv")
features_df = pd.read_csv(features_path, index_col=0, parse_dates=True)
print(f"[RF_EXPORT] ✅ 加载特征数据: {features_df.shape}")

# 2. 定义需要信任的类型（字符串）
trusted = [
    "sklearn.calibration._CalibratedClassifier",
    "sklearn.calibration._SigmoidCalibration",
    "sklearn.model_selection._split.TimeSeriesSplit"
]

# 3. 扫描模型文件
models_dir = Path(".")
model_files = sorted(models_dir.glob("rf_v4_complete_*.skops"))

all_probs = pd.DataFrame(index=features_df.index)
success_count = 0

# 4. 遍历每个模型并生成概率
for model_file in model_files:
    ticker = model_file.stem.split("_")[-1]

    try:
        model_data = skload(model_file, trusted=trusted)
        print(f"[RF_EXPORT] 📦 模型加载成功: {ticker}")

        # 获取模型保存的特征顺序
        if "selected_features" in model_data:
            selected_features = model_data["selected_features"]
        else:
            selected_features = [c for c in features_df.columns if c.startswith(f"{ticker}_")]

        # 对齐特征顺序
        X_full = features_df.reindex(columns=selected_features)
        X_full = np.nan_to_num(X_full)  # 避免 NaN

        # 缩放
        if "scaler" in model_data and model_data["scaler"] is not None:
            X_full = model_data["scaler"].transform(X_full)

        model = model_data["model"]

        # 预测概率
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_full)[:, 1]  # 取正类概率
        else:
            preds = model.predict(X_full)
            probs = np.array(preds, dtype=float)

        all_probs[ticker] = probs
        success_count += 1

    except Exception as e:
        print(f"[RF_EXPORT] ❌ 模型加载失败: {ticker} | {e}")

# 5. 导出结果
if success_count > 0:
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"rf_probs_5D_{timestamp}.csv"
    all_probs.to_csv(output_file)
    print(f"[RF_EXPORT] 🎯 成功导出 {success_count}/{len(model_files)} 个模型概率文件: {output_file}")
else:
    raise ValueError("❌ 没有生成任何概率数据")
