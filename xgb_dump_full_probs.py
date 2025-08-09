# xgb_dump_full_probs.py
import pandas as pd
from pathlib import Path
from XGboost_v5 import BankingXGBoostV5

# ===== 可配置 =====
BANKS = ['AXP','BAC','BK','C','CB','COF','GS','JPM','MS','PNC','SCHW','STT','TFC','USB','WFC']
HORIZON = '5D'          # 你也可以临时用 '1D'
TEST_SIZE = 0.4         # <- 关键：把样本外比例拉大（40%）
FEATURE_CSV = 'data/raw/banking_returns_10y.csv'
TARGET_CSV  = 'banking_targets_ai.csv'   # 你的 targets 路径
OUT_DIR = Path('results')

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    xgb = BankingXGBoostV5(
        top_k_features=50, n_splits=3, test_size=TEST_SIZE,
        pre_rfe_features=200, nested_cv=False,
        verbose=False, enable_tech_indicators=False,
        calibration_method='sigmoid'
    )
    xgb.load_data(feature_path=FEATURE_CSV, target_path=TARGET_CSV)

    series_map = {}
    for s in BANKS:
        try:
            p = xgb.get_full_period_probability(stock=s, horizon=HORIZON, task_type='direction')
            series_map[s] = p.rename(s)
            print(f"✅ {s}: {p.index.min().date()} ~ {p.index.max().date()}  共{len(p)}条")
        except Exception as e:
            print(f"⚠️ {s} 生成失败: {e}")

    if not series_map:
        raise RuntimeError("没有任何股票生成成功的概率序列")

    # 对齐所有股票的日期索引，列为股票
    probs_df = pd.concat(series_map, axis=1)
    probs_df.index.name = 'Date'
    probs_df.sort_index(inplace=True)

    fname = OUT_DIR / f"xgb_probs_{HORIZON}_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv"
    probs_df.to_csv(fname)
    print(f"\n🎉 全期概率已保存：{fname}  形状：{probs_df.shape}")

if __name__ == "__main__":
    main()
