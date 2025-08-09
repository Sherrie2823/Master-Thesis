import pandas as pd
from XGboost_v5 import BankingXGBoostV5

# 1) 初始化 + 加载数据
xgb = BankingXGBoostV5(
    top_k_features=50, n_splits=3, test_size=0.2,   # test_size 此处无关紧要了
    pre_rfe_features=200, nested_cv=False,
    verbose=True, enable_tech_indicators=False,
    calibration_method='sigmoid'
)

xgb.load_data(
    feature_path='data/raw/banking_returns_10y.csv',
    target_path='banking_targets_ai.csv'
)

# 2) 一篮子股票做“全期样本外”概率
BANKS = ['AXP','BAC','BK','C','CB','COF','GS','JPM','MS','PNC','SCHW','STT','TFC','USB','WFC']
probs_df = xgb.collect_probabilities_rolling(
    stocks=BANKS,
    horizon='5D',
    initial_window=252,      # 从第 1 年后开始 OOS
    step_size=1,             # 每天滚动
    retrain_every=21,        # 每 21 天再训练一次（速度和严谨的折中）
    save_dir='results',
    filename_prefix='xgb_probs_full_oos'
)

print(probs_df.shape)
print(probs_df.head())
