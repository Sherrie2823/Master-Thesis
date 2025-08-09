import pandas as pd
import numpy as np
from pathlib import Path
from XGboost_v5 import BankingXGBoostV5

ROOT = Path(__file__).resolve().parent
RAW  = ROOT / "data" / "raw"

# 自动匹配 returns 文件
ret_candidates = [RAW/"banking_returns_10y.csv", RAW/"banking_returns.csv"]
tgt_candidates = [RAW/"banking_targets_ai.csv",  ROOT/"banking_targets_ai.csv"]

ret_path = next((p for p in ret_candidates if p.exists()), None)
tgt_path = next((p for p in tgt_candidates if p.exists()), None)
print("RET:", ret_path, "exists:", ret_path and ret_path.exists())
print("TGT:", tgt_path, "exists:", tgt_path and tgt_path.exists())

if ret_path is None or tgt_path is None:
    raise FileNotFoundError("请确认 returns/targets 文件在上述路径之一。")

xgb = BankingXGBoostV5(enable_tech_indicators=False, verbose=False)
xgb = BankingXGBoostV5(
    top_k_features=50,
    n_splits=5,
    test_size=0.2,
    pre_rfe_features=200,
    nested_cv=False,
    verbose=False,
    enable_tech_indicators=False,
    calibration_method='sigmoid'
)
xgb.load_data(feature_path=str(ret_path), target_path=str(tgt_path))
jpm_cols = [c for c in xgb.targets.columns if c.startswith("JPM_Direction_")]
print("JPM 可用的目标列：", jpm_cols)
probs = xgb.get_probability_signal(stock='JPM', horizon='5D')
print(probs.head(), probs.describe())

# ===== 1) 取全市场（15只银行股）概率，拼成 DataFrame =====
BANKS = ['JPM','BAC','C']
# BANKS = ['AXP','BAC','BK','C','CB','COF','GS','JPM','MS','PNC','SCHW','STT','TFC','USB','WFC']
HORIZON = '5D'   # 先用 5D，等你补好 15D 再切过去

probs_map = {}
for s in BANKS:
    try:
        probs_map[s] = xgb.get_probability_signal(stock=s, horizon=HORIZON)
    except Exception as e:
        print(f"[skip] {s}: {e}")

# 按日期对齐拼表（只保留共同测试期）
probs_df = pd.DataFrame(probs_map).dropna(how='all')
print("probs_df shape:", probs_df.shape)
# 保险：去掉全0/全NaN的列
probs_df = probs_df.loc[:, probs_df.notna().sum()>0]

# ===== 2) 概率 -> 当日横截面权重（两种方式：normalize / topN）=====
def weights_from_probs(probs_row, cap=(0.0, 0.20), method='normalize', topN=5):
    p = probs_row.clip(lower=0)  # 负值保护（理论不会有）
    if method == 'topN':
        keep = p.nlargest(topN).index
        p = p.where(p.index.isin(keep), 0.0)
    if p.sum() == 0:
        w = p.copy()  # 全0就返回0（那天不持仓）
    else:
        w = p / p.sum()
    # 约束（可选）：单资产上限 20%，下限 0
    w = w.clip(lower=cap[0], upper=cap[1])
    if w.sum() > 0:
        w = w / w.sum()
    return w

# 每日生成权重（先给两套：纯归一化 & Top-5）
W_norm = probs_df.apply(lambda r: weights_from_probs(r, method='normalize'), axis=1)
W_top5 = probs_df.apply(lambda r: weights_from_probs(r, method='topN', topN=5), axis=1)

# ===== 3) 应用“持有期=15天、T+1生效”的交易机制 =====
HOLD = 15

def apply_holding(w_daily, hold=15):
    # 只在每 hold 天的第一天“计算/换仓”，其余天沿用上一期权重
    w = w_daily.copy()
    mask = np.zeros(len(w), dtype=bool)
    mask[::hold] = True
    w.loc[~w.index.isin(w.index[mask])] = np.nan
    w = w.ffill()
    return w.shift(1)     # T+1 生效，避免回看当天收益

W_norm_h = apply_holding(W_norm, HOLD)
W_top5_h = apply_holding(W_top5, HOLD)

# ===== 4) 回测：用日收益矩阵，计算净值/Sharpe/回撤（含交易成本）=====
rets = pd.read_csv(str(ret_path), index_col=0, parse_dates=True).sort_index()
rets = rets.reindex(W_norm_h.index).loc[W_norm_h.index]    # 对齐日期
rets = rets[W_norm_h.columns]                              # 对齐资产

TRANSCOST = 0.001  # 0.1% 每次换仓

def backtest(weights, returns, tc=0.001, rf_annual=0.02):
    # 组合日收益
    port_ret = (weights * returns).sum(axis=1).fillna(0.0)
    # 交易成本：|Δw|之和 * tc
    dw = weights.fillna(0).diff().abs().sum(axis=1)
    tc_series = dw * tc
    port_ret_net = port_ret - tc_series
    # 绩效
    v = (1 + port_ret_net).cumprod()
    ann = (1 + port_ret_net).prod() ** (252/len(port_ret_net)) - 1
    vol = port_ret_net.std() * np.sqrt(252)
    sharpe = (ann - rf_annual) / (vol + 1e-12)
    peak = np.maximum.accumulate(v.values)
    mdd = np.min(v.values/peak - 1.0)
    return {
        'series': port_ret_net,
        'value': v,
        'annual': ann,
        'vol': vol,
        'sharpe': sharpe,
        'mdd': mdd
    }

res_norm = backtest(W_norm_h, rets, TRANSCOST)
res_top5 = backtest(W_top5_h, rets, TRANSCOST)

print("\n== Performance (5D probs, HOLD=15, tc=10bps) ==")
print(f"Normalize  -> Ann: {res_norm['annual']:.2%}, Vol: {res_norm['vol']:.2%}, Sharpe: {res_norm['sharpe']:.2f}, MDD: {res_norm['mdd']:.2%}")
print(f"Top-5      -> Ann: {res_top5['annual']:.2%}, Vol: {res_top5['vol']:.2%}, Sharpe: {res_top5['sharpe']:.2f}, MDD: {res_top5['mdd']:.2%}")

if __name__ == "__main__":
    # 1) 初始化 & 加载数据
    xgb = BankingXGBoostV5(
        top_k_features=50, n_splits=3, test_size=0.2,
        pre_rfe_features=200, nested_cv=False,
        verbose=False, enable_tech_indicators=False,
        calibration_method='sigmoid'
    )

    # 注意路径：feature 在 data/raw，targets 在项目根目录（按你当前目录结构）
    xgb.load_data(
        feature_path='data/raw/banking_returns_10y.csv',
        target_path='banking_targets_ai.csv'
    )

    # 2) 一行拿到全市场概率矩阵（自动保存到 results/）
    BANKS = ['JPM','BAC','C']
    # BANKS = ['AXP','BAC','BK','C','CB','COF','GS','JPM','MS','PNC','SCHW','STT','TFC','USB','WFC']

    # —— 用“类里方法”的版本（你已按我说的放进 XGboost_v5.py）——
    probs_df = xgb.collect_probabilities_for_universe(
        stocks=BANKS, horizon='5D', save_dir='results', resume=True
    )

    print(probs_df.shape)
    print(probs_df.head())