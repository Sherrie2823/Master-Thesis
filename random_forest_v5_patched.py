import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.calibration import CalibratedClassifierCV
from skops.io import dump
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings('ignore')


class BankingRandomForestV4:
    def __init__(self, top_k_features=50, n_splits=5, test_size=0.2,
                 pre_rfe_features=200, nested_cv=True, verbose=True):
        """
        Random Forest V4 - 终极优化版（已对齐 XGB 流程）
        关键改动：
        - 默认读取 banking_returns_10y.csv
        - 训练后以 CalibratedClassifierCV 做 Sigmoid 概率校准
        - 保存每只股票的 y_prob_train / y_prob_test（带日期索引）
        - 新增 dump_full_probs() 导出全期概率 CSV，供回测脚本统一读取
        """
        self.features = None
        self.targets = None
        self.top_k_features = top_k_features
        self.n_splits = n_splits
        self.test_size = test_size
        self.pre_rfe_features = pre_rfe_features
        self.nested_cv = nested_cv
        self.verbose = verbose

        self.banking_stocks = ['AXP', 'BAC', 'BK', 'C', 'CB', 'COF', 'GS',
                               'JPM', 'MS', 'PNC', 'SCHW', 'STT', 'TFC', 'USB', 'WFC']
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.results = {}  # {stock: {'y_prob_train': Series, 'y_prob_test': Series}}

        # 两阶段超参数搜索
        self.coarse_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, 20],
            'min_samples_split': [5, 10, 20],
            'min_samples_leaf': [2, 5, 10],
            'max_features': ['sqrt', 'log2', 0.5],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        self.fine_param_grid = {}

    def log(self, message, level="INFO"):
        if self.verbose:
            print(f"[{level}] {message}")

    # === CHANGED: 默认特征文件改为 banking_returns_10y.csv ===
    def load_data(self, feature_path='banking_returns_10y.csv', target_path='banking_targets_ai.csv'):
        self.log("=" * 80)
        self.log("🌲 Random Forest V5 - 终极优化版（已对齐 XGB 输出）")
        self.log("=" * 80)
        self.log("📊 1. 加载数据...")

        self.features = pd.read_csv(feature_path, index_col='Date', parse_dates=True)
        self.targets = pd.read_csv(target_path, index_col='Date', parse_dates=True)

        self.log(f"   ✅ 特征数据: {self.features.shape}")
        self.log(f"   ✅ 目标数据: {self.targets.shape}")
        self.log(f"   ✅ 时间范围: {self.features.index[0]} 到 {self.features.index[-1]}")
        self.log(f"   ✅ 配置: Top-{self.top_k_features}特征, {self.n_splits}折CV, 预筛选{self.pre_rfe_features}特征, 嵌套CV={'开启' if self.nested_cv else '关闭'}")
        return True

    def prepare_data_no_leakage(self, stock, task_type='direction', horizon='5D'):
        """无数据泄漏的数据准备（默认 5D 与 XGB 对齐）"""
        target_col = f'{stock}_Direction_{horizon}'
        if target_col not in self.targets.columns:
            self.log(f"   ⚠️ 目标变量 {target_col} 不存在!", "WARNING")
            return None, None

        X = self.features.dropna()
        y = self.targets[target_col].dropna()
        valid_idx = X.index.intersection(y.index)
        X, y = X.loc[valid_idx], y.loc[valid_idx]

        self.log(f"   📈 {stock}: {X.shape[0]} 样本, {X.shape[1]} 特征")
        self.log(f"   📊 类别分布: {dict(y.value_counts().sort_index())}")
        return X, y

    def hierarchical_feature_selection(self, X_train, y_train, stock):
        """
        分层特征选择: SelectKBest -> RF Importance -> RFE
        """
        n_samples = len(X_train)
        original_features = X_train.shape[1]

        self.log(f"   🔍 开始分层特征选择 (原始特征: {original_features})")

        # 第一层: 统计筛选（互信息，速度快且非线性）
        if original_features > self.pre_rfe_features:
            selector_stat = SelectKBest(score_func=mutual_info_classif, k=self.pre_rfe_features)
            X_stat = selector_stat.fit_transform(X_train, y_train)
            selected_features_stat = X_train.columns[selector_stat.get_support()].tolist()
            self.log(f"   📉 统计筛选: {original_features} -> {len(selected_features_stat)}")
        else:
            X_stat = X_train
            selected_features_stat = X_train.columns.tolist()
            self.log(f"   📉 跳过统计筛选 (特征数已少于阈值)")

        # 第二层: 随机森林重要性
        rf_selector = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        rf_selector.fit(X_train[selected_features_stat], y_train)

        importances = pd.Series(rf_selector.feature_importances_, index=selected_features_stat)
        rf_top_features = importances.nlargest(min(self.top_k_features * 2, len(selected_features_stat))).index.tolist()
        self.log(f"   🌲 RF重要性筛选: {len(selected_features_stat)} -> {len(rf_top_features)}")

        # 第三层: RFE（样本数足够时）
        if n_samples >= 500 and len(rf_top_features) > self.top_k_features:
            self.log(f"   🔄 执行RFE (样本数充足: {n_samples})")
            X_rf = X_train[rf_top_features]
            rfe = RFE(
                estimator=RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
                n_features_to_select=self.top_k_features,
                step=0.1
            )
            with tqdm(desc="RFE进度", disable=not self.verbose) as pbar:
                rfe.fit(X_rf, y_train)
                pbar.update(1)
            final_features = X_rf.columns[rfe.support_].tolist()
            self.log(f"   🎯 RFE筛选: {len(rf_top_features)} -> {len(final_features)}")
        else:
            if n_samples < 500:
                self.log(f"   ⚠️ RFE跳过 (样本数不足: {n_samples})")
            elif len(rf_top_features) <= self.top_k_features:
                self.log(f"   ⚠️ RFE跳过 (特征数已达标: {len(rf_top_features)})")
            final_features = rf_top_features[:self.top_k_features]
            self.log(f"   ⚠️ 跳过RFE (样本不足或特征已达标)")

        self.log(f"   ✅ 最终选择特征: {len(final_features)}")
        return final_features

    def nested_cross_validation(self, X_train, y_train):
        outer_scores = []
        best_params_list = []
        outer_cv = TimeSeriesSplit(n_splits=self.n_splits)
        self.log(f"   🔄 嵌套CV: {self.n_splits}折外层验证")
        for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X_train)):
            if self.verbose:
                print(f"      折数 {fold_idx + 1}/{self.n_splits}")
            X_train_fold = X_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_train_fold = y_train.iloc[train_idx]
            y_val_fold = y_train.iloc[val_idx]

            inner_cv = TimeSeriesSplit(n_splits=3)
            model_coarse = RandomForestClassifier(random_state=42, n_jobs=-1)
            search_coarse = RandomizedSearchCV(
                estimator=model_coarse,
                param_distributions=self.coarse_param_grid,
                n_iter=20,
                scoring='f1',
                cv=inner_cv,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            search_coarse.fit(X_train_fold, y_train_fold)

            best_coarse = search_coarse.best_params_
            fine_grid = self.generate_fine_grid(best_coarse)

            model_fine = RandomForestClassifier(random_state=42, n_jobs=-1)
            search_fine = RandomizedSearchCV(
                estimator=model_fine,
                param_distributions=fine_grid,
                n_iter=20,
                scoring='f1',
                cv=inner_cv,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            search_fine.fit(X_train_fold, y_train_fold)

            best_model = search_fine.best_estimator_
            y_pred_val = best_model.predict(X_val_fold)
            fold_score = f1_score(y_val_fold, y_pred_val)

            outer_scores.append(fold_score)
            best_params_list.append(search_fine.best_params_)
            if self.verbose:
                print(f"         F1: {fold_score:.4f}, 参数: {search_fine.best_params_}")

        avg_score = np.mean(outer_scores)
        std_score = np.std(outer_scores)
        self.log(f"   ✅ 嵌套CV结果: F1 = {avg_score:.4f} (±{std_score:.4f})")
        best_params = best_params_list[0] if best_params_list else self.coarse_param_grid
        return best_params, avg_score

    def generate_fine_grid(self, coarse_best):
        fine_grid = {}
        base_n_est = coarse_best.get('n_estimators', 200)
        fine_grid['n_estimators'] = [max(50, base_n_est - 50), base_n_est, base_n_est + 50]
        base_depth = coarse_best.get('max_depth', 10)
        if base_depth is not None:
            fine_grid['max_depth'] = [max(3, base_depth - 2), base_depth, base_depth + 2, None]
        else:
            fine_grid['max_depth'] = [15, 20, None]
        base_split = coarse_best.get('min_samples_split', 10)
        fine_grid['min_samples_split'] = [max(2, base_split - 2), base_split, base_split + 5]
        base_leaf = coarse_best.get('min_samples_leaf', 5)
        fine_grid['min_samples_leaf'] = [max(1, base_leaf - 2), base_leaf, base_leaf + 2]
        fine_grid['max_features'] = [coarse_best.get('max_features', 'sqrt')]
        fine_grid['class_weight'] = [coarse_best.get('class_weight', 'balanced')]
        fine_grid['bootstrap'] = [True, False]
        return fine_grid

    # === CHANGED: horizon 默认改为 5D，与 XGB 一致 ===
    def train_model_v4(self, stock, task_type='direction', horizon='5D'):
        self.log(f"\n🌲 2. 训练 {stock} 模型 ({task_type}-{horizon})...")

        X, y = self.prepare_data_no_leakage(stock, task_type, horizon)
        if X is None:
            return None

        split_idx = int(len(X) * (1 - self.test_size))
        X_train_full, X_test_full = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train_full, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        self.log(f"   📊 训练集: {X_train_full.shape[0]} 样本")
        self.log(f"   📊 测试集: {X_test_full.shape[0]} 样本")

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_full),
            columns=X_train_full.columns,
            index=X_train_full.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test_full),
            columns=X_test_full.columns,
            index=X_test_full.index
        )
        self.scalers[stock] = scaler

        selected_features = self.hierarchical_feature_selection(X_train_scaled, y_train_full, stock)
        self.feature_selectors[stock] = selected_features

        X_train = X_train_scaled[selected_features]
        X_test = X_test_scaled[selected_features]

        if self.nested_cv:
            best_params, cv_score = self.nested_cross_validation(X_train, y_train_full)
        else:
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=self.coarse_param_grid,
                n_iter=30,
                scoring='f1',
                cv=tscv,
                random_state=42,
                n_jobs=-1,
                verbose=1 if self.verbose else 0
            )
            search.fit(X_train, y_train_full)
            best_params = search.best_params_
            cv_score = search.best_score_

        self.log(f"   ✅ 最佳参数: {best_params}")
        self.log(f"   ✅ CV F1得分: {cv_score:.4f}")

        # === CHANGED: 使用校准后的模型产生概率 ===
        final_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
        final_model.fit(X_train, y_train_full)

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        # === FIXED: 修改 base_estimator 为 estimator ===
        calibrated = CalibratedClassifierCV(estimator=final_model, method='sigmoid', cv=tscv)
        calibrated.fit(X_train, y_train_full)

        y_pred_train = calibrated.predict(X_train)
        y_pred_test = calibrated.predict(X_test)
        y_prob_train = calibrated.predict_proba(X_train)[:, 1]
        y_prob_test = calibrated.predict_proba(X_test)[:, 1]

        train_metrics = self.calculate_metrics(y_train_full, y_pred_train, y_prob_train)
        test_metrics = self.calculate_metrics(y_test, y_pred_test, y_prob_test)

        self.log(
            f"   ✅ 训练指标: Acc={train_metrics['accuracy']:.4f}, F1={train_metrics['f1']:.4f}, AUC={train_metrics['roc_auc']:.4f}")
        self.log(
            f"   ✅ 测试指标: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['roc_auc']:.4f}")
        self.log(f"   ✅ PR-AUC: {test_metrics['pr_auc']:.4f}")

        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)

        self.models[stock] = {
            'model': calibrated,  # 保存校准后的
            'selected_features': selected_features,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': feature_importance,
            'best_params': best_params,
            'cv_score': cv_score,
            'scaler': scaler
        }

        # === CHANGED: 保存训练/测试概率（带日期索引），用于导出全期概率 ===
        self.results.setdefault(stock, {})
        self.results[stock]['y_prob_train'] = pd.Series(y_prob_train, index=X_train.index, name=stock)
        self.results[stock]['y_prob_test'] = pd.Series(y_prob_test, index=X_test.index, name=stock)

        # === CHANGED: skops dump 保存校准后的模型 ===
        dump({
            'model': calibrated,
            'scaler': scaler,
            'selected_features': selected_features,
            'best_params': best_params
        }, f'rf_v4_complete_{stock}.skops')
        return test_metrics

    def calculate_metrics(self, y_true, y_pred, y_prob):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, average='binary'),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'pr_auc': average_precision_score(y_true, y_prob)
        }

    # === NEW: 拼接 & 导出全期概率 ===
    def get_full_period_probability(self, stock: str):
        if stock not in self.results:
            return None
        ser_train = self.results[stock].get('y_prob_train')
        ser_test = self.results[stock].get('y_prob_test')
        parts = [s for s in [ser_train, ser_test] if s is not None]
        if not parts:
            return None
        full = pd.concat(parts).sort_index()
        return full.rename(stock)

    def dump_full_probs(self, tickers, horizon='5D', outdir='results'):
        os.makedirs(outdir, exist_ok=True)
        cols = []
        for s in tickers:
            ser = self.get_full_period_probability(s)
            if ser is not None:
                cols.append(ser)
        if not cols:
            self.log('⚠️ 没有可导出的 RF 概率。')
            return None
        df = pd.concat(cols, axis=1).sort_index()
        ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        out = os.path.join(outdir, f'rf_probs_{horizon}_{ts}.csv')
        df.to_csv(out, index_label='Date')
        self.log(f'✅ 导出 RF 全期概率：{out}')
        return out

    # === CHANGED: train_all 增加 horizon 参数，并在结束后导出全期概率 ===
    def train_all_stocks_v4(self, task_type='direction', horizon='5D'):
        self.log(f"\n🚀 开始训练所有股票的V4模型... ({task_type}-{horizon})")
        summary = []
        progress_bar = tqdm(self.banking_stocks, desc="训练进度", disable=not self.verbose)
        for stock in progress_bar:
            progress_bar.set_description(f"训练 {stock}")
            try:
                metrics = self.train_model_v4(stock, task_type=task_type, horizon=horizon)
                if metrics:
                    row = {'Stock': stock}
                    row.update(metrics)
                    summary.append(row)
                    progress_bar.set_postfix({
                        'Acc': f"{metrics['accuracy']:.3f}",
                        'F1': f"{metrics['f1']:.3f}",
                        'AUC': f"{metrics['roc_auc']:.3f}"
                    })
            except Exception as e:
                self.log(f"   ❌ {stock} 训练失败: {e}", "ERROR")
                continue

        if summary:
            df_summary = pd.DataFrame(summary)
            df_summary.to_csv('rf_v4_performance.csv', index=False)
            self.log(f"\n📊 Random Forest V4 最终结果:")
            self.log(df_summary.round(4).to_string(index=False))
            self.log(f"\n📈 平均性能 (±标准差):")
            for metric in ['accuracy', 'f1', 'roc_auc', 'pr_auc']:
                mean_val = df_summary[metric].mean()
                std_val = df_summary[metric].std()
                self.log(f"   {metric.upper()}: {mean_val:.4f} (±{std_val:.4f})")
            best_stock = df_summary.loc[df_summary['f1'].idxmax()]
            self.log(f"\n🏆 最佳表现: {best_stock['Stock']} (F1: {best_stock['f1']:.4f})")

        # 导出全期概率 CSV（与 XGB 同款结构）
        self.dump_full_probs(self.banking_stocks, horizon=horizon, outdir='results')
        return summary


def main():
    print("🚀 启动 Random Forest V4 终极优化版（已对齐 XGB 输出）...")
    rf_v4 = BankingRandomForestV4(
        top_k_features=50,
        n_splits=5,
        test_size=0.2,
        pre_rfe_features=200,
        nested_cv=True,
        verbose=True
    )

    if not rf_v4.load_data():
        return None

    # === CHANGED: 默认 horizon=5D 与 XGB 对齐 ===
    performance_summary = rf_v4.train_all_stocks_v4(task_type='direction', horizon='5D')

    print("\n" + "=" * 80)
    print("🎉 Random Forest V4 终极优化版训练完成!")
    print("=" * 80)
    print("📊 生成文件:")
    print("   📄 rf_v4_performance.csv - 最终性能报告")
    print("   📄 rf_v4_complete_*.skops - 完整模型包 (模型+预处理+特征)")
    print("   📄 results/rf_probs_5D_*.csv - 全期概率（与 XGB 对齐）")
    return rf_v4


if __name__ == "__main__":
    _ = main()