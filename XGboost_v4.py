# 在 xgboost_v4.py 文件的最开头（import之前）添加以下代码：

import os
import warnings
import sys

# 抑制所有FutureWarning和UserWarning
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 抑制XGBoost的系统警告
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 如果有tensorflow相关警告

# 重定向stderr以减少输出
class SuppressOutput:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import joblib
from skops.io import dump
from tqdm.auto import tqdm
import warnings
from scipy.stats import uniform, randint
import traceback

warnings.filterwarnings('ignore')


class BankingXGBoostV4:
    def __init__(self, top_k_features=200, n_splits=5, test_size=0.2,
                 pre_rfe_features=200, nested_cv=False, verbose=True):
        """
        XGBoost V4 - 终极优化版
        
        Parameters:
        - top_k_features: 最终选择的特征数量
        - n_splits: 时序交叉验证折数
        - test_size: 最终测试集比例
        - pre_rfe_features: RFE前的预筛选特征数
        - nested_cv: 是否使用嵌套交叉验证
        - verbose: 是否显示详细进度
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
        self.results = {}
        
        # 两阶段超参数搜索
        self.coarse_param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_alpha': [0, 0.1],
            'reg_lambda': [1],
            'min_child_weight': [1]
        }
        
        self.fine_param_grid = {}  # 动态生成

    def log(self, message, level="INFO"):
        """日志输出"""
        if self.verbose:
            print(f"[{level}] {message}")

    def load_data(self, feature_path='banking_returns.csv', target_path='banking_targets_ai.csv'):
        self.log("=" * 80)
        self.log("🚀 XGBoost V4 - 终极优化版")
        self.log("=" * 80)
        self.log("📊 1. 加载数据...")
        
        self.features = pd.read_csv(feature_path, index_col='Date', parse_dates=True)
        self.targets = pd.read_csv(target_path, index_col='Date', parse_dates=True)
        
        self.log(f"   ✅ 特征数据: {self.features.shape}")
        self.log(f"   ✅ 目标数据: {self.targets.shape}")
        self.log(f"   ✅ 时间范围: {self.features.index[0]} 到 {self.features.index[-1]}")
        self.log(f"   ✅ 配置: Top-{self.top_k_features}特征, {self.n_splits}折CV, " +
                 f"预筛选{self.pre_rfe_features}特征, 嵌套CV={'开启' if self.nested_cv else '关闭'}")
        return True

    def prepare_data_no_leakage(self, stock, task_type='direction', horizon='1D'):
        """无数据泄漏的数据准备"""
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
        分层特征选择: SelectKBest -> XGB Importance -> RFE
        """
        n_samples = len(X_train)
        original_features = X_train.shape[1]
        
        self.log(f"   🔍 开始分层特征选择 (原始特征: {original_features})")
        
        # 第一层: 统计筛选 (快速)
        if original_features > self.pre_rfe_features:
            selector_stat = SelectKBest(score_func=f_classif, k=self.pre_rfe_features)
            X_stat = selector_stat.fit_transform(X_train, y_train)
            selected_features_stat = X_train.columns[selector_stat.get_support()].tolist()
            self.log(f"   📉 统计筛选: {original_features} -> {len(selected_features_stat)}")
        else:
            X_stat = X_train
            selected_features_stat = X_train.columns.tolist()
            self.log(f"   📉 跳过统计筛选 (特征数已少于阈值)")
        
        # 第二层: XGBoost重要性
        xgb_selector = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=4,
            eval_metric='logloss',
            verbosity=0
        )
        xgb_selector.fit(X_stat, y_train)
        
        # 选择重要性最高的特征
        importances = pd.Series(xgb_selector.feature_importances_, index=selected_features_stat)
        xgb_top_features = importances.nlargest(min(self.top_k_features * 2, len(selected_features_stat))).index.tolist()
        self.log(f"   🚀 XGB重要性筛选: {len(selected_features_stat)} -> {len(xgb_top_features)}")
        
        # 第三层: RFE (仅在合理样本数下使用)
        if n_samples >= 500 and len(xgb_top_features) > self.top_k_features:
            self.log(f"   🔄 执行RFE (样本数充足: {n_samples})")
            X_xgb = X_train[xgb_top_features]
            
            rfe = RFE(
                estimator=xgb.XGBClassifier(
                    n_estimators=50,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=4,
                    eval_metric='logloss',
                    verbosity=0
                ),
                n_features_to_select=self.top_k_features,
                step=0.1
            )
            
            with tqdm(desc="RFE进度", disable=not self.verbose) as pbar:
                rfe.fit(X_xgb, y_train)
                pbar.update(1)
            
            final_features = X_xgb.columns[rfe.support_].tolist()
            self.log(f"   🎯 RFE筛选: {len(xgb_top_features)} -> {len(final_features)}")
        else:
            if n_samples < 500:
                self.log(f"   ⚠️ RFE跳过 (样本数不足: {n_samples})")
            elif len(xgb_top_features) <= self.top_k_features:
                self.log(f"   ⚠️ RFE跳过 (特征数已达标: {len(xgb_top_features)})")
            final_features = xgb_top_features[:self.top_k_features]
        
        self.log(f"   ✅ 最终选择特征: {len(final_features)}")
        return final_features

    def nested_cross_validation(self, X_train, y_train):
        """
        嵌套交叉验证: 外层评估模型性能，内层调优超参数
        """
        outer_scores = []
        best_params_list = []
        
        # 外层时序分割
        outer_cv = TimeSeriesSplit(n_splits=5)
        
        self.log(f"   🔄 嵌套CV: {self.n_splits}折外层验证")
        
        for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X_train)):
            if self.verbose:
                print(f"      折数 {fold_idx + 1}/{self.n_splits}")
            
            # 分割数据
            X_train_fold = X_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_train_fold = y_train.iloc[train_idx]
            y_val_fold = y_train.iloc[val_idx]
            
            # 内层时序交叉验证进行超参数调优
            inner_cv = TimeSeriesSplit(n_splits=5)
            
            # 粗搜索
            model_coarse = xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss', verbosity=0)
            search_coarse = RandomizedSearchCV(
                estimator=model_coarse,
                param_distributions=self.coarse_param_grid,
                n_iter=20,
                scoring='f1',
                cv=inner_cv,
                random_state=42,
                n_jobs=4,
                verbose=0
            )
            search_coarse.fit(X_train_fold, y_train_fold)
            
            # 细搜索 (基于粗搜索结果)
            best_coarse = search_coarse.best_params_
            fine_grid = self.generate_fine_grid(best_coarse)
            
            model_fine = xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss', verbosity=0)
            search_fine = RandomizedSearchCV(
                estimator=model_fine,
                param_distributions=fine_grid,
                n_iter=20,
                scoring='f1',
                cv=inner_cv,
                random_state=42,
                n_jobs=4,
                verbose=0
            )
            search_fine.fit(X_train_fold, y_train_fold)
            
            # 在验证集上评估
            best_model = search_fine.best_estimator_
            y_pred_val = best_model.predict(X_val_fold)
            fold_score = f1_score(y_val_fold, y_pred_val)
            
            outer_scores.append(fold_score)
            best_params_list.append(search_fine.best_params_)
            
            if self.verbose:
                print(f"         F1: {fold_score:.4f}, 参数: {search_fine.best_params_}")
        
        # 返回平均最佳参数
        avg_score = np.mean(outer_scores)
        std_score = np.std(outer_scores)
        
        self.log(f"   ✅ 嵌套CV结果: F1 = {avg_score:.4f} (±{std_score:.4f})")
        
        # 选择最佳参数 (简化：选择第一个，实际可以用投票)
        best_params = best_params_list[0] if best_params_list else self.coarse_param_grid
        
        return best_params, avg_score

    def generate_fine_grid(self, coarse_best):
        """基于粗搜索结果生成精细搜索网格"""
        fine_grid = {}
        
        # n_estimators 细搜索
        base_n_est = coarse_best.get('n_estimators', 200)
        fine_grid['n_estimators'] = [max(50, base_n_est - 50), base_n_est, base_n_est + 50]
        
        # max_depth 细搜索
        base_depth = coarse_best.get('max_depth', 6)
        fine_grid['max_depth'] = [max(3, base_depth - 1), base_depth, base_depth + 1]
        
        # learning_rate 细搜索
        base_lr = coarse_best.get('learning_rate', 0.1)
        fine_grid['learning_rate'] = [max(0.01, base_lr - 0.05), base_lr, base_lr + 0.05]
        
        # 保持最佳的其他参数
        fine_grid['subsample'] = [coarse_best.get('subsample', 0.8)]
        fine_grid['colsample_bytree'] = [coarse_best.get('colsample_bytree', 0.8)]
        fine_grid['reg_alpha'] = [coarse_best.get('reg_alpha', 0)]
        fine_grid['reg_lambda'] = [coarse_best.get('reg_lambda', 1)]
        fine_grid['min_child_weight'] = [coarse_best.get('min_child_weight', 1)]
        
        return fine_grid

    def train_model_v4(self, stock, task_type='direction', horizon='1D'):
        """训练V4模型 - 终极优化版"""
        self.log(f"\n🚀 2. 训练 {stock} 模型 ({task_type}-{horizon})...")
        
        # 准备数据
        X, y = self.prepare_data_no_leakage(stock, task_type, horizon)
        if X is None:
            return None
        
        # 最终测试集分割
        split_idx = int(len(X) * (1 - self.test_size))
        X_train_full, X_test_full = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train_full, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        self.log(f"   📊 训练集: {X_train_full.shape[0]} 样本")
        self.log(f"   📊 测试集: {X_test_full.shape[0]} 样本")
        
        # 标准化 (只在训练集上fit)
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
        
        # 分层特征选择
        selected_features = self.hierarchical_feature_selection(X_train_scaled, y_train_full, stock)
        self.feature_selectors[stock] = selected_features
        
        X_train = X_train_scaled[selected_features]
        X_test = X_test_scaled[selected_features]
        
        # 超参数优化
        if self.nested_cv:
            best_params, cv_score = self.nested_cross_validation(X_train, y_train_full)
        else:
            # 标准方法
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            model = xgb.XGBClassifier(random_state=42, n_jobs=4, eval_metric='logloss', verbosity=0, tree_method='hist', use_label_encoder=False)
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=self.coarse_param_grid,
                n_iter=30,
                scoring='f1',
                cv=tscv,
                random_state=42,
                n_jobs=1,
                verbose=0
            )
            search.fit(X_train, y_train_full)
            best_params = search.best_params_
            cv_score = search.best_score_
        
        self.log(f"   ✅ 最佳参数: {best_params}")
        self.log(f"   ✅ CV F1得分: {cv_score:.4f}")
        
        # 训练最终模型
        final_model = xgb.XGBClassifier(**best_params, random_state=42, n_jobs=4, eval_metric='logloss', verbosity=0, tree_method='hist', use_label_encoder=False)
        final_model.fit(X_train, y_train_full)
        
        # 预测和评估
        y_pred_train = final_model.predict(X_train)
        y_pred_test = final_model.predict(X_test)
        y_prob_train = final_model.predict_proba(X_train)[:, 1]
        y_prob_test = final_model.predict_proba(X_test)[:, 1]
        
        # 计算指标
        train_metrics = self.calculate_metrics(y_train_full, y_pred_train, y_prob_train)
        test_metrics = self.calculate_metrics(y_test, y_pred_test, y_prob_test)
        
        self.log(
            f"   ✅ 训练指标: Acc={train_metrics['accuracy']:.4f}, F1={train_metrics['f1']:.4f}, AUC={train_metrics['roc_auc']:.4f}")
        self.log(
            f"   ✅ 测试指标: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['roc_auc']:.4f}")
        self.log(f"   ✅ PR-AUC: {test_metrics['pr_auc']:.4f}")
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 保存结果
        self.models[stock] = {
            'model': final_model,
            'selected_features': selected_features,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': feature_importance,
            'best_params': best_params,
            'cv_score': cv_score,
            'scaler': scaler
        }
        
        # 保存模型
        dump({
            'model': final_model,
            'scaler': scaler,
            'selected_features': selected_features,
            'best_params': best_params
        }, f'xgb_v4_complete_{stock}.skops')
        
        return test_metrics

    def calculate_metrics(self, y_true, y_pred, y_prob):
        """计算评估指标"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, average='binary'),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'pr_auc': average_precision_score(y_true, y_prob)
        }

    def train_all_stocks_v4(self):
        """训练所有股票的V4模型"""
        self.log(f"\n🚀 开始训练所有股票的XGBoost V4模型...")
        
        summary = []
        
        # 使用tqdm显示总体进度
        progress_bar = tqdm(self.banking_stocks, desc="训练进度", disable=not self.verbose)
        
        for stock in progress_bar:
            progress_bar.set_description(f"训练 {stock}")
            
            try:
                metrics = self.train_model_v4(stock)
                if metrics:
                    result_row = {'Stock': stock}
                    result_row.update(metrics)
                    summary.append(result_row)
                    
                    # 更新进度条后缀信息
                    progress_bar.set_postfix({
                        'Acc': f"{metrics['accuracy']:.3f}",
                        'F1': f"{metrics['f1']:.3f}",
                        'AUC': f"{metrics['roc_auc']:.3f}"
                    })
                    
            except Exception as e:
                self.log(f"   ❌ {stock} 训练失败: {e}", "ERROR")
                traceback.print_exc()
                continue
        
        # 保存和分析结果
        if summary:
            df_summary = pd.DataFrame(summary)
            df_summary.to_csv('xgb_v4_performance.csv', index=False)
            
            self.log(f"\n📊 XGBoost V4 最终结果:")
            self.log(df_summary.round(4).to_string(index=False))
            
            # 统计摘要
            self.log(f"\n📈 平均性能 (±标准差):")
            for metric in ['accuracy', 'f1', 'roc_auc', 'pr_auc']:
                mean_val = df_summary[metric].mean()
                std_val = df_summary[metric].std()
                self.log(f"   {metric.upper()}: {mean_val:.4f} (±{std_val:.4f})")
            
            # 最佳表现股票
            best_stock = df_summary.loc[df_summary['f1'].idxmax()]
            self.log(f"\n🏆 最佳表现: {best_stock['Stock']} (F1: {best_stock['f1']:.4f})")
        
        return summary


def main():
    """主函数"""
    print("🚀 启动 XGBoost V4 终极优化版...")
    
    # 创建模型
    xgb_v4 = BankingXGBoostV4(
        top_k_features=50,
        n_splits=5,
        test_size=0.2,
        pre_rfe_features=200,  # RFE前预筛选特征数
        nested_cv=False,  # 启用嵌套交叉验证
        verbose=True  # 显示详细进度
    )
    
    # 加载数据
    if not xgb_v4.load_data():
        return None
    
    # 训练所有股票
    performance_summary = xgb_v4.train_all_stocks_v4()
    
    print("\n" + "=" * 80)
    print("🎉 XGBoost V4 终极优化版训练完成!")
    print("=" * 80)
    print("📊 生成文件:")
    print("   📄 xgb_v4_performance.csv - 最终性能报告")
    print("   📄 xgb_v4_complete_*.skops - 完整模型包 (模型+预处理+特征)")
    
    return xgb_v4


if __name__ == "__main__":
    model_v4 = main()