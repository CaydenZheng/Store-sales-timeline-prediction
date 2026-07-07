# -*- coding: utf-8 -*-
"""
05_lightgbm.py —— LightGBM 模型
===================================
加载预处理数据，使用 LightGBM 训练和预测。
+ early_stopping_rounds 自动收敛 + -RMSLE 评分 + 自动回退安全阀
"""

import pandas as pd
import numpy as np
import os, warnings, time
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import make_scorer
import lightgbm as lgb
import joblib

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
FIGS_DIR = os.path.join(BASE_DIR, 'figs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)


def rmsle(y_true, y_pred):
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))


def rmsle_score(estimator, X, y):
    y_pred = estimator.predict(X)
    y_pred = np.maximum(y_pred, 0)
    y_true = np.maximum(y, 0)
    return -np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))


def main():
    print("\n" + "█" * 60)
    print("  LightGBM 模型训练")
    print("█" * 60 + "\n")

    # ---- 加载数据 ----
    print("[1] 加载预处理数据...")
    X = pd.read_pickle(os.path.join(PROCESSED_DIR, 'X_train.pkl'))
    y = pd.read_pickle(os.path.join(PROCESSED_DIR, 'y_train.pkl')).values.ravel()
    X_test = pd.read_pickle(os.path.join(PROCESSED_DIR, 'X_test.pkl'))
    test_ids = pd.read_pickle(os.path.join(PROCESSED_DIR, 'test_ids.pkl'))
    print(f"    X_train: {X.shape}, X_test: {X_test.shape}")

    # ---- 时间序列切分 ----
    print("\n[1.5] 时间序列切分...")
    date_series = X.pop('date')
    X_test = X_test.drop(columns=['date'], errors='ignore')
    sort_idx = np.argsort(date_series.values)
    X_sorted = X.iloc[sort_idx].reset_index(drop=True)
    y_sorted = y[sort_idx]

    n_total = len(X_sorted)
    n_train = int(n_total * 0.9)
    n_es    = int(n_train * 0.85)

    X_tr  = X_sorted.iloc[:n_es]
    y_tr  = y_sorted[:n_es]
    X_es  = X_sorted.iloc[n_es:n_train]
    y_es  = y_sorted[n_es:n_train]
    X_val = X_sorted.iloc[n_train:]
    y_val = y_sorted[n_train:]
    print(f"    训练: {X_tr.shape[0]:,}  早停: {X_es.shape[0]:,}  验证: {X_val.shape[0]:,}")

    for df in [X_tr, X_es, X_val, X_test]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

    # ---- 基础模型 ----
    print("\n[2] 训练基础 LightGBM + early stopping...")
    t0 = time.time()
    lgb_base = lgb.LGBMRegressor(
        n_estimators=3000, max_depth=10, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1)
    lgb_base.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], eval_metric='rmse',
                 callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    nt_base = lgb_base.best_iteration_
    print(f"    训练耗时: {time.time() - t0:.1f} 秒, 实际树数: {nt_base}")

    y_val_pred_base = lgb_base.predict(X_val)
    rmsle_base = rmsle(y_val, y_val_pred_base)
    mae_base = mean_absolute_error(y_val, np.maximum(y_val_pred_base, 0))
    print(f"\n    [基础+ES] RMSLE: {rmsle_base:.5f}, MAE: {mae_base:.2f}")

    # ---- 调参 ----
    print("\n[3] 超参数调优 (RandomizedSearchCV, -RMSLE)...")
    param_dist = {
        'max_depth': [6, 8, 10, 12, 15],
        'learning_rate': [0.05, 0.08, 0.1],
        'num_leaves': [31, 63, 127, 255],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'min_child_samples': [10, 20, 50],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0, 0.1, 0.5, 1.0],
    }
    lgb_search = RandomizedSearchCV(
        lgb.LGBMRegressor(n_estimators=200, random_state=42, n_jobs=1, verbose=-1),
        param_dist, n_iter=20, cv=3,
        scoring=make_scorer(rmsle_score, greater_is_better=False),
        random_state=42, n_jobs=-1, verbose=1)
    n_sub = min(800000, len(X_tr))
    idx = np.random.choice(len(X_tr), n_sub, replace=False)
    lgb_search.fit(X_tr.iloc[idx], y_tr[idx])
    print(f"\n    CV 最优 RMSLE: {-lgb_search.best_score_:.5f}")
    print(f"    最优参数: {lgb_search.best_params_}")

    # ---- 最终模型 ----
    print("\n[4] 使用最优参数训练最终模型（early_stopping_rounds=50, 最多 3000 树）...")
    t0 = time.time()
    best_params = lgb_search.best_params_.copy()
    lgb_final = lgb.LGBMRegressor(
        **best_params, n_estimators=3000, random_state=42, n_jobs=-1, verbose=-1)
    lgb_final.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], eval_metric='rmse',
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    nt_final = lgb_final.best_iteration_
    print(f"    训练耗时: {time.time() - t0:.1f} 秒, 实际树数: {nt_final}")

    y_val_pred = lgb_final.predict(X_val)
    rmsle_val = rmsle(y_val, y_val_pred)
    mae_val = mean_absolute_error(y_val, np.maximum(y_val_pred, 0))
    print(f"\n    [调优后] RMSLE: {rmsle_val:.5f}, MAE: {mae_val:.2f}")

    # 安全阀
    if rmsle_val > rmsle_base:
        print(f"\n    ⚠️ 调优后 RMSLE ({rmsle_val:.5f}) > 基础 ({rmsle_base:.5f}), 回退使用基础模型")
        lgb_final = lgb_base
        rmsle_val = rmsle_base
        y_val_pred = y_val_pred_base

    # ---- 测试预测 ----
    print("\n[5] 对测试集进行预测...")
    y_test_pred = lgb_final.predict(X_test)
    y_test_pred = np.maximum(y_test_pred, 0)

    submission = pd.DataFrame({'id': test_ids['id'], 'sales': y_test_pred})
    sub_path = os.path.join(OUTPUT_DIR, 'submission_lgb.csv')
    submission.to_csv(sub_path, index=False)
    print(f"    预测结果已保存: {sub_path}")

    model_path = os.path.join(OUTPUT_DIR, 'model_lgb.pkl')
    joblib.dump(lgb_final, model_path, compress=3)
    print(f"    模型已保存: {model_path}")

    # ---- 特征重要性 ----
    print("\n[6] 绘制特征重要性图...")
    feature_names = X_tr.columns.tolist()
    importances = lgb_final.feature_importances_
    indices = np.argsort(importances)[::-1][:30]

    fig, ax = plt.subplots(figsize=(10, 10))
    colors = sns.color_palette("mako", len(indices))
    ax.barh(range(len(indices)), importances[indices][::-1], color=colors[::-1], edgecolor='white')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(np.array(feature_names)[indices][::-1], fontsize=8)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('LightGBM — Top 30 Feature Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fpath = os.path.join(FIGS_DIR, 'lgb_feature_importance.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    图表已保存: {fpath}")

    # ---- 预测 vs 真实 ----
    print("[7] 绘制预测 vs 真实值散点图...")
    fig, ax = plt.subplots(figsize=(8, 8))
    sample_idx = np.random.choice(len(y_val), min(5000, len(y_val)), replace=False)
    ax.scatter(y_val[sample_idx], y_val_pred[sample_idx],
               alpha=0.3, s=10, c='mediumseagreen', edgecolors='none')
    ax.plot([0, y_val.max()], [0, y_val.max()], 'r--', linewidth=2)
    ax.set_xlabel('True Sales', fontsize=12)
    ax.set_ylabel('Predicted Sales', fontsize=12)
    ax.set_title(f'LightGBM — Predicted vs True (RMSLE={rmsle_val:.4f})',
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tight_layout()
    fpath = os.path.join(FIGS_DIR, 'lgb_pred_vs_true.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    图表已保存: {fpath}")

    # ---- 汇总 ----
    print("\n" + "=" * 60)
    print("  LightGBM 模型 — 最终结果")
    print("=" * 60)
    print(f"  基础+ES  RMSLE: {rmsle_base:.5f}")
    print(f"  调优后 RMSLE:   {rmsle_val:.5f}")
    print(f"  MAE:            {mae_val:.2f}")
    print(f"  最终模型树数:   {nt_final}")
    print(f"  测试集预测均值: {y_test_pred.mean():.2f}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
