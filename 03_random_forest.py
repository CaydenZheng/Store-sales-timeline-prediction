# -*- coding: utf-8 -*-
"""
03_random_forest.py —— 随机森林模型
========================================
加载预处理后的数据，使用随机森林进行训练和预测。
使用时间序列划分验证集，评估 RMSLE 指标。

"""

import pandas as pd
import numpy as np
import os, warnings, time
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
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


def main():
    print("\n" + "█" * 60)
    print("  随机森林 Random Forest 模型训练")
    print("█" * 60 + "\n")

    # ---- 加载数据 ----
    print("[1] 加载预处理数据...")
    X = pd.read_pickle(os.path.join(PROCESSED_DIR, 'X_train.pkl'))
    y = pd.read_pickle(os.path.join(PROCESSED_DIR, 'y_train.pkl')).values.ravel()
    X_test = pd.read_pickle(os.path.join(PROCESSED_DIR, 'X_test.pkl'))
    test_ids = pd.read_pickle(os.path.join(PROCESSED_DIR, 'test_ids.pkl'))
    print(f"    X_train: {X.shape}, y_train: {y.shape}")
    print(f"    X_test:  {X_test.shape}")

    # ---- 时间序列切分（前90%训练，后10%验证）----
    print("\n[1.5] 时间序列切分 (90%/10%)...")
    # 按日期排序
    date_series = X.pop('date')
    X_test = X_test.drop(columns=['date'], errors='ignore')
    sort_idx = np.argsort(date_series.values)
    X_sorted = X.iloc[sort_idx].reset_index(drop=True)
    y_sorted = y[sort_idx]
    n_train = int(len(X_sorted) * 0.9)
    X_train = X_sorted.iloc[:n_train]
    X_val   = X_sorted.iloc[n_train:]
    y_train = y_sorted[:n_train]
    y_val   = y_sorted[n_train:]
    print(f"    训练集: {X_train.shape[0]:,} (日期: {date_series.iloc[sort_idx].iloc[0].date()} ~ {date_series.iloc[sort_idx].iloc[n_train-1].date()})")
    print(f"    验证集: {X_val.shape[0]:,}   (日期: {date_series.iloc[sort_idx].iloc[n_train].date()} ~ {date_series.iloc[sort_idx].iloc[-1].date()})")

    for df in [X_train, X_val, X_test]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

    # ---- 基础模型 ----
    print("\n[2] 训练基础随机森林模型...")
    t0 = time.time()
    rf_base = RandomForestRegressor(
        n_estimators=100, max_depth=25, min_samples_split=10,
        min_samples_leaf=5, max_features='sqrt', random_state=42, n_jobs=-1, verbose=1)
    rf_base.fit(X_train, y_train)
    print(f"    训练耗时: {time.time() - t0:.1f} 秒")

    y_val_pred_base = rf_base.predict(X_val)
    print(f"\n    [基础模型] RMSLE: {rmsle(y_val, y_val_pred_base):.5f}, "
          f"MAE: {mean_absolute_error(y_val, np.maximum(y_val_pred_base, 0)):.2f}")

    # ---- 调参 ----
    print("\n[3] 超参数调优 (RandomizedSearchCV, 子集 30 万)...")
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [15, 20, 25, 30, None],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [2, 5, 10],
        'max_features': ['sqrt', 'log2', 0.5],
    }
    rf_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_dist, n_iter=20, cv=3,
        scoring='neg_mean_squared_error', random_state=42, n_jobs=-1, verbose=1)
    n_subset = min(300000, len(X_train))
    idx_subset = np.random.choice(len(X_train), n_subset, replace=False)
    rf_search.fit(X_train.iloc[idx_subset], y_train[idx_subset])
    print(f"\n    最优参数: {rf_search.best_params_}")

    # ---- 最终模型 ----
    print("\n[4] 使用最优参数训练最终模型...")
    t0 = time.time()
    rf_final = RandomForestRegressor(**rf_search.best_params_, random_state=42, n_jobs=-1, verbose=1)
    rf_final.fit(X_train, y_train)
    print(f"    训练耗时: {time.time() - t0:.1f} 秒")

    y_val_pred = rf_final.predict(X_val)
    rmsle_val = rmsle(y_val, y_val_pred)
    mae_val = mean_absolute_error(y_val, np.maximum(y_val_pred, 0))
    print(f"\n    [调优后] RMSLE: {rmsle_val:.5f}, MAE: {mae_val:.2f}")

    # ---- 测试集预测 ----
    print("\n[5] 对测试集进行预测...")
    y_test_pred = rf_final.predict(X_test)
    y_test_pred = np.maximum(y_test_pred, 0)

    submission = pd.DataFrame({'id': test_ids['id'], 'sales': y_test_pred})
    sub_path = os.path.join(OUTPUT_DIR, 'submission_rf.csv')
    submission.to_csv(sub_path, index=False)
    print(f"    预测结果已保存: {sub_path}")

    model_path = os.path.join(OUTPUT_DIR, 'model_rf.pkl')
    joblib.dump(rf_final, model_path, compress=3)
    print(f"    模型已保存: {model_path}")

    # ---- 特征重要性图 ----
    print("\n[6] 绘制特征重要性图...")
    feature_names = X_train.columns.tolist()
    importances = rf_final.feature_importances_
    indices = np.argsort(importances)[::-1][:30]

    fig, ax = plt.subplots(figsize=(10, 10))
    colors = sns.color_palette("viridis", len(indices))
    ax.barh(range(len(indices)), importances[indices][::-1], color=colors[::-1], edgecolor='white')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(np.array(feature_names)[indices][::-1], fontsize=8)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('Random Forest — Top 30 Feature Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fpath = os.path.join(FIGS_DIR, 'rf_feature_importance.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    图表已保存: {fpath}")

    # ---- 预测 vs 真实散点图 ----
    print("[7] 绘制预测 vs 真实值散点图...")
    fig, ax = plt.subplots(figsize=(8, 8))
    sample_idx = np.random.choice(len(y_val), min(5000, len(y_val)), replace=False)
    ax.scatter(y_val[sample_idx], y_val_pred[sample_idx],
               alpha=0.3, s=10, c='steelblue', edgecolors='none')
    ax.plot([0, y_val.max()], [0, y_val.max()], 'r--', linewidth=2)
    ax.set_xlabel('True Sales', fontsize=12)
    ax.set_ylabel('Predicted Sales', fontsize=12)
    ax.set_title(f'Random Forest — Predicted vs True (RMSLE={rmsle_val:.4f})',
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tight_layout()
    fpath = os.path.join(FIGS_DIR, 'rf_pred_vs_true.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    图表已保存: {fpath}")

    # ---- 汇总 ----
    print("\n" + "=" * 60)
    print("  随机森林模型 — 最终结果")
    print("=" * 60)
    print(f"  基础模型 RMSLE: {rmsle(y_val, y_val_pred_base):.5f}")
    print(f"  调优后 RMSLE:   {rmsle_val:.5f}")
    print(f"  MAE:            {mae_val:.2f}")
    print(f"  测试集预测样本数: {len(y_test_pred):,}")
    print(f"  预测均值:         {y_test_pred.mean():.2f}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
