# -*- coding: utf-8 -*-
"""
06_ensemble.py —— 模型融合 (Stacking & Weighted Averaging)
==============================================================
将随机森林、XGBoost、LightGBM 三个基模型进行融合：
  1. 加权平均 (Weighted Average)
  2. Stacking（二层集成，Ridge/Lasso 作为元学习器）
对比各模型和融合方案的效果，输出最优提交文件。
"""

import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
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


def load_models():
    """加载已训练的模型"""
    print("[1] 加载已训练模型...")
    models = {}
    model_files = {
        'Random Forest': os.path.join(OUTPUT_DIR, 'model_rf.pkl'),
        'XGBoost':       os.path.join(OUTPUT_DIR, 'model_xgb.pkl'),
        'LightGBM':      os.path.join(OUTPUT_DIR, 'model_lgb.pkl'),
    }
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
            print(f"    ✓ {name} 加载成功")
        else:
            print(f"    ✗ {name} 文件不存在: {path}")
    if len(models) == 0:
        raise FileNotFoundError("未找到任何模型文件，请先运行 03/04/05 脚本！")
    return models


def main():
    print("\n" + "█" * 60)
    print("  模型融合 Ensemble (Stacking & Weighted Averaging)")
    print("█" * 60 + "\n")

    # ---- 加载模型 ----
    models = load_models()

    # ---- 加载数据 ----
    print("\n[2] 加载预处理数据...")
    X = pd.read_pickle(os.path.join(PROCESSED_DIR, 'X_train.pkl'))
    y = pd.read_pickle(os.path.join(PROCESSED_DIR, 'y_train.pkl')).values.ravel()
    X_test = pd.read_pickle(os.path.join(PROCESSED_DIR, 'X_test.pkl'))
    test_ids = pd.read_pickle(os.path.join(PROCESSED_DIR, 'test_ids.pkl'))

    # ---- 时间序列切分 ----
    print("\n[2.5] 时间序列切分...")
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
    for df in [X_train, X_val, X_test]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
    print(f"    训练集: {X_train.shape[0]:,}  验证集: {X_val.shape[0]:,}")

    # ---- 验证集上各模型预测 ----
    print("\n[3] 在验证集上获取各模型预测...")
    model_names = list(models.keys())
    val_preds = {}
    single_scores = {}
    for name in model_names:
        yp = models[name].predict(X_val)
        yp = np.maximum(yp, 0)
        val_preds[name] = yp
        single_scores[name] = rmsle(y_val, yp)
        print(f"    {name:15s}  RMSLE: {single_scores[name]:.5f}")

    # ---- 1) 加权平均 ----
    print("\n" + "-" * 40)
    print("  [融合方案1] 加权平均 (Weighted Averaging)")
    print("-" * 40)

    n = len(model_names)

    # 等权平均
    yp_eq = np.zeros(len(y_val))
    for nm in model_names:
        yp_eq += val_preds[nm] / n
    r_eq = rmsle(y_val, yp_eq)
    print(f"    等权平均              RMSLE: {r_eq:.5f}")

    # RMSLE 反比加权
    inv_scores = np.array([1.0 / (single_scores[nm] + 1e-8) for nm in model_names])
    w_rmsle = inv_scores / inv_scores.sum()
    yp_w = np.zeros(len(y_val))
    for i, nm in enumerate(model_names):
        yp_w += w_rmsle[i] * val_preds[nm]
    r_w = rmsle(y_val, yp_w)
    print(f"    RMSLE 加权            RMSLE: {r_w:.5f}")
    print(f"    各模型权重: {dict(zip(model_names, w_rmsle.round(4)))}")

    # 网格搜索最优权重
    print("\n    网格搜索最优权重...")
    best_r = float('inf')
    best_w = None
    for w1 in np.arange(0, 1.01, 0.05):
        for w2 in np.arange(0, 1.01 - w1, 0.05):
            w3 = 1 - w1 - w2
            if w3 < 0: continue
            ws = [w1, w2, w3][:n]
            yp = np.zeros(len(y_val))
            for i, nm in enumerate(model_names):
                yp += ws[i] * val_preds[nm]
            sc = rmsle(y_val, yp)
            if sc < best_r:
                best_r = sc
                best_w = ws
    print(f"    网格搜索最优权重: {dict(zip(model_names, np.round(best_w, 4)))}")
    print(f"    最优加权 RMSLE:    {best_r:.5f}")

    # ---- 2) Stacking ----
    print("\n" + "-" * 40)
    print("  [融合方案2] Stacking（二层集成）")
    print("-" * 40)

    meta_val = np.column_stack([val_preds[nm] for nm in model_names])

    print("\n    [2a] 使用 Ridge 作为元学习器...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(meta_val, y_val)
    yp_ridge = ridge.predict(meta_val)
    r_ridge = rmsle(y_val, yp_ridge)
    print(f"    Ridge Stacking RMSLE:  {r_ridge:.5f}")
    print(f"    Ridge 系数: {dict(zip(model_names, ridge.coef_.round(4)))}")

    print("\n    [2b] 使用 Lasso 作为元学习器...")
    lasso = Lasso(alpha=0.01, max_iter=5000)
    lasso.fit(meta_val, y_val)
    yp_lasso = lasso.predict(meta_val)
    r_lasso = rmsle(y_val, yp_lasso)
    print(f"    Lasso Stacking RMSLE:  {r_lasso:.5f}")
    print(f"    Lasso 系数: {dict(zip(model_names, lasso.coef_.round(4)))}")

    # ---- 汇总比较 ----
    all_methods = {
        f'{nm}': sc for nm, sc in single_scores.items()
    }
    all_methods.update({
        'Equal Weight': r_eq,
        'RMSLE Weight': r_w,
        'Grid Weight': best_r,
        'Ridge Stack': r_ridge,
        'Lasso Stack': r_lasso,
    })
    best_method = min(all_methods, key=all_methods.get)
    best_score = all_methods[best_method]
    print(f"\n    >>> 最优融合方案: {best_method} (RMSLE={best_score:.5f})")

    # ---- 最终测试集预测 ----
    print("\n[4] 生成最终集成预测...")

    # 各模型在测试集上的预测
    test_preds = {}
    for nm in model_names:
        test_preds[nm] = np.maximum(models[nm].predict(X_test), 0)

    meta_test = np.column_stack([test_preds[nm] for nm in model_names])

    if best_method == 'Ridge Stack':
        y_test_final = ridge.predict(meta_test)
    elif best_method == 'Lasso Stack':
        y_test_final = lasso.predict(meta_test)
    elif best_method == 'Grid Weight':
        y_test_final = np.zeros(len(X_test))
        for i, nm in enumerate(model_names):
            y_test_final += best_w[i] * test_preds[nm]
    elif best_method == 'RMSLE Weight':
        y_test_final = np.zeros(len(X_test))
        for i, nm in enumerate(model_names):
            y_test_final += w_rmsle[i] * test_preds[nm]
    elif best_method == 'Equal Weight':
        y_test_final = np.zeros(len(X_test))
        for nm in model_names:
            y_test_final += test_preds[nm] / n
    else:
        best_model_name = best_method.split()[0] if ' ' in best_method else best_method
        y_test_final = test_preds.get(best_model_name, test_preds[model_names[0]])

    y_test_final = np.maximum(y_test_final, 0)

    final_submission = pd.DataFrame({'id': test_ids['id'], 'sales': y_test_final})
    final_path = os.path.join(OUTPUT_DIR, 'submission_final_ensemble.csv')
    final_submission.to_csv(final_path, index=False)
    print(f"    ✅ 最终提交文件已保存: {final_path}")

    # ---- 对比可视化 ----
    print("\n[5] 绘制融合对比图...")
    methods = list(all_methods.keys())
    scores = list(all_methods.values())
    n_models = len(model_names)
    colors_list = ['#3498db'] * n_models + ['#e67e22'] * 3 + ['#2ecc71'] * 2

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(methods)), scores, color=colors_list[:len(methods)], edgecolor='white')
    best_idx = scores.index(min(scores))
    bars[best_idx].set_color('#e74c3c')
    bars[best_idx].set_edgecolor('darkred')
    bars[best_idx].set_linewidth(2)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('RMSLE (lower is better)', fontsize=12)
    ax.set_title('Model Comparison — RMSLE Scores (Validation Set)', fontsize=14, fontweight='bold')
    for i, s in enumerate(scores):
        ax.text(i, s + 0.001, f'{s:.5f}', ha='center', fontsize=8)
    plt.tight_layout()
    fpath = os.path.join(FIGS_DIR, 'ensemble_comparison.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    图表已保存: {fpath}")

    # ---- 模型预测一致性 ----
    if len(model_names) >= 2:
        print("[6] 绘制模型间预测一致性散点图...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()
        names = list(model_names)
        pairs = [
            (test_preds[names[0]], test_preds[names[1]], f'{names[0]} vs {names[1]}'),
            (test_preds[names[0]], test_preds[names[2]], f'{names[0]} vs {names[2]}'),
            (test_preds[names[1]], test_preds[names[2]], f'{names[1]} vs {names[2]}'),
            (test_preds[names[0]], y_test_final, f'{names[0]} vs Final Ensemble'),
        ]
        for i, (x, yp, title) in enumerate(pairs):
            s = np.random.choice(len(x), min(3000, len(x)), replace=False)
            axes[i].scatter(x[s], yp[s], alpha=0.2, s=5, edgecolors='none')
            mx = max(x.max(), yp.max())
            axes[i].plot([0, mx], [0, mx], 'r--', lw=1)
            axes[i].set_xscale('log'); axes[i].set_yscale('log')
            axes[i].set_title(title, fontsize=11)
        plt.suptitle('Cross-Model Prediction Agreement (Test Set)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        fpath = os.path.join(FIGS_DIR, 'ensemble_cross_pred.png')
        plt.savefig(fpath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    图表已保存: {fpath}")

    # ---- 最终汇总 ----
    print("\n" + "=" * 60)
    print("  集成模型融合 — 最终结果汇总")
    print("=" * 60)
    print(f"  {'方法':<25s} {'RMSLE':>10s}")
    print("  " + "-" * 37)
    for method, score in all_methods.items():
        marker = "  ← BEST" if method == best_method else ""
        print(f"  {method:<25s} {score:>10.5f}{marker}")
    print("=" * 60)
    print(f"\n  最终提交文件: {final_path}")
    print(f"  预测均值: {y_test_final.mean():.2f}")
    print(f"  预测中位数: {np.median(y_test_final):.2f}\n")


if __name__ == '__main__':
    main()
