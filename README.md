# 门店销售时序预测

> DM3 班第 2 组 · 数据挖掘课程项目

基于 **随机森林 / XGBoost / LightGBM** 三大模型及其融合，对 Favorita 超市门店销量进行时序预测。

---

## 📁 项目结构

```
├── 01_eda.py              # 数据探索性分析（生成可视化图表）
├── 02_preprocessing.py    # 数据预处理 & 特征工程
├── 03_random_forest.py    # 随机森林模型训练与评估
├── 04_xgboost.py          # XGBoost 模型训练与评估
├── 05_lightgbm.py         # LightGBM 模型训练与评估
├── 06_ensemble.py         # 模型融合（加权平均 / Stacking）
├── dataset/               # 原始数据（不上传，见下方说明）
├── figs/                  # 分析图表
├── output/                # 模型文件 & 提交文件（不上传）
├── processed/             # 预处理后的特征（不上传）
├── requirements.txt       # Python 依赖
└── README.md
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 获取数据

本项目数据来自 Kaggle 竞赛 **Store Sales - Time Series Forecasting**：

🔗 [https://www.kaggle.com/competitions/store-sales-time-series-forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

下载后将以下文件放入 `dataset/` 目录：

| 文件 | 说明 |
|------|------|
| `train.csv` | 训练集 |
| `test.csv` | 测试集 |
| `stores.csv` | 门店信息 |
| `oil.csv` | 原油价格 |
| `holidays_events.csv` | 节假日事件 |
| `transactions.csv` | 交易量 |
| `sample_submission.csv` | 提交样例 |

### 3. 按顺序运行

```bash
python 01_eda.py           # 数据探索 → figs/
python 02_preprocessing.py # 特征工程 → processed/
python 03_random_forest.py # 随机森林   → output/
python 04_xgboost.py       # XGBoost     → output/
python 05_lightgbm.py      # LightGBM    → output/
python 06_ensemble.py      # 模型融合   → output/
```

## 🛠 技术栈

- **数据处理**: pandas, numpy
- **可视化**: matplotlib, seaborn
- **模型**: scikit-learn (RandomForest), XGBoost, LightGBM
- **融合**: 加权平均 + Stacking (Ridge/Lasso)

## 📊 模型一览

| 脚本 | 模型 | 核心思路 |
|------|------|----------|
| `03_random_forest.py` | Random Forest | 树模型基线，对异常值鲁棒 |
| `04_xgboost.py` | XGBoost | 梯度提升，支持缺失值 |
| `05_lightgbm.py` | LightGBM | 直方图算法，训练速度快 |
| `06_ensemble.py` | Stacking + WAvg | 多模型融合，降低方差 |

## 👥 组员

- （请填写组员姓名）

## 📝 致谢

数据源自 Kaggle **Store Sales - Time Series Forecasting** 竞赛，由 Corporación Favorita 提供。
