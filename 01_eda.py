# -*- coding: utf-8 -*-
"""
01_eda.py —— 数据探索性分析 (EDA)
=====================================
对 Favorita 超市销售数据集进行全面探索性分析，生成多张 seaborn 可视化图表。
图表保存在 ./figs/ 目录下。
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 若在无 GUI 服务器上运行，使用 Agg 后端
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings
warnings.filterwarnings('ignore')

# ---------- 中文字体设置 ----------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("Set2")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(BASE_DIR, 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)


def load_data():
    """加载所有数据文件"""
    print("=" * 60)
    print("正在加载数据...")
    train = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'),
                        parse_dates=['date'], dtype={'store_nbr': 'int8', 'onpromotion': 'int32'})
    test  = pd.read_csv(os.path.join(BASE_DIR, 'test.csv'),
                        parse_dates=['date'], dtype={'store_nbr': 'int8', 'onpromotion': 'int32'})
    stores = pd.read_csv(os.path.join(BASE_DIR, 'stores.csv'))
    oil = pd.read_csv(os.path.join(BASE_DIR, 'oil.csv'), parse_dates=['date'])
    holidays = pd.read_csv(os.path.join(BASE_DIR, 'holidays_events.csv'), parse_dates=['date'])
    transactions = pd.read_csv(os.path.join(BASE_DIR, 'transactions.csv'), parse_dates=['date'])
    submission = pd.read_csv(os.path.join(BASE_DIR, 'sample_submission.csv'))

    print(f"  train:          {train.shape}")
    print(f"  test:           {test.shape}")
    print(f"  stores:         {stores.shape}")
    print(f"  oil:            {oil.shape}")
    print(f"  holidays:       {holidays.shape}")
    print(f"  transactions:   {transactions.shape}")
    print(f"  submission:     {submission.shape}")
    print("=" * 60)
    return train, test, stores, oil, holidays, transactions, submission


def basic_info(train, test, oil, holidays, stores):
    """打印基本信息与缺失值"""
    print("\n【train 缺失值统计】")
    print(train.isnull().sum())
    print(f"\n  日期范围: {train['date'].min()} ~ {train['date'].max()}")
    print(f"  商店数量: {train['store_nbr'].nunique()}")
    print(f"  产品品类数: {train['family'].nunique()}")
    print(f"  品类列表: {sorted(train['family'].unique())}")

    print("\n【test 缺失值统计】")
    print(test.isnull().sum())
    print(f"  日期范围: {test['date'].min()} ~ {test['date'].max()}")
    print(f"  商店数量: {test['store_nbr'].nunique()}")

    print(f"\n【oil 缺失值】: {oil['dcoilwtico'].isnull().sum()} / {len(oil)}")
    print(f"  日期范围: {oil['date'].min()} ~ {oil['date'].max()}")

    print(f"\n【holidays 类型分布】")
    print(holidays['type'].value_counts())
    print(f"\n  locale 分布:\n{holidays['locale'].value_counts()}")

    print(f"\n【stores 信息】")
    print(stores.describe(include='all'))
    print(f"\n  type 分布:\n{stores['type'].value_counts()}")
    print(f"  cluster 分布:\n{stores['cluster'].value_counts()}")


# ======================== 可视化函数 ========================

def plot_sales_time_series(train):
    """图1: 每日总销售额时间序列"""
    daily_sales = train.groupby('date')['sales'].sum().reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(16, 8))

    # 整体趋势
    axes[0].plot(daily_sales['date'], daily_sales['sales'],
                 color='steelblue', linewidth=0.5, alpha=0.9)
    axes[0].set_title('Daily Total Sales Over Time (2013–2017)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Total Sales')
    axes[0].set_xlabel('')

    # 2016年地震前后（2016年4月附近）
    eq_mask = (daily_sales['date'] >= '2016-01-01') & (daily_sales['date'] <= '2016-07-31')
    axes[1].plot(daily_sales.loc[eq_mask, 'date'], daily_sales.loc[eq_mask, 'sales'],
                 color='coral', linewidth=0.7)
    axes[1].axvline(x=pd.Timestamp('2016-04-16'), color='red', linestyle='--',
                    linewidth=2, label='Earthquake (Apr 16)')
    axes[1].set_title('Daily Sales Around 2016 Earthquake', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Total Sales')
    axes[1].legend()

    plt.tight_layout()
    fpath = os.path.join(FIGS_DIR, '01_sales_time_series.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [保存] {fpath}")


def plot_sales_by_family(train):
    """图2: 各产品品类总销售额（柱状图）"""
    family_sales = train.groupby('family')['sales'].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = sns.color_palette("viridis", len(family_sales))
    bars = ax.bar(range(len(family_sales)), family_sales.values / 1e6, color=colors, edgecolor='white')

    ax.set_xticks(range(len(family_sales)))
    ax.set_xticklabels(family_sales.index, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Total Sales (Millions)', fontsize=12)
    ax.set_title('Total Sales by Product Family', fontsize=14, fontweight='bold')

    for i, v in enumerate(family_sales.values / 1e6):
        ax.text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=7)

    plt.tight_layout()
    fpath = os.path.join(FIGS_DIR, '02_sales_by_family.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [保存] {fpath}")


def plot_sales_by_store_type(train, stores):
    """图3: 不同商店类型的每日销售额分布（箱线图）"""
    merged = train.merge(stores[['store_nbr', 'type', 'cluster']], on='store_nbr', how='left')
    daily = merged.groupby(['date', 'type'])['sales'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=daily, x='type', y='sales', palette='Set2', ax=ax,
                order=sorted(daily['type'].unique()))
    ax.set_yscale('log')
    ax.set_title('Daily Sales Distribution by Store Type (log scale)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Store Type', fontsize=12)
    ax.set_ylabel('Daily Sales (log scale)', fontsize=12)

    plt.tight_layout()
    fpath = os.path.join(FIGS_DIR, '03_sales_by_store_type.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [保存] {fpath}")


def plot_promotion_vs_sales(train):
    """图4: 促销商品数量 vs 销售额 散点图"""
    # 采样以减少绘图负担
    sample = train.sample(n=50000, random_state=42)
    sample = sample[sample['onpromotion'] > 0]  # 只看有促销的

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=sample, x='onpromotion', y='sales', alpha=0.3,
                    s=10, color='steelblue', ax=ax)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('On Promotion Count (log)', fontsize=12)
    ax.set_ylabel('Sales (log)', fontsize=12)
    ax.set_title('Promotion vs Sales (sampled 50k, log-log scale)', fontsize=14, fontweight='bold')

    # 添加趋势线
    from numpy.polynomial.polynomial import polyfit
    log_x = np.log10(sample['onpromotion'].values + 1)
    log_y = np.log10(sample['sales'].values + 1)
    b, m = polyfit(log_x, log_y, 1)
    x_line = np.linspace(log_x.min(), log_x.max(), 100)
    ax.plot(10**x_line, 10**(b + m * x_line), 'r--', linewidth=2, label='Trend')

    ax.legend()
    plt.tight_layout()
    fpath = os.path.join(FIGS_DIR, '04_promotion_vs_sales.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [保存] {fpath}")


def plot_oil_price(oil):
    """图5: 油价时间序列及缺失值"""
    oil = oil.copy()
    # 简单线性插值用于展示
    oil['dcoilwtico_interp'] = oil['dcoilwtico'].interpolate(method='linear')

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(oil['date'], oil['dcoilwtico_interp'], color='darkgreen', linewidth=0.8, label='Interpolated')
    # 标记原始有值的位置
    mask_valid = oil['dcoilwtico'].notna()
    ax.scatter(oil.loc[mask_valid, 'date'], oil.loc[mask_valid, 'dcoilwtico'],
               s=3, alpha=0.6, color='limegreen', label='Original')

    ax.set_title('Daily Oil Price (WTI)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.legend(fontsize=10)

    plt.tight_layout()
    fpath = os.path.join(FIGS_DIR, '05_oil_price.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [保存] {fpath}")


def plot_day_of_week(train):
    """图6: 星期几对销售的影响"""
    train = train.copy()
    train['dayofweek'] = train['date'].dt.dayofweek
    train['day_name'] = train['date'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    daily = train.groupby(['date', 'dayofweek', 'day_name'])['sales'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=daily, x='day_name', y='sales', order=day_order,
                palette='coolwarm', ax=ax)
    ax.set_yscale('log')
    ax.set_title('Daily Sales by Day of Week (log scale)', fontsize=14, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Total Daily Sales (log)', fontsize=12)

    plt.tight_layout()
    fpath = os.path.join(FIGS_DIR, '06_sales_by_dayofweek.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [保存] {fpath}")


def plot_monthly_seasonality(train):
    """图7: 月度销售模式"""
    train = train.copy()
    train['year'] = train['date'].dt.year
    train['month'] = train['date'].dt.month

    monthly = train.groupby(['year', 'month'])['sales'].sum().reset_index()
    monthly['year_month'] = monthly['year'].astype(str) + '-' + monthly['month'].astype(str).str.zfill(2)
    pivot = monthly.pivot(index='month', columns='year', values='sales')

    fig, ax = plt.subplots(figsize=(12, 6))
    for year_col in pivot.columns:
        ax.plot(pivot.index, pivot[year_col] / 1e6, marker='o', linewidth=2,
                markersize=6, label=str(year_col))
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_title('Monthly Total Sales by Year', fontsize=14, fontweight='bold')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Total Sales (Millions)', fontsize=12)
    ax.legend(title='Year', fontsize=9)

    plt.tight_layout()
    fpath = os.path.join(FIGS_DIR, '07_monthly_sales.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [保存] {fpath}")


def plot_payday_effect(train):
    """图8: 发薪日（15日/月末）效应"""
    train = train.copy()
    train['day'] = train['date'].dt.day
    train['is_payday'] = (train['day'] == 15) | (train['day'] == train['date'].dt.days_in_month)
    daily_pay = train.groupby(['date', 'is_payday'])['sales'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=daily_pay, x='is_payday', y='sales', palette=['skyblue', 'coral'], ax=ax)
    ax.set_yscale('log')
    ax.set_xticklabels(['Non-payday', 'Payday (15th/月末)'])
    ax.set_title('Sales: Payday vs Non-payday (log scale)', fontsize=14, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Total Daily Sales (log)', fontsize=12)

    plt.tight_layout()
    fpath = os.path.join(FIGS_DIR, '08_payday_effect.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [保存] {fpath}")


def plot_holiday_effect(train, holidays):
    """图9: 节假日效应"""
    # 匹配 National 假日
    national_holidays = holidays[holidays['locale'] == 'National']
    holiday_dates = set(national_holidays['date'].dt.date)

    train = train.copy()
    train['date_only'] = train['date'].dt.date
    train['is_national_holiday'] = train['date_only'].apply(lambda x: x in holiday_dates)
    daily_hol = train.groupby(['date', 'is_national_holiday'])['sales'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=daily_hol, x='is_national_holiday', y='sales',
                palette=['mediumseagreen', 'tomato'], ax=ax)
    ax.set_yscale('log')
    ax.set_xticklabels(['Non-holiday', 'National Holiday'])
    ax.set_title('Sales: National Holiday vs Regular Day (log scale)', fontsize=14, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Total Daily Sales (log)', fontsize=12)

    plt.tight_layout()
    fpath = os.path.join(FIGS_DIR, '09_holiday_effect.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [保存] {fpath}")


def plot_correlation_heatmap(train, oil):
    """图10: 相关性热力图"""
    train = train.copy()
    train['dayofweek'] = train['date'].dt.dayofweek
    train['month'] = train['date'].dt.month
    train['day'] = train['date'].dt.day

    daily = train.groupby('date').agg({
        'sales': 'sum', 'onpromotion': 'sum',
        'dayofweek': 'first', 'month': 'first', 'day': 'first'
    }).reset_index()

    # merge oil
    oil_filled = oil.copy()
    oil_filled['dcoilwtico'] = oil_filled['dcoilwtico'].interpolate(method='linear')
    daily = daily.merge(oil_filled[['date', 'dcoilwtico']], on='date', how='left')
    daily['dcoilwtico'] = daily['dcoilwtico'].fillna(method='ffill')

    corr = daily[['sales', 'onpromotion', 'dayofweek', 'month', 'day', 'dcoilwtico']].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, ax=ax,
                vmin=-1, vmax=1)
    ax.set_title('Correlation Heatmap (Daily Aggregated)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    fpath = os.path.join(FIGS_DIR, '10_correlation_heatmap.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [保存] {fpath}")


def plot_transactions(train, transactions):
    """图11: 交易量与销售额关系"""
    daily_tr = transactions.groupby('date')['transactions'].sum().reset_index()
    daily_sales = train.groupby('date')['sales'].sum().reset_index()
    merged = daily_sales.merge(daily_tr, on='date', how='inner')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 时间序列对比
    ax = axes[0]
    ax2 = ax.twinx()
    ax.plot(merged['date'], merged['sales'] / 1e6, color='steelblue', linewidth=0.7, label='Sales')
    ax2.plot(merged['date'], merged['transactions'] / 1e3, color='coral', linewidth=0.7, label='Transactions')
    ax.set_ylabel('Total Sales (Millions)', color='steelblue', fontsize=11)
    ax2.set_ylabel('Total Transactions (Thousands)', color='coral', fontsize=11)
    ax.set_title('Sales vs Transactions Over Time', fontsize=13, fontweight='bold')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # 散点图
    ax = axes[1]
    ax.scatter(merged['transactions'], merged['sales'], alpha=0.4, s=10,
               c='steelblue', edgecolors='none')
    ax.set_xlabel('Transactions', fontsize=11)
    ax.set_ylabel('Sales', fontsize=11)
    ax.set_title('Sales vs Transactions Scatter', fontsize=13, fontweight='bold')

    plt.tight_layout()
    fpath = os.path.join(FIGS_DIR, '11_transactions.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [保存] {fpath}")


def plot_family_time_series(train):
    """图12: TOP 6 品类的时间序列"""
    top_families = train.groupby('family')['sales'].sum().nlargest(6).index.tolist()
    family_daily = train[train['family'].isin(top_families)].groupby(
        ['date', 'family'])['sales'].sum().reset_index()

    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    axes = axes.flatten()

    for i, fam in enumerate(top_families):
        data = family_daily[family_daily['family'] == fam]
        axes[i].fill_between(data['date'], data['sales'], alpha=0.3, color=f'C{i}')
        axes[i].plot(data['date'], data['sales'], linewidth=0.5, color=f'C{i}')
        axes[i].set_title(f'{fam}', fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Daily Sales')

    plt.suptitle('Top 6 Product Families — Daily Sales Time Series', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    fpath = os.path.join(FIGS_DIR, '12_family_time_series.png')
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [保存] {fpath}")


# ======================== 主函数 ========================

def main():
    print("\n" + "█" * 60)
    print("  数据挖掘大作业 — 探索性数据分析 (EDA)")
    print("█" * 60 + "\n")

    train, test, stores, oil, holidays, transactions, submission = load_data()

    # 1. 基本信息
    basic_info(train, test, oil, holidays, stores)

    # 2. 可视化
    print("\n>>> 开始生成可视化图表...")
    plot_sales_time_series(train)
    plot_sales_by_family(train)
    plot_sales_by_store_type(train, stores)
    plot_promotion_vs_sales(train)
    plot_oil_price(oil)
    plot_day_of_week(train)
    plot_monthly_seasonality(train)
    plot_payday_effect(train)
    plot_holiday_effect(train, holidays)
    plot_correlation_heatmap(train, oil)
    plot_transactions(train, transactions)
    plot_family_time_series(train)

    print(f"\n✅ 全部图表已保存至: {FIGS_DIR}")
    print("   共 12 张可视化图表\n")


if __name__ == '__main__':
    main()
