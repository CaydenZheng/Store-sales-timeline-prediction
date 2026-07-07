# -*- coding: utf-8 -*-
"""
02_preprocessing.py —— 数据预处理与特征工程
===============================================
对原始数据进行清洗、合并、特征构造，输出可供模型直接使用的训练集和测试集。
处理后的数据保存为 .pkl 格式到 ./processed/ 目录。

"""

import pandas as pd
import numpy as np
import os, warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ======================== 数据加载 ========================

def load_raw_data():
    print("=" * 60)
    print("正在加载原始数据...")
    train = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'), parse_dates=['date'])
    test  = pd.read_csv(os.path.join(BASE_DIR, 'test.csv'),  parse_dates=['date'])
    stores = pd.read_csv(os.path.join(BASE_DIR, 'stores.csv'))
    oil = pd.read_csv(os.path.join(BASE_DIR, 'oil.csv'), parse_dates=['date'])
    holidays = pd.read_csv(os.path.join(BASE_DIR, 'holidays_events.csv'), parse_dates=['date'])
    transactions = pd.read_csv(os.path.join(BASE_DIR, 'transactions.csv'), parse_dates=['date'])
    print(f"  train({train.shape})  test({test.shape})")
    return train, test, stores, oil, holidays, transactions


# ======================== 特征工程 ========================

def add_date_features(df):
    """从 date 列提取时间特征"""
    df = df.copy()
    df['year']        = df['date'].dt.year.astype('int16')
    df['month']       = df['date'].dt.month.astype('int8')
    df['day']         = df['date'].dt.day.astype('int8')
    df['dayofweek']   = df['date'].dt.dayofweek.astype('int8')
    df['weekofyear']  = df['date'].dt.isocalendar().week.astype('int8')
    df['dayofyear']   = df['date'].dt.dayofyear.astype('int16')
    df['quarter']     = df['date'].dt.quarter.astype('int8')
    df['is_weekend']  = (df['dayofweek'] >= 5).astype('int8')
    df['is_month_start'] = (df['day'] <= 3).astype('int8')
    df['is_month_end']   = (df['day'] >= df['date'].dt.days_in_month - 2).astype('int8')

    days_in_month = df['date'].dt.days_in_month
    df['is_payday'] = ((df['day'] == 15) | (df['day'] == days_in_month)).astype('int8')
    df['near_payday'] = ((df['day'] >= 13) & (df['day'] <= 17) |
                         (df['day'] >= days_in_month - 2)).astype('int8')

    # P0-2: 每月第几周（1~5），捕获月内消费节奏
    df['week_of_month'] = ((df['day'] - 1) // 7 + 1).astype('int8')

    # P1-5: 距离最近发薪日天数（15号或月末）
    day = df['day'].values
    dim = days_in_month.values
    days_from_payday = np.where(day == 15, 0,
                        np.where(day == dim, 0,
                        np.where(day < 15, day, day - 15)))
    df['days_from_payday'] = days_from_payday.astype('int8')

    # P1-4: 2016-04-16 地震后 8 周标记
    eq_start = pd.Timestamp('2016-04-16')
    eq_end   = pd.Timestamp('2016-06-11')  # +8 周
    df['earthquake_impact'] = ((df['date'] >= eq_start) &
                               (df['date'] <= eq_end)).astype('int8')

    return df


def process_oil(oil):
    """油价数据：线性插值 + 前向/后向填充 + 滞后特征"""
    oil = oil.copy().sort_values('date')
    oil['dcoilwtico'] = oil['dcoilwtico'].interpolate(method='linear', limit_direction='both')
    oil['dcoilwtico'] = oil['dcoilwtico'].fillna(method='ffill').fillna(method='bfill')

    oil['oil_lag7']  = oil['dcoilwtico'].shift(7)
    oil['oil_lag30'] = oil['dcoilwtico'].shift(30)
    oil['oil_lag90'] = oil['dcoilwtico'].shift(90)
    oil['oil_roll_mean7']  = oil['dcoilwtico'].rolling(7, min_periods=1).mean()
    oil['oil_roll_mean30'] = oil['dcoilwtico'].rolling(30, min_periods=1).mean()
    oil['oil_roll_std7']   = oil['dcoilwtico'].rolling(7, min_periods=1).std()
    oil['oil_change_7d']  = oil['dcoilwtico'] - oil['oil_lag7']
    oil['oil_change_30d'] = oil['dcoilwtico'] - oil['oil_lag30']

    for col in ['oil_lag7', 'oil_lag30', 'oil_lag90', 'oil_roll_mean7',
                'oil_roll_mean30', 'oil_roll_std7', 'oil_change_7d', 'oil_change_30d']:
        oil[col] = oil[col].fillna(method='bfill').fillna(oil[col].mean())

    return oil


def process_holidays(holidays):
    """处理节假日数据，构造按日期聚合的假日特征"""
    holidays = holidays.copy()
    date_hol = holidays.groupby('date').agg(
        num_holidays=('type', 'count'),
        has_national=('locale', lambda x: int(any(x == 'National'))),
        has_regional=('locale', lambda x: int(any(x == 'Regional'))),
        has_local=('locale', lambda x: int(any(x == 'Local'))),
    ).reset_index()

    for htype in ['Holiday', 'Additional', 'Bridge', 'Transfer', 'Event', 'Work Day']:
        mask = holidays['type'] == htype
        cnt = holidays[mask].groupby('date').size().reset_index(name=f'is_{htype.replace(" ", "_").lower()}')
        cnt[f'is_{htype.replace(" ", "_").lower()}'] = 1
        date_hol = date_hol.merge(cnt, on='date', how='left')

    for col in date_hol.columns:
        if col != 'date':
            date_hol[col] = date_hol[col].fillna(0).astype('int8')

    return date_hol


def process_transactions(transactions):
    """处理交易数据"""
    tr = transactions.copy()
    tr = tr.groupby(['date', 'store_nbr'], as_index=False)['transactions'].sum()
    return tr


def add_store_features(df, stores):
    """合并商店特征并编码"""
    df = df.merge(stores, on='store_nbr', how='left')
    for col in ['city', 'state', 'type']:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))
    df['cluster'] = df['cluster'].astype('int8')
    return df


def add_family_dow_mean(train, test):
    """P1-6: 每个品类 x 星期几的历史平均销售额（捕获周模式）"""
    family_dow = train.groupby(['family', 'dayofweek'])['sales'].mean().reset_index()
    family_dow.columns = ['family', 'dayofweek', 'family_dow_mean']
    global_mean = train['sales'].mean()
    train = train.merge(family_dow, on=['family', 'dayofweek'], how='left')
    test  = test.merge(family_dow, on=['family', 'dayofweek'], how='left')
    train['family_dow_mean'] = train['family_dow_mean'].fillna(global_mean)
    test['family_dow_mean']  = test['family_dow_mean'].fillna(global_mean)
    return train, test


def create_global_mean_features(train, test):
    """构造全局统计特征（品类-商店组合的历史均值等）"""
    grp = train.groupby(['store_nbr', 'family']).agg(
        mean_sales_store_family=('sales', 'mean'),
        std_sales_store_family=('sales', 'std'),
        median_sales_store_family=('sales', 'median'),
        max_sales_store_family=('sales', 'max'),
        sum_promotion_store_family=('onpromotion', 'sum'),
    ).reset_index()

    grp_family = train.groupby('family').agg(
        mean_sales_family=('sales', 'mean'),
        std_sales_family=('sales', 'std'),
    ).reset_index()

    grp_store = train.groupby('store_nbr').agg(
        mean_sales_store=('sales', 'mean'),
    ).reset_index()

    train = train.merge(grp, on=['store_nbr', 'family'], how='left')
    test  = test.merge(grp, on=['store_nbr', 'family'], how='left')
    train = train.merge(grp_family, on='family', how='left')
    test  = test.merge(grp_family, on='family', how='left')
    train = train.merge(grp_store, on='store_nbr', how='left')
    test  = test.merge(grp_store, on='store_nbr', how='left')

    for col in ['mean_sales_store_family', 'std_sales_store_family',
                'median_sales_store_family', 'max_sales_store_family',
                'sum_promotion_store_family']:
        global_val = train[col].mean()
        test[col] = test[col].fillna(global_val)
        train[col] = train[col].fillna(global_val)

    return train, test


def create_lag_features(train, test):
    """构造时序滞后特征（每个 store_nbr x family 组合上）"""
    train = train.copy()
    train = train.sort_values(['store_nbr', 'family', 'date']).reset_index(drop=True)

    def make_lags(grp):
        grp = grp.sort_values('date')
        for lag in [1, 7, 14, 30]:
            grp[f'sales_lag{lag}'] = grp['sales'].shift(lag)
        for window in [7, 14, 30]:
            grp[f'sales_roll_mean{window}'] = grp['sales'].shift(1).rolling(window, min_periods=1).mean()
            grp[f'sales_roll_std{window}']  = grp['sales'].shift(1).rolling(window, min_periods=1).std()
        # P0-1: 促销滞后特征
        for lag in [1, 7, 14]:
            grp[f'promo_lag{lag}'] = grp['onpromotion'].shift(lag)
        p_shift1 = grp['onpromotion'].shift(1)
        for window in [7, 14]:
            grp[f'promo_roll_mean{window}'] = p_shift1.rolling(window, min_periods=1).mean()
        return grp

    train = train.groupby(['store_nbr', 'family'], group_keys=False).apply(make_lags)

    lag_cols = [c for c in train.columns if 'lag' in c or 'roll' in c]
    for col in lag_cols:
        train[col] = train[col].fillna(0)

    # P0-1: 促销 lag 也填充 0
    promo_lag_cols = [c for c in train.columns if c.startswith('promo_lag') or c.startswith('promo_roll')]
    for col in promo_lag_cols:
        train[col] = train[col].fillna(0)

    last_train_date = train['date'].max()
    recent_30 = train[train['date'] >= last_train_date - pd.Timedelta(days=30)]

    test_with_lag = test.copy()
    test_with_lag['sales_lag1'] = 0
    test_with_lag['sales_lag7'] = 0
    test_with_lag['sales_lag14'] = 0
    test_with_lag['sales_lag30'] = 0
    for w in [7, 14, 30]:
        test_with_lag[f'sales_roll_mean{w}'] = 0
        test_with_lag[f'sales_roll_std{w}'] = 0

    last_sales = recent_30.groupby(['store_nbr', 'family']).tail(1)[
        ['store_nbr', 'family', 'sales']].rename(columns={'sales': 'last_known_sales'})

    test_with_lag = test_with_lag.merge(last_sales, on=['store_nbr', 'family'], how='left')

    for lag in [1, 7, 14, 30]:
        test_with_lag[f'sales_lag{lag}'] = test_with_lag['last_known_sales'].fillna(0)

    for w in [7, 14, 30]:
        test_with_lag[f'sales_roll_mean{w}'] = test_with_lag['last_known_sales'].fillna(0)
        test_with_lag[f'sales_roll_std{w}'] = 0

    # P0-1: test 的 promo lag — 用训练集最后已知的 onpromotion
    last_promo = recent_30.groupby(['store_nbr', 'family']).tail(1)[
        ['store_nbr', 'family', 'onpromotion']].rename(
        columns={'onpromotion': 'last_known_promo'})
    test_with_lag = test_with_lag.merge(last_promo, on=['store_nbr', 'family'], how='left')
    for lag in [1, 7, 14]:
        test_with_lag[f'promo_lag{lag}'] = test_with_lag['last_known_promo'].fillna(0)
    for w in [7, 14]:
        test_with_lag[f'promo_roll_mean{w}'] = test_with_lag['last_known_promo'].fillna(0)

    test_with_lag.drop(columns=['last_known_sales', 'last_known_promo'], inplace=True, errors='ignore')

    return train, test_with_lag


# ======================== 主流程 ========================

def main():
    print("\n" + "█" * 60)
    print("  数据预处理与特征工程")
    print("█" * 60 + "\n")

    train, test, stores, oil, holidays, transactions = load_raw_data()

    # ---- 1. 日期特征 ----
    print("\n[1/9] 构造日期特征...")
    train = add_date_features(train)
    test  = add_date_features(test)

    # ---- 2. 油价 ----
    print("[2/9] 处理油价数据...")
    oil = process_oil(oil)
    train = train.merge(oil, on='date', how='left')
    test  = test.merge(oil, on='date', how='left')
    oil_cols = [c for c in oil.columns if c != 'date']
    for col in oil_cols:
        train[col] = train[col].fillna(method='ffill').fillna(method='bfill')
        test[col]  = test[col].fillna(method='ffill').fillna(method='bfill')

    # ---- 3. 节假日 ----
    print("[3/9] 处理节假日特征...")
    date_hol = process_holidays(holidays)
    train = train.merge(date_hol, on='date', how='left')
    test  = test.merge(date_hol, on='date', how='left')
    hol_cols = [c for c in date_hol.columns if c != 'date']
    for col in hol_cols:
        train[col] = train[col].fillna(0).astype('int8')
        test[col]  = test[col].fillna(0).astype('int8')

    # ---- 4. 商店特征 ----
    print("[4/9] 合并商店特征...")
    train = add_store_features(train, stores)
    test  = add_store_features(test, stores)

    # ---- 5. 交易量 ----
    print("[5/9] 处理交易数据...")
    tr = process_transactions(transactions)
    train = train.merge(tr, on=['date', 'store_nbr'], how='left')
    test  = test.merge(tr, on=['date', 'store_nbr'], how='left')
    train['transactions'] = train['transactions'].fillna(0)
    test['transactions']  = test['transactions'].fillna(0)

    # ---- 6. 全局统计特征 ----
    print("[6/9] 构造全局统计特征...")
    train, test = create_global_mean_features(train, test)

    # ---- 7. 滞后特征 ----
    print("[7/9] 构造时序滞后特征...")
    train, test = create_lag_features(train, test)

    # P0-3: 销售动量（加速/减速趋势）
    print("[8/9] 构造销售动量特征...")
    for df in [train, test]:
        df['sales_momentum_1_7']  = df['sales_lag1'] - df['sales_lag7']
        df['sales_momentum_7_30'] = df['sales_lag7'] - df['sales_lag30']

    # P1-6: 品类 x 星期几 周模式
    print("[9/9] 构造 family x dow 特征 & family 编码...")
    train, test = add_family_dow_mean(train, test)
    family_le = LabelEncoder()
    train['family_enc'] = family_le.fit_transform(train['family'].astype(str))
    test['family_enc']  = family_le.transform(test['family'].astype(str))
    print(f"    family 类别数: {len(family_le.classes_)}")

    # ---- 准备最终特征和目标 ----
    print("\n>>> 准备特征矩阵...")

    # 保留 date 列供模型脚本做时间序列切分（模型训练前会 drop）
    drop_cols = ['id', 'city', 'state', 'type', 'family', 'sales']
    test_ids = test['id'].copy()

    feature_cols = [c for c in train.columns if c not in drop_cols and c != 'id']
    feature_cols = sorted(set(feature_cols) & set(test.columns))

    X_train = train[feature_cols].copy()
    y_train = train['sales'].values.astype(np.float32)
    X_test = test[feature_cols].copy()

    print(f"\n  训练集特征矩阵: {X_train.shape}")
    print(f"  测试集特征矩阵: {X_test.shape}")
    print(f"  特征列数: {len(feature_cols)}")
    print(f"  特征列表:\n    {feature_cols}")

    # ---- 保存 ----
    print("\n>>> 保存处理后的数据...")
    X_train.to_pickle(os.path.join(PROCESSED_DIR, 'X_train.pkl'))
    pd.Series(y_train, name='sales').to_pickle(os.path.join(PROCESSED_DIR, 'y_train.pkl'))
    X_test.to_pickle(os.path.join(PROCESSED_DIR, 'X_test.pkl'))
    test_ids.to_frame('id').to_pickle(os.path.join(PROCESSED_DIR, 'test_ids.pkl'))
    pd.Series(feature_cols).to_csv(os.path.join(PROCESSED_DIR, 'feature_cols.txt'),
                                   index=False, header=False)

    print(f"\n✅ 预处理完成! 数据已保存至: {PROCESSED_DIR}")
    print(f"    X_train.pkl  — 训练特征 ({X_train.shape[0]:,} 行 x {X_train.shape[1]} 列)")
    print(f"    y_train.pkl  — 训练目标 ({len(y_train):,})")
    print(f"    X_test.pkl   — 测试特征 ({X_test.shape[0]:,} 行 x {X_test.shape[1]} 列)")
    print(f"    test_ids.pkl — 测试ID\n")


if __name__ == '__main__':
    main()
