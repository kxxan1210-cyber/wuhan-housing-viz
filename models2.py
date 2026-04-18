import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = 'C:/Users/Kxxan/PycharmProjects/pythonProject/筛选后数据26.csv'

FEATURES = [
    '容积率', '绿化率', '总户数', '在售房源', '在租房源',
    '离学校距离', '离交通距离', '学校', '餐饮', '风景',
    '交通', '购物', '生活服务', '医疗',
    '到光谷核心区距离(km)', '光谷辐射度', '学区等级',
    # 新增特征
    '房龄', '物业费数值', '停车比', '是否商品房', '人均GDP',
    'log_离学校距离', 'log_离交通距离', 'log_到光谷距离',
    '学区_距离交互'
]
TARGET = '每平方单价'

# 特征中文标签和默认值（用于前端表单）
FEATURE_META = {
    '容积率':           {'label': '容积率',           'min': 0.1,  'max': 10,    'step': 0.1,  'default': 2.5},
    '绿化率':           {'label': '绿化率 (%)',        'min': 0,    'max': 100,   'step': 1,    'default': 35},
    '总户数':           {'label': '总户数 (户)',        'min': 10,   'max': 5000,  'step': 10,   'default': 500},
    '在售房源':         {'label': '在售房源 (套)',      'min': 0,    'max': 500,   'step': 1,    'default': 20},
    '在租房源':         {'label': '在租房源 (套)',      'min': 0,    'max': 500,   'step': 1,    'default': 10},
    '离学校距离':       {'label': '离学校距离 (m)',     'min': 0,    'max': 5000,  'step': 50,   'default': 500},
    '离交通距离':       {'label': '离交通距离 (m)',     'min': 0,    'max': 5000,  'step': 50,   'default': 800},
    '学校':             {'label': '周边学校数量',       'min': 0,    'max': 30,    'step': 1,    'default': 5},
    '餐饮':             {'label': '周边餐饮数量',       'min': 0,    'max': 200,   'step': 1,    'default': 30},
    '风景':             {'label': '周边风景数量',       'min': 0,    'max': 50,    'step': 1,    'default': 3},
    '交通':             {'label': '周边交通设施数量',   'min': 0,    'max': 100,   'step': 1,    'default': 10},
    '购物':             {'label': '周边购物数量',       'min': 0,    'max': 100,   'step': 1,    'default': 15},
    '生活服务':         {'label': '周边生活服务数量',   'min': 0,    'max': 100,   'step': 1,    'default': 20},
    '医疗':             {'label': '周边医疗数量',       'min': 0,    'max': 50,    'step': 1,    'default': 5},
    '到光谷核心区距离(km)': {'label': '到光谷核心区距离 (km)', 'min': 0, 'max': 50, 'step': 0.1, 'default': 5.0},
    '光谷辐射度':       {'label': '光谷辐射度',         'min': 0,    'max': 1,     'step': 0.01, 'default': 0.5},
    '学区等级':         {'label': '学区等级 (1-5)',     'min': 1,    'max': 5,     'step': 1,    'default': 3},
}


def load_data():
    df = pd.read_csv(DATA_PATH, encoding='utf-8')
    df['绿化率'] = df['绿化率'].astype(str).str.replace('%', '').astype(float)
    df['总户数'] = df['总户数'].astype(str).str.extract(r'(\d+)').astype(float)
    for f in ['容积率', '绿化率', '总户数', '在售房源', '在租房源',
              '离学校距离', '离交通距离', '学校', '餐饮', '风景',
              '交通', '购物', '生活服务', '医疗',
              '到光谷核心区距离(km)', '光谷辐射度', '学区等级', '人均GDP']:
        df[f] = pd.to_numeric(df[f], errors='coerce')
    df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce')

    # 新增特征工程
    # 1. 房龄
    df['房龄'] = 2024 - df['竣工时间'].astype(str).str.extract(r'(\d{4})')[0].astype(float)

    # 2. 物业费（元/平米/月）
    df['物业费数值'] = df['物业费'].astype(str).str.extract(r'([\d.]+)')[0].astype(float)

    # 3. 停车比（1:X 中的 X）
    df['停车比'] = df['停车位'].astype(str).str.extract(r'1:([\d.]+)')[0].astype(float)

    # 4. 是否商品房
    df['是否商品房'] = (df['权属类别'].astype(str).str.contains('商品房住宅')).astype(int)

    # 5. 对数变换（缓解距离的非线性影响）
    df['log_离学校距离'] = np.log1p(df['离学校距离'])
    df['log_离交通距离'] = np.log1p(df['离交通距离'])
    df['log_到光谷距离'] = np.log1p(df['到光谷核心区距离(km)'])

    # 6. 交互项：学区等级 × 离学校距离（越好的学区离学校越近越值钱）
    df['学区_距离交互'] = df['学区等级'] / (df['离学校距离'] + 1)

    df = df.dropna(subset=FEATURES + [TARGET])
    return df


def build_ols(df):
    X = df[FEATURES].values
    y = df[TARGET].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    y_pred_all = model.predict(X_scaled)
    return {
        'model': model, 'scaler': scaler,
        'r2': round(r2_score(y_test, y_pred_test), 4),
        'rmse': round(float(np.sqrt(mean_squared_error(y_test, y_pred_test))), 2),
        'intercept': round(float(model.intercept_), 2),
        'coef': {k: round(float(v), 4) for k, v in zip(FEATURES, model.coef_)},
        'y_pred': y_pred_all.tolist(),
        'y_true': y.tolist()
    }


def build_lasso(df):
    X = df[FEATURES].values
    y = df[TARGET].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = LassoCV(cv=5, random_state=42, max_iter=10000)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    y_pred_all = model.predict(X_scaled)
    return {
        'model': model, 'scaler': scaler,
        'r2': round(r2_score(y_test, y_pred_test), 4),
        'rmse': round(float(np.sqrt(mean_squared_error(y_test, y_pred_test))), 2),
        'alpha': round(float(model.alpha_), 6),
        'coef': {k: round(float(v), 4) for k, v in zip(FEATURES, model.coef_)},
        'y_pred': y_pred_all.tolist(),
        'y_true': y.tolist()
    }


def _gwr_cv_score(bandwidth, X_scaled, y, dists):
    """Leave-one-out CV 误差，用于选最优带宽"""
    n = len(y)
    y_pred = []
    for i in range(n):
        w = np.exp(-0.5 * (dists[i] / bandwidth) ** 2)
        w[i] = 0  # 排除自身
        W = np.diag(w)
        Xw = X_scaled.T @ W @ X_scaled
        try:
            beta = np.linalg.solve(Xw + np.eye(X_scaled.shape[1]) * 1e-6,
                                   X_scaled.T @ W @ y)
        except np.linalg.LinAlgError:
            beta = np.zeros(X_scaled.shape[1])
        y_pred.append(float(X_scaled[i] @ beta))
    return float(np.sqrt(mean_squared_error(y, y_pred)))


def build_gwr(df):
    df_gwr = df.drop_duplicates(subset='小区').reset_index(drop=True)
    coords = df_gwr[['经度', '纬度']].values
    X = df_gwr[FEATURES].values
    y = df_gwr[TARGET].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n = len(y)
    dists = cdist(coords, coords)

    # 交叉验证选最优带宽（在 10%~90% 分位数范围内搜索）
    flat = dists[dists > 0]
    candidates = np.percentile(flat, [10, 20, 30, 40, 50, 60, 70, 80, 90])
    best_bw, best_score = candidates[0], np.inf
    print("GWR 带宽搜索中...")
    for bw in candidates:
        score = _gwr_cv_score(bw, X_scaled, y, dists)
        print(f"  bandwidth={bw:.4f}  CV-RMSE={score:.2f}")
        if score < best_score:
            best_score = score
            best_bw = bw
    bandwidth = best_bw
    print(f"最优带宽: {bandwidth:.4f}  CV-RMSE: {best_score:.2f}")

    local_r2, y_pred_gwr = [], []
    local_coef = {f: [] for f in FEATURES}

    for i in range(n):
        w = np.exp(-0.5 * (dists[i] / bandwidth) ** 2)
        W = np.diag(w)
        Xw = X_scaled.T @ W @ X_scaled
        try:
            beta = np.linalg.solve(Xw + np.eye(len(FEATURES)) * 1e-6, X_scaled.T @ W @ y)
        except np.linalg.LinAlgError:
            beta = np.zeros(len(FEATURES))
        y_pred_gwr.append(float(X_scaled[i] @ beta))
        ss_res = np.sum(w * (y - X_scaled @ beta) ** 2)
        ss_tot = np.sum(w * (y - np.average(y, weights=w)) ** 2)
        local_r2.append(round(float(1 - ss_res / ss_tot) if ss_tot > 0 else 0, 4))
        for j, f in enumerate(FEATURES):
            local_coef[f].append(round(float(beta[j]), 4))

    points = [
        {
            '小区': df_gwr.iloc[i]['小区'],
            '商圈板块': df_gwr.iloc[i]['商圈板块'],
            '经度': float(df_gwr.iloc[i]['经度']),
            '纬度': float(df_gwr.iloc[i]['纬度']),
            '每平方单价': float(y[i]),
            'local_r2': local_r2[i],
            'pred': round(y_pred_gwr[i], 2)
        } for i in range(n)
    ]

    return {
        'scaler': scaler, 'coords': coords, 'X_scaled': X_scaled, 'y': y,
        'bandwidth': round(float(bandwidth), 4),
        'r2': round(float(r2_score(y, y_pred_gwr)), 4),
        'rmse': round(float(np.sqrt(mean_squared_error(y, y_pred_gwr))), 2),
        'local_coef': local_coef,
        'points': points,
        'y_pred': y_pred_gwr,
        'y_true': y.tolist()
    }


def predict_ols(ols, input_vals):
    x = np.array(input_vals).reshape(1, -1)
    x_scaled = ols['scaler'].transform(x)
    return round(float(ols['model'].predict(x_scaled)[0]), 2)


def predict_lasso(lasso, input_vals):
    x = np.array(input_vals).reshape(1, -1)
    x_scaled = lasso['scaler'].transform(x)
    return round(float(lasso['model'].predict(x_scaled)[0]), 2)


def predict_gwr(gwr, input_vals, coord):
    """coord: [经度, 纬度]"""
    x = np.array(input_vals).reshape(1, -1)
    x_scaled = gwr['scaler'].transform(x)
    pt = np.array(coord).reshape(1, -1)
    d = cdist(pt, gwr['coords'])[0]
    w = np.exp(-0.5 * (d / gwr['bandwidth']) ** 2)
    W = np.diag(w)
    Xw = gwr['X_scaled'].T @ W @ gwr['X_scaled']
    try:
        beta = np.linalg.solve(Xw + np.eye(len(FEATURES)) * 1e-6, gwr['X_scaled'].T @ W @ gwr['y'])
    except np.linalg.LinAlgError:
        beta = np.zeros(len(FEATURES))
    return round(float(x_scaled @ beta), 2)


def get_map_data(df):
    return df[['小区', '行政区域', '商圈板块', '经度', '纬度', TARGET]].drop_duplicates(subset='小区').to_dict(orient='records')


def get_feature_stats(df):
    stats = {}
    for f in FEATURES:
        col = pd.to_numeric(df[f], errors='coerce')
        stats[f] = {
            'mean': round(float(col.mean()), 4),
            'std': round(float(col.std()), 4),
            'corr': round(float(col.corr(pd.to_numeric(df[TARGET], errors='coerce'))), 4)
        }
    return stats
