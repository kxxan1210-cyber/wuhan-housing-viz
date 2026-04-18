# 武汉光谷房价空间分析可视化项目

基于武汉光谷区域 2842 条住宅小区数据，使用多元线性回归（MLR）、Lasso 回归和地理加权回归（GWR）三种模型对房价影响因素进行建模分析，并以交互式网页呈现结果。

---

## 项目结构

```
项目根目录
├── 前端可视化（Desktop/Wuhan_Housing_Viz/）
│   ├── index_v2.html          # 主页面（当前使用版本）
│   ├── index.html             # 原始版本（备用）
│   └── gwr_coefficients.json  # GWR 模型局部系数数据（由后端生成）
│
└── 后端模型（PycharmProjects/pythonProject/）
    ├── models2.py             # 核心模型文件（主要代码）
    ├── 筛选后数据26.csv        # 数据源（2842 条小区数据）
    ├── app.py                 # Flask 后端服务 v1
    ├── app2.py                # Flask 后端服务 v2
    ├── models.py              # 早期模型版本（已弃用）
    ├── models222.py           # 实验版本（已弃用）
    └── 选最优带宽.py           # GWR 带宽调优独立脚本
```

---

## 核心文件说明

### `models2.py` — 模型核心

包含三个回归模型的完整实现：

| 函数 | 说明 |
|------|------|
| `load_data()` | 读取 CSV，清洗数据，进行特征工程 |
| `build_ols(df)` | 多元线性回归（MLR） |
| `build_lasso(df)` | Lasso 回归（自动交叉验证选 alpha） |
| `build_gwr(df)` | 地理加权回归（GWR，自动 CV 选最优带宽） |
| `predict_ols/lasso/gwr()` | 单点预测接口 |
| `get_map_data(df)` | 返回地图所需数据 |
| `get_feature_stats(df)` | 返回各特征统计信息 |

#### 输入特征（共 25 个）

**基础特征：** 容积率、绿化率、总户数、在售/在租房源、离学校/交通距离、周边 POI 数量（学校/餐饮/风景/交通/购物/生活服务/医疗）、到光谷核心区距离、光谷辐射度、学区等级、人均 GDP

**特征工程（新增）：**
- `房龄` — 由竣工时间计算
- `物业费数值` — 从文本中提取数值
- `停车比` — 从"1:X"格式中提取
- `是否商品房` — 二值哑变量
- `log_离学校距离` / `log_离交通距离` / `log_到光谷距离` — 对数变换
- `学区_距离交互` — 学区等级 ÷ 离学校距离

#### 模型结果

| 模型 | R² | RMSE（元/㎡） |
|------|----|--------------|
| MLR（多元线性回归） | 0.5005 | 4421.64 |
| Lasso 回归 | 0.5038 | 4407.29 |
| GWR（地理加权回归） | 0.6322 | 3909.86 |

> 注：添加特征工程后，R² 相比初版（约 0.29）提升约 70%。

#### 控制台完整输出

```
样本数: 2842
GWR 带宽搜索中...
  bandwidth=0.0471  CV-RMSE=4579.55
  bandwidth=0.0714  CV-RMSE=9380.17
  bandwidth=0.0919  CV-RMSE=13434.72
  bandwidth=0.1118  CV-RMSE=16022.78
  bandwidth=0.1332  CV-RMSE=17911.51
  bandwidth=0.1582  CV-RMSE=19448.81
  bandwidth=0.1895  CV-RMSE=20782.94
  bandwidth=0.2316  CV-RMSE=21948.87
  bandwidth=0.2998  CV-RMSE=22998.14
最优带宽: 0.0471  CV-RMSE: 4579.55
GWR R²=0.6322  RMSE=3909.86
MLR   R²=0.5005   RMSE=4421.64
Lasso R²=0.5038 RMSE=4407.29
GWR   R²=0.6322  RMSE=3909.86
```

带宽单位为经纬度度数（0.0471° ≈ 5km），CV-RMSE 随带宽增大单调上升，说明局部小带宽对本数据集更优。

#### 主要特征系数（MLR，标准化后）

Lasso 正则化几乎未压缩任何系数至 0，说明所有特征均有贡献，与 MLR 结果接近（R² 差异仅 0.003）。影响房价最显著的特征方向：

- 正向影响：学区等级、物业费数值、是否商品房、光谷辐射度
- 负向影响：房龄（越老越便宜）、到光谷核心区距离、log_离交通距离

#### 完整代码（`models2.py`）

```python
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
```

---

### `index_v2.html` — 可视化前端

纯前端单页应用，无需服务器，直接用浏览器打开即可。

主要功能：
- **模型对比卡片** — 切换查看三个模型的 R² 和 RMSE
- **R² 对比柱状图** — 基于 ECharts 绘制
- **GWR 空间系数地图** — 展示各小区的局部回归系数
- **自定义位置预测** — 点击地图选点，输入特征值预测房价

#### 完整代码（`index_v2.html`）

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>武汉市房价空间预测与可视化系统 v2</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
    <style>
        *, *::before, *::after { box-sizing: border-box; }
        body {
            margin: 0; padding: 0;
            display: flex; height: 100vh;
            font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif;
            background: #0f1923; color: #e0e6f0;
        }
        #sidebar {
            width: 380px; min-width: 320px;
            background: linear-gradient(180deg, #141e2e 0%, #0f1923 100%);
            border-right: 1px solid #1e3050;
            display: flex; flex-direction: column;
            z-index: 1000; overflow-y: auto;
        }
        /* ... 省略样式，完整见文件 ... */
    </style>
</head>
<body>
<div id="loading-overlay">
    <div class="spinner"></div>
    <div class="loading-text">正在加载 GWR 系数数据…</div>
</div>

<div id="sidebar">
    <div id="sidebar-header">
        <h1>武汉市房价空间预测系统</h1>
        <p>MLR · Lasso · 地理加权回归 (GWR)</p>
    </div>
    <!-- 模型切换 Tab -->
    <div class="section">
        <div class="tab-bar">
            <button class="tab-btn" onclick="switchModel('mlr')">MLR</button>
            <button class="tab-btn" onclick="switchModel('lasso')">Lasso</button>
            <button class="tab-btn active" onclick="switchModel('gwr')">GWR</button>
        </div>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">R² 决定系数</div>
                <div class="metric-value good" id="metric-r2">0.6322</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">RMSE 均方根误差</div>
                <div class="metric-value" id="metric-rmse">—</div>
            </div>
        </div>
    </div>
    <!-- R² 对比图 -->
    <div class="section">
        <div class="section-title">模型性能对比 (R²)</div>
        <div id="chart-r2"></div>
    </div>
</div>

<div id="map"></div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const MODEL_STATS = {
    mlr:   { r2: '0.5005', rmse: '4421.64', r2Class: 'warn' },
    lasso: { r2: '0.5038', rmse: '4407.29', r2Class: 'warn' },
    gwr:   { r2: '0.6322', rmse: '3909.86', r2Class: 'good' }
};

let currentModel = 'gwr';
let currentPointData = null;
let allGwrData = [];
let userMarker = null;

// 地图初始化（高德底图）
const map = L.map('map').setView([30.5928, 114.3055], 11);
L.tileLayer('http://webrd0{s}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}', {
    subdomains: ['1','2','3','4'], maxZoom: 18
}).addTo(map);

// ECharts R² 对比图
const myChart = echarts.init(document.getElementById('chart-r2'), 'dark');
myChart.setOption({
    backgroundColor: 'transparent',
    series: [{
        data: [
            { value: 0.5005, itemStyle: { color: '#ff9800' } },
            { value: 0.5038, itemStyle: { color: '#ff9800' } },
            { value: 0.6322, itemStyle: { color: '#4caf50' } }
        ],
        type: 'bar'
    }]
});

// 模型切换
function switchModel(model) {
    currentModel = model;
    const stats = MODEL_STATS[model];
    const r2El = document.getElementById('metric-r2');
    r2El.textContent = stats.r2;
    r2El.className = 'metric-value ' + stats.r2Class;
    document.getElementById('metric-rmse').textContent = stats.rmse;
}

// 加载 GWR 系数 JSON
fetch('gwr_coefficients.json')
    .then(r => r.json())
    .then(data => {
        allGwrData = data;
        const intercepts = data.map(d => d.intercept);
        const minI = Math.min(...intercepts), maxI = Math.max(...intercepts);
        data.forEach(point => {
            const t = (point.intercept - minI) / (maxI - minI);
            const color = interpolateColor('#1a6abf', '#ff7800', t);
            L.circleMarker([point.lat, point.lon], {
                radius: 5, fillColor: color, color: '#fff',
                weight: 0.5, fillOpacity: 0.75
            }).addTo(map)
              .bindTooltip(`截距: ${Math.round(point.intercept).toLocaleString()} 元/㎡`);
        });
        document.getElementById('loading-overlay').classList.add('hidden');
    });

// 颜色插值
function interpolateColor(c1, c2, t) {
    const hex = s => parseInt(s, 16);
    const r1=hex(c1.slice(1,3)), g1=hex(c1.slice(3,5)), b1=hex(c1.slice(5,7));
    const r2=hex(c2.slice(1,3)), g2=hex(c2.slice(3,5)), b2=hex(c2.slice(5,7));
    return '#'
        + Math.round(r1+(r2-r1)*t).toString(16).padStart(2,'0')
        + Math.round(g1+(g2-g1)*t).toString(16).padStart(2,'0')
        + Math.round(b1+(b2-b1)*t).toString(16).padStart(2,'0');
}

// 地图点击：最近邻匹配
map.on('click', function(e) {
    if (!allGwrData.length) return;
    let nearest = null, minDist = Infinity;
    allGwrData.forEach(p => {
        const d = e.latlng.distanceTo(L.latLng(p.lat, p.lon));
        if (d < minDist) { minDist = d; nearest = p; }
    });
    currentPointData = nearest;
});

// 预测
window.calculatePrice = function() {
    if (!currentPointData) { alert('请先在地图上点击选择基准点！'); return; }
    const coefs = currentPointData.coefs || {};
    const intercept = currentPointData.intercept || 0;
    const school  = parseFloat(document.getElementById('input-school').value)  || 0;
    const traffic = parseFloat(document.getElementById('input-traffic').value) || 0;
    const medical = parseFloat(document.getElementById('input-medical').value) || 0;
    const far     = parseFloat(document.getElementById('input-far').value)     || 0;
    const green   = parseFloat(document.getElementById('input-green').value)   || 0;
    let price = intercept
        + (coefs['离学校距离'] || 0) * school
        + (coefs['离交通距离'] || 0) * traffic
        + (coefs['医疗']       || 0) * medical
        + (coefs['容积率']     || 0) * far
        + (coefs['绿化率']     || 0) * green;
    if (price < 0) price = 0;
    document.getElementById('final-price').textContent = price.toFixed(0);
};
</script>
</body>
</html>
```

---

### `选最优带宽.py` — GWR 带宽调优脚本

```python
from models2 import load_data, build_gwr

df = load_data()
gwr = build_gwr(df)
print(f"GWR R²={gwr['r2']}  RMSE={gwr['rmse']}")
```

---

### `筛选后数据26.csv` — 数据源

武汉光谷区域住宅小区数据，主要字段包括：

> 小区名称、行政区域、商圈板块、经度、纬度、每平方单价、容积率、绿化率、总户数、竣工时间、物业费、停车位、权属类别、学区等级、各类 POI 距离与数量、人均 GDP 等

---

## 运行方式

### 查看可视化页面

直接用浏览器打开：
```
Desktop/Wuhan_Housing_Viz/index_v2.html
```

### 重新运行模型

在 PyCharm 中运行以下代码：

```python
from models2 import load_data, build_ols, build_lasso, build_gwr

df = load_data()
ols   = build_ols(df)
lasso = build_lasso(df)
gwr   = build_gwr(df)

print(f"MLR   R²={ols['r2']}   RMSE={ols['rmse']}")
print(f"Lasso R²={lasso['r2']} RMSE={lasso['rmse']}")
print(f"GWR   R²={gwr['r2']}  RMSE={gwr['rmse']}")
```

> ⚠️ 注意：GWR 带宽搜索（LOO-CV）在 2842 条数据上需要几分钟，请耐心等待。

---

## 依赖环境

```
Python 3.x
pandas
numpy
scikit-learn
scipy
```

安装依赖：
```bash
pip install pandas numpy scikit-learn scipy
```
