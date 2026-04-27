import json
import numpy as np

JSON_PATH = 'C:/Users/Kxxan/Desktop/Wuhan_Housing_Viz/gwr_coefficients.json'

GUANGGU_LON, GUANGGU_LAT = 114.4064, 30.5229

with open(JSON_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

points = data['points']

# 计算每个点到光谷广场的距离（km，简化平面近似）
def dist_km(lon, lat):
    dlat = (lat - GUANGGU_LAT) * 111.0
    dlon = (lon - GUANGGU_LON) * 111.0 * np.cos(np.radians(GUANGGU_LAT))
    return np.sqrt(dlat**2 + dlon**2)

dists = np.array([dist_km(p['lon'], p['lat']) for p in points])
bandwidth = np.median(dists)  # 带宽取中位数，与 guanggufushedu.py 一致

radiations = np.exp(-0.5 * (dists / bandwidth) ** 2)

for i, p in enumerate(points):
    p['radiation'] = round(float(radiations[i]), 4)
    p['dist_to_guanggu_km'] = round(float(dists[i]), 4)

with open(JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"完成，带宽 bandwidth={bandwidth:.4f} km")
print(f"辐射度范围: {radiations.min():.4f} ~ {radiations.max():.4f}")
print(f"共更新 {len(points)} 个点")
