import csv
import json
import re
import os

csv_path = r'C:\Users\Kxxan\PycharmProjects\pythonProject\筛选后数据26_radiation.csv'
out_path = os.path.join(os.path.dirname(__file__), 'housing_data.json')

def parse_year(s):
    m = re.search(r'(\d{4})', str(s))
    return int(m.group(1)) if m else None

def parse_green(s):
    m = re.search(r'([\d.]+)', str(s))
    return float(m.group(1)) if m else None

def parse_fee(s):
    m = re.search(r'([\d.]+)', str(s))
    return float(m.group(1)) if m else None

records = []
seen = set()

with open(csv_path, encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader, 1):
        name = row.get('小区', '').strip()
        district = row.get('行政区域', '').strip()
        business = row.get('商圈板块', '').strip()
        price_raw = row.get('每平方单价', '')
        year = parse_year(row.get('竣工时间', ''))
        far_raw = row.get('容积率', '')
        green = parse_green(row.get('绿化率', ''))
        fee = parse_fee(row.get('物业费', ''))
        schools = row.get('学校', '0')
        medical = row.get('医疗', '0')
        lon = row.get('经度', '')
        lat = row.get('纬度', '')
        total_units = row.get('总户数', '')
        dist_guanggu = row.get('到光谷核心区距离(km)', '')
        radiation = row.get('光谷辐射度', '')
        school_level = row.get('学区等级', '')

        try:
            price = float(price_raw)
        except:
            continue
        try:
            far = float(far_raw)
        except:
            far = None

        # 去重（小区名+区域+经纬度）
        key = (name, district, lon, lat)
        if key in seen:
            continue
        seen.add(key)

        try:
            schools_n = int(float(schools))
        except:
            schools_n = 0
        try:
            medical_n = int(float(medical))
        except:
            medical_n = 0
        try:
            lon_f = float(lon)
            lat_f = float(lat)
        except:
            lon_f = lat_f = None
        try:
            dist_f = float(dist_guanggu)
        except:
            dist_f = None
        try:
            rad_f = float(radiation)
        except:
            rad_f = None
        try:
            sl = int(float(school_level))
        except:
            sl = None
        try:
            units = int(re.search(r'(\d+)', str(total_units)).group(1))
        except:
            units = None

        records.append({
            'id': i,
            'name': name,
            'district': district,
            'business': business,
            'price': int(price),
            'year': year,
            'far': far,
            'green': green,
            'fee': fee,
            'schools': schools_n,
            'medical': medical_n,
            'lon': lon_f,
            'lat': lat_f,
            'total_units': units,
            'dist_guanggu': dist_f,
            'radiation': rad_f,
            'school_level': sl,
        })

with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(records, f, ensure_ascii=False, separators=(',', ':'))

print(f'Done: {len(records)} records -> {out_path}')
