import json
with open('housing_data.json', encoding='utf-8') as f:
    data = json.load(f)
print('总记录数:', len(data))
print('第1条:', data[0])
print('name 为空:', sum(1 for r in data if not r.get('name')))
