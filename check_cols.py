import csv
with open(r'C:\Users\Kxxan\PycharmProjects\pythonProject\筛选后数据26_radiation.csv', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    cols = reader.fieldnames
    print('列名:', cols[:5])
    row = next(reader)
    print('小区值:', repr(row.get('小区', 'KEY_NOT_FOUND')))
    print('第一列key:', repr(list(row.keys())[0]))
    print('第一列值:', repr(list(row.values())[0]))
