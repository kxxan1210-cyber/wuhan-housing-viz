from models2 import load_data, build_gwr

df = load_data()
gwr = build_gwr(df)
print(f"GWR R²={gwr['r2']}  RMSE={gwr['rmse']}")