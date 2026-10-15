"""Time series forecasting: monthly airline passengers (lag features + ML)."""
import urllib.request, io
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
try:
    df = pd.read_csv(io.BytesIO(urllib.request.urlopen(URL, timeout=15).read()))
except Exception:
    print("Download failed -> using synthetic series")
    t = np.arange(144)
    df = pd.DataFrame({"Month": pd.date_range("1949-01", periods=144, freq="MS").strftime("%Y-%m"),
                       "Passengers": 100 + 2.5 * t + (30 + 0.3 * t) * np.sin(2 * np.pi * t / 12) + np.random.default_rng(0).normal(0, 6, 144)})
df["Month"] = pd.to_datetime(df["Month"]); df = df.set_index("Month")
y = df["Passengers"]

feat = pd.DataFrame({f"lag{l}": y.shift(l) for l in (1, 2, 3, 12)})
feat["month"] = y.index.month; feat["t"] = np.arange(len(y)); feat["target"] = y
feat = feat.dropna()
cut = int(len(feat) * 0.8)
tr, te = feat.iloc[:cut], feat.iloc[cut:]
Xc = [c for c in feat.columns if c != "target"]

preds = {"SeasonalNaive(lag12)": te["lag12"].values}
for name, m in {"LinearRegression": LinearRegression(), "RandomForest": RandomForestRegressor(300, random_state=42)}.items():
    preds[name] = m.fit(tr[Xc], tr["target"]).predict(te[Xc])
for n, p in preds.items():
    print(f"{n}: MAE={mean_absolute_error(te.target, p):.1f} RMSE={np.sqrt(mean_squared_error(te.target, p)):.1f}")

plt.figure(figsize=(10, 4)); plt.plot(y.index, y, label="actual", color="k")
for n, p in preds.items(): plt.plot(te.index, p, "--", label=n)
plt.legend(); plt.title("Forecast on last 20% (time-ordered split)"); plt.tight_layout(); plt.savefig("results.png")
