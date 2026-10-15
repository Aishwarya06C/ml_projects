"""House price prediction - California Housing (regression)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
print(X.describe().T)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

models = {"LinearRegression": LinearRegression(),
          "RandomForest": RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)}
for name, m in models.items():
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    print(f"{name}: MAE={mean_absolute_error(yte, p):.3f}  R2={r2_score(yte, p):.3f}")

rf = models["RandomForest"]
imp = sorted(zip(rf.feature_importances_, X.columns))
plt.barh([n for _, n in imp], [v for v, _ in imp])
plt.title("Random Forest feature importance"); plt.tight_layout()
plt.savefig("results.png"); print("Saved results.png")
