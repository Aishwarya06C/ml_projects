"""Customer segmentation with K-Means.
Uses synthetic mall-style data so it runs anywhere. To use the real Kaggle
'Mall Customers' dataset, replace the DataFrame creation with:
    df = pd.read_csv("Mall_Customers.csv").rename(columns={"Annual Income (k$)": "income", "Spending Score (1-100)": "spending", "Age": "age"})
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

rng = np.random.default_rng(42)
centers = [(25, 25, 80), (45, 85, 20), (30, 85, 85), (50, 30, 20), (40, 55, 50)]  # age, income, spending
df = pd.concat([pd.DataFrame(rng.normal(c, (5, 8, 8), size=(60, 3)), columns=["age", "income", "spending"]) for c in centers],
               ignore_index=True).clip(lower=1)
Z = StandardScaler().fit_transform(df)

ks = range(2, 9)
inertia, sil = [], []
for k in ks:
    km = KMeans(k, n_init=10, random_state=42).fit(Z)
    inertia.append(km.inertia_); sil.append(silhouette_score(Z, km.labels_))
best_k = list(ks)[int(np.argmax(sil))]
print("Silhouette by k:", dict(zip(ks, np.round(sil, 3))), "-> best k =", best_k)

df["cluster"] = KMeans(best_k, n_init=10, random_state=42).fit_predict(Z)
print(df.groupby("cluster").mean().round(1).assign(size=df.cluster.value_counts().sort_index()))

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(list(ks), inertia, "o-"); ax[0].set_title("Elbow (inertia)")
ax[1].plot(list(ks), sil, "o-"); ax[1].set_title("Silhouette score")
ax[2].scatter(df.income, df.spending, c=df.cluster, cmap="tab10"); ax[2].set_xlabel("income"); ax[2].set_ylabel("spending"); ax[2].set_title("Segments")
plt.tight_layout(); plt.savefig("results.png"); print("Saved results.png")
