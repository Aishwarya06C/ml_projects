"""Credit card fraud detection (highly imbalanced data)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             classification_report)

d = fetch_openml(data_id=1597, as_frame=True)      # creditcard (284k rows)
X, y = d.data.astype(float), d.target.astype(int)
print("Fraud rate: %.4f%%" % (100 * y.mean()))
X["Amount"] = StandardScaler().fit_transform(X[["Amount"]])
X["Time"] = StandardScaler().fit_transform(X[["Time"]])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

models = {"LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
          "RandomForest": RandomForestClassifier(n_estimators=50, max_depth=12, class_weight="balanced_subsample",
                                                 n_jobs=-1, random_state=42)}
for name, m in models.items():
    m.fit(Xtr, ytr); pr = m.predict_proba(Xte)[:, 1]
    print(f"\n{name}: PR-AUC={average_precision_score(yte, pr):.3f}")
    print(classification_report(yte, m.predict(Xte), digits=3))
    p, r, _ = precision_recall_curve(yte, pr); plt.plot(r, p, label=name)
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.title("Precision-Recall (accuracy is misleading here!)")
plt.savefig("results.png")
