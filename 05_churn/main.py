"""Customer churn prediction (telecom)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

d = fetch_openml(data_id=40701, as_frame=True)   # churn dataset
X = pd.get_dummies(d.data, drop_first=True).astype(float)
y = pd.factorize(d.target, sort=True)[0]          # churn = 1
print("Churn rate:", y.mean().round(3))
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

models = {"LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
          "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1),
          "GradientBoosting": GradientBoostingClassifier(random_state=42)}
for name, m in models.items():
    m.fit(Xtr, ytr)
    pr = m.predict_proba(Xte)[:, 1]
    print(f"{name}: ROC-AUC={roc_auc_score(yte, pr):.3f}")
    fpr, tpr, _ = roc_curve(yte, pr); plt.plot(fpr, tpr, label=name)
print(classification_report(yte, m.predict(Xte)))
plt.plot([0, 1], [0, 1], "k--"); plt.legend(); plt.title("ROC curves - churn")
plt.savefig("results.png"); print("Saved results.png")
