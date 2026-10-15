"""Diabetes prediction - Pima Indians dataset."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score

d = fetch_openml(data_id=37, as_frame=True)       # 'diabetes' (Pima)
X, y = d.data.copy(), (d.target == "tested_positive").astype(int)
for c in ["plas", "pres", "skin", "insu", "mass"]:  # zeros are really missing values
    X[c] = X[c].replace(0, np.nan)
print(X.isna().sum())
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

prep = lambda: [SimpleImputer(strategy="median"), StandardScaler()]
models = {"LogisticRegression": LogisticRegression(max_iter=1000),
          "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
          "SVM": SVC()}
names, cvs = [], []
for name, m in models.items():
    pipe = make_pipeline(*prep(), m)
    cv = cross_val_score(pipe, Xtr, ytr, cv=5).mean()
    p = pipe.fit(Xtr, ytr).predict(Xte)
    print(f"{name}: cv={cv:.3f} test_acc={accuracy_score(yte, p):.3f} f1={f1_score(yte, p):.3f} recall={recall_score(yte, p):.3f}")
    names.append(name); cvs.append(cv)
plt.bar(names, cvs); plt.ylim(0.6, 0.85); plt.title("5-fold CV accuracy"); plt.savefig("results.png")
