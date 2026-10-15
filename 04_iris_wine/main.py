"""Iris & Wine: compare KNN, SVM, Decision Tree with 5-fold CV."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

rows = []
for dname, loader in {"Iris": load_iris, "Wine": load_wine}.items():
    X, y = loader(return_X_y=True)
    models = {"KNN": KNeighborsClassifier(), "SVM": SVC(), "DecisionTree": DecisionTreeClassifier(random_state=42)}
    for mname, m in models.items():
        s = cross_val_score(make_pipeline(StandardScaler(), m), X, y, cv=5)
        rows.append((dname, mname, s.mean(), s.std()))
df = pd.DataFrame(rows, columns=["dataset", "model", "cv_accuracy", "std"])
print(df.round(3).to_string(index=False))
df.pivot(index="model", columns="dataset", values="cv_accuracy").plot.bar(ylim=(0.8, 1.0))
plt.title("5-fold CV accuracy"); plt.tight_layout(); plt.savefig("results.png"); print("Saved results.png")
