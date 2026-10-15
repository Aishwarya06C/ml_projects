"""Titanic survival prediction (binary classification)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

df = fetch_openml("titanic", version=1, as_frame=True).frame
num, cat = ["age", "sibsp", "parch", "fare", "pclass"], ["sex", "embarked"]
X, y = df[num + cat].copy(), df["survived"].astype(int)
X[cat] = X[cat].astype(object)  # avoid categorical-dtype imputer issues
X[num] = X[num].astype(float)
print(df.groupby("sex")["survived"].apply(lambda s: s.astype(int).mean()))
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), cat)])
models = {"LogisticRegression": LogisticRegression(max_iter=1000),
          "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42)}
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, (name, m) in zip(axes, models.items()):
    pipe = make_pipeline(pre, m).fit(Xtr, ytr)
    p = pipe.predict(Xte)
    print(f"{name}: acc={accuracy_score(yte, p):.3f} f1={f1_score(yte, p):.3f}")
    ConfusionMatrixDisplay.from_predictions(yte, p, ax=ax); ax.set_title(name)
plt.tight_layout(); plt.savefig("results.png"); print("Saved results.png")
