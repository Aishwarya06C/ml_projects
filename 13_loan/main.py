"""Loan approval / credit risk prediction (German Credit dataset)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report, ConfusionMatrixDisplay

d = fetch_openml(data_id=31, as_frame=True)       # credit-g
X, y = d.data, (d.target == "good").astype(int)   # 1 = approve
cat = X.select_dtypes(["category", "object"]).columns.tolist()
num = X.columns.difference(cat).tolist()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
pre = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])

models = {"LogisticRegression": LogisticRegression(max_iter=2000),
          "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
          "GradientBoosting": GradientBoostingClassifier(random_state=42)}
best, best_auc = None, 0
for name, m in models.items():
    pipe = make_pipeline(pre, m).fit(Xtr, ytr)
    auc = roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])
    print(f"{name}: ROC-AUC={auc:.3f}")
    if auc > best_auc: best, best_auc, best_name = pipe, auc, name
print(f"\nBest: {best_name}"); print(classification_report(yte, best.predict(Xte), target_names=["reject", "approve"]))
ConfusionMatrixDisplay.from_estimator(best, Xte, yte); plt.title(best_name); plt.savefig("results.png")
