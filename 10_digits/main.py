"""Handwritten digit recognition (sklearn 8x8 digits; SVM vs MLP)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

digits = load_digits()
Xtr, Xte, ytr, yte = train_test_split(digits.data, digits.target, test_size=0.25, random_state=42, stratify=digits.target)
models = {"SVM": make_pipeline(StandardScaler(), SVC(gamma=0.001, C=10)),
          "MLP": make_pipeline(StandardScaler(), MLPClassifier((128, 64), max_iter=400, random_state=42))}
preds = {}
for name, m in models.items():
    preds[name] = m.fit(Xtr, ytr).predict(Xte)
    print(f"{name}: accuracy={accuracy_score(yte, preds[name]):.3f}")

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 2, 1); ConfusionMatrixDisplay.from_predictions(yte, preds["SVM"], ax=ax); ax.set_title("SVM confusion matrix")
for i in range(8):
    a = fig.add_subplot(4, 8, 17 + i)
    a.imshow(Xte[i].reshape(8, 8), cmap="gray_r"); a.set_title(f"p={preds['SVM'][i]}", fontsize=8); a.axis("off")
plt.tight_layout(); plt.savefig("results.png"); print("Saved results.png")
