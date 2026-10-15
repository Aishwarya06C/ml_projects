"""Fake news detector (TF-IDF + linear models).
Best: download Kaggle 'Fake and Real News Dataset' (Fake.csv, True.csv) into this
folder. Without them, it falls back to a 20-newsgroups topic classifier so the
script always runs.
"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

if os.path.exists("Fake.csv") and os.path.exists("True.csv"):
    df = pd.concat([pd.read_csv("Fake.csv").assign(label=0), pd.read_csv("True.csv").assign(label=1)])
    texts, labels, names = df["text"].fillna(""), df["label"], ["fake", "real"]
else:
    from sklearn.datasets import fetch_20newsgroups
    d = fetch_20newsgroups(subset="all", categories=["sci.space", "rec.sport.hockey", "talk.politics.misc", "comp.graphics"],
                           remove=("headers", "footers", "quotes"))
    texts, labels, names = pd.Series(d.data), pd.Series(d.target), d.target_names
    print("Kaggle files not found -> using 20newsgroups fallback")

Xtr, Xte, ytr, yte = train_test_split(texts, labels, test_size=0.2, stratify=labels, random_state=42)
models = {"PassiveAggressive": PassiveAggressiveClassifier(max_iter=50, random_state=42),
          "LogisticRegression": LogisticRegression(max_iter=1000)}
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, clf) in zip(axes, models.items()):
    pipe = make_pipeline(TfidfVectorizer(stop_words="english", max_df=0.7), clf).fit(Xtr, ytr)
    p = pipe.predict(Xte)
    print(name); print(classification_report(yte, p, target_names=[str(n) for n in names]))
    ConfusionMatrixDisplay.from_predictions(yte, p, ax=ax, colorbar=False); ax.set_title(name)
plt.tight_layout(); plt.savefig("results.png")
