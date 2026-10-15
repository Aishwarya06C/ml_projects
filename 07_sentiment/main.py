"""Sentiment analysis on Amazon/IMDB/Yelp review sentences."""
import io, zipfile, urllib.request
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

URL = "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip"
z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL).read()))
parts = [pd.read_csv(z.open(n), sep="\t", names=["text", "label"], quoting=3)
         for n in z.namelist() if n.endswith("_labelled.txt")]
df = pd.concat(parts, ignore_index=True)
print(df.shape, df.label.value_counts().to_dict())
Xtr, Xte, ytr, yte = train_test_split(df.text, df.label, test_size=0.2, stratify=df.label, random_state=42)

vec = lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
models = {"NaiveBayes": make_pipeline(vec(), MultinomialNB()),
          "LogisticRegression": make_pipeline(vec(), LogisticRegression(max_iter=1000))}
for name, m in models.items():
    m.fit(Xtr, ytr); p = m.predict(Xte)
    print(f"{name}: acc={accuracy_score(yte, p):.3f} f1={f1_score(yte, p):.3f}")

lr = models["LogisticRegression"]
words = np.array(lr[0].get_feature_names_out()); coef = lr[1].coef_[0]
order = np.argsort(coef); top = np.r_[order[:10], order[-10:]]
plt.barh(words[top], coef[top], color=["r"] * 10 + ["g"] * 10)
plt.title("Most negative / positive words"); plt.tight_layout(); plt.savefig("results.png")
print(lr.predict(["I absolutely loved this", "Terrible, waste of money"]))
