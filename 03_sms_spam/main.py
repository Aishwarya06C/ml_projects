"""SMS spam classifier (TF-IDF + Naive Bayes / Logistic Regression)."""
import io, zipfile, urllib.request
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL).read()))
df = pd.read_csv(z.open("SMSSpamCollection"), sep="\t", names=["label", "text"])
print(df.label.value_counts())
y = (df.label == "spam").astype(int)
Xtr, Xte, ytr, yte = train_test_split(df.text, y, test_size=0.2, stratify=y, random_state=42)

models = {"NaiveBayes": MultinomialNB(), "LogisticRegression": LogisticRegression(max_iter=1000)}
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, (name, clf) in zip(axes, models.items()):
    pipe = make_pipeline(TfidfVectorizer(stop_words="english"), clf).fit(Xtr, ytr)
    p = pipe.predict(Xte)
    print(name); print(classification_report(yte, p, target_names=["ham", "spam"]))
    ConfusionMatrixDisplay.from_predictions(yte, p, ax=ax); ax.set_title(name)
plt.tight_layout(); plt.savefig("results.png")
print(pipe.predict(["WINNER!! Claim your free prize now", "Are we meeting for lunch?"]))
