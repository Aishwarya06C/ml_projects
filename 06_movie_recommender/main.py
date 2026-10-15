"""Content-based movie recommender (genres, TF-IDF + cosine similarity)."""
import io, zipfile, urllib.request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL).read()))
movies = pd.read_csv(z.open("ml-latest-small/movies.csv"))
ratings = pd.read_csv(z.open("ml-latest-small/ratings.csv"))
stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"])
movies = movies.merge(stats, on="movieId", how="left").fillna({"mean": 0, "count": 0})
movies["genres_clean"] = movies.genres.str.replace("|", " ", regex=False).str.replace("-", "")
tfidf = TfidfVectorizer().fit_transform(movies.genres_clean)

def recommend(title, n=5, min_ratings=20):
    idx = movies.index[movies.title.str.contains(title, case=False, regex=False)]
    if len(idx) == 0:
        return f"'{title}' not found"
    i = idx[0]
    sim = cosine_similarity(tfidf[i], tfidf).ravel()      # one row only -> low memory
    cand = movies.assign(similarity=sim).drop(index=i)
    cand = cand[cand["count"] >= min_ratings]
    cand = cand.sort_values(["similarity", "mean"], ascending=False).head(n)
    print(f"\nBecause you liked: {movies.title[i]}")
    return cand[["title", "genres", "similarity", "mean"]].round(2)

for t in ["Toy Story", "Matrix, The", "Dark Knight"]:
    print(recommend(t))
