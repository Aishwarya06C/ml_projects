# Sentiment Analysis

**Problem:** Classify review sentences as positive or negative.
**Dataset:** UCI Sentiment Labelled Sentences (Amazon, IMDB, Yelp)
**Approach:** TF-IDF (1-2 grams) + Naive Bayes / Logistic Regression

## Run
```bash
pip install -r ../requirements.txt
python main.py
```
Outputs metrics in the terminal and saves `results.png`.

## Results
Running `main.py` prints the metrics and saves them (via `run_all.sh`) to `metrics.txt`; the chart is saved as `results.png`.
