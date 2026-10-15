# Fake News Detector

**Problem:** Classify news text.
**Dataset:** Kaggle Fake/Real News CSVs (fallback: 20 Newsgroups)
**Approach:** TF-IDF + Passive Aggressive / Logistic Regression

## Run
```bash
pip install -r ../requirements.txt
python main.py
```
Outputs metrics in the terminal and saves `results.png`.

## Results
Running `main.py` prints the metrics and saves them (via `run_all.sh`) to `metrics.txt`; the chart is saved as `results.png`.
