# Machine Learning Projects

15 small, end-to-end ML projects (regression, classification, NLP, clustering, recommender, time series, computer vision).

| # | Project | Task | Models |
|---|---------|------|--------|
| 01 | House Price | Regression | Linear Reg, Random Forest |
| 02 | Titanic | Classification | LogReg, Random Forest |
| 03 | SMS Spam | NLP | Naive Bayes, LogReg |
| 04 | Iris & Wine | Classification | KNN, SVM, Tree |
| 05 | Churn | Classification | LogReg, RF, GBM |
| 06 | Movie Recommender | Recommendation | TF-IDF + cosine |
| 07 | Sentiment | NLP | NB, LogReg |
| 08 | Fraud Detection | Imbalanced | LogReg, RF |
| 09 | Diabetes | Classification | LogReg, RF, SVM |
| 10 | Digits | Image | SVM, MLP |
| 11 | Segmentation | Clustering | K-Means |
| 12 | Fake News | NLP | PassiveAggressive, LogReg |
| 13 | Loan Approval | Classification | LogReg, RF, GBM |
| 14 | Forecasting | Time series | Lag features + LR/RF |
| 15 | Image Transfer Learning | Deep learning | MobileNetV2 |

## How to run
```bash
pip install -r requirements.txt
bash run_all.sh          # runs all projects, saves metrics.txt + results.png, writes RESULTS.md
```
Or run one: `cd 01_house_price && python main.py`.
Project 15 needs TensorFlow: `pip install -r requirements-deeplearning.txt`.

Each folder has its own README, `main.py`, and saves `results.png`. See `RESULTS.md` for a summary of all metrics.
