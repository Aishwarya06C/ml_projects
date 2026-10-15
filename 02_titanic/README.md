# Titanic Survival Prediction

**Problem:** Predict passenger survival (classification).
**Dataset:** Titanic (OpenML)
**Approach:** Imputation + one-hot pipeline; Logistic Regression vs Random Forest

## Run
```bash
pip install -r ../requirements.txt
python main.py
```
Outputs metrics in the terminal and saves `results.png`.

## Results
Running `main.py` prints the metrics and saves them (via `run_all.sh`) to `metrics.txt`; the chart is saved as `results.png`.
