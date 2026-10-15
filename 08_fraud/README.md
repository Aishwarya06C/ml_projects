# Credit Card Fraud Detection

**Problem:** Detect rare fraudulent transactions.
**Dataset:** Credit Card Fraud (OpenML 1597)
**Approach:** Class weighting, PR-AUC, precision/recall (SMOTE is an optional extension)

## Run
```bash
pip install -r ../requirements.txt
python main.py
```
Outputs metrics in the terminal and saves `results.png`.

## Results
Running `main.py` prints the metrics and saves them (via `run_all.sh`) to `metrics.txt`; the chart is saved as `results.png`.
