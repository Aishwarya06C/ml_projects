# Loan Approval Prediction

**Problem:** Predict good vs bad credit risk.
**Dataset:** German Credit (OpenML 31)
**Approach:** ColumnTransformer; LR, RF, Gradient Boosting; ROC-AUC

## Run
```bash
pip install -r ../requirements.txt
python main.py
```
Outputs metrics in the terminal and saves `results.png`.

## Results
Running `main.py` prints the metrics and saves them (via `run_all.sh`) to `metrics.txt`; the chart is saved as `results.png`.
