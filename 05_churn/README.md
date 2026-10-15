# Customer Churn Prediction

**Problem:** Predict which telecom customers will leave.
**Dataset:** Telecom churn (OpenML 40701)
**Approach:** Logistic Regression, Random Forest, Gradient Boosting; ROC-AUC

## Run
```bash
pip install -r ../requirements.txt
python main.py
```
Outputs metrics in the terminal and saves `results.png`.

## Results
Running `main.py` prints the metrics and saves them (via `run_all.sh`) to `metrics.txt`; the chart is saved as `results.png`.
