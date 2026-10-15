# Diabetes Prediction

**Problem:** Predict diabetes from medical measurements.
**Dataset:** Pima Indians Diabetes (OpenML 37)
**Approach:** Median imputation of invalid zeros; LR, RF, SVM; recall focus

## Run
```bash
pip install -r ../requirements.txt
python main.py
```
Outputs metrics in the terminal and saves `results.png`.

## Results
Running `main.py` prints the metrics and saves them (via `run_all.sh`) to `metrics.txt`; the chart is saved as `results.png`.
