# Customer Segmentation

**Problem:** Group customers by behaviour.
**Dataset:** Synthetic mall-style data (swap in Kaggle Mall Customers)
**Approach:** K-Means; elbow + silhouette to pick k

## Run
```bash
pip install -r ../requirements.txt
python main.py
```
Outputs metrics in the terminal and saves `results.png`.

## Results
Running `main.py` prints the metrics and saves them (via `run_all.sh`) to `metrics.txt`; the chart is saved as `results.png`.
