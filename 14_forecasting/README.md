# Time Series Forecasting

**Problem:** Forecast monthly airline passengers.
**Dataset:** Airline passengers (1949-1960)
**Approach:** Lag features; seasonal naive baseline vs LR vs RF; time-ordered split

## Run
```bash
pip install -r ../requirements.txt
python main.py
```
Outputs metrics in the terminal and saves `results.png`.

## Results
Running `main.py` prints the metrics and saves them (via `run_all.sh`) to `metrics.txt`; the chart is saved as `results.png`.
