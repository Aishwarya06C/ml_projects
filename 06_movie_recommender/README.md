# Movie Recommender

**Problem:** Recommend similar movies.
**Dataset:** MovieLens small (GroupLens)
**Approach:** Content-based: TF-IDF on genres + cosine similarity

## Run
```bash
pip install -r ../requirements.txt
python main.py
```
Outputs metrics in the terminal and saves `results.png`.

## Results
Running `main.py` prints the metrics and saves them (via `run_all.sh`) to `metrics.txt`; the chart is saved as `results.png`.
