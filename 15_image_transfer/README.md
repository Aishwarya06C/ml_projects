# Image Classification with Transfer Learning

**Problem:** Classify flower images.
**Dataset:** TensorFlow flower_photos
**Approach:** MobileNetV2 (frozen) + new head; augmentation

## Run
```bash
pip install -r ../requirements.txt
python main.py
```
Outputs metrics in the terminal and saves `results.png`.

## Results
Running `main.py` prints the metrics and saves them (via `run_all.sh`) to `metrics.txt`; the chart is saved as `results.png`.
