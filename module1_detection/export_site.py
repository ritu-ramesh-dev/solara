"""Publish Module 1 results to the website (website/data/results.json + sample images).

Run after evaluate.py:  python -m module1_detection.export_site
"""
import json
import shutil

import pandas as pd

from . import config

SITE = config.ROOT / "website"
SAMPLES_PER_CLASS = 2


def main():
    metrics = json.loads((config.OUT_DIR / "metrics.json").read_text())
    names = metrics["class_names"]
    preds = pd.read_csv(config.OUT_DIR / "test_predictions.csv")
    index = pd.read_csv(config.INDEX_CSV)
    usable = index[index["usable"]]

    # Clear, confident examples of each class from the test days, plus the
    # most confident mistake so the page shows where the model goes wrong.
    picks = []
    for k in range(len(names)):
        good = preds[(preds.severity == k) & (preds.predicted == k) & (preds.irradiance > 0.25)]
        picks += good.sample(min(SAMPLES_PER_CLASS, len(good)), random_state=config.SEED).to_dict("records")
    misses = preds[preds.severity != preds.predicted].sort_values("confidence", ascending=False)
    picks = picks[:len(names) * SAMPLES_PER_CLASS - 1] + misses.head(1).to_dict("records")

    out_dir = SITE / "assets" / "samples"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    samples = []
    for i, r in enumerate(picks):
        dst = out_dir / f"sample_{i}.jpg"
        shutil.copy(config.ROOT / r["path"], dst)
        samples.append({
            "src": f"assets/samples/{dst.name}",
            "loss": round(float(r["power_loss_pct"]), 1),
            "true": names[int(r["severity"])],
            "pred": names[int(r["predicted"])],
        })

    data = {
        "class_names": names,
        "metrics": {k: metrics[k] for k in (
            "accuracy", "macro_f1", "macro_precision", "macro_recall",
            "test_images", "test_days", "majority_class_baseline_accuracy")},
        "confusion": metrics["confusion_matrix"],
        "class_counts": {n: int((usable["severity"] == k).sum()) for k, n in enumerate(names)},
        "samples": samples,
    }
    (SITE / "data").mkdir(exist_ok=True)
    (SITE / "data" / "results.json").write_text(json.dumps(data, indent=2))
    print(f"Wrote {SITE / 'data' / 'results.json'} and {len(samples)} sample images")


if __name__ == "__main__":
    main()
