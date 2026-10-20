"""Publish Module 2 results to the website (website/data/powerloss.json).

Run after evaluate.py:  python -m module2_powerloss.export_site
"""
import json

import pandas as pd

from . import config

SITE = config.ROOT / "website"
SCATTER_POINTS = 1200


def main():
    m = json.loads((config.OUT_DIR / "metrics.json").read_text())
    preds = pd.read_csv(config.OUT_DIR / "test_predictions.csv")
    best = m["best_model"]
    pts = preds.sample(min(SCATTER_POINTS, len(preds)), random_state=config.SEED)
    data = {
        "test_images": m["test_images"],
        "best_model": best,
        "module1_classifier": m["module1_classifier"],
        "models": [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "mae_by_class"}
            | {"key": key, "mae_by_class": {c: round(x, 2) for c, x in r["mae_by_class"].items()}}
            for key, r in m["models"].items()
        ],
        "scatter": [[round(a, 1), round(b, 1)] for a, b in zip(pts["power_loss_pct"], pts[best])],
    }
    (SITE / "data").mkdir(exist_ok=True)
    (SITE / "data" / "powerloss.json").write_text(json.dumps(data))
    print(f"Wrote {SITE / 'data' / 'powerloss.json'}")


if __name__ == "__main__":
    main()
