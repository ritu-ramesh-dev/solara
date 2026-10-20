"""Publish Module 2 results to the website (website/data/powerloss.json).

Run after evaluate.py:  python -m module2_powerloss.export_site
"""
import json

import pandas as pd

from . import config

SITE = config.ROOT / "website"
SCATTER_POINTS = 1200


def export_samples(preds, best, targets=(2, 8, 16, 28, 48, 75)):
    """Six test photographs spread over the loss range, with the best model's
    prediction, copied into website/assets/predict/."""
    import shutil

    out_dir = SITE / "assets" / "predict"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    bright = preds[preds["irradiance"] > 0.3]
    rows = []
    for i, t in enumerate(targets):
        r = bright.iloc[(bright["power_loss_pct"] - t).abs().argsort().iloc[0]]
        dst = out_dir / f"photo_{i}.jpg"
        shutil.copy(config.ROOT / r["path"], dst)
        rows.append({"src": f"assets/predict/{dst.name}", "measured": round(float(r["power_loss_pct"]), 1),
                     "predicted": round(float(r[best]), 1)})
    return rows


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
        "samples": export_samples(preds, best),
    }
    (SITE / "data").mkdir(exist_ok=True)
    (SITE / "data" / "powerloss.json").write_text(json.dumps(data))
    print(f"Wrote {SITE / 'data' / 'powerloss.json'}")


if __name__ == "__main__":
    main()
