"""Compare every power-loss model on the shared held-out test images.

Writes to module2_powerloss/outputs/:
  metrics.json               MAE, RMSE, R² per model; MAE per severity class;
                             accuracy / macro-F1 when predictions are binned
                             into Module 1's four classes
  test_predictions.csv       per-image predictions of every model
  model_comparison.png       MAE and RMSE per model
  predicted_vs_actual.png    best model: predicted vs measured loss
  error_by_class.png         MAE per severity class for each model
  cnn_training_curves.png    CNN train / val MAE per epoch

Usage:  python -m module2_powerloss.evaluate   (after train_cnn and train_trees)
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from module1_detection import config as m1
from module1_detection.data import eval_transform

from . import config
from .data import PowerLossDataset, load_index
from .model import load_checkpoint

LABELS = {
    "mean_baseline": "Mean baseline",
    "env_xgboost": "XGBoost, environment only",
    "random_forest": "Random Forest, image + env",
    "xgboost": "XGBoost, image + env",
    "cnn": "CNN regressor, image + env",
}


@torch.no_grad()
def cnn_predictions(test_df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _ = load_checkpoint(device=device)
    loader = DataLoader(PowerLossDataset(test_df, eval_transform()), batch_size=128,
                        num_workers=config.NUM_WORKERS, pin_memory=True)
    out = []
    for x, env, _ in loader:
        out.append(net(x.to(device), env.to(device)).float().cpu())
    return torch.cat(out).numpy() * 100


def to_class(loss_pct):
    return np.digitize(loss_pct, m1.LOSS_BINS)


def main():
    df = load_index()
    test = df[df["split"] == "test"].reset_index(drop=True)
    trees = pd.read_csv(config.OUT_DIR / "tree_test_predictions.csv").set_index("path")
    preds = test[["path", "power_loss_pct", "severity", "irradiance"]].copy()
    for k in ("mean_baseline", "env_xgboost", "random_forest", "xgboost"):
        preds[k] = trees.loc[preds["path"], k].to_numpy()
    preds["cnn"] = cnn_predictions(test)
    preds.to_csv(config.OUT_DIR / "test_predictions.csv", index=False)

    y = preds["power_loss_pct"].to_numpy()
    y_cls = preds["severity"].to_numpy()
    names = m1.CLASS_NAMES
    results = {}
    for k in LABELS:
        p = preds[k].to_numpy()
        p_cls = to_class(p)
        results[k] = {
            "label": LABELS[k],
            "mae": mean_absolute_error(y, p),
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "r2": r2_score(y, p),
            "mae_by_class": {n: float(np.abs(y - p)[y_cls == i].mean()) for i, n in enumerate(names)},
            "binned_accuracy": float((p_cls == y_cls).mean()),
            "binned_macro_f1": f1_score(y_cls, p_cls, average="macro"),
        }
    best = min((k for k in LABELS if k != "mean_baseline"), key=lambda k: results[k]["mae"])
    m1_metrics = json.loads((m1.OUT_DIR / "metrics.json").read_text())
    summary = {
        "test_images": int(len(y)),
        "best_model": best,
        "models": results,
        "module1_classifier": {"accuracy": m1_metrics["accuracy"], "macro_f1": m1_metrics["macro_f1"]},
    }
    (config.OUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))

    print(f"{'model':30s} {'MAE':>6s} {'RMSE':>6s} {'R2':>6s} {'bin acc':>8s} {'bin F1':>7s}")
    for k, r in results.items():
        print(f"{r['label']:30s} {r['mae']:6.2f} {r['rmse']:6.2f} {r['r2']:6.3f} "
              f"{r['binned_accuracy']:8.3f} {r['binned_macro_f1']:7.3f}")
    print(f"Module 1 classifier for reference: accuracy {m1_metrics['accuracy']:.3f}, "
          f"macro-F1 {m1_metrics['macro_f1']:.3f}")
    print(f"Best: {LABELS[best]}")

    plot_comparison(results, config.OUT_DIR / "model_comparison.png")
    plot_scatter(y, preds[best].to_numpy(), LABELS[best], results[best], config.OUT_DIR / "predicted_vs_actual.png")
    plot_by_class(results, names, config.OUT_DIR / "error_by_class.png")
    plot_history(config.OUT_DIR / "cnn_training_curves.png")
    print(f"Saved figures to {config.OUT_DIR}")


def plot_comparison(results, path):
    keys = list(results)
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - 0.2, [results[k]["mae"] for k in keys], 0.4, label="MAE")
    ax.bar(x + 0.2, [results[k]["rmse"] for k in keys], 0.4, label="RMSE")
    for i, k in enumerate(keys):
        ax.text(i - 0.2, results[k]["mae"] + 0.4, f"{results[k]['mae']:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x, [results[k]["label"].replace(", ", "\n") for k in keys], fontsize=9)
    ax.set_ylabel("error (percentage points of power loss)")
    ax.set_title("Power-loss prediction error on the test set (lower is better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scatter(y, p, label, r, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    hb = ax.hexbin(y, p, gridsize=60, bins="log", cmap="viridis", mincnt=1)
    ax.plot([0, 100], [0, 100], "w--", lw=1)
    for b in m1.LOSS_BINS:
        ax.axvline(b, c="gray", lw=0.6, ls=":")
        ax.axhline(b, c="gray", lw=0.6, ls=":")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("measured power loss (%)")
    ax.set_ylabel("predicted power loss (%)")
    ax.set_title(f"{label}\nMAE {r['mae']:.2f}  RMSE {r['rmse']:.2f}  R² {r['r2']:.3f}", fontsize=10)
    fig.colorbar(hb, ax=ax, label="images (log)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_by_class(results, names, path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(names))
    keys = [k for k in results if k != "mean_baseline"]
    w = 0.8 / len(keys)
    for i, k in enumerate(keys):
        ax.bar(x + (i - (len(keys) - 1) / 2) * w, [results[k]["mae_by_class"][n] for n in names], w,
               label=results[k]["label"])
    ax.set_xticks(x, [f"{n}\n({lo})" for n, lo in zip(names, ["<5%", "5-15%", "15-30%", ">30%"])])
    ax.set_ylabel("MAE (percentage points)")
    ax.set_title("Prediction error by soiling severity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(path):
    f = config.OUT_DIR / "cnn_history.json"
    if not f.exists():
        return
    h = json.loads(f.read_text())
    x = np.arange(1, len(h) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, [e["train"]["mae"] for e in h], label="train")
    ax.plot(x, [e["val"]["mae"] for e in h], label="val")
    ax.axvline(config.HEAD_EPOCHS + 0.5, ls="--", c="gray", lw=1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MAE (percentage points)")
    ax.set_title("CNN power-loss regressor")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
