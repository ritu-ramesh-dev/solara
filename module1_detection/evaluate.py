"""Evaluate the trained classifier on the held-out test set.

Writes to module1_detection/outputs/:
  metrics.json             accuracy, macro precision/recall/F1, per-class report, baseline
  confusion_matrix.png     counts and row-normalised
  training_curves.png      loss / macro-F1 per epoch
  sample_predictions.png   a grid of test images with true vs predicted class

Usage:  python -m module1_detection.evaluate
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from . import config
from .data import make_loaders
from .model import load_checkpoint


@torch.no_grad()
def predict_loader(net, loader, device):
    probs, ys = [], []
    for x, y, _ in loader:
        probs.append(torch.softmax(net(x.to(device)), 1).cpu())
        ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def plot_confusion(cm, names, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names, ax=axes[0])
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=names, yticklabels=names,
                vmin=0, vmax=1, ax=axes[1])
    for ax, t in zip(axes, ["Counts", "Row-normalised (recall)"]):
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(t)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(path):
    hist_file = config.OUT_DIR / "history.json"
    if not hist_file.exists():
        return
    hist = json.loads(hist_file.read_text())
    x = np.arange(1, len(hist) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key, ax in zip(["loss", "f1"], axes):
        ax.plot(x, [h["train"][key] for h in hist], label="train")
        ax.plot(x, [h["val"][key] for h in hist], label="val")
        ax.axvline(config.HEAD_EPOCHS + 0.5, ls="--", c="gray", lw=1)
        ax.set_xlabel("epoch")
        ax.set_title("Loss" if key == "loss" else "Macro-F1")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_samples(test_df, preds, names, path, n=16, seed=config.SEED):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(test_df), size=min(n, len(test_df)), replace=False)
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for ax, i in zip(axes.flat, idx):
        row = test_df.iloc[i]
        ax.imshow(Image.open(config.ROOT / row["path"]).convert("RGB"))
        ok = preds[i] == row["severity"]
        ax.set_title(f"true {names[row['severity']]} ({row['power_loss_pct']:.0f}%)\n"
                     f"pred {names[preds[i]]}", color="green" if ok else "red", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, ckpt = load_checkpoint(device=device)
    names = ckpt["class_names"]
    loaders, parts = make_loaders()
    probs, y = predict_loader(net, loaders["test"], device)
    pred = probs.argmax(1)

    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    labels = list(range(len(names)))
    cm = confusion_matrix(y, pred, labels=labels)
    majority = parts["train"]["severity"].mode()[0]
    metrics = {
        "test_images": int(len(y)),
        "test_days": int(parts["test"]["date"].nunique()),
        "accuracy": accuracy_score(y, pred),
        "macro_precision": p,
        "macro_recall": r,
        "macro_f1": f1,
        "majority_class_baseline_accuracy": float((y == majority).mean()),
        "per_class": classification_report(y, pred, labels=labels, target_names=names,
                                           output_dict=True, zero_division=0),
        "loss_bins_pct": ckpt["loss_bins"],
        "class_names": names,
        "confusion_matrix": cm.tolist(),
    }
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    test = parts["test"].reset_index(drop=True)
    test[["path", "date", "power_loss_pct", "irradiance", "severity"]].assign(
        predicted=pred, confidence=probs.max(1).round(4),
    ).to_csv(config.OUT_DIR / "test_predictions.csv", index=False)

    print(classification_report(y, pred, labels=labels, target_names=names, zero_division=0))
    print(f"Accuracy {metrics['accuracy']:.4f} | macro-F1 {f1:.4f} | "
          f"majority baseline {metrics['majority_class_baseline_accuracy']:.4f}")

    plot_confusion(cm, names, config.OUT_DIR / "confusion_matrix.png")
    plot_history(config.OUT_DIR / "training_curves.png")
    plot_samples(parts["test"].reset_index(drop=True), pred, names, config.OUT_DIR / "sample_predictions.png")
    print(f"Saved figures to {config.OUT_DIR}")


if __name__ == "__main__":
    main()
