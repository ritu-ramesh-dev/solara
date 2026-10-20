"""Classify the soiling severity of one or more panel images.

Usage:  python -m module1_detection.predict path/to/panel.jpg [more.jpg ...]
"""
import argparse

import torch
from PIL import Image

from .data import eval_transform
from .model import load_checkpoint


@torch.no_grad()
def predict(paths, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, ckpt = load_checkpoint(device=device)
    tf = eval_transform()
    results = []
    for path in paths:
        x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        probs = torch.softmax(net(x), 1)[0].cpu()
        k = int(probs.argmax())
        results.append({
            "path": str(path),
            "severity": ckpt["class_names"][k],
            "confidence": float(probs[k]),
            "probabilities": dict(zip(ckpt["class_names"], map(float, probs))),
        })
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    for r in predict(ap.parse_args().images):
        probs = "  ".join(f"{k} {v:.2f}" for k, v in r["probabilities"].items())
        print(f"{r['path']}: {r['severity']} ({r['confidence']:.1%})  [{probs}]")
