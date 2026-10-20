"""Predict the power loss (%) of a panel image with the CNN regressor.

Irradiance and time of day are read from DeepSolarEye-style filenames; for
other images pass them yourself.

Usage:
    python -m module2_powerloss.predict panel.jpg --irradiance 0.45 --time 12:30
    python -m module2_powerloss.predict data/raw/images/.../solar_Fri_Jun_16_..._I_0.30.jpg
"""
import argparse
import math

import torch
from PIL import Image

from module1_detection import config as m1
from module1_detection.data import eval_transform
from module1_detection.prepare_data import parse_name

from .model import load_checkpoint


def env_vector(irradiance, hour):
    frac = hour / 24
    return torch.tensor([[irradiance, math.sin(2 * math.pi * frac), math.cos(2 * math.pi * frac)]],
                        dtype=torch.float32)


@torch.no_grad()
def predict(path, irradiance=None, time=None, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _ = load_checkpoint(device=device)
    parsed = parse_name(str(path).replace("\\", "/").rsplit("/", 1)[-1])
    if parsed and irradiance is None:
        irradiance = parsed[2]
    if parsed and time is None:
        ts = parsed[0]
        hour = ts.hour + ts.minute / 60
    else:
        hh, mm = (time or "12:00").split(":")
        hour = int(hh) + int(mm) / 60
    if irradiance is None:
        raise SystemExit("Pass --irradiance (normalised 0-1) for images without DeepSolarEye filenames.")
    x = eval_transform()(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    loss = float(net(x, env_vector(irradiance, hour).to(device))[0]) * 100
    k = sum(loss >= b for b in m1.LOSS_BINS)
    return {"path": str(path), "power_loss_pct": loss, "severity": m1.CLASS_NAMES[k]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--irradiance", type=float)
    ap.add_argument("--time", help="HH:MM, local time the photo was taken")
    a = ap.parse_args()
    for img in a.images:
        r = predict(img, a.irradiance, a.time)
        print(f"{r['path']}: {r['power_loss_pct']:.1f}% power lost ({r['severity']})")
