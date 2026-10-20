"""Train the CNN power-loss regressor.

Two phases, as in Module 1:
  1. backbone frozen (it starts from the Module 1 classifier), train the
     environment branch and regression head
  2. unfreeze everything, lower learning rate, cosine decay

Loss: Smooth L1 (Huber) on the loss fraction, robust to the few noisy labels.
The checkpoint with the lowest validation MAE is saved.

Usage:  python -m module2_powerloss.train_cnn
"""
import json
import time

import numpy as np
import torch
from torch import nn

from . import config
from .data import make_loaders
from .model import build_model, freeze_backbone


def run_epoch(net, loader, device, criterion, optimizer=None, scaler=None):
    training = optimizer is not None
    net.train(training)
    total, preds, targets = 0.0, [], []
    with torch.set_grad_enabled(training):
        for x, env, y in loader:
            x, env, y = (t.to(device, non_blocking=True) for t in (x, env, y))
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                out = net(x, env)
            loss = criterion(out.float(), y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total += loss.item() * len(y)
            preds.append(out.detach().float().cpu())
            targets.append(y.cpu())
    p, t = torch.cat(preds).numpy() * 100, torch.cat(targets).numpy() * 100
    return {
        "loss": total / len(t),
        "mae": float(np.abs(p - t).mean()),
        "rmse": float(np.sqrt(((p - t) ** 2).mean())),
    }


def main():
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loaders, parts = make_loaders()
    for s, d in parts.items():
        print(f"{s:5s}: {len(d):6d} images")

    net = build_model().to(device)
    criterion = nn.SmoothL1Loss(beta=0.05)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)

    history, best = [], float("inf")
    phases = [
        ("head", config.HEAD_EPOCHS, config.HEAD_LR, True),
        ("finetune", config.FINETUNE_EPOCHS, config.FINETUNE_LR, False),
    ]
    for phase, epochs, lr, frozen in phases:
        freeze_backbone(net, frozen)
        params = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr = run_epoch(net, loaders["train"], device, criterion, optimizer, scaler)
            va = run_epoch(net, loaders["val"], device, criterion)
            scheduler.step()
            history.append({"phase": phase, "epoch": epoch, "train": tr, "val": va})
            flag = ""
            if va["mae"] < best:
                best = va["mae"]
                torch.save({"state_dict": net.state_dict(), "env_features": config.ENV_FEATURES}, config.CHECKPOINT)
                flag = "  * saved"
            print(f"[{phase} {epoch:2d}/{epochs}] "
                  f"train MAE {tr['mae']:.2f} RMSE {tr['rmse']:.2f} | "
                  f"val MAE {va['mae']:.2f} RMSE {va['rmse']:.2f} ({time.time() - t0:.0f}s){flag}")

    (config.OUT_DIR / "cnn_history.json").write_text(json.dumps(history, indent=2))
    print(f"Best val MAE: {best:.3f} percentage points. Checkpoint: {config.CHECKPOINT}")


if __name__ == "__main__":
    main()
