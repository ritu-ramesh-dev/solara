"""Fine-tune MobileNetV2 on DeepSolarEye soiling-severity classes.

Two phases:
  1. backbone frozen, train only the new head (fast, stable start)
  2. unfreeze everything and fine-tune with a lower learning rate + cosine decay

The checkpoint with the best validation macro-F1 is saved.

Usage:  python -m module1_detection.train
"""
import json
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from . import config
from .data import make_loaders
from .model import build_model, freeze_backbone


def run_epoch(net, loader, device, criterion, optimizer=None, scaler=None):
    training = optimizer is not None
    net.train(training)
    total_loss, preds, targets = 0.0, [], []
    with torch.set_grad_enabled(training):
        for x, y, _ in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = net(x)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * len(y)
            preds.append(logits.argmax(1).cpu())
            targets.append(y.cpu())
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return {
        "loss": total_loss / len(targets),
        "acc": float((preds == targets).mean()),
        "f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
    }


def main():
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loaders, parts = make_loaders()
    for s, d in parts.items():
        print(f"{s:5s}: {len(d):6d} images, {d['date'].nunique()} days")

    net = build_model().to(device)
    # The sampler already balances classes, so a plain loss is used here.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)

    history, best_f1 = [], -1.0
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
            if va["f1"] > best_f1:
                best_f1 = va["f1"]
                torch.save({
                    "state_dict": net.state_dict(),
                    "class_names": config.CLASS_NAMES,
                    "loss_bins": config.LOSS_BINS,
                    "img_size": config.IMG_SIZE,
                }, config.CHECKPOINT)
                flag = "  * saved"
            print(f"[{phase} {epoch:2d}/{epochs}] "
                  f"train loss {tr['loss']:.3f} acc {tr['acc']:.3f} f1 {tr['f1']:.3f} | "
                  f"val loss {va['loss']:.3f} acc {va['acc']:.3f} f1 {va['f1']:.3f} "
                  f"({time.time() - t0:.0f}s){flag}")

    with open(config.OUT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Best val macro-F1: {best_f1:.4f}. Checkpoint: {config.CHECKPOINT}")


if __name__ == "__main__":
    main()
