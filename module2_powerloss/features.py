"""Extract image embeddings from the Module 1 classifier for the tree models.

Each usable image goes through Module 1's MobileNetV2 backbone; the 1280-d
global-pooled feature vector (the layer right before the classifier head) is
cached in data/module1_embeddings.npz.

Usage:  python -m module2_powerloss.features
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from module1_detection.data import eval_transform
from module1_detection.model import load_checkpoint

from . import config
from .data import PowerLossDataset, load_index


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _ = load_checkpoint(config.MODULE1_CHECKPOINT, device=device)
    df = load_index()
    loader = DataLoader(PowerLossDataset(df, eval_transform()), batch_size=128,
                        num_workers=config.NUM_WORKERS, pin_memory=True)
    feats = []
    for i, (x, _, _) in enumerate(loader):
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            f = net.features(x.to(device, non_blocking=True))
        feats.append(f.float().mean((2, 3)).cpu().numpy())
        if i % 50 == 0:
            print(f"{i * 128:6d} / {len(df)}")
    emb = np.concatenate(feats).astype(np.float32)
    np.savez_compressed(config.EMBEDDINGS, paths=df["path"].to_numpy(), embeddings=emb)
    print(f"Saved {emb.shape} embeddings to {config.EMBEDDINGS}")


if __name__ == "__main__":
    main()
