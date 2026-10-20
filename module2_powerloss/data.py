"""Dataset for power-loss regression: image + environmental features -> loss %."""
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from module1_detection.data import eval_transform, train_transform

from . import config


def add_env_features(df):
    ts = pd.to_datetime(df["timestamp"])
    frac = (ts.dt.hour + ts.dt.minute / 60 + ts.dt.second / 3600) / 24
    df = df.copy()
    df["tod_sin"] = np.sin(2 * np.pi * frac)
    df["tod_cos"] = np.cos(2 * np.pi * frac)
    return df


def load_index():
    df = pd.read_csv(config.INDEX_CSV)
    return add_env_features(df[df["usable"]]).reset_index(drop=True)


class PowerLossDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.env = torch.tensor(self.frame[config.ENV_FEATURES].to_numpy(dtype=np.float32))
        self.target = torch.tensor(self.frame["power_loss_pct"].to_numpy(dtype=np.float32) / 100.0)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        img = Image.open(config.ROOT / self.frame.at[i, "path"]).convert("RGB")
        return self.transform(img), self.env[i], self.target[i]


def make_loaders(df=None):
    df = load_index() if df is None else df
    parts = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    kw = dict(batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,
              pin_memory=True, persistent_workers=config.NUM_WORKERS > 0)
    return {
        "train": DataLoader(PowerLossDataset(parts["train"], train_transform()), shuffle=True, **kw),
        "val": DataLoader(PowerLossDataset(parts["val"], eval_transform()), shuffle=False, **kw),
        "test": DataLoader(PowerLossDataset(parts["test"], eval_transform()), shuffle=False, **kw),
    }, parts
