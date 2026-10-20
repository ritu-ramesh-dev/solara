"""PyTorch dataset and transforms for the indexed DeepSolarEye images."""
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from . import config

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class AddGaussianNoise:
    def __init__(self, std=0.03):
        self.std = std

    def __call__(self, x):
        return (x + torch.randn_like(x) * self.std).clamp(0, 1)


def train_transform():
    # Augmentations for lighting and camera-angle robustness (report Section 3.2).
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomRotation(8),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.25, saturation=0.15, hue=0.02),
        transforms.ToTensor(),
        AddGaussianNoise(0.02),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class SoilingDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        img = Image.open(config.ROOT / row["path"]).convert("RGB")
        return self.transform(img), int(row["severity"]), float(row["power_loss_pct"])


def load_index(usable_only=True):
    df = pd.read_csv(config.INDEX_CSV)
    return df[df["usable"]] if usable_only else df


def class_weights(train_df):
    counts = train_df["severity"].value_counts().reindex(range(len(config.CLASS_NAMES)), fill_value=0)
    w = counts.sum() / (len(counts) * counts.clip(lower=1))
    return torch.tensor(w.values, dtype=torch.float32)


def make_loaders(df=None):
    df = load_index() if df is None else df
    parts = {s: df[df["split"] == s] for s in ("train", "val", "test")}

    # Oversample rare severities so every class is seen evenly during training.
    w = class_weights(parts["train"])[parts["train"]["severity"].values]
    sampler = WeightedRandomSampler(w.double(), num_samples=len(w), replacement=True)

    kw = dict(batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,
              pin_memory=True, persistent_workers=config.NUM_WORKERS > 0)
    return {
        "train": DataLoader(SoilingDataset(parts["train"], train_transform()), sampler=sampler, **kw),
        "val": DataLoader(SoilingDataset(parts["val"], eval_transform()), shuffle=False, **kw),
        "test": DataLoader(SoilingDataset(parts["test"], eval_transform()), shuffle=False, **kw),
    }, parts
