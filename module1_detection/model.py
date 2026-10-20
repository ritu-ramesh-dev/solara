"""MobileNetV2 transfer-learning classifier for soiling severity."""
import torch
from torch import nn
from torchvision import models

from . import config


def build_model(num_classes=len(config.CLASS_NAMES), pretrained=True):
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    net = models.mobilenet_v2(weights=weights)
    in_features = net.classifier[1].in_features
    net.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return net


def freeze_backbone(net, frozen=True):
    for p in net.features.parameters():
        p.requires_grad = not frozen


def load_checkpoint(path=config.CHECKPOINT, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    net = build_model(len(ckpt["class_names"]), pretrained=False)
    net.load_state_dict(ckpt["state_dict"])
    return net.to(device).eval(), ckpt
