"""CNN power-loss regressor: MobileNetV2 image features fused with environment data.

    image -> MobileNetV2 features -> global pool (1280) ─┐
                                                          ├─ concat -> MLP -> sigmoid -> loss fraction
    [irradiance, tod_sin, tod_cos] -> small MLP (32) ─────┘

The backbone starts from the Module 1 soiling classifier, so the network
already knows what dust looks like before it learns to put a number on it.
"""
import torch
from torch import nn
from torchvision import models

from . import config


class PowerLossNet(nn.Module):
    def __init__(self, n_env=len(config.ENV_FEATURES)):
        super().__init__()
        self.features = models.mobilenet_v2(weights=None).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.env = nn.Sequential(nn.Linear(n_env, 32), nn.ReLU(inplace=True))
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280 + 32, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x, env):
        f = self.pool(self.features(x)).flatten(1)
        z = torch.cat([f, self.env(env)], 1)
        return torch.sigmoid(self.head(z)).squeeze(1)   # fraction of power lost, 0..1


def build_model(init_from_module1=True):
    net = PowerLossNet()
    if init_from_module1 and config.MODULE1_CHECKPOINT.exists():
        state = torch.load(config.MODULE1_CHECKPOINT, map_location="cpu")["state_dict"]
        backbone = {k[len("features."):]: v for k, v in state.items() if k.startswith("features.")}
        net.features.load_state_dict(backbone)
    else:
        imagenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
        net.features.load_state_dict(imagenet.features.state_dict())
    return net


def freeze_backbone(net, frozen=True):
    for p in net.features.parameters():
        p.requires_grad = not frozen


def load_checkpoint(path=config.CHECKPOINT, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    net = PowerLossNet()
    net.load_state_dict(ckpt["state_dict"])
    return net.to(device).eval(), ckpt
