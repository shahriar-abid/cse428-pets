"""Model factory + bonus-task classifier backbones."""

from ..data import NUM_CLASSES
from .attention_unet import AttentionUNet
from .unet import UNet

__all__ = ["UNet", "AttentionUNet", "build_model", "build_backbone_classifier"]


def build_model(cfg):
    name = cfg["model"]["name"]
    kwargs = dict(
        base_channels=cfg["model"].get("base_channels", 32),
        num_classes=cfg["model"].get("num_classes", NUM_CLASSES),
    )
    if name == "unet":
        return UNet(**kwargs)
    if name == "attention_unet":
        return AttentionUNet(**kwargs)
    raise ValueError(f"unknown model: {name}")


def build_backbone_classifier(name, num_classes=NUM_CLASSES, pretrained=True):
    """Bonus: standalone classifiers (ResNet18 / MobileNetV3 / EfficientNet-B0)
    for the architecture-comparison task. Usable with the same Trainer."""
    import torch.nn as nn
    from torchvision.models import (
        EfficientNet_B0_Weights,
        MobileNet_V3_Small_Weights,
        ResNet18_Weights,
        efficientnet_b0,
        mobilenet_v3_small,
        resnet18,
    )

    if name == "resnet18":
        m = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "mobilenet_v3_small":
        m = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        )
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif name == "efficientnet_b0":
        m = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"unknown backbone: {name}")
    return m
