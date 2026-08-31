from ..data import NUM_CLASSES
from .attention_unet import AttentionUNet
from .heads import build_backbone_classifier
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
