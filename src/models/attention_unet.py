"""Attention U-Net (Oktay et al., 2018): attention gates on the skip connections."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import UNet


class AttentionGate(nn.Module):
    """Additive attention gate: filters skip features using the coarser gating signal."""

    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch)
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        a = self.relu(self.w_g(g) + self.w_x(x))
        return x * self.psi(a)


class AttentionUNet(UNet):
    """Same encoder/decoder/classifier as UNet, but every skip connection is
    re-weighted by an attention gate before concatenation."""

    def __init__(self, in_ch=3, base_channels=32, num_classes=37):
        super().__init__(in_ch=in_ch, base_channels=base_channels, num_classes=num_classes)
        c = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.a4 = AttentionGate(c[3] * 2, c[3], c[3] // 2)
        self.a3 = AttentionGate(c[3], c[2], c[2] // 2)
        self.a2 = AttentionGate(c[2], c[1], c[1] // 2)
        self.a1 = AttentionGate(c[1], c[0], c[0] // 2)

    def forward(self, x):
        skips, b = self.encode(x)
        s1, s2, s3, s4 = skips
        d = self.dec4(torch.cat([self.up4(b), self.a4(b, s4)], dim=1))
        d = self.dec3(torch.cat([self.up3(d), self.a3(d, s3)], dim=1))
        d = self.dec2(torch.cat([self.up2(d), self.a2(d, s2)], dim=1))
        d = self.dec1(torch.cat([self.up1(d), self.a1(d, s1)], dim=1))
        seg_logits = self.seg_head(d)
        cls_logits = self.cls_head(F.adaptive_avg_pool2d(b, 1).flatten(1))
        return {"seg_logits": seg_logits, "cls_logits": cls_logits, "bottleneck": b}
