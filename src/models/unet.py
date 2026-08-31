"""U-Net (Ronneberger et al., 2015) with a breed-classifier head on the encoder output."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv 3x3 -> BatchNorm -> ReLU) x2, the basic U-Net block."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Joint model: binary segmentation decoder + 37-breed classifier on the bottleneck.

    Returns {"seg_logits": (B,1,H,W), "cls_logits": (B,num_classes), "bottleneck": ...}.
    Input H and W must be divisible by 16.
    """

    def __init__(self, in_ch=3, base_channels=32, num_classes=37):
        super().__init__()
        c = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.enc1 = DoubleConv(in_ch, c[0])
        self.enc2 = DoubleConv(c[0], c[1])
        self.enc3 = DoubleConv(c[1], c[2])
        self.enc4 = DoubleConv(c[2], c[3])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(c[3], c[3] * 2)

        self.up4 = nn.ConvTranspose2d(c[3] * 2, c[3], 2, stride=2)
        self.dec4 = DoubleConv(c[3] * 2, c[3])
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.dec3 = DoubleConv(c[2] * 2, c[2])
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.dec2 = DoubleConv(c[1] * 2, c[1])
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.dec1 = DoubleConv(c[0] * 2, c[0])

        self.seg_head = nn.Conv2d(c[0], 1, 1)
        self.cls_head = nn.Linear(c[3] * 2, num_classes)

    def encode(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        b = self.bottleneck(self.pool(s4))
        return (s1, s2, s3, s4), b

    def decode(self, b, skips):
        s1, s2, s3, s4 = skips
        x = self.dec4(torch.cat([self.up4(b), s4], dim=1))
        x = self.dec3(torch.cat([self.up3(x), s3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), s2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), s1], dim=1))
        return x

    def forward(self, x):
        skips, b = self.encode(x)
        seg_logits = self.seg_head(self.decode(b, skips))
        cls_logits = self.cls_head(F.adaptive_avg_pool2d(b, 1).flatten(1))
        return {"seg_logits": seg_logits, "cls_logits": cls_logits, "bottleneck": b}
