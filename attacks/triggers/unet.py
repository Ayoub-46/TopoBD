"""Lightweight U-Net for generative backdoor trigger generation.

Architecture
-----------
Two encoder stages → bottleneck → two decoder stages with skip connections.
Fully convolutional — adapts to any spatial size divisible by 4.
Designed for 32×32 (CIFAR-10, GTSRB) and 28×28 (MNIST, FEMNIST) inputs.

Output is unbounded (no final activation).  The caller is responsible for
clamping after scaling, e.g. ``clamp(x + α·G(x), 0, 1)``.
"""

import torch
import torch.nn as nn


class _ConvBlock(nn.Sequential):
    """Two conv-BN-ReLU layers — the basic U-Net building block."""

    def __init__(self, in_c: int, out_c: int):
        super().__init__(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """U-Net perturbation generator for the IBA trigger.

    Args:
        in_channels:   Number of input/output image channels (1 or 3).
        base_features: Feature map width of the first encoder block.
                       Subsequent blocks double this value.
    """

    def __init__(self, in_channels: int = 3, base_features: int = 32):
        super().__init__()
        b = base_features

        # Encoder
        self.enc1 = _ConvBlock(in_channels, b)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _ConvBlock(b, b * 2)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = _ConvBlock(b * 2, b * 4)

        # Decoder — ConvTranspose2d doubles spatial dims; skip connections
        # concatenate encoder features, doubling the channel count.
        self.up2  = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.dec2 = _ConvBlock(b * 4, b * 2)   # b*2 (up) + b*2 (skip)
        self.up1  = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.dec1 = _ConvBlock(b * 2, b)        # b (up) + b (skip)

        # 1×1 projection to output channels — no activation, unbounded output
        self.out_conv = nn.Conv2d(b, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b  = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)
