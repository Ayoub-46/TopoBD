"""ResNet implementations for the FL backdoor research framework.

Supports ResNet-18 and ResNet-34.  The standard torchvision ResNet is designed
for ImageNet (224×224 inputs); this implementation replaces the aggressive
7×7/stride-2 stem with a lightweight 3×3 stem so that small-image datasets
(CIFAR-10/100, 32×32) do not lose spatial resolution before the first residual
block.  For larger inputs (≥ 64px) the full ImageNet stem is used automatically.

References
----------
He et al., "Deep Residual Learning for Image Recognition", CVPR 2016.
"""

from typing import List, Optional, Type, Union

import torch
import torch.nn as nn

from .base import BaseModel, ModelConfig


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    """Standard two-layer residual block (used in ResNet-18 / 34)."""

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


# ---------------------------------------------------------------------------
# ResNet body
# ---------------------------------------------------------------------------

class ResNet(BaseModel):
    """Configurable ResNet that adapts its stem to the input resolution.

    Args:
        config:      :class:`~models.base.ModelConfig` instance.
        block:       Residual block class (only :class:`BasicBlock` for now).
        layers:      Number of blocks per stage, e.g. ``[2, 2, 2, 2]`` for
                     ResNet-18.
        base_width:  Number of filters in the first stage (default 64).

    The stem automatically switches from a lightweight 3×3 convolution (for
    inputs with height < 64px, e.g. CIFAR) to the standard 7×7/stride-2
    ImageNet stem for larger inputs.
    """

    def __init__(
        self,
        config: ModelConfig,
        block: Type[BasicBlock],
        layers: List[int],
        base_width: int = 64,
    ):
        super().__init__(config)
        self._in_channels = base_width
        in_channels = config.input_shape[0]   # C from (C, H, W)
        h = config.input_shape[1]             # H

        # ---- Stem: small-image vs large-image --------------------------------
        if h < 64:
            # CIFAR-style: preserve spatial resolution through the stem
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, base_width, kernel_size=3,
                          stride=1, padding=1, bias=False),
                nn.BatchNorm2d(base_width),
                nn.ReLU(inplace=True),
            )
            self.maxpool = nn.Identity()
        else:
            # ImageNet-style stem
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, base_width, kernel_size=7,
                          stride=2, padding=3, bias=False),
                nn.BatchNorm2d(base_width),
                nn.ReLU(inplace=True),
            )
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ---- Residual stages -------------------------------------------------
        self.layer1 = self._make_layer(block, base_width * 1, layers[0], stride=1)
        self.layer2 = self._make_layer(block, base_width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_width * 8, layers[3], stride=2)

        # ---- Classifier ------------------------------------------------------
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_width * 8 * block.expansion, config.num_classes)

        self.reset_parameters()

    # ------------------------------------------------------------------
    # Layer factory
    # ------------------------------------------------------------------

    def _make_layer(
        self,
        block: Type[BasicBlock],
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self._in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self._in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = [block(self._in_channels, out_channels, stride, downsample)]
        self._in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self._in_channels, out_channels))
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def reset_parameters(self) -> None:
        """Kaiming-normal init for Conv2d, constant init for BN, Xavier for FC."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Public constructors
# ---------------------------------------------------------------------------

def resnet18(config: ModelConfig) -> ResNet:
    """ResNet-18: four stages of [2, 2, 2, 2] BasicBlocks."""
    return ResNet(config, block=BasicBlock, layers=[2, 2, 2, 2])


def resnet34(config: ModelConfig) -> ResNet:
    """ResNet-34: four stages of [3, 4, 6, 3] BasicBlocks."""
    return ResNet(config, block=BasicBlock, layers=[3, 4, 6, 3])