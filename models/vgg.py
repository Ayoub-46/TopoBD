"""VGG implementation for the FL backdoor research framework.

Configuration 'B' (VGG-13): two 3x3 conv layers per stage, five stages,
2x2 max-pool between stages. Every conv is followed by BatchNorm + ReLU
("VGG13-BN"), consistent with this framework's ResNet implementation and
standard practice for stable CIFAR-scale training.

Reference
---------
Simonyan and Zisserman, "Very Deep Convolutional Networks for Large-Scale
Image Recognition", ICLR 2015.
"""

from typing import List, Union

import torch
import torch.nn as nn

from .base import BaseModel, ModelConfig

# fmt: off
_VGG13_CFG: List[Union[int, str]] = [
    64, 64, "M",
    128, 128, "M",
    256, 256, "M",
    512, 512, "M",
    512, 512, "M",
]
# fmt: on


class VGG(BaseModel):
    """Configurable VGG, adapted for small (CIFAR-scale) inputs.

    Uses a single ``Linear(512, num_classes)`` classifier after global
    average pooling, not the original three-FC 4096-unit head -- that head
    was sized for ImageNet's 224x224 inputs (7x7 final feature maps); for
    32x32 inputs, the five max-pools already collapse spatial resolution to
    ~1x1, so the original head would be ~120M redundant parameters that
    would also dominate per-round FL communication cost.

    Args:
        config:     :class:`~models.base.ModelConfig` instance.
        cfg:        Layer configuration list (ints = conv output channels,
                    ``"M"`` = 2x2 max-pool).
        batch_norm: When ``True`` (default), every conv is ``bias=False``
                    followed by BatchNorm2d ("VGG-BN"). When ``False``, no
                    BatchNorm is used at all and every conv has ``bias=True``
                    instead -- the original (pre-2015) VGG design, where
                    bias-space is built entirely from per-channel conv
                    biases rather than BatchNorm beta.
    """

    def __init__(
        self, config: ModelConfig, cfg: List[Union[int, str]], batch_norm: bool = True,
    ):
        super().__init__(config)
        in_channels = config.input_shape[0]
        self.batch_norm = batch_norm

        layers: List[nn.Module] = []
        c = in_channels
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(c, v, kernel_size=3, padding=1, bias=not batch_norm))
                if batch_norm:
                    layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(inplace=True))
                c = v
        self.features = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(c, config.num_classes)

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

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


def vgg13(config: ModelConfig) -> VGG:
    """VGG-13 (configuration 'B') with BatchNorm."""
    return VGG(config, cfg=_VGG13_CFG, batch_norm=True)


def vgg13_nobn(config: ModelConfig) -> VGG:
    """VGG-13 (configuration 'B') with no BatchNorm -- bias=True on every conv."""
    return VGG(config, cfg=_VGG13_CFG, batch_norm=False)
