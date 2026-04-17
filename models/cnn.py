"""Lightweight CNN for fast ablation runs and sanity checks.

``SimpleCNN`` is a small but non-trivial convolutional network that trains
quickly on CIFAR-10 / MNIST while still being representive enough that
backdoor attacks behave realistically.  It is not intended to match
state-of-the-art accuracy; use ResNet-18 for final results.

Architecture (default, CIFAR-10)::

    Conv(3→32, 3×3) → BN → ReLU → MaxPool(2)
    Conv(32→64, 3×3) → BN → ReLU → MaxPool(2)
    Conv(64→128, 3×3) → BN → ReLU → AdaptiveAvgPool(1)
    Linear(128 → num_classes)

The number of conv stages and filter widths are configurable via
``ModelConfig.kwargs`` so the same class covers MNIST and CIFAR without
subclassing.
"""

from typing import List

import torch
import torch.nn as nn

from .base import BaseModel, ModelConfig


class SimpleCNN(BaseModel):
    """Configurable small CNN.

    Accepted ``ModelConfig.kwargs``:

    ``channels`` (``List[int]``, default ``[32, 64, 128]``):
        Output channels for each convolutional stage.
    ``dropout`` (``float``, default ``0.0``):
        Dropout probability applied before the final linear layer.
        Use ``0.5`` to regularise on small datasets.

    Example::

        cfg = ModelConfig.from_adapter("simple_cnn", adapter,
                                       channels=[32, 64], dropout=0.3)
        model = get_model(cfg)
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        channels: List[int] = config.kwargs.get("channels", [32, 64, 128])
        dropout: float = float(config.kwargs.get("dropout", 0.0))

        if len(channels) < 1:
            raise ValueError("SimpleCNN requires at least one convolutional stage.")

        in_c = config.input_shape[0]   # number of input channels (C)
        stages: List[nn.Module] = []

        for out_c in channels:
            stages += [
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
            in_c = out_c

        # Collapse remaining spatial dimensions regardless of input resolution
        stages.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.features = nn.Sequential(*stages)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.classifier = nn.Linear(in_c, config.num_classes)

        self.reset_parameters()

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)

    def reset_parameters(self) -> None:
        """Kaiming-normal for Conv2d, constant for BN, Xavier for Linear."""
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