import torch
import torch.nn as nn

from .base import BaseModel, ModelConfig


class GTSRBNet(BaseModel):
    """Two-stage CNN for GTSRB (32×32 RGB → 43 classes).

    Architecture matches the user's prior work that achieved ~94% clean accuracy:
        Conv(3→32, 5×5) → ReLU → MaxPool(2) → Dropout(0.25)
        Conv(32→64, 3×3) → ReLU → MaxPool(2) → Dropout(0.25)
        Linear(8*8*64→512) → ReLU → Dropout(0.5) → Linear(512→num_classes)
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        in_c = config.input_shape[0]

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_c, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(8 * 8 * 64, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, config.num_classes),
        )
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.flatten(1)
        return self.fc_layers(x)

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
