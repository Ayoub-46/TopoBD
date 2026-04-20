import torch
import torch.nn as nn

from .base import BaseModel, ModelConfig


class LeNet5(BaseModel):
    """LeNet-5 adapted for variable input shapes and class counts.

    Architecture (for 1×28×28 FEMNIST input):
        Conv(in_c→6, 5×5, pad=2) → Tanh → AvgPool(2)   [→ 6×14×14]
        Conv(6→16, 5×5)          → Tanh → AvgPool(2)   [→ 16×5×5 = 400]
        Linear(400→120) → Tanh → Linear(120→84) → Tanh → Linear(84→num_classes)

    The padding=2 on the first conv keeps spatial dimensions intact so that a
    28×28 input produces the classic 400-d embedding before the FC layers.
    The flatten size is inferred from input_shape at construction time, so the
    same class works for any spatial resolution.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        in_c, H, W = config.input_shape

        self.features = nn.Sequential(
            nn.Conv2d(in_c, 6, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
        )

        # Derive the flatten dimension without hardcoding it
        with torch.no_grad():
            flat_dim = self.features(torch.zeros(1, in_c, H, W)).flatten(1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, config.num_classes),
        )

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="tanh")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
