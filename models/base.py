"""Base class and configuration dataclass for all models in the framework.

Every model registered in the model registry must subclass :class:`BaseModel`.
The :class:`ModelConfig` dataclass is the single source of truth for
architecture hyperparameters and is the object passed to the experiment runner
so that configs can be serialised/logged without carrying live objects.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Hyperparameters that fully describe a model architecture.

    Designed to be constructed from a :class:`~datasets.adapter.DatasetAdapter`
    so that architecture decisions (number of output neurons, expected input
    shape) are always consistent with the dataset being used.

    Args:
        name:        Registry key used to look this architecture up via
                     :func:`~models.get_model`.
        num_classes: Number of output logits (= dataset number of classes).
        input_shape: CHW shape of a single input sample, e.g. ``(3, 32, 32)``.
        kwargs:      Any additional architecture-specific hyperparameters
                     (e.g. ``{"num_layers": 18}`` for ResNet).

    Example::

        adapter = CIFAR10Dataset()
        cfg = ModelConfig.from_adapter("resnet", adapter, num_layers=18)
        model = get_model(cfg)
    """

    name: str
    num_classes: int
    input_shape: Tuple[int, ...]
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_adapter(
        cls,
        name: str,
        adapter,          # DatasetAdapter — avoid circular import with string hint
        **kwargs: Any,
    ) -> "ModelConfig":
        """Convenience constructor that pulls ``num_classes`` and ``input_shape``
        directly from a :class:`~datasets.adapter.DatasetAdapter` instance,
        eliminating the risk of a mismatch between dataset and model.
        """
        return cls(
            name=name,
            num_classes=adapter.num_classes,
            input_shape=adapter.input_shape,
            kwargs=kwargs,
        )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseModel(nn.Module, ABC):
    """Abstract base for all models in the framework.

    Extends ``nn.Module`` with:

    * A :meth:`reset_parameters` contract so the experiment runner can
      re-initialise a model between trials without reconstructing it.
    * A :meth:`num_parameters` convenience property.
    * A :meth:`output_shape` abstract property so downstream code (e.g. attack
      clients that need to inspect logit dimensions) can introspect without
      running a forward pass.

    Subclasses must call ``super().__init__(config)`` and implement
    :meth:`forward` and :meth:`reset_parameters`.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass.  Input shape must match ``config.input_shape``
        (ignoring the batch dimension).
        """

    @abstractmethod
    def reset_parameters(self) -> None:
        """Re-initialise all learnable parameters to their default values.

        Must be idempotent: calling it twice is equivalent to calling it once.
        The experiment runner calls this between independent trials to ensure
        each run starts from the same random state (given the same seed).
        """

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    @property
    def num_parameters(self) -> int:
        """Total number of learnable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_classes(self) -> int:
        return self.config.num_classes

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return self.config.input_shape

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}, "
            f"input_shape={self.input_shape}, "
            f"params={self.num_parameters:,})"
        )