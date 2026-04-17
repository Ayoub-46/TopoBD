"""Base classes for the trigger subsystem.

Hierarchy
---------
:class:`BaseTrigger`
    Abstract root.  Defines the ``apply`` contract and provides a default
    no-op ``train_trigger`` so that static triggers do not need to implement it.

:class:`LearnableTrigger`
    Intermediate ABC for triggers whose pattern must be optimised before use
    (e.g. A3FL, WaNet, Input-Aware).  Marks ``is_static = False`` and makes
    ``train_trigger`` abstract.

Value-range contract
--------------------
``apply`` and ``train_trigger`` operate in **whatever pixel space the
incoming tensors occupy**.  In this framework, ``BackdoorDataset`` calls
``trigger_fn`` on already-transformed (normalised) tensors, so trigger
implementations must NOT hard-clamp outputs to ``[0, 1]``.  If a trigger
is intended for a raw-pixel ``[0, 1]`` workflow, the caller is responsible
for applying it before the normalisation transform.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch

Tensor = torch.Tensor


class BaseTrigger(ABC):
    """Abstract base class for all backdoor triggers.

    Args:
        position: Top-left corner ``(x, y)`` of the trigger patch where
                  ``x`` is the column offset from the left and ``y`` is the
                  row offset from the top.  Maps to tensor indices as
                  ``tensor[:, y : y+height, x : x+width]``.
        size:     ``(width, height)`` of the trigger patch in pixels.
        pattern:  Initial trigger pattern tensor of shape ``(C, H, W)``
                  matching the full image size.  Subclasses may replace this
                  with a learnable parameter after construction.
        alpha:    Blending factor ``∈ [0, 1]``.  ``1.0`` (default) means the
                  trigger fully replaces masked pixels; ``< 1.0`` blends the
                  trigger with the original image.
    """

    def __init__(
        self,
        position: Tuple[int, int],
        size: Tuple[int, int],
        pattern: Tensor,
        alpha: float = 1.0,
    ):
        self.position = position   # (x, y)  →  (col_offset, row_offset)
        self.size = size           # (width, height)
        self.pattern = pattern     # (C, H, W)
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def apply(self, image: Tensor) -> Tensor:
        """Apply the trigger to a single image tensor.

        Args:
            image: Float tensor of shape ``(C, H, W)`` on any device.
                   Must already be in the same value space as ``self.pattern``
                   (normalised or raw pixel).

        Returns:
            Triggered image of the same shape and device as ``image``.
            Must not alter the input tensor in-place.
        """

    # ------------------------------------------------------------------
    # Optional training interface
    # ------------------------------------------------------------------

    def train_trigger(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        target_class: int,
    ) -> None:
        """Optimise the trigger pattern (no-op for static triggers).

        Static triggers do not need to override this method.  Learnable
        triggers should subclass :class:`LearnableTrigger` instead of
        overriding this directly, so that the ``is_static`` property is set
        correctly.

        Args:
            model:        The current global model on the client device.
            dataloader:   Client's local training DataLoader.
            target_class: Target class index for the backdoor.
        """

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_static(self) -> bool:
        """``True`` for fixed patterns, ``False`` for patterns that require
        optimisation via :meth:`train_trigger` before use.
        """
        return True

    # ------------------------------------------------------------------
    # Convenience: batch application
    # ------------------------------------------------------------------

    def apply_batch(self, images: Tensor) -> Tensor:
        """Apply the trigger to a batch of images.

        The default implementation loops over the batch dimension by calling
        :meth:`apply`.  Subclasses may override with a fully-vectorised
        version for performance.

        Args:
            images: Float tensor of shape ``(N, C, H, W)``.

        Returns:
            Triggered batch of the same shape and device.
        """
        return torch.stack([self.apply(img) for img in images])


# ---------------------------------------------------------------------------
# Intermediate ABC for learnable triggers
# ---------------------------------------------------------------------------

class LearnableTrigger(BaseTrigger, ABC):
    """Abstract base for triggers that require adversarial optimisation.

    Subclasses must implement both :meth:`apply` and :meth:`train_trigger`.
    The :attr:`is_static` property is fixed to ``False``.
    """

    @property
    def is_static(self) -> bool:
        return False

    @abstractmethod
    def train_trigger(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        target_class: int,
    ) -> None:
        """Optimise the trigger pattern against ``model``."""