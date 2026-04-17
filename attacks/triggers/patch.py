"""Simple patch trigger (BadNets / NeuralToxin / Model Replacement style).

Operates in raw ``[0, 1]`` pixel space — apply before ``Normalize``.
"""

import torch

from .base import BaseTrigger


class PatchTrigger(BaseTrigger):
    """A solid-colour rectangular patch placed at a fixed position.

    This is the canonical trigger used in BadNets, NeuralToxin, and
    model-replacement attacks.

    Args:
        position: Top-left corner ``(x, y)`` in pixel coordinates.
        size:     ``(width, height)`` of the patch in pixels.
        color:    ``(R, G, B)`` fill colour in **raw [0, 1] space**.
                  To use a colour specified in uint8 [0, 255] space, divide
                  each channel by 255 before passing it here.
        alpha:    Blending factor — ``1.0`` (default) fully replaces the patch
                  region; ``< 1.0`` blends with the original pixels.
    """

    def __init__(
        self,
        position: tuple = (28, 28),
        size: tuple = (3, 3),
        color: tuple = (1.0, 0.0, 0.0),
        alpha: float = 1.0,
    ):
        # Pattern shape is (C, 1, 1) — broadcasts over the patch ROI.
        # The base-class docstring says (C, H, W) for the full image; this
        # sub-class uses a compact representation for static solid colours.
        pattern = torch.tensor(color, dtype=torch.float32).view(-1, 1, 1)
        super().__init__(position, size, pattern, alpha)

    # is_static == True is already provided by BaseTrigger; no need to set it
    # as an instance attribute (doing so would clash with the property).

    def apply(self, image: torch.Tensor) -> torch.Tensor:
        """Apply the patch to a single ``(C, H, W)`` image in [0, 1] space.

        The patch is clipped to the image boundary so that out-of-bounds
        positions are silently handled rather than raising an exception.

        Args:
            image: Float tensor of shape ``(C, H, W)`` with values in [0, 1].

        Returns:
            Cloned tensor with the patch applied.  Input is not modified.
        """
        if image.dim() != 3:
            raise ValueError(
                f"apply() expects a 3-D tensor (C, H, W), got {image.shape}."
            )

        poisoned = image.clone()
        _, img_h, img_w = poisoned.shape
        x, y = self.position
        w, h = self.size

        # Clip patch coordinates to valid image bounds
        y0 = max(0, min(y, img_h))
        y1 = max(0, min(y + h, img_h))
        x0 = max(0, min(x, img_w))
        x1 = max(0, min(x + w, img_w))

        if y1 <= y0 or x1 <= x0:
            # Patch is entirely outside the image — return unchanged clone
            return poisoned

        pattern = self.pattern.to(image.device)   # (C, 1, 1) broadcasts
        region  = poisoned[:, y0:y1, x0:x1]       # (C, roi_h, roi_w)

        poisoned[:, y0:y1, x0:x1] = (
            self.alpha * pattern + (1.0 - self.alpha) * region
        ).clamp(0.0, 1.0)   # clamp is valid: we are in [0, 1] pre-norm space

        return poisoned