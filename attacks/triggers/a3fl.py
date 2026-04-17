"""A3FL adversarially adaptive trigger.

Reference
---------
Zhang et al., "A3FL: Adversarially Adaptive Backdoor Attacks to
Federated Learning", NeurIPS 2023.
https://github.com/sail-sg/A3FL

Design notes
------------
* The adversarial model is built once (before the trigger optimisation loop)
  using the initial pattern.  This matches the official repository's
  ``get_adversarial_model`` call structure.
* Cosine similarity between the original and adversarial model is computed
  once before the optimisation loop — both models are frozen, so the
  similarity is constant across all epochs and batches.
* Pattern updates use a projected-gradient-descent (PGD) sign step, as in
  the official implementation.
* The clamp that was previously applied after the blending formula has been
  removed.  Trigger values must already live in the same space as the
  incoming image tensors (see value-range contract in ``base.py``).
"""

import copy
import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .base import LearnableTrigger, Tensor

logger = logging.getLogger(__name__)


class A3FLTrigger(LearnableTrigger):
    """Adversarially adaptive learnable backdoor trigger (A3FL).

    The trigger is a full-image-sized learnable pattern applied through a
    binary mask.  It is optimised so that it:

    1. Causes the victim model to predict ``target_class`` on triggered images
       (*backdoor loss*).
    2. Remains effective even against a hardened adversarial model trained to
       resist the trigger (*adaptation loss*).

    Args:
        position:         Top-left ``(x, y)`` corner of the trigger patch.
        size:             ``(width, height)`` of the trigger patch.
        in_channels:      Number of image channels (1 for greyscale, 3 for RGB).
        image_size:       ``(H, W)`` of the full image.
        trigger_epochs:   Number of outer optimisation epochs.
        trigger_lr:       PGD step size for the pattern update.
        lambda_balance:   Scaling factor for the adaptation loss term.
        adv_epochs:       Number of fine-tuning epochs for the adversarial model.
        adv_lr:           Learning rate for the adversarial model fine-tuning.
        alpha:            Blending factor (passed to :class:`BaseTrigger`).
    """

    def __init__(
        self,
        position: Tuple[int, int] = (2, 2),
        size: Tuple[int, int] = (5, 5),
        in_channels: int = 3,
        image_size: Tuple[int, int] = (32, 32),
        trigger_epochs: int = 10,
        trigger_lr: float = 0.01,
        lambda_balance: float = 0.1,
        adv_epochs: int = 100,
        adv_lr: float = 0.01,
        alpha: float = 1.0,
    ):
        # Initialise pattern to 0.5 (mid-range), matching the official repo.
        # Shape: (C, H, W) — full image size, not patch size.
        initial_pattern = torch.full(
            (in_channels, image_size[0], image_size[1]), fill_value=0.5
        )

        # Binary mask: 1 inside the patch, 0 outside.
        # position = (x, y) → tensor[:, y : y+h, x : x+w]
        x, y = position
        w, h = size
        mask = torch.zeros(in_channels, image_size[0], image_size[1])
        mask[:, y : y + h, x : x + w] = 1.0
        self.mask = mask

        super().__init__(position, size, initial_pattern, alpha)

        self.trigger_epochs = trigger_epochs
        self.trigger_lr = trigger_lr
        self.lambda_balance = lambda_balance
        self.adv_epochs = adv_epochs
        self.adv_lr = adv_lr

    # ------------------------------------------------------------------
    # LearnableTrigger interface
    # ------------------------------------------------------------------

    def train_trigger(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        target_class: int,
    ) -> None:
        """Optimise the trigger pattern using the A3FL adversarial scheme.

        Args:
            model:        Victim model (frozen during optimisation).
            dataloader:   Client's local training DataLoader.
            target_class: Backdoor target class index.
        """
        device = next(model.parameters()).device
        pattern = self.pattern.clone().detach().to(device).requires_grad_(True)

        logger.info("A3FL: building adversarial model (%d epochs).", self.adv_epochs)
        adversarial_model = self._build_adversarial_model(model, dataloader, pattern)

        # Freeze the victim model for the duration of trigger optimisation.
        self._set_grad(model, requires_grad=False)
        model.eval()

        loss_fn = nn.CrossEntropyLoss()

        # FIX: similarity is constant (both models frozen) — compute once.
        similarity = self._cosine_similarity(model, adversarial_model)
        dynamic_lambda = float(self.lambda_balance * similarity)
        logger.info(
            "A3FL: model similarity=%.4f, dynamic_lambda=%.4f.",
            similarity.item(),
            dynamic_lambda,
        )

        logger.info("A3FL: optimising trigger (%d epochs).", self.trigger_epochs)
        for epoch in range(self.trigger_epochs):
            epoch_loss = 0.0
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                poisoned = self._blend(inputs, pattern)
                targets = torch.full(
                    (inputs.size(0),), target_class, dtype=torch.long, device=device
                )

                backdoor_loss = loss_fn(model(poisoned), targets)
                adapt_loss = loss_fn(adversarial_model(poisoned), targets)
                total_loss = backdoor_loss + dynamic_lambda * adapt_loss

                # Manual PGD step — pattern is a leaf tensor, not in any optimizer
                if pattern.grad is not None:
                    pattern.grad.zero_()
                total_loss.backward()

                with torch.no_grad():
                    pattern.sub_(self.trigger_lr * pattern.grad.sign())
                    # No hard clamp here — see value-range contract in base.py.
                    # If operating in raw-pixel [0,1] space, uncomment:
                    # pattern.clamp_(0.0, 1.0)

                epoch_loss += total_loss.item()

            logger.debug(
                "A3FL epoch [%d/%d]: avg_loss=%.4f",
                epoch + 1,
                self.trigger_epochs,
                epoch_loss / max(len(dataloader), 1),
            )

        # Restore model state
        self._set_grad(model, requires_grad=True)

        # Persist optimised pattern (CPU, detached from computation graph)
        self.pattern = pattern.detach().cpu()
        logger.info("A3FL: trigger optimisation complete.")

    # ------------------------------------------------------------------
    # BaseTrigger interface
    # ------------------------------------------------------------------

    def apply(self, image: Tensor) -> Tensor:
        """Apply the optimised trigger to a single ``(C, H, W)`` image.

        Args:
            image: Float tensor of shape ``(C, H, W)``.

        Returns:
            Triggered image on the same device as ``image``.

        Raises:
            ValueError: if ``image`` is not a 3-D tensor.
        """
        if image.dim() != 3:
            raise ValueError(
                f"apply() expects a 3-D tensor (C, H, W), got shape {image.shape}."
            )
        # FIX: delegate to _blend to avoid duplicating the blending formula.
        return self._blend(image.unsqueeze(0), self.pattern).squeeze(0)

    # Override with a vectorised version for performance.
    def apply_batch(self, images: Tensor) -> Tensor:
        """Apply the trigger to a batch ``(N, C, H, W)`` of images."""
        return self._blend(images, self.pattern)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _blend(self, images: Tensor, pattern: Tensor) -> Tensor:
        """Alpha-blend ``pattern`` into ``images`` using the binary mask.

        Works for both single images ``(C, H, W)`` and batches ``(N, C, H, W)``.
        Moves ``pattern`` and ``mask`` to the same device as ``images``.
        Does NOT clamp — see value-range contract.
        """
        p = pattern.to(images.device)
        m = self.mask.to(images.device)
        return images * (1.0 - m) + p * m * self.alpha

    def _build_adversarial_model(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        pattern: Tensor,
    ) -> nn.Module:
        """Return a copy of ``model`` fine-tuned to resist the current trigger.

        The adversarial model is trained on triggered images labelled with
        their *original* (clean) labels, forcing it to learn to ignore the
        trigger.  This is then used in the outer optimisation loop to push
        the trigger to be resilient against such hardening.
        """
        adv_model = copy.deepcopy(model)
        device = next(adv_model.parameters()).device
        self._set_grad(adv_model, requires_grad=True)
        adv_model.train()

        optimizer = optim.SGD(
            adv_model.parameters(), lr=self.adv_lr, momentum=0.9, weight_decay=5e-4
        )
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(self.adv_epochs):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                # Triggered images, but trained to predict ORIGINAL labels —
                # this teaches the adversarial model to ignore the trigger.
                poisoned = self._blend(inputs, pattern.detach())
                optimizer.zero_grad()
                loss_fn(adv_model(poisoned), labels).backward()
                optimizer.step()

        self._set_grad(adv_model, requires_grad=False)
        adv_model.eval()
        return adv_model

    @staticmethod
    def _cosine_similarity(model_a: nn.Module, model_b: nn.Module) -> Tensor:
        """Scalar cosine similarity between the flattened parameter vectors
        of two models.
        """
        params_a = torch.cat([p.detach().view(-1) for p in model_a.parameters()])
        params_b = torch.cat(
            [p.detach().to(params_a.device).view(-1) for p in model_b.parameters()]
        )
        return F.cosine_similarity(params_a, params_b, dim=0, eps=1e-8)

    @staticmethod
    def _set_grad(model: nn.Module, requires_grad: bool) -> None:
        for param in model.parameters():
            param.requires_grad = requires_grad