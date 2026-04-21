"""IBA generative backdoor trigger.

Reference
---------
Nguyen et al., "IBA: Towards Irreversible Backdoor Attacks in Federated
Learning." NeurIPS 2023.

How it works
-----------
A U-Net G maps each clean image x to an input-specific perturbation.
The poisoned image is::

    x̃ = clamp(x + α · G(x),  0, 1)

G is re-trained each active round by minimising::

    L = CE(model(norm(x̃)), y_t)  +  λ · ||G(x)||₂

where ``norm`` is the dataset normalisation so the classifier always
receives properly normalised inputs.

Value-range contract
--------------------
``apply`` receives and returns images in raw ``[0, 1]`` pixel space
(post-ToTensor, pre-Normalize), consistent with ``BackdoorDataset``.
Normalisation is applied ONLY inside ``train_trigger`` (for the classifier
forward pass) and NOT in ``apply`` — the caller (BackdoorDataset) handles
post-trigger normalisation via ``post_trigger_transform``.
"""

import logging
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .base import LearnableTrigger, Tensor
from .unet import UNet

logger = logging.getLogger(__name__)


class IBATrigger(LearnableTrigger):
    """Generative backdoor trigger backed by a U-Net perturbation generator.

    Args:
        unet:                U-Net instance used as generator G.
        normalize_transform: Dataset normalisation transform applied after
                             trigger injection inside ``train_trigger`` so
                             the classifier receives normalised inputs.
                             Pass ``None`` to skip normalisation (not
                             recommended — leads to distribution mismatch).
        alpha:               Perturbation scale: ``x̃ = clamp(x + α·G(x), 0, 1)``.
        lambda_noise:        Weight for the L2 noise-regularisation term.
        generator_epochs:    Adam epochs per ``train_trigger`` call.
        generator_lr:        Adam learning rate for the generator.
    """

    def __init__(
        self,
        unet: UNet,
        normalize_transform: Optional[Callable] = None,
        alpha: float = 0.2,
        lambda_noise: float = 0.01,
        generator_epochs: int = 5,
        generator_lr: float = 1e-3,
    ):
        # BaseTrigger stores: position=(0,0), size=(0,0), pattern=unet, alpha=alpha
        super().__init__(position=(0, 0), size=(0, 0), pattern=unet, alpha=alpha)
        self.generator = unet
        self.normalize_transform = normalize_transform
        self.lambda_noise = lambda_noise
        self.generator_epochs = generator_epochs
        self.generator_lr = generator_lr

    # ------------------------------------------------------------------
    # LearnableTrigger interface
    # ------------------------------------------------------------------

    def train_trigger(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        target_class: int,
    ) -> None:
        """Train the U-Net generator against the current global model.

        Trains for ``self.generator_epochs`` epochs, minimising the combined
        adversarial + noise-regularisation loss.  The classifier is frozen;
        only the generator parameters are updated.

        Args:
            model:        Current global model — frozen during training.
            dataloader:   Pre-normalisation DataLoader (images in ``[0, 1]``).
            target_class: Backdoor target class index.
        """
        device = next(model.parameters()).device
        self.generator.to(device).train()

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        # Pre-compute batch normalisation constants to avoid per-image calls
        sample_batch = next(iter(dataloader))[0]
        C = sample_batch.shape[1]
        if self.normalize_transform is not None:
            nt   = self.normalize_transform
            mean = torch.tensor(nt.mean, dtype=torch.float32, device=device).view(1, C, 1, 1)
            std  = torch.tensor(nt.std,  dtype=torch.float32, device=device).view(1, C, 1, 1)
        else:
            logger.warning(
                "IBATrigger: normalize_transform is None — classifier receives "
                "[0,1] inputs instead of normalised inputs."
            )
            mean, std = None, None

        optimizer = optim.Adam(self.generator.parameters(), lr=self.generator_lr)
        loss_fn   = nn.CrossEntropyLoss()

        logger.info(
            "IBA: training generator for %d epochs (target=%d, α=%.3f, λ=%.4f).",
            self.generator_epochs, target_class, self.alpha, self.lambda_noise,
        )

        for epoch in range(self.generator_epochs):
            epoch_loss = 0.0
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                optimizer.zero_grad()

                perturbation = self.generator(inputs)                             # [N, C, H, W]
                poisoned_pre = torch.clamp(inputs + self.alpha * perturbation, 0.0, 1.0)
                poisoned_in  = (poisoned_pre - mean) / std if mean is not None else poisoned_pre

                target_t = torch.full(
                    (inputs.size(0),), target_class, dtype=torch.long, device=device
                )
                l_cls   = loss_fn(model(poisoned_in), target_t)
                l_noise = perturbation.view(perturbation.size(0), -1).norm(p=2, dim=1).mean()
                loss    = l_cls + self.lambda_noise * l_noise

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            logger.debug(
                "IBA generator epoch [%d/%d]: avg_loss=%.4f",
                epoch + 1, self.generator_epochs,
                epoch_loss / max(len(dataloader), 1),
            )

        for p in model.parameters():
            p.requires_grad_(True)
        self.generator.eval()
        logger.info("IBA: generator training complete.")

    # ------------------------------------------------------------------
    # BaseTrigger interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def apply(self, image: Tensor) -> Tensor:
        """Apply the generative trigger to a single ``(C, H, W)`` image.

        Expects ``image`` in raw ``[0, 1]`` space (pre-normalisation).
        Returns the triggered image in ``[0, 1]``.

        Args:
            image: Float tensor ``(C, H, W)`` in ``[0, 1]``.

        Returns:
            Triggered image ``(C, H, W)`` in ``[0, 1]``.
        """
        device = image.device
        self.generator.to(device).eval()
        perturbation = self.generator(image.unsqueeze(0)).squeeze(0)
        return torch.clamp(image + self.alpha * perturbation, 0.0, 1.0)
