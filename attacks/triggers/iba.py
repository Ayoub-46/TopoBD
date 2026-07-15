"""IBA generative backdoor trigger.

Reference
---------
Nguyen et al., "IBA: Towards Irreversible Backdoor Attacks in Federated
Learning." NeurIPS 2023.

How it works
-----------
A U-Net G maps each clean image x to an input-specific perturbation.
Per the paper Eq. 1, the perturbation is **L∞-bounded** and added directly
(no scaling); the poisoned image is::

    x̃ = clamp(x + Π_ε(G(x)),  0, 1)

where ``Π_ε(·) = clamp(·, -ε, +ε)`` is the elementwise projection onto the
L∞ ball of radius ε (i.e. ``||Π_ε(G(x))||_∞ ≤ ε``).

G is re-trained each active round against the CURRENT global model by
minimising ONLY the adversarial term::

    L = CE(model(norm(x̃)), y_t)

The L∞ radius is self-adjusting (paper Eq. 4): ``ε_t = max(ε̂, ε_0·(1-λ)^(t-t_I))``
decays from ``ε_0`` toward the floor ``ε̂`` over the attacker's active rounds.

The magnitude constraint is the hard L∞ clamp, NOT a soft penalty.  The paper
(§3.1) explicitly rejects an L2 norm ("can result in localized artifacts …
more susceptible to detection") in favour of the L∞ bound, so the earlier
``α``-scaling + ``λ·||G(x)||₂`` regularisation has been removed.  ``norm`` is
the dataset normalisation so the classifier always receives properly
normalised inputs.

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
        epsilon:             L∞ bound on the additive perturbation (paper
                             Eq. 1: ``||G(x)||_∞ ≤ ε``).  The generator output
                             is projected onto the L∞ ball by an elementwise
                             clamp to ``[-ε, +ε]`` before being added to the
                             image (raw ``[0,1]`` pixel space).  Default
                             ``8/255`` — the standard adversarial L∞ budget.
                             NOTE: a large budget such as ``0.3`` makes the
                             CE-trained generator a *universal adversarial
                             perturbation* that a clean (un-backdoored) model
                             already classifies as the target (clean-model
                             ASR ≈ 1.0), i.e. an input artifact rather than a
                             model-resident backdoor; ``8/255`` keeps the
                             trigger non-transferable so the backdoor lives in
                             the model (clean-model ASR ≈ chance).  This value
                             is also the FLOOR ``ε̂`` of the decay schedule.
        eps_0:               Initial (largest) L∞ radius of the self-adjusting
                             decay schedule (paper Eq. 4).  Default ``0.3``
                             (paper §4.1).  A large early ε implants the
                             backdoor quickly; it then decays toward the floor.
        lambda_decay:        Per-round decay rate ``λ_ξ`` of the schedule
                             (paper Eq. 4).  ``ε_t = max(ε̂, ε_0·(1-λ)^(t-t_I))``.
        generator_epochs:    Adam epochs per ``train_trigger`` call.
        generator_lr:        Adam learning rate for the generator.

    Note on ε̂ vs the paper's 0.05: the paper §4.1 lists ε̂=0.05, but our
    validity gate (clean-model ASR must be ≈ chance) fails at 0.05
    (clean-model ASR ≈ 0.19).  We therefore floor the schedule at ε̂=8/255,
    the largest budget that keeps the DEPLOYED trigger non-transferable, and
    document the deviation rather than silently reintroducing the artifact.
    Because the deployed/eval ε is this floor, the decay changes only the
    training curriculum, not the clean-model gate.
    """

    def __init__(
        self,
        unet: UNet,
        normalize_transform: Optional[Callable] = None,
        epsilon: float = 8 / 255,
        eps_0: float = 0.3,
        lambda_decay: float = 0.4,
        generator_epochs: int = 5,
        generator_lr: float = 1e-3,
    ):
        # BaseTrigger requires an alpha; it is unused by IBA (the magnitude
        # constraint is the L∞ clamp, not a blend/scale), so fix it to 1.0.
        super().__init__(position=(0, 0), size=(0, 0), pattern=unet, alpha=1.0)
        self.generator = unet
        self.normalize_transform = normalize_transform
        # ε̂ (floor / deployed radius); self.epsilon is the CURRENT deployed
        # radius used by apply() and _project(). Outside training it equals ε̂.
        self.eps_hat = float(epsilon)
        self.eps_0 = float(eps_0)
        self.lambda_decay = float(lambda_decay)
        self.epsilon = self.eps_hat
        self.generator_epochs = generator_epochs
        self.generator_lr = generator_lr

    # ------------------------------------------------------------------
    # L∞ projection (paper Eq. 1) — used IDENTICALLY in train + deploy
    # ------------------------------------------------------------------

    def _project(self, perturbation: Tensor) -> Tensor:
        """Project the generator output onto the L∞ ball of radius ε.

        Elementwise clamp to ``[-ε, +ε]`` (the exact L∞-ball projection).
        Called by BOTH :meth:`train_trigger` (classifier forward pass) and
        :meth:`apply` (deployment) so the trained and deployed triggers are
        magnitude-constrained identically.  ``torch.clamp`` is differentiable
        (gradient passes through unclamped coordinates, zero at the boundary),
        which realises projected-gradient training of the generator.
        """
        return torch.clamp(perturbation, -self.epsilon, self.epsilon)

    def _epsilon_at(self, round_idx: int, attack_start_round: int) -> float:
        """Self-adjusting L∞ radius for round ``round_idx`` (paper Eq. 4).

            ε_t = max(ε̂,  ε_0 · (1 - λ_ξ)^(t - t_I))

        with ``t_I = attack_start_round``.  Monotonically decays from ``eps_0``
        toward the floor ``eps_hat`` as the attacker participates in more
        rounds, so early rounds implant with a large budget and later rounds
        refine within a small, non-transferable budget.
        """
        t = max(0, round_idx - attack_start_round)
        decayed = self.eps_0 * (1.0 - self.lambda_decay) ** t
        return max(self.eps_hat, decayed)

    # ------------------------------------------------------------------
    # LearnableTrigger interface
    # ------------------------------------------------------------------

    def train_trigger(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        target_class: int,
        round_idx: int = 0,
        attack_start_round: int = 0,
    ) -> None:
        """Train the U-Net generator against the current global model.

        Trains for ``self.generator_epochs`` epochs, minimising ONLY the
        adversarial cross-entropy loss ``CE(model(norm(x̃)), y_t)``; the
        perturbation magnitude is bounded by the hard L∞ projection
        (:meth:`_project`), not a soft penalty.  The classifier is frozen;
        only the generator parameters are updated.

        Before training, the current deployed radius ``self.epsilon`` is set to
        the decay-schedule value ``_epsilon_at(round_idx, attack_start_round)``
        (paper Eq. 4), so both this round's generator training and the
        subsequent ``apply`` use the same ε_t.  ``model`` is the CURRENT global
        model broadcast this round (§3.2 "keep the generator in sync with the
        state of the global model").

        Args:
            model:              Current global model — frozen during training.
            dataloader:         Pre-normalisation DataLoader (images in ``[0,1]``).
            target_class:       Backdoor target class index.
            round_idx:          Current FL round (for the ε decay schedule).
            attack_start_round: First active round ``t_I`` (schedule origin).
        """
        # Self-adjusting L∞ radius for this round (paper Eq. 4). Sets the
        # deployed radius used by both this training pass and apply().
        self.epsilon = self._epsilon_at(round_idx, attack_start_round)

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
            "IBA: training generator for %d epochs (target=%d, round=%d, "
            "ε_t=%.4f [ε_0=%.3f→ε̂=%.4f, λ=%.2f], L∞-bounded).",
            self.generator_epochs, target_class, round_idx, self.epsilon,
            self.eps_0, self.eps_hat, self.lambda_decay,
        )

        for epoch in range(self.generator_epochs):
            epoch_loss = 0.0
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                optimizer.zero_grad()

                perturbation = self.generator(inputs)                             # [N, C, H, W]
                # L∞ projection (paper Eq. 1) then clamp to valid pixel range.
                poisoned_pre = torch.clamp(inputs + self._project(perturbation), 0.0, 1.0)
                poisoned_in  = (poisoned_pre - mean) / std if mean is not None else poisoned_pre

                target_t = torch.full(
                    (inputs.size(0),), target_class, dtype=torch.long, device=device
                )
                # CE only — the L∞ clamp replaces the former L2 noise penalty.
                loss = loss_fn(model(poisoned_in), target_t)

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
        # Move generator back to CPU so that apply() is safe to call from
        # DataLoader worker processes, which are forked and cannot initialise
        # a CUDA context inherited from the parent.
        self.generator.cpu().eval()
        logger.info("IBA: generator training complete.")

    # ------------------------------------------------------------------
    # BaseTrigger interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def apply(self, image: Tensor) -> Tensor:
        """Apply the generative trigger to a single ``(C, H, W)`` image.

        Expects ``image`` in raw ``[0, 1]`` space (pre-normalisation).
        Returns the triggered image in ``[0, 1]``.

        The generator is always on CPU here: ``train_trigger`` moves it back
        to CPU after each training call so that this method is safe to invoke
        from DataLoader worker processes (forked, no CUDA context available).

        Args:
            image: Float tensor ``(C, H, W)`` in ``[0, 1]``.

        Returns:
            Triggered image ``(C, H, W)`` in ``[0, 1]``.
        """
        perturbation = self.generator(image.unsqueeze(0)).squeeze(0)
        # Same L∞ projection (paper Eq. 1) as train_trigger, then valid-range clamp.
        return torch.clamp(image + self._project(perturbation), 0.0, 1.0)
