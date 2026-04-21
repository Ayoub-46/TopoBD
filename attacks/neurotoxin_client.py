"""Neurotoxin attack client.

Reference
---------
Zhang et al., "Neurotoxin: Durable Backdoors in Federated Learning", ICML 2022.

Core idea
---------
Neurotoxin constrains the backdoor update to the *least important* model
parameters — those that the global model is not actively updating.  This
makes the poisoned update harder to detect and more durable, because
defenses and future benign rounds are unlikely to overwrite low-importance
parameters.

Importance of parameter θ_i is defined as::

    score_i = |Δ_i| / (|θ_i| + ε)

where Δ = global_params_t − global_params_{t-1} is the previous global
update.  The top-k% most important parameters are *masked out* (their
gradients are zeroed), so only the remaining (1-k)% get updated during
poisoned local training.

The previous global delta is tracked internally: each time the runner calls
``set_params(new_global)``, the client stores the delta from the last seen
global snapshot.  No changes to the runner are required.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Optional

import torch
from torch.utils.data import DataLoader

from fl.client import BenignClient, ClientUpdate
from datasets.backdoor import BackdoorDataset
from attacks.triggers.base import BaseTrigger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attack configuration
# ---------------------------------------------------------------------------

@dataclass
class NeurotoxinConfig:
    """Typed configuration for the Neurotoxin attack client.

    Args:
        trigger:             An instantiated static :class:`BaseTrigger`
                             (e.g. ``PatchTrigger``).
        target_label:        Backdoor target class index.
        normalize_transform: Dataset normalisation transform applied after
                             the trigger so the model receives normalised
                             inputs.
        poison_fraction:     Fraction of local training samples to poison
                             each round.
        attack_start_round:  First FL round in which the attack is active.
        attack_end_round:    Last FL round in which the attack is active
                             (inclusive).  ``float('inf')`` means no end.
        mask_k_percent:      Fraction of the *most important* parameters
                             (by normalised gradient magnitude) whose
                             gradients are zeroed.  Default 0.95 leaves only
                             the bottom 5% of parameters free to update.
        seed:                Base RNG seed for reproducible poisoning.
    """

    trigger: BaseTrigger
    target_label: int
    normalize_transform: Callable
    poison_fraction: float = 0.5
    attack_start_round: int = 0
    attack_end_round: float = float("inf")
    mask_k_percent: float = 0.95
    seed: int = 42


# ---------------------------------------------------------------------------
# Attack client
# ---------------------------------------------------------------------------

class NeurotoxinClient(BenignClient):
    """FL client that mounts the Neurotoxin backdoor attack.

    The ``trainloader`` must be built from a **pre-normalisation** dataset
    (i.e. returned by ``DatasetAdapter.get_client_pre_loaders()``), because
    the trigger is applied in ``[0, 1]`` pixel space before normalisation.

    Args:
        config: :class:`NeurotoxinConfig` instance.
        All other args forwarded to :class:`~fl.client.BenignClient`.
    """

    def __init__(
        self,
        config: NeurotoxinConfig,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config = config
        # Tracks the previous global snapshot to compute the inter-round delta
        self._prev_global_params: Optional[Dict[str, torch.Tensor]] = None
        self._global_delta: Optional[Dict[str, torch.Tensor]] = None

    # ------------------------------------------------------------------
    # set_params override — delta tracking
    # ------------------------------------------------------------------

    def set_params(self, params: Dict[str, torch.Tensor]) -> None:
        """Load new global params and update the stored inter-round delta."""
        if self._prev_global_params is not None:
            self._global_delta = {
                k: params[k].cpu().float() - self._prev_global_params[k]
                for k in params
            }
        # Store a float32 CPU copy of the incoming params for next round
        self._prev_global_params = {k: v.cpu().float().clone() for k, v in params.items()}
        super().set_params(params)

    # ------------------------------------------------------------------
    # local_train override
    # ------------------------------------------------------------------

    def local_train(
        self,
        epochs: Optional[int] = None,
        round_idx: int = 0,
        **kwargs,
    ) -> ClientUpdate:
        """Poisoned local training with Neurotoxin gradient masking.

        Falls back to standard benign training when outside the configured
        attack window.
        """
        cfg = self.config

        # ---- Benign fallback outside attack window --------------------------
        if not (cfg.attack_start_round <= round_idx <= cfg.attack_end_round):
            return super().local_train(epochs=epochs, round_idx=round_idx)

        n_epochs = epochs if epochs is not None else self.epochs_default

        # ---- Build importance mask ------------------------------------------
        grad_mask = self._build_grad_mask()
        if grad_mask is None:
            logger.warning(
                "Neurotoxin client [%d] round %d: no previous global delta — "
                "attacking without gradient mask.",
                self.id, round_idx,
            )

        # ---- Build poisoned dataset -----------------------------------------
        round_seed = cfg.seed + round_idx
        poisoned_dataset = BackdoorDataset(
            original_dataset=self.trainloader.dataset,
            trigger_fn=cfg.trigger.apply,
            target_label=cfg.target_label,
            post_trigger_transform=cfg.normalize_transform,
            poison_fraction=cfg.poison_fraction,
            seed=round_seed,
            poison_exclude_target=True,
        )
        poisoned_loader = DataLoader(
            poisoned_dataset,
            batch_size=self.trainloader.batch_size,
            shuffle=True,
            num_workers=getattr(self.trainloader, "num_workers", 0),
        )

        # ---- Poisoned training with gradient masking ------------------------
        self._model.train()
        train_loss, correct, total, steps = 0.0, 0, 0, 0

        for _ in range(n_epochs):
            for inputs, targets in poisoned_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self._model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()

                if grad_mask is not None:
                    self._apply_grad_mask(grad_mask)

                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                steps += 1

            if self.scheduler is not None:
                self.scheduler.step()

        avg_loss = train_loss / steps if steps > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        num_poisoned = len(poisoned_dataset.poisoned_indices)
        logger.info(
            "Neurotoxin client [%d] round %d: poisoned %d/%d samples "
            "(target=%d, masked=%s).",
            self.id, round_idx, num_poisoned, len(poisoned_dataset),
            cfg.target_label, grad_mask is not None,
        )

        result = ClientUpdate(
            client_id=self.get_id(),
            num_samples=self.num_samples(),
            weights=self.get_params(),
            metrics={"loss": avg_loss, "accuracy": accuracy},
            round_idx=round_idx,
            is_malicious=True,
        )
        result.metadata.update({
            "attack": "neurotoxin",
            "target_label": cfg.target_label,
            "poison_fraction": cfg.poison_fraction,
            "num_poisoned": num_poisoned,
            "masked": grad_mask is not None,
            "mask_k_percent": cfg.mask_k_percent,
        })
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_grad_mask(self) -> Optional[Dict[str, torch.Tensor]]:
        """Build a boolean importance mask from the previous global delta.

        Returns a dict mapping parameter name → bool tensor where ``True``
        means the gradient is *kept* (parameter is unimportant) and
        ``False`` means the gradient is *zeroed* (parameter is important).

        Returns ``None`` when no delta is available yet (first active round).
        """
        if self._global_delta is None:
            return None

        eps = 1e-12
        importance_parts = []
        key_to_importance: Dict[str, torch.Tensor] = {}

        current_params = {n: p for n, p in self._model.named_parameters()}

        for name, delta in self._global_delta.items():
            if name not in current_params:
                continue
            param = current_params[name].detach().cpu().float()
            importance = delta.abs() / (param.abs() + eps)
            importance_parts.append(importance.flatten())
            key_to_importance[name] = importance

        if not importance_parts:
            return None

        all_importances = torch.cat(importance_parts)
        k = max(1, int(self.config.mask_k_percent * all_importances.numel()))
        # kth-largest value is the threshold: params with score >= threshold
        # are "important" and will be masked out
        threshold = torch.topk(all_importances, k, largest=True)[0][-1].item()

        return {
            name: (imp < threshold)   # True → keep (unimportant)
            for name, imp in key_to_importance.items()
        }

    def _apply_grad_mask(self, grad_mask: Dict[str, torch.Tensor]) -> None:
        """Zero out gradients for high-importance parameters in-place."""
        with torch.no_grad():
            for name, param in self._model.named_parameters():
                if param.grad is not None and name in grad_mask:
                    mask = grad_mask[name].to(
                        dtype=param.grad.dtype, device=param.grad.device
                    )
                    param.grad.mul_(mask)
