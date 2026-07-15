"""IBA backdoor attack client.

Reference
---------
Nguyen et al., "IBA: Towards Irreversible Backdoor Attacks in Federated
Learning." NeurIPS 2023.

Each active round the client performs two stages:

1. **Generator training** — adapts the shared U-Net trigger against the
   current global model using a small local subset.
2. **Poisoned local training** — trains on a mix of clean and triggered
   samples; normalisation is applied correctly after trigger injection via
   ``BackdoorDataset.post_trigger_transform``.

All malicious clients receive the same ``IBATrigger`` instance (shared
reference).  Training is sequential within a round, so each client fine-
tunes the same generator in turn.  This is consistent with the assumption
in the paper that the adversary controls all malicious clients and
coordinates the attack.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from fl.client import BenignClient, ClientUpdate
from datasets.backdoor import BackdoorDataset
from attacks.triggers.iba import IBATrigger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attack configuration
# ---------------------------------------------------------------------------

@dataclass
class IBAConfig:
    """Typed configuration for the IBA attack client.

    Args:
        trigger:             Shared :class:`IBATrigger` instance.  All
                             malicious clients in the same federation must
                             reference the *same* object so that generator
                             training is cumulative and the ASR evaluator
                             in the runner reflects the trained state.
        target_label:        Backdoor target class index.
        normalize_transform: Dataset normalisation applied after the trigger
                             inside ``BackdoorDataset``.
        poison_fraction:     Fraction of local samples to poison per round.
        attack_start_round:  First FL round in which the attack is active.
        attack_end_round:    Last FL round (inclusive).  ``float('inf')``
                             means no end.
        trigger_sample_size: Local samples used for generator fine-tuning.
                             Smaller values are faster; larger values give a
                             better gradient estimate.
        seed:                Base RNG seed.  Effective seed per round is
                             ``seed + round_idx``.
        delta:               Poisoning-SPACE constraint (paper §3.2): after
                             poisoned local training the malicious update is
                             projected onto the L2 ball of radius ``delta``
                             around the received global model, so the update's
                             norm cannot exceed a benign-looking magnitude.
                             ``None`` disables the projection.
        dimension_k_percent: Poisoning-DIMENSION constraint (paper Eq. 6): the
                             fraction of coordinates — the *bottom* ``k`` by
                             running main-task gradient magnitude (the
                             least-frequently-updated ones) — the poison update
                             is confined to; all other coordinates are zeroed.
                             ``0`` or ``1`` disables the masking.
        clean_grad_batches:  Clean (main-task) mini-batches used each active
                             round to update the running per-coordinate
                             gradient-magnitude estimate that defines the mask.
    """
    trigger: IBATrigger
    target_label: int
    normalize_transform: Callable
    poison_fraction: float = 0.5
    attack_start_round: int = 0
    attack_end_round: float = float("inf")
    trigger_sample_size: int = 512
    seed: int = 42
    attack_epochs: int = 10
    # ---- Stage 3: constrained model poisoning (paper §3.2) ----
    # delta = L2 vicinity radius. Default 2.3 is calibrated to a benign
    # update's L2 norm for CIFAR-10 / VGG-13-noBN in this federation (measured
    # ~2.32); recalibrate for other datasets/models. ``None`` disables it.
    delta: Optional[float] = 2.3
    dimension_k_percent: float = 0.5
    clean_grad_batches: int = 2


# ---------------------------------------------------------------------------
# Attack client
# ---------------------------------------------------------------------------

class IBAClient(BenignClient):
    """FL client that mounts the IBA backdoor attack.

    The ``trainloader`` must be built from a **pre-normalisation** dataset
    (returned by ``DatasetAdapter.get_client_pre_loaders()``), because the
    trigger is applied in raw ``[0, 1]`` pixel space before normalisation.

    Args:
        config: :class:`IBAConfig` instance.
        All other args forwarded to :class:`~fl.client.BenignClient`.
    """

    def __init__(self, config: IBAConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        # Stage 3 poisoning-dimension: running per-coordinate estimate of the
        # main-task (clean) gradient magnitude, maintained across the
        # attacker's active rounds (paper Eq. 6). Lazily initialised.
        self._grad_running: Optional[dict] = None
        self._grad_rounds: int = 0

        # Outside the attack window this client must train exactly like a
        # BenignClient -- but self.trainloader is the pre-normalisation
        # loader (needed so the trigger can be pasted in [0,1] space).
        # poison_fraction=0.0 poisons nothing while still applying
        # normalize_transform to every sample (BackdoorDataset applies
        # post_trigger_transform unconditionally), giving a clean loader
        # equivalent to what a real benign client trains on.
        clean_dataset = BackdoorDataset(
            original_dataset=self.trainloader.dataset,
            trigger_fn=config.trigger.apply,
            target_label=config.target_label,
            post_trigger_transform=config.normalize_transform,
            poison_fraction=0.0,
            seed=config.seed,
            poison_exclude_target=True,
        )
        self._clean_loader = DataLoader(
            clean_dataset,
            batch_size=self.trainloader.batch_size,
            shuffle=True,
            num_workers=getattr(self.trainloader, "num_workers", 0),
        )

    # ------------------------------------------------------------------
    # local_train override
    # ------------------------------------------------------------------

    def local_train(
        self,
        epochs: Optional[int] = None,
        round_idx: int = 0,
        **kwargs,
    ) -> ClientUpdate:
        """Two-stage IBA attack, falling back to benign training outside
        the configured attack window.

        Args:
            epochs:    Local epochs.  ``None`` uses the constructor default.
            round_idx: Current FL round index.

        Returns:
            :class:`ClientUpdate` with ``is_malicious=True`` during attack
            rounds, ``False`` otherwise.
        """
        cfg = self.config

        # ---- Benign fallback ------------------------------------------------
        if not (cfg.attack_start_round <= round_idx <= cfg.attack_end_round):
            original_loader = self.trainloader
            self.trainloader = self._clean_loader
            result = super().local_train(epochs=epochs, round_idx=round_idx)
            self.trainloader = original_loader
            return result

        # Snapshot the received global model BEFORE any local update. Used by
        # the Stage-3 constrained-poisoning projections (space + dimension).
        # The runner calls set_params(global) immediately before local_train,
        # and generator training below freezes the model, so self._model here
        # is exactly the current global model.
        w_global = {k: v.detach().cpu().clone()
                    for k, v in self._model.state_dict().items()}

        # ---- Stage 1+2: fine-tune the shared U-Net generator ----------------
        # (against the CURRENT global model; ε follows the Eq. 4 decay via
        #  round_idx / attack_start_round.)
        logger.info(
            "IBA client [%d] — round %d: training generator.", self.id, round_idx
        )
        trigger_loader = self._build_trigger_dataloader()
        if trigger_loader is not None:
            cfg.trigger.train_trigger(
                model=self._model,
                dataloader=trigger_loader,
                target_class=cfg.target_label,
                round_idx=round_idx,
                attack_start_round=cfg.attack_start_round,
            )
        else:
            logger.warning(
                "IBA client [%d] — trigger training skipped: no local data.", self.id
            )

        # ---- Stage 3 (dimension): refresh the running main-task gradient
        #      magnitude estimate at the current global model (paper Eq. 6). ----
        if 0.0 < cfg.dimension_k_percent < 1.0:
            self._update_clean_grad_estimate(cfg.clean_grad_batches)

        # ---- Poisoned local training (loss mixture α=β=0.5 via poison_fraction
        #      data mixing, paper Eq. 3) ----------------------------------------
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

        original_loader = self.trainloader
        try:
            self.trainloader = poisoned_loader
            result = super().local_train(epochs=cfg.attack_epochs, round_idx=round_idx)
        finally:
            self.trainloader = original_loader

        # ---- Stage 3: constrained model poisoning (paper §3.2) --------------
        # Project the raw poisoned update (w_local - w_global) onto (i) the
        # bottom-k% least-updated coordinates, then (ii) the L2 δ-ball.
        constrained_weights, raw_norm, proj_norm, n_coords = \
            self._constrain_poison_update(w_global, result.weights)
        result.weights = constrained_weights

        result.is_malicious = True
        result.metadata.update({
            "attack": "iba",
            "target_label": cfg.target_label,
            "poison_fraction": cfg.poison_fraction,
            "num_poisoned": len(poisoned_dataset.poisoned_indices),
            "round_seed": round_seed,
            "update_norm_raw": raw_norm,
            "update_norm_projected": proj_norm,
            "poisoned_coords": n_coords,
        })

        logger.info(
            "IBA client [%d] — round %d: poisoned %d / %d samples (target=%d); "
            "update L2 %.3f→%.3f (δ=%s), %d poisoned coords (k=%.2f).",
            self.id, round_idx,
            len(poisoned_dataset.poisoned_indices), len(poisoned_dataset),
            cfg.target_label, raw_norm, proj_norm, cfg.delta, n_coords,
            cfg.dimension_k_percent,
        )
        return result

    # ------------------------------------------------------------------
    # Stage 3 — constrained model poisoning (paper §3.2)
    # ------------------------------------------------------------------

    def _update_clean_grad_estimate(self, n_batches: int) -> None:
        """Update the running per-coordinate main-task gradient magnitude.

        Runs ``n_batches`` CE backward passes on CLEAN local data (true labels)
        at the current global model and accumulates the running MEAN of
        ``|∂L_main/∂θ|`` per coordinate across the attacker's active rounds.
        This is our defensible reading of paper Eq. 6: the "infrequently
        updated" coordinates are those with the smallest running main-task
        gradient magnitude.  (Eq. 6's running-average is stated ambiguously in
        the paper; cross-check against the authors' code at
        github.com/sail-research/iba before treating this as canonical.)

        Kept on CPU so the per-coordinate estimate does not consume GPU memory.
        """
        model = self._model
        model.train()
        batch_accum = {name: torch.zeros_like(p, device="cpu")
                       for name, p in model.named_parameters()}
        seen = 0
        loader_iter = iter(self._clean_loader)
        for _ in range(max(1, n_batches)):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                break
            x, y = x.to(self.device), y.to(self.device)
            model.zero_grad(set_to_none=True)
            self.loss_fn(model(x), y).backward()
            for name, p in model.named_parameters():
                if p.grad is not None:
                    batch_accum[name] += p.grad.detach().abs().cpu()
            seen += 1
        model.zero_grad(set_to_none=True)
        if seen == 0:
            return
        batch_mean = {name: acc / seen for name, acc in batch_accum.items()}

        if self._grad_running is None:
            self._grad_running = batch_mean
            self._grad_rounds = 1
        else:
            r = self._grad_rounds
            self._grad_running = {
                name: (self._grad_running[name] * r + batch_mean[name]) / (r + 1)
                for name in batch_mean
            }
            self._grad_rounds = r + 1

    def _constrain_poison_update(self, w_global: dict, w_local: dict):
        """Apply the poisoning-dimension and poisoning-space constraints.

        Returns ``(new_weights, raw_norm, projected_norm, n_poisoned_coords)``.

        (i) Dimension (paper Eq. 6): confine the update ``w_local - w_global`` to
            the bottom-``k``% coordinates by running main-task gradient
            magnitude (a single GLOBAL threshold across all float parameters,
            matching the concatenated-ranking convention); zero the rest.
        (ii) Space (paper §3.2, PGD vicinity): if the (masked) update's global
            L2 norm exceeds ``delta``, rescale it onto the δ-ball around
            ``w_global``.  We use a post-hoc projection of the final update
            rather than per-step PGD: for a fixed vicinity radius the two reach
            the same feasible set, and the post-hoc form is deterministic and
            leaves the inner poisoned-training loop (and its α=β=0.5 mixture)
            untouched.
        """
        cfg = self.config
        float_keys = [k for k, v in w_global.items() if torch.is_floating_point(v)]
        update = {k: (w_local[k].cpu() - w_global[k].cpu()) for k in float_keys}

        # (i) dimension mask -------------------------------------------------
        n_coords = 0
        if (self._grad_running is not None
                and 0.0 < cfg.dimension_k_percent < 1.0):
            all_mag = torch.cat([
                self._grad_running[k].flatten()
                for k in float_keys if k in self._grad_running
            ])
            kth = max(1, int(cfg.dimension_k_percent * all_mag.numel()))
            threshold = torch.kthvalue(all_mag, kth).values
            for k in float_keys:
                if k in self._grad_running:
                    mask = (self._grad_running[k] <= threshold).to(update[k].dtype)
                    update[k] = update[k] * mask
                    n_coords += int(mask.sum().item())

        # (ii) space projection onto the L2 δ-ball ---------------------------
        raw_norm = float(torch.sqrt(sum((update[k] ** 2).sum()
                                        for k in float_keys)).item())
        proj_norm = raw_norm
        if cfg.delta is not None and raw_norm > cfg.delta:
            scale = cfg.delta / (raw_norm + 1e-12)
            for k in float_keys:
                update[k] = update[k] * scale
            proj_norm = cfg.delta

        new_weights = {}
        for k, v in w_global.items():
            if k in float_keys:
                new_weights[k] = v.cpu() + update[k]
            else:  # non-float buffers (e.g. num_batches_tracked): keep local
                new_weights[k] = w_local[k].cpu()
        return new_weights, raw_norm, proj_norm, n_coords

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_trigger_dataloader(self) -> Optional[DataLoader]:
        """Sample a small subset for generator fine-tuning.

        Returns:
            DataLoader over at most ``config.trigger_sample_size`` local
            (pre-norm) samples, or ``None`` if the local dataset is empty.
        """
        base_dataset = self.trainloader.dataset
        num_samples  = len(base_dataset)
        if num_samples == 0:
            return None

        k   = min(self.config.trigger_sample_size, num_samples)
        rng = np.random.RandomState(self.config.seed)
        idx = rng.choice(num_samples, size=k, replace=False).tolist()

        batch_size = min(getattr(self.trainloader, "batch_size", 32), k)
        return DataLoader(Subset(base_dataset, idx), batch_size=batch_size, shuffle=True)
