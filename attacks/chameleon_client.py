"""Chameleon backdoor attack client.

Reference
---------
Dai, Yanbo, and Songze Li. "Chameleon: Adapting to peer images for planting
durable backdoors in federated learning." ICML 2023.

Core idea
---------
For each active round, the client optimises a sample-specific perturbation δ
for every poisoned source image x such that:

  1.  model(normalize(x + δ)) predicts the target class      [classification]
  2.  ||x + δ − p||²₂ is small, where p is a peer image
      drawn from the target class in the client's local data  [similarity]
  3.  ||δ||∞ ≤ ε                                             [budget]

The combined loss is: L_cls + λ_sim * L_sim

Optimisation is sign-based PGD in [0, 1] pixel space; normalisation is
applied inside the forward pass so the model always receives normalised
inputs.  Because poisoned images are nudged toward target-class peers, the
resulting backdoored updates are statistically similar to benign ones,
degrading norm- and distance-based defences (e.g. Multi-Krum, FLTrust).

ASR evaluation note
-------------------
Chameleon has no fixed trigger; ASR must be approximated at evaluation time.
Set ``trigger_type: patch`` in the YAML config to use a patch trigger as a
proxy, or ``trigger_type: none`` to skip ASR logging entirely.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from fl.client import BenignClient, ClientUpdate
from models import get_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attack configuration
# ---------------------------------------------------------------------------

@dataclass
class ChameleonConfig:
    """Typed configuration for the Chameleon attack client.

    Args:
        target_label:        Backdoor target class index.
        normalize_transform: Dataset normalisation transform (e.g.
                             ``Normalize(mean, std)``).  Used both inside the
                             PGD loop and when building the training batch.
        poison_fraction:     Fraction of non-target local samples to poison
                             each round.
        attack_start_round:  First FL round in which the attack is active.
        attack_end_round:    Last FL round in which the attack is active
                             (inclusive).  ``float('inf')`` means no end.
        epsilon:             Maximum L∞ perturbation in [0, 1] pixel space.
                             Paper default: 0.3.
        lambda_sim:          Weight for the image-similarity loss term.
                             Higher values make poisoned images closer to peers
                             at the cost of weaker classification pressure.
                             Paper default: 1.0.
        num_pgd_steps:       Number of sign-PGD steps to optimise δ.
                             Paper default: 100.
        pgd_lr:              PGD step size (multiplied by sign of gradient).
                             A value of ``epsilon / num_pgd_steps * 2.5``
                             gives a good balance; default 0.01 works well for
                             ε = 0.3 over 100 steps.
        peer_pool_size:      Maximum number of target-class images to keep as
                             the peer pool.  Excess images are subsampled.
        seed:                Base RNG seed.  The effective seed each round is
                             ``seed + round_idx`` so the poisoned subset varies.
    """
    target_label: int
    normalize_transform: Callable
    poison_fraction: float = 0.5
    attack_start_round: int = 0
    attack_end_round: float = float("inf")
    epsilon: float = 0.3
    lambda_sim: float = 1.0
    num_pgd_steps: int = 100
    pgd_lr: float = 0.01
    peer_pool_size: int = 128
    seed: int = 42


# ---------------------------------------------------------------------------
# Attack client
# ---------------------------------------------------------------------------

class ChameleonClient(BenignClient):
    """FL client that mounts the Chameleon backdoor attack.

    The ``trainloader`` must be built from a **pre-normalisation** dataset
    (returned by ``DatasetAdapter.get_client_pre_loaders()``), because the
    perturbation is optimised in raw [0, 1] pixel space and normalisation is
    applied internally.

    Args:
        config: :class:`ChameleonConfig` instance.
        All other args forwarded to :class:`~fl.client.BenignClient`.
    """

    def __init__(self, config: ChameleonConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

    # ------------------------------------------------------------------
    # local_train override
    # ------------------------------------------------------------------

    def local_train(
        self,
        epochs: Optional[int] = None,
        round_idx: int = 0,
        **kwargs,
    ) -> ClientUpdate:
        """Chameleon two-phase attack with benign fallback outside the window.

        Phase 1 — Perturbation optimisation:
            For each poisoned source sample, run sign-PGD against the current
            global model to find a δ that minimises classification + similarity
            loss.

        Phase 2 — Local training:
            Construct a mixed TensorDataset (clean normalised + poisoned
            normalised) and run standard SGD for ``epochs`` epochs.
        """
        cfg = self.config

        # ---- Benign fallback outside attack window --------------------------
        if not (cfg.attack_start_round <= round_idx <= cfg.attack_end_round):
            return super().local_train(epochs=epochs, round_idx=round_idx)

        n_epochs = epochs if epochs is not None else self.epochs_default
        round_seed = cfg.seed + round_idx

        # ---- Collect all local samples (pre-normalisation) ------------------
        all_images, all_labels = self._collect_local_data()
        # all_images: [N, C, H, W] in [0, 1]  |  all_labels: [N] int

        # ---- Identify peer pool and source candidates -----------------------
        target_mask = (all_labels == cfg.target_label)
        peer_images = all_images[target_mask]

        if len(peer_images) == 0:
            logger.warning(
                "Chameleon client [%d] round %d: no target-class samples "
                "in local data — falling back to benign training.",
                self.id, round_idx,
            )
            return super().local_train(epochs=epochs, round_idx=round_idx)

        # Sub-sample peer pool if it exceeds the configured cap
        if len(peer_images) > cfg.peer_pool_size:
            rng_peer = np.random.RandomState(round_seed + 1)
            peer_idx = rng_peer.choice(len(peer_images), cfg.peer_pool_size, replace=False)
            peer_images = peer_images[peer_idx]

        # ---- Select poisoned subset from non-target samples -----------------
        source_indices = torch.where(~target_mask)[0]  # global indices into all_images
        n_poison = max(1, int(len(source_indices) * cfg.poison_fraction))
        n_poison = min(n_poison, len(source_indices))

        rng = np.random.RandomState(round_seed)
        chosen = rng.choice(len(source_indices), n_poison, replace=False)
        poison_global_idx = source_indices[chosen]       # [n_poison] — indices into all_images
        source_to_poison  = all_images[poison_global_idx]  # [n_poison, C, H, W]

        logger.info(
            "Chameleon client [%d] round %d: optimising δ for %d/%d samples "
            "(ε=%.3f, λ_sim=%.2f, steps=%d).",
            self.id, round_idx, n_poison, len(all_images),
            cfg.epsilon, cfg.lambda_sim, cfg.num_pgd_steps,
        )

        # ---- Phase 1: optimise sample-specific perturbations ----------------
        poisoned_norm, avg_delta = self._optimize_perturbations(
            source_to_poison, peer_images, round_seed
        )  # poisoned_norm: [n_poison, C, H, W] normalised; avg_delta: [C, H, W] in [0,1]

        # ---- Build mixed normalised TensorDataset ---------------------------
        # Normalise the complete local dataset
        norm_t = cfg.normalize_transform
        C = all_images.shape[1]
        mean_t = torch.tensor(norm_t.mean, dtype=torch.float32).view(1, C, 1, 1)
        std_t  = torch.tensor(norm_t.std,  dtype=torch.float32).view(1, C, 1, 1)

        combined_images = (all_images - mean_t) / std_t       # [N, C, H, W]
        combined_labels = all_labels.clone()

        # Replace poisoned positions with their adversarially perturbed versions
        combined_images[poison_global_idx] = poisoned_norm
        combined_labels[poison_global_idx] = cfg.target_label

        mixed_loader = DataLoader(
            TensorDataset(combined_images, combined_labels),
            batch_size=getattr(self.trainloader, "batch_size", 32),
            shuffle=True,
        )

        # ---- Phase 2: local training ----------------------------------------
        self._model.train()
        train_loss, correct, total, steps = 0.0, 0, 0, 0

        for _ in range(n_epochs):
            for inputs, targets in mixed_loader:
                inputs  = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self._model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total   += targets.size(0)
                correct += (predicted == targets).sum().item()
                steps   += 1

            if self.scheduler is not None:
                self.scheduler.step()

        avg_loss = train_loss / steps if steps > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        logger.info(
            "Chameleon client [%d] round %d: poisoned %d/%d samples "
            "(target=%d, loss=%.4f, acc=%.3f).",
            self.id, round_idx, n_poison, len(all_images),
            cfg.target_label, avg_loss, accuracy,
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
            "attack": "chameleon",
            "target_label": cfg.target_label,
            "poison_fraction": cfg.poison_fraction,
            "num_poisoned": n_poison,
            "epsilon": cfg.epsilon,
            "lambda_sim": cfg.lambda_sim,
            # avg_delta is the mean per-sample perturbation this round; the runner
            # forwards it to ChameleonASREvaluator so ASR is evaluated by replaying
            # the actual learned trigger rather than running PGD at eval time.
            "avg_delta": avg_delta,
        })
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect_local_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Materialise the full local pre-normalisation dataset into tensors.

        Returns:
            ``(images, labels)`` — both on CPU.
            ``images``: float32 [N, C, H, W] in [0, 1].
            ``labels``: int64 [N].
        """
        images_list, labels_list = [], []
        for imgs, lbls in self.trainloader:
            images_list.append(imgs.cpu())
            labels_list.append(lbls.cpu())
        return torch.cat(images_list, dim=0), torch.cat(labels_list, dim=0)

    def _optimize_perturbations(
        self,
        source_imgs: torch.Tensor,   # [N, C, H, W] in [0, 1], CPU
        peer_imgs:   torch.Tensor,   # [M, C, H, W] in [0, 1], CPU
        round_seed:  int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sign-PGD optimisation of sample-specific perturbations.

        For each source image the nearest peer is assigned randomly; then PGD
        minimises ``L_cls(normalize(x+δ), y_t) + λ·||x+δ − p||²``.

        The model is put in ``eval()`` mode during optimisation and its
        parameters are frozen to prevent gradient accumulation.  Both are
        restored after the loop.

        Args:
            source_imgs: Pre-norm source images to poison.
            peer_imgs:   Pre-norm target-class peer pool.
            round_seed:  Seed for peer assignment (reproducible per round).

        Returns:
            ``(poisoned_norm, avg_delta)`` where ``poisoned_norm`` is
            normalised poisoned images ``[N, C, H, W]`` on CPU, and
            ``avg_delta`` is the mean per-sample perturbation ``[C, H, W]``
            in ``[0, 1]`` space, used as the round's universal trigger for
            faithful ASR evaluation.
        """
        cfg    = self.config
        device = self.device
        N, C   = source_imgs.shape[0], source_imgs.shape[1]

        source_imgs = source_imgs.to(device)
        peer_imgs   = peer_imgs.to(device)

        # Randomly assign one peer per source sample
        rng = np.random.RandomState(round_seed + 2)
        peer_idx       = rng.choice(len(peer_imgs), size=N, replace=True)
        assigned_peers = peer_imgs[peer_idx]  # [N, C, H, W]

        # Batch normalisation constants (avoids per-image transform calls)
        norm_t = cfg.normalize_transform
        mean = torch.tensor(norm_t.mean, dtype=torch.float32, device=device).view(1, C, 1, 1)
        std  = torch.tensor(norm_t.std,  dtype=torch.float32, device=device).view(1, C, 1, 1)

        # Freeze model parameters — we only need gradients w.r.t. δ
        for p in self._model.parameters():
            p.requires_grad_(False)
        self._model.eval()

        target_t = torch.full((N,), cfg.target_label, dtype=torch.long, device=device)
        delta    = torch.zeros_like(source_imgs)   # [N, C, H, W]

        for _ in range(cfg.num_pgd_steps):
            delta.requires_grad_(True)

            x_adv  = (source_imgs + delta).clamp(0.0, 1.0)
            x_norm = (x_adv - mean) / std

            cls_loss = F.cross_entropy(self._model(x_norm), target_t)
            sim_loss = F.mse_loss(x_adv, assigned_peers)
            loss = cls_loss + cfg.lambda_sim * sim_loss

            loss.backward()

            with torch.no_grad():
                # Sign-PGD step + L∞ projection
                delta = (delta - cfg.pgd_lr * delta.grad.sign()).clamp(
                    -cfg.epsilon, cfg.epsilon
                )

        # Restore model state
        for p in self._model.parameters():
            p.requires_grad_(True)
        self._model.train()

        # Build final normalised poisoned tensors + mean trigger for ASR eval
        with torch.no_grad():
            x_final      = (source_imgs + delta).clamp(0.0, 1.0)
            poisoned_norm = ((x_final - mean) / std).cpu()
            avg_delta     = delta.mean(dim=0).cpu()   # [C, H, W] in [-ε, ε]

        return poisoned_norm, avg_delta


# ---------------------------------------------------------------------------
# Adaptive ASR evaluator
# ---------------------------------------------------------------------------

class ChameleonASREvaluator:
    """Faithful ASR evaluator for Chameleon via trigger replay.

    Instead of running PGD at evaluation time (which measures adversarial
    vulnerability — trivially 100% on any standard model at ε=0.3 — not
    backdoor success), this evaluator stores the **average perturbation**
    computed by malicious clients each active round and replays it at test
    time.

    The key insight: a model that has learned the Chameleon backdoor will
    respond to the trigger pattern even after aggregation.  A clean model
    will not, because the pattern was optimised against a different (earlier)
    model state that never saw backdoored updates.

    Usage in the runner:
    1. After each active Chameleon round, call :meth:`update_trigger` with
       the ``avg_delta`` collected from malicious client updates.
    2. In the eval loop, call :meth:`evaluate_asr` with the current
       ``global_params``.  Returns ``float('nan')`` until the first trigger
       is stored (i.e. before the attack window starts).

    Args:
        model_cfg:           Model configuration (used to build the eval copy).
        test_images:         Pre-normalisation test images ``[N, C, H, W]``
                             in ``[0, 1]`` — only non-target-class samples.
        target_label:        Backdoor target class index.
        normalize_transform: Dataset normalisation transform.
        device:              Torch device.
    """

    def __init__(
        self,
        model_cfg,
        test_images: torch.Tensor,
        target_label: int,
        normalize_transform: Callable,
        device: torch.device = torch.device("cpu"),
    ):
        self._model = get_model(model_cfg).to(device)
        self.test_images = test_images           # [N, C, H, W] in [0, 1], CPU
        self.target_label = target_label
        self.normalize_transform = normalize_transform
        self.device = device
        self._trigger: Optional[torch.Tensor] = None   # [C, H, W] in [-ε, ε]

    def update_trigger(self, avg_delta: torch.Tensor) -> None:
        """Store the latest round's mean perturbation as the current trigger.

        Called by the runner after each active Chameleon training round.

        Args:
            avg_delta: Mean per-sample perturbation ``[C, H, W]`` from
                       malicious client metadata, in ``[−ε, ε]`` range.
        """
        self._trigger = avg_delta.cpu().clone()

    def evaluate_asr(self, global_params: Dict[str, torch.Tensor]) -> float:
        """Apply the stored trigger to test images and return ASR.

        Loads ``global_params`` into a private model copy, adds
        ``self._trigger`` to every test image (clamped to ``[0, 1]``),
        normalises, and reports the fraction classified as the target class.

        Returns ``float('nan')`` when no trigger has been stored yet
        (before the attack window starts).

        Args:
            global_params: Aggregated global model state dict.

        Returns:
            ASR in ``[0, 1]``, or ``float('nan')`` if no trigger stored.
        """
        if self._trigger is None:
            return float("nan")

        self._model.load_state_dict(
            {k: v.to(self.device) for k, v in global_params.items()}
        )
        self._model.eval()

        images  = self.test_images.to(self.device)         # [N, C, H, W]
        trigger = self._trigger.to(self.device)            # [C, H, W]
        N, C    = images.shape[0], images.shape[1]

        norm_t = self.normalize_transform
        mean = torch.tensor(norm_t.mean, dtype=torch.float32, device=self.device).view(1, C, 1, 1)
        std  = torch.tensor(norm_t.std,  dtype=torch.float32, device=self.device).view(1, C, 1, 1)

        with torch.no_grad():
            triggered = (images + trigger.unsqueeze(0)).clamp(0.0, 1.0)
            x_norm    = (triggered - mean) / std
            preds     = self._model(x_norm).argmax(dim=1)
            asr       = (preds == self.target_label).float().mean().item()

        return asr
