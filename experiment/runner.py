"""FL experiment runner.

The :class:`FLRunner` accepts a single :class:`~experiment.config.ExperimentConfig`
and constructs every component — adapter, model, server, clients, eval loaders —
before executing the full FL loop with optional attack and defense.

Extension points
----------------
* **Defense**: pass a defense-aware server subclass as ``defense_server``.
  Detection-based defenses must implement ``filter_updates() -> DetectionResult``;
  robust-aggregation defenses (trimmed mean, etc.) need not — TPR/FPR columns
  will simply remain ``NaN``.
* **New datasets**: add an entry to :func:`~experiment.utils.build_adapter`.
* **New models**: register with :func:`~models.register_model`.
* **New attacks**: add a branch in :func:`~experiment.utils.build_clients` and
  register the trigger with :func:`~attacks.triggers.register_trigger`.
"""

from __future__ import annotations

import logging
import math
import os
import random
from typing import Dict, FrozenSet, List, Optional

import torch
from torch.utils.data import DataLoader

from fl.server import FedAvgAggregator
from models import get_model, ModelConfig

from .config import ExperimentConfig
from .metrics import MetricsTracker, RoundMetrics
from .utils import (
    DetectionResult,
    build_adapter,
    build_clients,
    build_server,
    resolve_device,
    seed_everything,
)

logger = logging.getLogger(__name__)


class FLRunner:
    """Orchestrates a complete FL experiment from a single config object.

    Args:
        config:         Full experiment specification.
        defense_server: Optional pre-constructed defense server.  When
                        ``None`` a plain :class:`~fl.server.FedAvgAggregator`
                        is used.

    Example::

        from experiment import ExperimentConfig, AttackConfig, FLRunner

        cfg = ExperimentConfig(
            name="patch_attack_dirichlet",
            dataset="cifar10",
            model="resnet18",
            num_clients=100,
            clients_per_round=10,
            num_rounds=200,
            local_epochs=5,
            partition="dirichlet",
            dirichlet_alpha=0.5,
            attack=AttackConfig(
                attack_type="patch",
                num_malicious=10,
                target_label=0,
                poison_fraction=0.5,
                trigger_kwargs={
                    "position": (29, 29),
                    "size": (3, 3),
                    "color": (1.0, 0.0, 0.0),
                },
            ),
        )
        runner = FLRunner(cfg)
        metrics = runner.run()
        metrics.print_summary()
    """

    def __init__(
        self,
        config: ExperimentConfig,
        defense_server: Optional[FedAvgAggregator] = None,
    ) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        seed_everything(config.seed)

        logger.info(
            "Initialising '%s' | device=%s | rounds=%d | "
            "clients=%d/%d | attack=%s | defense=%s",
            config.name, self.device, config.num_rounds,
            config.clients_per_round, config.num_clients,
            config.attack.attack_type,
            config.defense.defense_type,
        )

        # ---- Dataset --------------------------------------------------------
        self.adapter = build_adapter(config)

        # ---- Model config (single source of truth) --------------------------
        # Computed here and passed to build_clients so it is never constructed
        # twice with potentially divergent settings.
        self.model_cfg = ModelConfig.from_adapter(config.model, self.adapter)

        # ---- Server ---------------------------------------------------------
        global_model = get_model(self.model_cfg).to(self.device)

        if defense_server is not None:
            self.server = defense_server
            # Initialise the defense server with the same starting weights
            self.server.set_params(
                {k: v.to(self.device) for k, v in global_model.state_dict().items()}
            )
        else:
            self.server = build_server(config, global_model, self.device)

        # Detection-based defenses expose filter_updates(); aggregation-only
        # defenses do not — checked once here rather than every round.
        self._has_detection_defense = hasattr(self.server, "filter_updates")

        # ---- Clients --------------------------------------------------------
        # FIX: pass model_cfg so build_clients does not recompute it
        self.malicious_ids, self.clients = build_clients(
            config, self.adapter, self.model_cfg, self.device
        )

        # ---- Evaluation loaders ---------------------------------------------
        self.test_loader = self.adapter.get_test_loader(
            batch_size=config.batch_size, num_workers=2
        )
        self.backdoor_test_loader = self._build_backdoor_loader()
        self.chameleon_evaluator  = self._build_chameleon_evaluator()

        logger.info(
            "Setup complete — %d clients (%d malicious) | "
            "backdoor eval: %s",
            len(self.clients),
            len(self.malicious_ids),
            "chameleon-adaptive" if self.chameleon_evaluator is not None
            else ("static-trigger" if self.backdoor_test_loader is not None else "none"),
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> MetricsTracker:
        """Execute the full FL training loop.

        Metrics are flushed to CSV incrementally so partial results survive
        an interrupted run.

        Returns:
            :class:`MetricsTracker` with one entry per evaluated round.
        """
        cfg = self.config
        output_dir = os.path.join(cfg.output_dir, cfg.name)
        os.makedirs(output_dir, exist_ok=True)

        # Persist config for reproducibility
        cfg.save(os.path.join(output_dir, "config.json"))

        tracker = MetricsTracker(output_dir=output_dir)
        global_params = self.server.get_params()

        for round_idx in range(cfg.num_rounds):

            # ---- Client selection ------------------------------------------
            selected_ids: List[int] = random.sample(
                list(self.clients.keys()), cfg.clients_per_round
            )
            malicious_selected: FrozenSet[int] = frozenset(
                cid for cid in selected_ids if cid in self.malicious_ids
            )

            # Clients outside the attack window train benignly — they are not
            # adversarial this round even if they are in the malicious set.
            atk = cfg.attack
            attack_window_active = (
                atk.attack_start_round <= round_idx
                and (atk.attack_end_round is None or round_idx <= atk.attack_end_round)
            )
            true_malicious_this_round: FrozenSet[int] = (
                malicious_selected if attack_window_active else frozenset()
            )

            n_mal = len(true_malicious_this_round)
            n_ben = len(selected_ids) - n_mal

            # ---- Local training --------------------------------------------
            total_samples = 0
            round_updates = []
            for cid in selected_ids:
                client = self.clients[cid]
                client.set_params(global_params)
                update = client.local_train(
                    epochs=cfg.local_epochs, round_idx=round_idx
                )
                self.server.receive_update(
                    client_id=update.client_id,
                    params=update.weights,
                    length=update.num_samples,
                )
                total_samples += update.num_samples
                round_updates.append(update)

            # ---- Feed Chameleon trigger to evaluator -----------------------
            if self.chameleon_evaluator is not None and attack_window_active:
                mal_updates = [
                    u for u in round_updates
                    if u.is_malicious and "avg_delta" in u.metadata
                ]
                if mal_updates:
                    avg_delta = torch.stack(
                        [u.metadata["avg_delta"] for u in mal_updates]
                    ).mean(dim=0)
                    self.chameleon_evaluator.update_trigger(avg_delta)

            # ---- Defense filtering (detection-based) -----------------------
            detection: Optional[DetectionResult] = None
            if self._has_detection_defense:
                detection = self.server.filter_updates(
                    true_malicious=true_malicious_this_round
                )

            # ---- Aggregation + round reset ---------------------------------
            agg = self.server.aggregate()
            self.server.reset()
            global_params = agg.aggregated_params

            # ---- Evaluation ------------------------------------------------
            # Always evaluate on the final round; otherwise every eval_every rounds.
            should_eval = (
                round_idx % cfg.eval_every == 0
                or round_idx == cfg.num_rounds - 1
            )
            if not should_eval:
                continue

            eval_result = self.server.evaluate(self.test_loader)
            clean_acc  = eval_result.metrics["main_accuracy"]
            clean_loss = eval_result.metrics["loss"]

            asr = math.nan
            asr_loss = math.nan
            if self.backdoor_test_loader is not None:
                bd = self.server.evaluate(self.backdoor_test_loader)
                asr      = bd.metrics["main_accuracy"]
                asr_loss = bd.metrics["loss"]
            elif self.chameleon_evaluator is not None:
                asr = self.chameleon_evaluator.evaluate_asr(global_params)

            # ---- TPR / FPR -------------------------------------------------
            d_tpr = math.nan
            d_fpr = math.nan
            if detection is not None:
                d_tpr = detection.tpr
                d_fpr = detection.compute_fpr(n_ben)

            # ---- Record and log --------------------------------------------
            metrics = RoundMetrics(
                round=round_idx,
                clean_loss=clean_loss,
                clean_acc=clean_acc,
                asr=asr,
                asr_loss=asr_loss,
                is_attack_round=int(n_mal > 0),
                n_selected=len(selected_ids),
                n_malicious_selected=n_mal,
                total_samples=total_samples,
                defense_tpr=d_tpr,
                defense_fpr=d_fpr,
            )
            tracker.record(metrics)
            self._log_round(round_idx, cfg.num_rounds, metrics)

        tracker.save()
        tracker.print_summary()

        model_path = os.path.join(output_dir, "final_model.pt")
        torch.save(global_params, model_path)
        logger.info("Final model saved to %s", model_path)

        return tracker

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_backdoor_loader(self):
        """Build the ASR evaluation loader, or return ``None``."""
        cfg = self.config
        atk = cfg.attack

        if atk.attack_type == "none" or atk.num_malicious == 0:
            return None

        # trigger_type="none" signals that this attack has no fixed trigger
        # (e.g. Chameleon uses sample-specific perturbations). Skip ASR eval.
        trigger_name = atk.trigger_type or atk.attack_type
        if trigger_name == "none":
            return None

        # IBA uses a shared generator trained in-place each round.  Obtain
        # the trigger from the first malicious client so the backdoor test
        # loader always calls trigger.apply on the CURRENT trained generator
        # (BackdoorDataset is not cached, so apply() is called fresh per item).
        if atk.attack_type == "iba":
            first_mal = next(
                (cid for cid in sorted(self.malicious_ids) if cid in self.clients), None
            )
            if first_mal is None:
                return None
            trigger = self.clients[first_mal].config.trigger
            return self.adapter.get_backdoor_test_loader(
                trigger_fn=trigger.apply,
                target_label=atk.target_label,
                batch_size=cfg.batch_size,
                num_workers=2,
            )

        from attacks.triggers import get_trigger

        # Inject dataset shape defaults; trigger factories tolerate unknown
        # kwargs and discard what their class does not accept.
        C, H, W = self.adapter.input_shape
        trigger_kw = {"in_channels": C, "image_size": (H, W)}
        trigger_kw.update(atk.trigger_kwargs)

        trigger = get_trigger(trigger_name, **trigger_kw)
        return self.adapter.get_backdoor_test_loader(
            trigger_fn=trigger.apply,
            target_label=atk.target_label,
            batch_size=cfg.batch_size,
            num_workers=2,
        )

    def _build_chameleon_evaluator(self):
        """Build a :class:`ChameleonASREvaluator` when the attack is Chameleon.

        Collects non-target test images from the adapter's pre-normalisation
        test dataset and sub-samples to ``num_eval_samples`` for speed.
        Returns ``None`` for all other attack types.
        """
        atk = self.config.attack
        if atk.attack_type != "chameleon":
            return None

        from attacks.chameleon_client import ChameleonASREvaluator

        # Materialise pre-norm test images on CPU
        pre_loader = DataLoader(
            self.adapter.test_pre_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=2,
        )
        images_list, labels_list = [], []
        for imgs, lbls in pre_loader:
            images_list.append(imgs.cpu())
            labels_list.append(lbls.cpu())
        all_images = torch.cat(images_list)
        all_labels = torch.cat(labels_list)

        # Keep only non-target samples
        non_target = (all_labels != atk.target_label).nonzero(as_tuple=True)[0]
        eval_images = all_images[non_target]

        # Sub-sample for evaluation speed
        num_eval = atk.trigger_kwargs.get("num_eval_samples", 500)
        if len(eval_images) > num_eval:
            rng = torch.Generator()
            rng.manual_seed(self.config.seed)
            idx = torch.randperm(len(eval_images), generator=rng)[:num_eval]
            eval_images = eval_images[idx]

        logger.info(
            "Chameleon ASR evaluator: %d non-target test images "
            "(trigger-replay mode — no PGD at eval time).",
            len(eval_images),
        )

        return ChameleonASREvaluator(
            model_cfg=self.model_cfg,
            test_images=eval_images,
            target_label=atk.target_label,
            normalize_transform=self.adapter.normalize_transform,
            device=self.device,
        )

    @staticmethod
    def _log_round(round_idx: int, total: int, m: RoundMetrics) -> None:
        attack_str = (
            f"  |  ASR: {m.asr * 100:6.2f}%  asr_loss: {m.asr_loss:.4f}"
            if not math.isnan(m.asr) else ""
        )
        defense_str = (
            f"  |  TPR: {m.defense_tpr * 100:.1f}%  FPR: {m.defense_fpr * 100:.1f}%"
            if not math.isnan(m.defense_tpr) else ""
        )
        logger.info(
            "Round %4d/%d  |  acc: %6.2f%%  loss: %.4f%s%s",
            round_idx + 1, total,
            m.clean_acc * 100, m.clean_loss,
            attack_str, defense_str,
        )