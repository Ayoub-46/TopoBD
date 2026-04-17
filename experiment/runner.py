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

from fl.server import FedAvgAggregator
from models import get_model, ModelConfig

from .config import ExperimentConfig
from .metrics import MetricsTracker, RoundMetrics
from .utils import (
    DetectionResult,
    build_adapter,
    build_clients,
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
            self.server = FedAvgAggregator(model=global_model, device=self.device)

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

        logger.info(
            "Setup complete — %d clients (%d malicious) | "
            "backdoor eval loader: %s",
            len(self.clients),
            len(self.malicious_ids),
            "yes" if self.backdoor_test_loader is not None else "no",
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
            n_mal = len(malicious_selected)
            n_ben = len(selected_ids) - n_mal

            # ---- Local training --------------------------------------------
            total_samples = 0
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

            # ---- Defense filtering (detection-based) -----------------------
            detection: Optional[DetectionResult] = None
            if self._has_detection_defense:
                detection = self.server.filter_updates(
                    true_malicious=malicious_selected
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

        from attacks.triggers import get_trigger

        # Inject dataset shape defaults; trigger factories tolerate unknown
        # kwargs and discard what their class does not accept.
        C, H, W = self.adapter.input_shape
        trigger_kw = {"in_channels": C, "image_size": (H, W)}
        trigger_kw.update(atk.trigger_kwargs)

        trigger = get_trigger(atk.attack_type, **trigger_kw)
        return self.adapter.get_backdoor_test_loader(
            trigger_fn=trigger.apply,
            target_label=atk.target_label,
            batch_size=cfg.batch_size,
            num_workers=2,
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