"""FL experiment runner.

The :class:`FLRunner` accepts a single :class:`~experiment.config.ExperimentConfig`
and constructs every component — adapter, model, server, clients, eval loaders —
before executing the full FL loop with optional attack and defense.

Metrics written per run
-----------------------
* ``metrics.csv``            — one row per evaluated round (clean_acc, asr,
                               macro_f1, raw detection counts, …)
* ``detection_raw.csv``      — one row per FL round, TP/FP/TN/FN counts
* ``per_class_metrics.csv``  — one row per (eval_round, class)
* ``convergence.json``       — rounds-to-target-acc, AUC, pooled TPR/FPR
* ``client_atypicality.json``— TV-distance from global distribution per client
* ``client_inclusion.csv``   — cumulative benign-client inclusion rates
* ``client_fp_atypicality.csv`` — per-client FPR vs atypicality
* ``trigger.pt``             — serialised trigger state for learnable attacks
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
from collections import defaultdict
from typing import Dict, FrozenSet, List, Optional

import torch
from torch.utils.data import DataLoader

from fl.server import FedAvgAggregator
from models import get_model, ModelConfig

from .config import ExperimentConfig
from .metrics import MetricsTracker, PerClassEvalResult, RoundMetrics
from .utils import (
    DetectionResult,
    build_adapter,
    build_clients,
    build_server,
    compute_client_atypicality,
    resolve_device,
    seed_everything,
)

logger = logging.getLogger(__name__)


class FLRunner:
    """Orchestrates a complete FL experiment from a single config object."""

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

        # ---- Model config ---------------------------------------------------
        self.model_cfg = ModelConfig.from_adapter(config.model, self.adapter)

        # ---- Server ---------------------------------------------------------
        global_model = get_model(self.model_cfg).to(self.device)

        if defense_server is not None:
            self.server = defense_server
            self.server.set_params(
                {k: v.to(self.device) for k, v in global_model.state_dict().items()}
            )
        else:
            self.server = build_server(config, global_model, self.device, self.adapter)

        self._has_detection_defense = hasattr(self.server, "filter_updates")

        # ---- Clients --------------------------------------------------------
        self.malicious_ids, self.clients = build_clients(
            config, self.adapter, self.model_cfg, self.device
        )

        # ---- Evaluation loaders ---------------------------------------------
        self.test_loader = self.adapter.get_test_loader(
            batch_size=config.batch_size, num_workers=2
        )
        self.backdoor_test_loader = self._build_backdoor_loader()
        self.chameleon_evaluator  = self._build_chameleon_evaluator()

        # ---- Client atypicality (computed once at setup) --------------------
        self._client_atypicality: Dict[int, float] = compute_client_atypicality(
            self.adapter,
            num_clients=config.num_clients,
            strategy=config.partition,
            seed=config.seed,
            alpha=config.dirichlet_alpha,
        )

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
        """Execute the full FL training loop."""
        cfg = self.config
        output_dir = os.path.join(cfg.output_dir, cfg.name)
        os.makedirs(output_dir, exist_ok=True)
        cfg.save(os.path.join(output_dir, "config.json"))

        tracker      = MetricsTracker(output_dir=output_dir)
        global_params = self.server.get_params()

        # Per-client tracking (benign clients only)
        # rounds a benign client was sampled / included (not rejected) / false-positive
        ben_sampled:  Dict[int, int] = defaultdict(int)
        ben_included: Dict[int, int] = defaultdict(int)
        ben_fp:       Dict[int, int] = defaultdict(int)

        # Anomaly scores per round (for AUPRC CSV, written at run end)
        score_rows: List[dict] = []

        for round_idx in range(cfg.num_rounds):

            # ---- Client selection ------------------------------------------
            selected_ids: List[int] = random.sample(
                list(self.clients.keys()), cfg.clients_per_round
            )
            malicious_selected: FrozenSet[int] = frozenset(
                cid for cid in selected_ids if cid in self.malicious_ids
            )

            atk = cfg.attack
            attack_window_active = (
                atk.attack_start_round <= round_idx
                and (atk.attack_end_round is None or round_idx <= atk.attack_end_round)
            )
            true_malicious_this_round: FrozenSet[int] = (
                malicious_selected if attack_window_active else frozenset()
            )

            n_mal = len(true_malicious_this_round)
            # n_ben = benign clients sampled this round (all selected − true malicious)
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

            # ---- Chameleon trigger feed ------------------------------------
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

            # ---- Defense filtering (every round) ---------------------------
            detection: Optional[DetectionResult] = None
            if self._has_detection_defense:
                detection = self.server.filter_updates(
                    true_malicious=true_malicious_this_round
                )

            # ---- Raw detection counts (§6a / A7) ---------------------------
            # FPR defined every round the defense is active.
            # TPR defined only when malicious clients were sampled (n_mal > 0).
            tp = fp = tn = fn = -1
            if detection is not None:
                tp, fp, tn, fn = detection.raw_counts(n_ben)

                # Per-client tracking (benign clients only)
                for cid in selected_ids:
                    is_benign = cid not in true_malicious_this_round
                    if is_benign:
                        ben_sampled[cid] += 1
                        if cid in detection.rejected_ids:
                            ben_fp[cid] += 1
                        else:
                            ben_included[cid] += 1
            else:
                # No detection defense — all benign updates reach the aggregator
                for cid in selected_ids:
                    if cid not in true_malicious_this_round:
                        ben_sampled[cid]  += 1
                        ben_included[cid] += 1

            # Record raw counts every round (not just eval rounds)
            tracker.record_detection_raw(round_idx, tp, fp, tn, fn, n_mal, n_ben)

            # Anomaly scores (for AUPRC CSV)
            if detection is not None and detection.client_scores:
                for cid, score in detection.client_scores.items():
                    is_mal = cid in true_malicious_this_round
                    score_rows.append({
                        "round": round_idx, "client_id": cid,
                        "score": score, "is_malicious": int(is_mal),
                    })

            # ---- Aggregation -----------------------------------------------
            agg = self.server.aggregate()
            self.server.reset()
            global_params = agg.aggregated_params

            # ---- Conditional evaluation ------------------------------------
            should_eval = (
                round_idx % cfg.eval_every == 0
                or round_idx == cfg.num_rounds - 1
            )
            if not should_eval:
                continue

            # Clean test evaluation
            eval_result = self.server.evaluate(self.test_loader)
            clean_acc  = eval_result.metrics["main_accuracy"]
            clean_loss = eval_result.metrics["loss"]

            # Per-class metrics (§6b)
            per_class_result: Optional[PerClassEvalResult] = None
            per_class_result = self.server.evaluate_per_class(self.test_loader)
            macro_f1 = macro_p = macro_r = math.nan
            if per_class_result is not None:
                macro_f1 = per_class_result.macro_f1
                macro_p  = per_class_result.macro_precision
                macro_r  = per_class_result.macro_recall

            # ASR
            asr = asr_loss = math.nan
            if self.backdoor_test_loader is not None:
                bd       = self.server.evaluate(self.backdoor_test_loader)
                asr      = bd.metrics["main_accuracy"]
                asr_loss = bd.metrics["loss"]
            elif self.chameleon_evaluator is not None:
                asr = self.chameleon_evaluator.evaluate_asr(global_params)

            # Detection scalars for this eval round (per A7):
            # FPR every round; TPR only when malicious sampled
            d_tpr = d_fpr = math.nan
            if detection is not None:
                if n_mal > 0 and tp >= 0:
                    d_tpr = tp / n_mal                  # = TP / (TP+FN)
                if fp >= 0:
                    d_fpr = fp / n_ben if n_ben > 0 else math.nan

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
                defense_tp=tp,
                defense_fp=fp,
                defense_tn=tn,
                defense_fn=fn,
                defense_tpr=d_tpr,
                defense_fpr=d_fpr,
                macro_precision=macro_p,
                macro_recall=macro_r,
                macro_f1=macro_f1,
            )
            tracker.record(metrics)
            tracker.record_per_class(round_idx, per_class_result)
            self._log_round(round_idx, cfg.num_rounds, metrics)

        # ---- End-of-run outputs --------------------------------------------
        tracker.save()
        tracker.print_summary()

        model_path = os.path.join(output_dir, "final_model.pt")
        torch.save(global_params, model_path)
        logger.info("Final model saved to %s", model_path)

        self._save_trigger(output_dir)
        self._write_client_atypicality(output_dir)
        self._write_client_inclusion(output_dir, ben_sampled, ben_included, ben_fp)
        self._write_anomaly_scores(output_dir, score_rows)

        return tracker

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_backdoor_loader(self):
        cfg = self.config
        atk = cfg.attack

        if atk.attack_type == "none" or atk.num_malicious == 0:
            return None

        # neurotoxin uses a patch trigger for ASR evaluation (the gradient-masking
        # part is training-time only and does not affect the trigger shape).
        _trigger_fallback = {"neurotoxin": "patch"}
        trigger_name = atk.trigger_type or _trigger_fallback.get(atk.attack_type, atk.attack_type)
        if trigger_name == "none":
            return None

        if atk.attack_type in ("iba", "a3fl"):
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
        atk = self.config.attack
        if atk.attack_type != "chameleon":
            return None

        from attacks.chameleon_client import ChameleonASREvaluator
        pre_loader = DataLoader(
            self.adapter.test_pre_dataset,
            batch_size=256, shuffle=False, num_workers=2,
        )
        images_list, labels_list = [], []
        for imgs, lbls in pre_loader:
            images_list.append(imgs.cpu())
            labels_list.append(lbls.cpu())
        all_images = torch.cat(images_list)
        all_labels = torch.cat(labels_list)

        non_target = (all_labels != atk.target_label).nonzero(as_tuple=True)[0]
        eval_images = all_images[non_target]

        num_eval = atk.trigger_kwargs.get("num_eval_samples", 500)
        if len(eval_images) > num_eval:
            rng = torch.Generator()
            rng.manual_seed(self.config.seed)
            idx = torch.randperm(len(eval_images), generator=rng)[:num_eval]
            eval_images = eval_images[idx]

        logger.info(
            "Chameleon ASR evaluator: %d non-target test images.",
            len(eval_images),
        )
        return ChameleonASREvaluator(
            model_cfg=self.model_cfg,
            test_images=eval_images,
            target_label=atk.target_label,
            normalize_transform=self.adapter.normalize_transform,
            device=self.device,
        )

    def _save_trigger(self, output_dir: str) -> None:
        atk = self.config.attack
        if atk.attack_type not in ("a3fl", "iba"):
            return
        first_mal = next(
            (cid for cid in sorted(self.malicious_ids) if cid in self.clients), None
        )
        if first_mal is None:
            return
        client = self.clients[first_mal]
        if not hasattr(client, "config") or not hasattr(client.config, "trigger"):
            return
        trigger = client.config.trigger
        trigger_path = os.path.join(output_dir, "trigger.pt")
        if atk.attack_type == "a3fl":
            torch.save({"type": "a3fl", "pattern": trigger.pattern.cpu()}, trigger_path)
        elif atk.attack_type == "iba":
            torch.save(
                {"type": "iba", "unet_state_dict": trigger.generator.state_dict()},
                trigger_path,
            )
        logger.info("Trigger state saved to %s", trigger_path)

    def _write_client_atypicality(self, output_dir: str) -> None:
        """Write per-client TV-distance from global label distribution."""
        if not self._client_atypicality:
            return
        path = os.path.join(output_dir, "client_atypicality.json")
        data = {
            str(cid): {
                "atypicality": v,
                "is_malicious": cid in self.malicious_ids,
            }
            for cid, v in self._client_atypicality.items()
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Client atypicality written to %s", path)

    def _write_client_inclusion(
        self,
        output_dir: str,
        ben_sampled: Dict[int, int],
        ben_included: Dict[int, int],
        ben_fp: Dict[int, int],
    ) -> None:
        """Write cumulative benign-client inclusion and FPR, paired with atypicality."""
        path = os.path.join(output_dir, "client_inclusion.csv")
        cols = [
            "client_id", "rounds_sampled", "rounds_included", "rounds_fp",
            "inclusion_rate", "empirical_fpr", "atypicality",
        ]
        benign_ids = sorted(
            cid for cid in self.clients if cid not in self.malicious_ids
        )
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for cid in benign_ids:
                sampled  = ben_sampled.get(cid, 0)
                included = ben_included.get(cid, 0)
                fp_count = ben_fp.get(cid, 0)
                writer.writerow({
                    "client_id":      cid,
                    "rounds_sampled": sampled,
                    "rounds_included": included,
                    "rounds_fp":      fp_count,
                    "inclusion_rate": f"{included / sampled:.6f}" if sampled else "NaN",
                    "empirical_fpr":  f"{fp_count / sampled:.6f}" if sampled else "NaN",
                    "atypicality":    f"{self._client_atypicality.get(cid, float('nan')):.6f}",
                })
        logger.info("Client inclusion rates written to %s", path)

    def _write_anomaly_scores(self, output_dir: str, rows: List[dict]) -> None:
        """Write per-round per-client anomaly scores (for AUPRC computation)."""
        if not rows:
            return
        path = os.path.join(output_dir, "anomaly_scores.csv")
        cols = ["round", "client_id", "score", "is_malicious"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Anomaly scores written to %s (%d rows)", path, len(rows))

    @staticmethod
    def _log_round(round_idx: int, total: int, m: RoundMetrics) -> None:
        attack_str = (
            f"  |  ASR: {m.asr * 100:6.2f}%  asr_loss: {m.asr_loss:.4f}"
            if not math.isnan(m.asr) else ""
        )
        defense_str = ""
        if m.defense_fp >= 0:
            tpr_str = f"TPR: {m.defense_tpr * 100:.1f}%  " if not math.isnan(m.defense_tpr) else ""
            defense_str = f"  |  {tpr_str}FPR: {m.defense_fpr * 100:.1f}%"
        macro_str = (
            f"  |  mF1: {m.macro_f1 * 100:.1f}%"
            if not math.isnan(m.macro_f1) else ""
        )
        logger.info(
            "Round %4d/%d  |  acc: %6.2f%%  loss: %.4f%s%s%s",
            round_idx + 1, total,
            m.clean_acc * 100, m.clean_loss,
            attack_str, defense_str, macro_str,
        )
