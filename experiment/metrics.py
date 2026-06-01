"""Per-round metrics collection and CSV / JSON export.

Tracked quantities
------------------
Every row in the output CSV corresponds to one evaluated round.

Detection metrics follow §6a of the run plan:
  * Raw TP/FP/TN/FN counts are written every round (detection_raw.csv).
  * Pooled period TPR/FPR are derived by summing counts, never averaging rates.
  * FPR is defined and recorded every round that a detection defense is active,
    including warmup and attack=none cells.
  * TPR is recorded only on rounds where malicious clients were actually sampled
    (otherwise the field is -1 = N/A).

Learning-task metrics (per §6b) are written to per_class_metrics.csv alongside
the main metrics.csv, and convergence statistics are written to convergence.json.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Per-round container
# ---------------------------------------------------------------------------

@dataclass
class RoundMetrics:
    """All measurable quantities for one FL round."""

    round: int

    # Model quality
    clean_loss: float
    clean_acc: float

    # Backdoor effectiveness
    asr: float = math.nan           # NaN → no attack configured
    asr_loss: float = math.nan

    # Attack activity
    is_attack_round: int = 0        # 1 if ≥1 malicious client selected
    n_selected: int = 0
    n_malicious_selected: int = 0
    total_samples: int = 0

    # Detection counts (§6a).
    # Use -1 as "not applicable" sentinel (no detection defense, or no
    # malicious sampled for TP/FN).  FP/TN are defined whenever a detection
    # defense is active; TP/FN require malicious clients to have been sampled.
    defense_tp: int = -1
    defense_fp: int = -1
    defense_tn: int = -1
    defense_fn: int = -1

    # Derived detection rates (scalars for convenience; pool from raw counts
    # across the attack period for the headline metric).
    defense_tpr: float = math.nan   # NaN when n_mal == 0 or no detection defense
    defense_fpr: float = math.nan   # NaN when no detection defense

    # Per-class / macro learning-task metrics (§6b)
    macro_precision: float = math.nan
    macro_recall: float = math.nan
    macro_f1: float = math.nan


# ---------------------------------------------------------------------------
# Per-class result container (not persisted in main CSV — separate file)
# ---------------------------------------------------------------------------

@dataclass
class PerClassEvalResult:
    """Per-class precision/recall/F1 from one evaluation pass."""
    per_class: Dict[int, Dict[str, float]]  # {class_id: {tp, fp, fn, precision, recall, f1}}
    macro_precision: float
    macro_recall: float
    macro_f1: float


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class MetricsTracker:
    """Collects :class:`RoundMetrics` across all rounds and exports results.

    Writes:
      * ``metrics.csv``        — one row per eval round (incremental flush).
      * ``metrics.json``       — full array at run end.
      * ``detection_raw.csv``  — one row per FL round, every round,
                                  raw TP/FP/TN/FN counts.
      * ``per_class_metrics.csv`` — one row per (eval_round, class).
      * ``convergence.json``   — rounds-to-target-acc, AUC of acc-vs-round.

    Usage::

        tracker = MetricsTracker(output_dir="results/run_01")
        # inside the FL loop (every round, even non-eval):
        tracker.record_detection_raw(round_idx, tp, fp, tn, fn, n_mal, n_ben)
        # at eval rounds:
        tracker.record(RoundMetrics(...))
        tracker.record_per_class(round_idx, per_class_result)
        # after the loop:
        tracker.save()
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self._rows: List[RoundMetrics] = []
        self._per_class_rows: List[dict] = []
        self._detection_raw_rows: List[dict] = []

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def record(self, metrics: RoundMetrics) -> None:
        """Append one eval round's metrics and immediately flush to CSV."""
        self._rows.append(metrics)
        self._append_csv_row(metrics)

    def record_per_class(
        self, round_idx: int, result: Optional[PerClassEvalResult]
    ) -> None:
        """Append per-class metrics for one eval round."""
        if result is None:
            return
        for cls_id, d in result.per_class.items():
            row = {"round": round_idx, "class": cls_id, **d}
            self._per_class_rows.append(row)

    def record_detection_raw(
        self,
        round_idx: int,
        tp: int,
        fp: int,
        tn: int,
        fn: int,
        n_mal_selected: int,
        n_ben_selected: int,
    ) -> None:
        """Append per-FL-round raw detection counts (every round, not just eval)."""
        self._detection_raw_rows.append({
            "round":          round_idx,
            "n_mal_selected": n_mal_selected,
            "n_ben_selected": n_ben_selected,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        })

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def rows(self) -> List[RoundMetrics]:
        return list(self._rows)

    # ------------------------------------------------------------------
    # Derived statistics
    # ------------------------------------------------------------------

    @property
    def best_clean_acc(self) -> float:
        return max((r.clean_acc for r in self._rows), default=0.0)

    @property
    def final_clean_acc(self) -> float:
        return self._rows[-1].clean_acc if self._rows else 0.0

    @property
    def final_asr(self) -> float:
        return self._rows[-1].asr if self._rows else math.nan

    def asr_over_attack_window(self) -> List[float]:
        """Per-eval-round ASR values during attack rounds only."""
        return [r.asr for r in self._rows if r.is_attack_round and not math.isnan(r.asr)]

    def mean_asr_attack_window(self) -> float:
        vals = self.asr_over_attack_window()
        return sum(vals) / len(vals) if vals else math.nan

    def max_asr_attack_window(self) -> float:
        vals = self.asr_over_attack_window()
        return max(vals) if vals else math.nan

    def pooled_tpr(self) -> float:
        """Attack-period TPR: sum(TP) / sum(TP+FN) over rounds with malicious sampled.

        Pools raw counts — never averages per-round rates (per §6a).
        """
        total_tp = total_tp_fn = 0
        for row in self._detection_raw_rows:
            if row["tp"] >= 0 and row["fn"] >= 0 and row["n_mal_selected"] > 0:
                total_tp    += row["tp"]
                total_tp_fn += row["tp"] + row["fn"]
        return total_tp / total_tp_fn if total_tp_fn > 0 else math.nan

    def pooled_fpr(self) -> float:
        """Global FPR: sum(FP) / sum(FP+TN) over all rounds with detection defense active.

        Includes warmup and attack=none cells per §6a / A7.
        """
        total_fp = total_fp_tn = 0
        for row in self._detection_raw_rows:
            if row["fp"] >= 0 and row["tn"] >= 0:
                total_fp    += row["fp"]
                total_fp_tn += row["fp"] + row["tn"]
        return total_fp / total_fp_tn if total_fp_tn > 0 else math.nan

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write all output files to ``output_dir``."""
        os.makedirs(self.output_dir, exist_ok=True)
        self._write_csv_full()
        self._write_json()
        self._write_detection_raw()
        self._write_per_class()
        self._write_convergence()

    def _csv_path(self) -> str:
        return os.path.join(self.output_dir, "metrics.csv")

    def _json_path(self) -> str:
        return os.path.join(self.output_dir, "metrics.json")

    def _detection_raw_path(self) -> str:
        return os.path.join(self.output_dir, "detection_raw.csv")

    def _per_class_path(self) -> str:
        return os.path.join(self.output_dir, "per_class_metrics.csv")

    def _convergence_path(self) -> str:
        return os.path.join(self.output_dir, "convergence.json")

    def _header(self) -> List[str]:
        return [f.name for f in fields(RoundMetrics)]

    def _append_csv_row(self, row: RoundMetrics) -> None:
        """Append a single row to the CSV (creates header on first write)."""
        path = self._csv_path()
        os.makedirs(self.output_dir, exist_ok=True)
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._header())
            if write_header:
                writer.writeheader()
            writer.writerow(self._format_row(row))

    def _write_csv_full(self) -> None:
        with open(self._csv_path(), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._header())
            writer.writeheader()
            for row in self._rows:
                writer.writerow(self._format_row(row))

    def _write_json(self) -> None:
        with open(self._json_path(), "w") as f:
            json.dump([asdict(r) for r in self._rows], f, indent=2)

    def _write_detection_raw(self) -> None:
        if not self._detection_raw_rows:
            return
        cols = ["round", "n_mal_selected", "n_ben_selected", "tp", "fp", "tn", "fn"]
        with open(self._detection_raw_path(), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(self._detection_raw_rows)

    def _write_per_class(self) -> None:
        if not self._per_class_rows:
            return
        cols = ["round", "class", "tp", "fp", "fn", "precision", "recall", "f1"]
        with open(self._per_class_path(), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._per_class_rows)

    def _write_convergence(self) -> None:
        """Compute and write rounds-to-target-accuracy and AUC of acc-vs-round."""
        if not self._rows:
            return
        rounds = [r.round for r in self._rows]
        accs   = [r.clean_acc for r in self._rows]

        # Rounds-to-target (70% default; also compute 80% if reached)
        rtt: Dict[str, Optional[int]] = {}
        for target in (0.50, 0.70, 0.80, 0.90):
            hit = next((r for r, a in zip(rounds, accs) if a >= target), None)
            rtt[f"rounds_to_{int(target * 100)}pct"] = hit

        # AUC of accuracy vs round (trapezoidal integration normalised by total rounds)
        total_rounds = max(rounds) - min(rounds) if len(rounds) > 1 else 1
        auc = 0.0
        for i in range(1, len(rounds)):
            dr = rounds[i] - rounds[i - 1]
            auc += 0.5 * (accs[i] + accs[i - 1]) * dr
        auc /= total_rounds

        data = {
            "final_clean_acc":          self.final_clean_acc,
            "best_clean_acc":           self.best_clean_acc,
            "acc_auc_normalised":       auc,
            "pooled_tpr":               _nan_safe(self.pooled_tpr()),
            "pooled_fpr":               _nan_safe(self.pooled_fpr()),
            "mean_asr_attack_window":   _nan_safe(self.mean_asr_attack_window()),
            "max_asr_attack_window":    _nan_safe(self.max_asr_attack_window()),
            **{k: v for k, v in rtt.items()},
        }
        with open(self._convergence_path(), "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _format_row(row: RoundMetrics) -> dict:
        """Format floats to 6 d.p.; keep NaN as literal 'NaN'; keep ints as-is."""
        out = {}
        for f in fields(row):
            val = getattr(row, f.name)
            if isinstance(val, float):
                out[f.name] = "NaN" if math.isnan(val) else f"{val:.6f}"
            else:
                out[f.name] = val
        return out

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        if not self._rows:
            print("MetricsTracker: no rounds recorded.")
            return

        last = self._rows[-1]
        has_attack  = not math.isnan(last.asr)
        has_defense = last.defense_fp >= 0

        sep = "=" * 62
        print(f"\n{sep}")
        print(f"  Summary — {len(self._rows)} rounds evaluated")
        print(sep)
        print(f"  {'Final clean accuracy':<32} {last.clean_acc * 100:>7.2f}%")
        print(f"  {'Best  clean accuracy':<32} {self.best_clean_acc * 100:>7.2f}%")
        print(f"  {'Final clean loss':<32} {last.clean_loss:>8.4f}")

        if not math.isnan(last.macro_f1):
            print(f"  {'Macro F1':<32} {last.macro_f1 * 100:>7.2f}%")

        if has_attack:
            mean_asr = self.mean_asr_attack_window()
            max_asr  = self.max_asr_attack_window()
            print(f"  {'Final ASR':<32} {last.asr * 100:>7.2f}%")
            if not math.isnan(mean_asr):
                print(f"  {'Mean ASR (attack window)':<32} {mean_asr * 100:>7.2f}%")
                print(f"  {'Max  ASR (attack window)':<32} {max_asr * 100:>7.2f}%")

        if has_defense:
            ptpr = self.pooled_tpr()
            pfpr = self.pooled_fpr()
            if not math.isnan(ptpr):
                print(f"  {'Pooled TPR (attack period)':<32} {ptpr * 100:>7.2f}%")
            print(f"  {'Pooled FPR (all rounds)':<32} {pfpr * 100:>7.2f}%")

        print(sep + "\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nan_safe(v: float) -> object:
    """Convert NaN to None so JSON serialisation doesn't produce 'NaN'."""
    return None if math.isnan(v) else v
