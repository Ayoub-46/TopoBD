"""Per-round metrics collection and CSV / JSON export.

Tracked quantities
------------------
Every row in the output CSV corresponds to one evaluated round:

+-------------------------------+-------------------------------------------+
| Column                        | Description                               |
+===============================+===========================================+
| ``round``                     | Round index (0-based).                    |
+-------------------------------+-------------------------------------------+
| ``clean_loss``                | Global model loss on the clean test set.  |
+-------------------------------+-------------------------------------------+
| ``clean_acc``                 | Global model accuracy on clean test set.  |
+-------------------------------+-------------------------------------------+
| ``asr``                       | Attack success rate on backdoor test set. |
|                               | ``NaN`` when no attack is configured.     |
+-------------------------------+-------------------------------------------+
| ``asr_loss``                  | Loss on the backdoor test set.            |
+-------------------------------+-------------------------------------------+
| ``is_attack_round``           | ``1`` if ≥ 1 malicious client was sampled |
|                               | this round, else ``0``.                   |
+-------------------------------+-------------------------------------------+
| ``n_selected``                | Clients sampled this round.               |
+-------------------------------+-------------------------------------------+
| ``n_malicious_selected``      | Malicious clients among those sampled.    |
+-------------------------------+-------------------------------------------+
| ``total_samples``             | Training samples aggregated this round.   |
+-------------------------------+-------------------------------------------+
| ``defense_tpr``               | True-positive rate of the defense filter  |
|                               | (correctly rejected / total malicious     |
|                               | in round). ``NaN`` when no detection      |
|                               | defense is active.                        |
+-------------------------------+-------------------------------------------+
| ``defense_fpr``               | False-positive rate (incorrectly rejected |
|                               | benign / total benign in round).          |
+-------------------------------+-------------------------------------------+
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass, field, fields
from typing import List, Optional


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

    # Defense detection (NaN → no detection defense active)
    defense_tpr: float = math.nan
    defense_fpr: float = math.nan


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class MetricsTracker:
    """Collects :class:`RoundMetrics` across all rounds and exports results.

    Usage::

        tracker = MetricsTracker(output_dir="results/run_01")
        # inside the FL loop:
        tracker.record(RoundMetrics(round=r, clean_loss=..., clean_acc=...))
        # after the loop:
        tracker.save()          # writes metrics.csv and metrics.json
        tracker.print_summary() # console table
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self._rows: List[RoundMetrics] = []

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def record(self, metrics: RoundMetrics) -> None:
        """Append one round's metrics and immediately flush to CSV.

        Flushing every round means partial results are available even if
        the experiment crashes mid-run.
        """
        self._rows.append(metrics)
        self._append_csv_row(metrics)

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

    @property
    def avg_defense_tpr(self) -> float:
        vals = [r.defense_tpr for r in self._rows if not math.isnan(r.defense_tpr)]
        return sum(vals) / len(vals) if vals else math.nan

    @property
    def avg_defense_fpr(self) -> float:
        vals = [r.defense_fpr for r in self._rows if not math.isnan(r.defense_fpr)]
        return sum(vals) / len(vals) if vals else math.nan

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write ``metrics.csv`` and ``metrics.json`` to ``output_dir``."""
        os.makedirs(self.output_dir, exist_ok=True)
        # CSV is already written row-by-row; rewrite cleanly for portability
        self._write_csv_full()
        self._write_json()

    def _csv_path(self) -> str:
        return os.path.join(self.output_dir, "metrics.csv")

    def _json_path(self) -> str:
        return os.path.join(self.output_dir, "metrics.json")

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

    @staticmethod
    def _format_row(row: RoundMetrics) -> dict:
        """Format floats to 6 d.p.; keep NaN as the literal string 'NaN'."""
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
        has_defense = not math.isnan(last.defense_tpr)

        sep = "=" * 58
        print(f"\n{sep}")
        print(f"  Summary — {len(self._rows)} rounds evaluated")
        print(sep)
        print(f"  {'Final clean accuracy':<30} {last.clean_acc * 100:>7.2f}%")
        print(f"  {'Best  clean accuracy':<30} {self.best_clean_acc * 100:>7.2f}%")
        print(f"  {'Final clean loss':<30} {last.clean_loss:>8.4f}")

        if has_attack:
            print(f"  {'Final ASR':<30} {last.asr * 100:>7.2f}%")

        if has_defense:
            tpr = self.avg_defense_tpr
            fpr = self.avg_defense_fpr
            print(f"  {'Avg defense TPR':<30} {tpr * 100:>7.2f}%")
            print(f"  {'Avg defense FPR':<30} {fpr * 100:>7.2f}%")

        print(sep + "\n")