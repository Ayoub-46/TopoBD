from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import copy as cp
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Typed return containers
# ---------------------------------------------------------------------------

@dataclass
class AggregationResult:
    """Structured return value from an aggregator's aggregate() call.

    Carrying provenance metadata alongside the averaged parameters makes it
    straightforward for defense modules to hook into the aggregation pipeline
    and inspect per-client statistics without re-implementing bookkeeping.
    """
    aggregated_params: Dict[str, torch.Tensor]
    num_clients: int
    total_samples: int
    # Per-client weight used during aggregation, keyed by client_id.
    # Useful for defenses that re-weight or clip contributions.
    client_weights: Dict[int, float] = field(default_factory=dict)
    # Open-ended slot for defense/attack subclasses to attach extra info.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServerEvalResult:
    """Structured return value from a server-side evaluation pass."""
    num_samples: int
    metrics: Dict[str, float]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseServer(ABC):
    @abstractmethod
    def set_params(self, state_dict: Dict[str, torch.Tensor]) -> None:
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def aggregate(self) -> AggregationResult:
        pass

    @abstractmethod
    def reset(self) -> None:
        """Discard all buffered client updates to prepare for the next round."""


# ---------------------------------------------------------------------------
# Concrete FedAvg aggregator
# ---------------------------------------------------------------------------

class FedAvgAggregator(BaseServer):
    def __init__(
        self,
        model: nn.Module,
        testloader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        self.device = device if device is not None else torch.device("cpu")
        self.model: nn.Module = cp.deepcopy(model).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.testloader = testloader
        # Keyed by client_id; holds {'params': ..., 'length': ...}
        self._received_updates: Dict[int, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # BaseServer interface
    # ------------------------------------------------------------------

    def get_params(self) -> Dict[str, torch.Tensor]:
        """Return a CPU copy of the current global model state."""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def set_params(self, params: Dict[str, torch.Tensor]) -> None:
        """Load params onto the server model (moved to server device)."""
        self.model.load_state_dict({k: v.to(self.device) for k, v in params.items()})

    def aggregate(self) -> AggregationResult:
        """Weighted-average all buffered client updates (FedAvg).

        Handles integer/boolean tensors (e.g. BatchNorm's
        ``num_batches_tracked``) correctly by skipping floating-point
        averaging for them and instead forwarding the first client's value.

        Returns:
            An :class:`AggregationResult` that carries the averaged parameters
            alongside provenance metadata for downstream defense hooks.

        Raises:
            RuntimeError: if no client updates have been received.
        """

        if not self._received_updates:
            raise RuntimeError(
                "aggregate() called with no client updates buffered. "
                "Call receive_update() at least once before aggregating."
            )

        total_samples: int = sum(
            d["length"] for d in self._received_updates.values()
        )
        client_weights: Dict[int, float] = {
            cid: d["length"] / total_samples
            for cid, d in self._received_updates.items()
        }

        first_params = next(iter(self._received_updates.values()))["params"]
        averaged: Dict[str, torch.Tensor] = {}

        for key in first_params.keys():
            ref = first_params[key]

            # Non-floating-point params (e.g. num_batches_tracked,
            # running_mean for int8 models) must not be cast to float32.
            # Strategy: majority vote — use the value from the client whose
            # weight is largest (i.e. the client with most samples).
            if not ref.is_floating_point():
                majority_cid = max(client_weights, key=client_weights.__getitem__)
                averaged[key] = self._received_updates[majority_cid]["params"][key].clone()
                continue

            acc = torch.zeros_like(ref, dtype=torch.float32)
            for cid, data in self._received_updates.items():
                if key in data["params"]:
                    acc += data["params"][key].float() * client_weights[cid]

            # Restore the original dtype so downstream load_state_dict is
            # type-safe (e.g. float16 models stay float16)
            averaged[key] = acc.to(ref.dtype)

        # Push averaged params to the server model
        self.set_params({k: v.to(self.device) for k, v in averaged.items()})

        result = AggregationResult(
            aggregated_params={k: v.cpu().clone() for k, v in averaged.items()},
            num_clients=len(self._received_updates),
            total_samples=total_samples,
            client_weights=client_weights,
        )

        # FIX #9: reset is an explicit separate step so callers are never
        # surprised by a side-effect. Aggregate no longer clears the buffer;
        # the experiment runner should call reset() explicitly.
        return result

    def reset(self) -> None:
        """Discard all buffered client updates.

        Should be called by the experiment runner at the *end* of every round,
        *after* aggregate() has been called and the result has been processed.
        Keeping reset() separate from aggregate() means defenses can
        inspect self._received_updates after aggregation if needed.
        """
        self._received_updates.clear()

    # ------------------------------------------------------------------
    # Additional public methods
    # ------------------------------------------------------------------

    def receive_update(
        self,
        client_id: int,
        params: Dict[str, torch.Tensor],
        length: int,
    ) -> None:
        """Buffer a client update for the current round.

        Args:
            client_id: Unique identifier of the submitting client.
            params:    Model state dict (will be cloned to CPU float32 for
                       numerically stable accumulation).
            length:    Number of training samples used — the FedAvg weight.

        Raises:
            ValueError: if ``client_id`` has already submitted an update this
                        round, guarding against accidental double-counting.
        """
        if client_id in self._received_updates:
            raise ValueError(
                f"Client {client_id} has already submitted an update for this "
                "round. Call reset() before starting a new round."
            )

        self._received_updates[client_id] = {
            "params": {k: v.cpu().clone().float() for k, v in params.items()},
            "length": int(length),
        }

    def load_testdata(self, testloader: DataLoader) -> None:
        self.testloader = testloader

    def evaluate(self, valloader: Optional[DataLoader] = None) -> ServerEvalResult:
        """Run an evaluation pass on the server-side global model.

        Args:
            valloader: DataLoader to evaluate on.  Falls back to the loader
                       provided at construction time if *None*.

        Returns:
            A :class:`ServerEvalResult` dataclass instance.
        """
        loader = valloader or self.testloader
        self.model.eval()

        if loader is None:
            return ServerEvalResult(
                num_samples=0,
                metrics={"loss": float("nan"), "main_accuracy": float("nan")},
            )

        loss_sum, correct, total, iters = 0.0, 0, 0, 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs.data, 1)
                correct += (preds == targets).sum().item()
                loss_sum += self.loss_fn(outputs, targets).item()
                total += targets.size(0)
                iters += 1

        return ServerEvalResult(
            num_samples=total,
            metrics={
                "loss": loss_sum / iters if iters > 0 else float("nan"),
                "main_accuracy": correct / total if total > 0 else float("nan"),
            },
        )

    def save_model(self, path: str) -> None:
        """Persist the global model's state dict to disk."""
        torch.save(self.model.state_dict(), path)

    @classmethod
    def load_model(cls, model: nn.Module, path: str, device: Optional[torch.device] = None) -> "FedAvgAggregator":
        """Convenience factory: restore a saved aggregator from a checkpoint.

        Args:
            model:  Architecture instance (parameters will be overwritten).
            path:   Path previously passed to save_model().
            device: Target device.

        Returns:
            A new :class:`FedAvgAggregator` with the saved weights loaded.
        """
        instance = cls(model=model, device=device)
        state = torch.load(path, map_location=instance.device)
        instance.model.load_state_dict(state)
        return instance