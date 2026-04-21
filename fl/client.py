from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Typed return containers
# ---------------------------------------------------------------------------

@dataclass
class ClientUpdate:
    """Structured return value from a client's local_train call.

    The `metadata` field is intentionally open-ended so that attack subclasses
    can attach extra information (e.g. scaling factors, trigger masks) without
    changing the contract.
    """
    client_id: int
    num_samples: int
    weights: Dict[str, torch.Tensor]
    metrics: Dict[str, float]
    round_idx: int
    is_malicious: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Structured return value from a client's local_evaluate call."""
    client_id: int
    num_samples: int
    metrics: Dict[str, float]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseClient(ABC):
    @abstractmethod
    def get_id(self) -> int:
        pass

    @abstractmethod
    def num_samples(self) -> int:
        pass

    @abstractmethod
    def set_params(self, state_dict: Dict[str, torch.Tensor]) -> None:
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def local_train(
        self,
        epochs: Optional[int],
        round_idx: int,
        **kwargs,
    ) -> ClientUpdate:
        """Train locally for `epochs` rounds (falls back to constructor default
        when *None*; passing 0 is valid and skips training entirely).
        """

    @abstractmethod
    def local_evaluate(self) -> EvalResult:
        pass


# ---------------------------------------------------------------------------
# Concrete benign client
# ---------------------------------------------------------------------------

class BenignClient(BaseClient):
    def __init__(
        self,
        id: int,
        trainloader: Optional[DataLoader],
        testloader: Optional[DataLoader],
        model: nn.Module,
        lr: float,
        weight_decay: float,
        epochs: int = 1,
        device: Optional[torch.device] = None,
    ):
        self.id = id
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device if device is not None else torch.device("cpu")
        self.epochs_default = epochs
        self.lr = lr
        self.weight_decay = weight_decay

        self._model: nn.Module = model.to(self.device)
        self.dataset_len: int = len(trainloader.dataset) if trainloader is not None else 0

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self._create_optimizer()

        self.loss_fn = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def model(self) -> nn.Module:
        return self._model

    def _create_optimizer(self) -> None:
        """(Re)create the optimizer bound to the current model parameters.

        Called at construction and after every set_params() to ensure the
        momentum buffer never carries over between rounds — standard FedAvg
        behaviour expected by FL security benchmarks.
        """
        self.optimizer = torch.optim.SGD(
            self._model.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=self.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.97
        )

    # ------------------------------------------------------------------
    # BaseClient interface
    # ------------------------------------------------------------------

    def get_id(self) -> int:
        return self.id

    def num_samples(self) -> int:
        return self.dataset_len

    def set_params(self, params: Dict[str, torch.Tensor]) -> None:
        """Load a new parameter snapshot and reset the optimizer state."""
        self._model.load_state_dict(
            {k: v.to(self.device) for k, v in params.items()}
        )
        self._create_optimizer()

    def get_params(self) -> Dict[str, torch.Tensor]:
        """Return a CPU copy of the current model state."""
        return {k: v.cpu().clone() for k, v in self._model.state_dict().items()}

    def local_train(
        self,
        epochs: Optional[int] = None,
        round_idx: int = 0,
        **kwargs,
    ) -> ClientUpdate:
        """Run local SGD for `epochs` passes over the training set.

        Args:
            epochs:    Number of local epochs.  Pass *None* to use the
                       constructor default.  Pass *0* to skip training (the
                       client will still return its current weights).
            round_idx: Global round index, forwarded into the returned update.
            **kwargs:  Reserved for subclass use (e.g. attack-specific args).

        Returns:
            A :class:`ClientUpdate` dataclass instance.
        """
        n_epochs: int = epochs if epochs is not None else self.epochs_default

        self._model.train()
        train_loss, correct, total, steps = 0.0, 0, 0, 0

        for _ in range(n_epochs):
            if self.trainloader is None:
                break
            for inputs, targets in self.trainloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self._model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
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

        return ClientUpdate(
            client_id=self.get_id(),
            num_samples=self.num_samples(),
            weights=self.get_params(),
            metrics={"loss": avg_loss, "accuracy": accuracy},
            round_idx=round_idx,
            is_malicious=False,
        )

    def local_evaluate(self) -> EvalResult:
        """Evaluate on the local test set (falls back to train set if absent)."""
        self._model.eval()
        loss_sum, correct, total, iters = 0.0, 0, 0, 0

        valloader = self.testloader or self.trainloader
        if valloader is None:
            return EvalResult(
                client_id=self.get_id(),
                num_samples=0,
                metrics={"loss": float("nan"), "accuracy": float("nan")},
            )

        with torch.no_grad():
            for inputs, targets in valloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self._model(inputs)
                _, preds = torch.max(outputs.data, 1)

                correct += (preds == targets).sum().item()
                loss_sum += self.loss_fn(outputs, targets).item()
                total += targets.size(0)
                iters += 1

        return EvalResult(
            client_id=self.get_id(),
            num_samples=total,
            metrics={
                "loss": loss_sum / iters if iters > 0 else float("nan"),
                "accuracy": correct / total if total > 0 else float("nan"),
            },
        )