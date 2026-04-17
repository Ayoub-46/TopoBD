"""The ``fl`` package — core federated learning primitives.

Public API
----------
Client side
~~~~~~~~~~~
:class:`BaseClient`
    Abstract base — subclass to implement custom client behaviour.

:class:`BenignClient`
    Standard honest client running local SGD.

:class:`ClientUpdate`
    Typed return value from :meth:`~BenignClient.local_train`.

:class:`EvalResult`
    Typed return value from :meth:`~BenignClient.local_evaluate`.

Server side
~~~~~~~~~~~
:class:`BaseServer`
    Abstract aggregator base.

:class:`FedAvgAggregator`
    Weighted-average aggregator (FedAvg).  Extended by defense servers
    in the ``defenses`` package.

:class:`AggregationResult`
    Typed return value from :meth:`~FedAvgAggregator.aggregate`.
    Carries ``aggregated_params``, ``client_weights``, and ``total_samples``
    so defense hooks have full provenance information.

:class:`ServerEvalResult`
    Typed return value from :meth:`~FedAvgAggregator.evaluate`.

Typical usage::

    from fl import BenignClient, FedAvgAggregator

    server = FedAvgAggregator(model=global_model, device=device)
    client = BenignClient(id=0, trainloader=..., model=..., lr=0.01, ...)

    client.set_params(server.get_params())
    update = client.local_train(epochs=5, round_idx=0)

    server.receive_update(update.client_id, update.weights, update.num_samples)
    result = server.aggregate()
    server.reset()
"""

from .client import BaseClient, BenignClient, ClientUpdate, EvalResult
from .server import (
    BaseServer,
    FedAvgAggregator,
    AggregationResult,
    ServerEvalResult,
)

__all__ = [
    # Client
    "BaseClient",
    "BenignClient",
    "ClientUpdate",
    "EvalResult",
    # Server
    "BaseServer",
    "FedAvgAggregator",
    "AggregationResult",
    "ServerEvalResult",
]