"""The ``defenses`` package — Byzantine-robust aggregation servers.

Every defense is a subclass of :class:`~fl.server.FedAvgAggregator`.

Detection-based defenses (Krum, FLTrust, FLAME, …) additionally implement
``filter_updates(true_malicious) -> DetectionResult``.  The runner detects
this method via ``hasattr`` and calls it between local training and
aggregation, then uses the returned :class:`~experiment.utils.DetectionResult`
to compute per-round TPR / FPR.

Robust-aggregation defenses (trimmed mean, median, …) do NOT implement
``filter_updates`` — they override ``aggregate()`` instead.  TPR / FPR
columns remain ``NaN`` for these defenses.

Usage::

    from defenses import MKrumServer
    from models import get_model, ModelConfig

    server = MKrumServer(
        model=get_model(model_cfg),
        num_byzantine=10,
        num_to_select=5,
        device=device,
    )

    runner = FLRunner(cfg, defense_server=server)
    metrics = runner.run()
"""

from .mkrum import MKrumServer

__all__ = [
    "MKrumServer",
]