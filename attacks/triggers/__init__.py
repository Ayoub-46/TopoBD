"""Trigger registry for the FL backdoor research framework.

Usage::

    from attacks.triggers import get_trigger, register_trigger

    # Built-in:
    trigger = get_trigger("patch", position=(29, 29), size=(3, 3))
    trigger = get_trigger("a3fl", in_channels=3, image_size=(32, 32))

    # Register a custom trigger:
    @register_trigger("my_trigger")
    def build_my_trigger(**kwargs):
        return MyTrigger(**kwargs)
"""

from typing import Callable, Dict

from .base import BaseTrigger, LearnableTrigger
from .patch import PatchTrigger
from .a3fl import A3FLTrigger

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Callable[..., BaseTrigger]] = {}


def register_trigger(name: str) -> Callable:
    """Decorator that registers a trigger factory function under ``name``.

    The decorated callable accepts keyword arguments and returns a
    :class:`BaseTrigger` instance.

    Example::

        @register_trigger("my_trigger")
        def build(**kwargs) -> BaseTrigger:
            return MyTrigger(**kwargs)
    """
    name = name.lower()

    def decorator(fn: Callable[..., BaseTrigger]) -> Callable:
        if name in _REGISTRY:
            raise KeyError(
                f"A trigger is already registered as '{name}'. "
                "Choose a unique name or remove the existing entry first."
            )
        _REGISTRY[name] = fn
        return fn

    return decorator


def get_trigger(name: str, **kwargs) -> BaseTrigger:
    """Instantiate a trigger by registry name.

    Args:
        name:     Registry key (case-insensitive).
        **kwargs: Forwarded to the trigger's constructor.

    Returns:
        An initialised :class:`BaseTrigger` subclass.

    Raises:
        KeyError: if ``name`` is not registered.
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"No trigger registered as '{key}'. "
            f"Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key](**kwargs)


def list_triggers():
    """Return a sorted list of all registered trigger names."""
    return sorted(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Built-in registrations
# ---------------------------------------------------------------------------

@register_trigger("patch")
def _build_patch(**kwargs) -> PatchTrigger:
    # Filter to only the kwargs PatchTrigger accepts.
    # The runner always injects in_channels/image_size for shape-aware
    # triggers; static triggers must silently discard what they don't need.
    valid = {"position", "size", "color", "alpha"}
    return PatchTrigger(**{k: v for k, v in kwargs.items() if k in valid})


@register_trigger("a3fl")
def _build_a3fl(**kwargs) -> A3FLTrigger:
    # A3FLTrigger accepts in_channels and image_size — forward what it knows.
    valid = {
        "position", "size", "in_channels", "image_size",
        "trigger_epochs", "trigger_lr", "lambda_balance",
        "adv_epochs", "adv_lr", "alpha",
    }
    return A3FLTrigger(**{k: v for k, v in kwargs.items() if k in valid})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "BaseTrigger",
    "LearnableTrigger",
    "PatchTrigger",
    "A3FLTrigger",
    "get_trigger",
    "register_trigger",
    "list_triggers",
]