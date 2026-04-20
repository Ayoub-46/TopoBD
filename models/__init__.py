"""Model registry for the FL backdoor research framework.

Usage::

    from models import get_model, register_model, ModelConfig

    # Built-in:
    cfg = ModelConfig.from_adapter("resnet18", cifar10_adapter)
    model = get_model(cfg)

    # Register a custom architecture:
    @register_model("my_vgg")
    def build_my_vgg(config: ModelConfig):
        return MyVGG(config)

    model = get_model(ModelConfig.from_adapter("my_vgg", adapter))
"""

from typing import Callable, Dict

from .base import BaseModel, ModelConfig
from .cnn import SimpleCNN
from .gtsrb_cnn import GTSRBNet
from .lenet5 import LeNet5
from .resnet import resnet18, resnet34

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Maps lowercase string keys → callables that accept a ModelConfig and return
# an nn.Module (specifically a BaseModel subclass).
_REGISTRY: Dict[str, Callable[[ModelConfig], BaseModel]] = {}


def register_model(name: str) -> Callable:
    """Decorator that registers a model-builder function under ``name``.

    The decorated callable must accept a single :class:`ModelConfig` argument
    and return a :class:`BaseModel` instance.

    Example::

        @register_model("my_net")
        def build_my_net(config: ModelConfig) -> BaseModel:
            return MyNet(config)
    """
    name = name.lower()

    def decorator(fn: Callable[[ModelConfig], BaseModel]) -> Callable:
        if name in _REGISTRY:
            raise KeyError(
                f"A model is already registered under the name '{name}'. "
                "Choose a unique name or explicitly remove the existing entry "
                "from models._REGISTRY before re-registering."
            )
        _REGISTRY[name] = fn
        return fn

    return decorator


def get_model(config: ModelConfig) -> BaseModel:
    """Instantiate a model from a :class:`ModelConfig`.

    Args:
        config: Fully-specified model configuration.  ``config.name`` is used
                as the registry lookup key (case-insensitive).

    Returns:
        An initialised :class:`BaseModel` subclass instance.

    Raises:
        KeyError: if ``config.name`` is not found in the registry.
    """
    key = config.name.lower()
    if key not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise KeyError(
            f"No model registered under the name '{key}'. "
            f"Available models: {available}"
        )
    return _REGISTRY[key](config)


def list_models():
    """Return a sorted list of all registered model names."""
    return sorted(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Built-in registrations
# ---------------------------------------------------------------------------

@register_model("resnet18")
def _build_resnet18(config: ModelConfig) -> BaseModel:
    return resnet18(config)


@register_model("resnet34")
def _build_resnet34(config: ModelConfig) -> BaseModel:
    return resnet34(config)


@register_model("simple_cnn")
def _build_simple_cnn(config: ModelConfig) -> BaseModel:
    return SimpleCNN(config)


@register_model("gtsrb_cnn")
def _build_gtsrb_cnn(config: ModelConfig) -> BaseModel:
    return GTSRBNet(config)


@register_model("lenet5")
def _build_lenet5(config: ModelConfig) -> BaseModel:
    return LeNet5(config)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "BaseModel",
    "ModelConfig",
    "get_model",
    "register_model",
    "list_models",
    # concrete classes exposed for type-checking / isinstance guards
    "SimpleCNN",
    "GTSRBNet",
    "LeNet5",
]