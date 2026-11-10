import importlib
from types import ModuleType
from typing import TYPE_CHECKING

from .initializer import init_torch_env

init_torch_env()

if TYPE_CHECKING:
    from . import initializer, models, modules, training, transforms

__version__ = "0.0.1"

__all__ = ["models", "modules", "transforms", "training", "initializer"]


def __getattr__(name: str) -> ModuleType:
    if name in __all__:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
