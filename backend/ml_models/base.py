import time

import torch

from backend.core.config import settings


def resolve_device() -> str:
    if settings.device != "auto":
        return settings.device
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Service:
    """Base class for ML services. Subclasses implement load() lazily.

    All errors must be translated into typed AppErrors here, never raw
    library stack traces (rules.md §3).
    """

    name = "base"

    def _timed(self, fn, *args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        return result, (time.perf_counter() - start) * 1000