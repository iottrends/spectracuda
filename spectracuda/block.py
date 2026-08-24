"""Base class for all spectracuda processing blocks.

Every block operates on a batch dimension as its primary shape -- e.g.
`(n_batch, fft_size)` complex arrays in and out -- never a single scalar
sample/symbol. This is a hard requirement, not a style preference: GPU
acceleration on Jetson only pays off once operations are batched (kernel
launch and Python/CuPy dispatch overhead otherwise dominates). See
docs/architecture.md, "Batch-shape contract".

Subclasses must override `batch_shape_doc` with a short, precise
description of their exact input/output batch-shape contract.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .backend import BackendName, default_backend, get_xp


class Block(ABC):
    """Common interface for every swappable (Layer 2) and fixed-infra
    (Layer 1) block.

    Parameters
    ----------
    backend:
        "numpy" or "cupy". Defaults to the best available backend
        (cupy if a working CUDA runtime is present, else numpy).
    """

    #: Subclasses should override this with their exact batch-shape
    #: contract, e.g.:
    #: "(n_batch, fft_size) complex64 in -> (n_batch, fft_size) complex64 out"
    batch_shape_doc: str = "unspecified -- subclass should document this"

    def __init__(self, *, backend: BackendName | None = None) -> None:
        self.backend: BackendName = backend or default_backend()
        self.xp = get_xp(self.backend)

    @abstractmethod
    def process(self, batch: Any, **kwargs: Any) -> Any:
        """Run this block over a batch. Shape contract: see `batch_shape_doc`."""
        raise NotImplementedError

    def __call__(self, batch: Any, **kwargs: Any) -> Any:
        return self.process(batch, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(backend={self.backend!r})"
