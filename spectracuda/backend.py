"""Backend abstraction: numpy/cupy array-module selection.

Every Block holds `self.xp`, resolved once at construction and threaded
down through the whole pipeline (see docs/architecture.md, "Backend
abstraction"). This keeps every block runnable in backend="numpy" mode
with zero GPU present -- the correctness test suite depends on this,
since Jetson hardware won't be in a normal CI runner -- while
backend="cupy" is the real GPU execution path on Jetson.
"""
from __future__ import annotations

from typing import Any, Literal

BackendName = Literal["numpy", "cupy"]

_CUPY_AVAILABLE: bool | None = None


def cupy_available() -> bool:
    """Return True if a working CuPy + CUDA runtime is importable.

    Result is cached after the first call (import + device probe is not
    free). Cleared implicitly per-process only -- this is fine since the
    CUDA runtime doesn't appear/disappear mid-process in practice.
    """
    global _CUPY_AVAILABLE
    if _CUPY_AVAILABLE is None:
        try:
            import cupy  # noqa: F401

            cupy.cuda.runtime.getDeviceCount()
            _CUPY_AVAILABLE = True
        except Exception:
            _CUPY_AVAILABLE = False
    return _CUPY_AVAILABLE


def get_xp(backend: BackendName = "numpy") -> Any:
    """Resolve a backend name to its array module (numpy or cupy).

    Raises RuntimeError with a clear message if backend="cupy" is
    requested but no working CUDA/CuPy runtime is available. This fails
    loudly rather than silently falling back to numpy, since a silent
    fallback would hide a real performance regression (or a broken
    Jetson setup) rather than surface it.
    """
    if backend == "numpy":
        import numpy as np

        return np
    if backend == "cupy":
        if not cupy_available():
            raise RuntimeError(
                "backend='cupy' was requested but no working CuPy/CUDA "
                "runtime is available in this environment. Install the "
                "cupy build matching your CUDA/JetPack version (see the "
                "'cuda' extra in pyproject.toml), or use backend='numpy'."
            )
        import cupy as cp

        return cp
    raise ValueError(f"Unknown backend {backend!r}; expected 'numpy' or 'cupy'")


def default_backend() -> BackendName:
    """Best available backend: cupy if a working GPU is present, else numpy."""
    return "cupy" if cupy_available() else "numpy"
