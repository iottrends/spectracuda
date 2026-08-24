import numpy as np
import pytest

import spectracuda.backend as backend_mod
from spectracuda.backend import cupy_available, default_backend, get_xp


def test_numpy_backend_always_available():
    xp = get_xp("numpy")
    assert xp is np


def test_default_backend_matches_cupy_availability():
    assert default_backend() == ("cupy" if cupy_available() else "numpy")


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        get_xp("tensorflow")


def test_cupy_backend_without_cuda_raises_clearly(monkeypatch):
    monkeypatch.setattr(backend_mod, "cupy_available", lambda: False)
    with pytest.raises(RuntimeError, match="cupy"):
        get_xp("cupy")


def test_cupy_backend_smoke(backend):
    """Exercises the real cupy path whenever a working CUDA runtime is
    present (skipped otherwise via the `backend` fixture)."""
    xp = get_xp(backend)
    arr = xp.arange(4)
    total = arr.sum()
    if backend == "cupy":
        total = xp.asnumpy(total)
    assert int(total) == 6
