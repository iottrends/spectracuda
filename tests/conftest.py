import pytest

from spectracuda.backend import cupy_available


@pytest.fixture(params=["numpy", "cupy"])
def backend(request):
    """Parametrize a test over both backends, skipping cupy when no
    working CUDA runtime is present (e.g. this dev machine, until the
    Jetson arrives)."""
    name = request.param
    if name == "cupy" and not cupy_available():
        pytest.skip("cupy/CUDA not available in this environment")
    return name
