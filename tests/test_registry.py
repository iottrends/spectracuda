import pytest

from spectracuda.block import Block
from spectracuda.registry import available, register, resolve


class _DummyBlock(Block):
    batch_shape_doc = "(n_batch,) any dtype in -> (n_batch,) same dtype out"

    def __init__(self, *, backend=None, gain: float = 1.0):
        super().__init__(backend=backend)
        self.gain = gain

    def process(self, batch, **kwargs):
        return batch * self.gain


@pytest.fixture(autouse=True)
def _register_dummy():
    register("dummy_category", "dummy")(_DummyBlock)
    yield


def test_register_and_resolve_by_string():
    inst = resolve("dummy_category", "dummy")
    assert isinstance(inst, _DummyBlock)


def test_resolve_passthrough_instance():
    built = _DummyBlock(gain=2.0)
    assert resolve("dummy_category", built) is built


def test_resolve_unknown_string_raises_with_known_list():
    with pytest.raises(KeyError):
        resolve("dummy_category", "nope")


def test_resolve_wrong_type_raises_typeerror():
    with pytest.raises(TypeError):
        resolve("dummy_category", 123)


def test_available_lists_registered_names():
    assert "dummy" in available("dummy_category")


def test_default_kwargs_passed_through_on_string_resolution():
    inst = resolve("dummy_category", "dummy", gain=3.0)
    assert inst.gain == 3.0
