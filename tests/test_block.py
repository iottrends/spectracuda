import pytest

from spectracuda.block import Block


def test_block_is_abstract():
    with pytest.raises(TypeError):
        Block()  # process() not implemented


def test_call_delegates_to_process():
    class _Passthrough(Block):
        def process(self, batch, **kwargs):
            return batch

    inst = _Passthrough(backend="numpy")
    assert inst(42) == 42
    assert inst.backend == "numpy"
    assert inst.xp.__name__ == "numpy"
