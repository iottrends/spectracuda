import numpy as np
import pytest

from spectracuda.mac.tm import TmEntity


def test_round_trip():
    tm = TmEntity(max_segment_bits=64)
    sdu = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype="uint8")
    pdus = tm.transmit(sdu)
    assert len(pdus) == 1
    np.testing.assert_array_equal(pdus[0], sdu)  # no header at all -- truly transparent
    out = tm.receive(pdus[0])
    np.testing.assert_array_equal(out, sdu)


def test_oversized_sdu_raises_rather_than_silently_truncating():
    tm = TmEntity(max_segment_bits=16)
    with pytest.raises(ValueError, match="exceeding TM's max_segment_bits"):
        tm.transmit(np.zeros(24, dtype="uint8"))


def test_non_byte_aligned_sdu_raises():
    tm = TmEntity(max_segment_bits=64)
    with pytest.raises(ValueError, match="not a multiple of 8"):
        tm.transmit(np.zeros(13, dtype="uint8"))


def test_kwargs_sink_absorbs_shared_construction_kwargs():
    # Mac(mode=...) constructs any mode from one shared kwargs dict --
    # TmEntity must accept and ignore params it doesn't use.
    tm = TmEntity(max_segment_bits=64, window_size=32, max_retries=4)
    assert tm.max_segment_bits == 64
