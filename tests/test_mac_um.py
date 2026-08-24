import random

import numpy as np
import pytest

from spectracuda.mac.um import UmEntity


def _sdu(n_bits: int, seed: int = 0) -> np.ndarray:
    return (np.random.default_rng(seed).integers(0, 2, size=n_bits)).astype("uint8")


def test_single_segment_round_trip():
    tx, rx = UmEntity(max_segment_bits=64), UmEntity(max_segment_bits=64)
    sdu = _sdu(32)
    pdus = tx.transmit(sdu)
    assert len(pdus) == 1
    delivered = rx.receive(pdus[0])
    assert len(delivered) == 1
    np.testing.assert_array_equal(delivered[0], sdu)


def test_multi_segment_round_trip_in_order():
    tx, rx = UmEntity(max_segment_bits=16), UmEntity(max_segment_bits=16)
    sdu = _sdu(88, seed=1)  # 88 bits, not an exact multiple of max_segment_bits=16
    pdus = tx.transmit(sdu)
    assert len(pdus) > 1
    delivered = []
    for p in pdus:
        delivered.extend(rx.receive(p))
    assert len(delivered) == 1
    np.testing.assert_array_equal(delivered[0], sdu)


def test_multi_segment_reassembly_survives_out_of_order_delivery():
    """The real bug this project's own reassembly buffer used to have:
    seeding `expected_sn` from whichever segment arrives FIRST (not the
    stream's true starting SN) desyncs reassembly the moment delivery
    order != send order -- this is a genuine RLC/MAC requirement (real
    channels reorder), not a hypothetical edge case."""
    tx, rx = UmEntity(max_segment_bits=16, window_size=8), UmEntity(max_segment_bits=16, window_size=8)
    sdu = _sdu(88, seed=2)
    pdus = tx.transmit(sdu)
    assert len(pdus) >= 3
    order = list(range(len(pdus)))
    random.Random(7).shuffle(order)
    delivered = []
    for i in order:
        delivered.extend(rx.receive(pdus[i]))
    assert len(delivered) == 1
    np.testing.assert_array_equal(delivered[0], sdu)


def test_dropped_segment_means_sdu_never_completes():
    """UM's defining trade-off: no retransmission, so a lost segment
    means permanent loss of that SDU -- not silently wrong data, just no
    data."""
    tx, rx = UmEntity(max_segment_bits=16, window_size=8), UmEntity(max_segment_bits=16, window_size=8)
    sdu = _sdu(88, seed=3)
    pdus = tx.transmit(sdu)
    assert len(pdus) >= 3
    delivered = []
    for i, p in enumerate(pdus):
        if i == 1:
            continue  # drop one segment
        delivered.extend(rx.receive(p))
    assert delivered == []


def test_sn_wraps_across_many_transmits():
    """Both ends start at SN 0 (the documented shared assumption -- see
    ReassemblyBuffer's constructor docstring) and stay in sync as SN
    genuinely wraps past 1023 back to 0 through ordinary use, not a
    hand-patched internal counter."""
    tx, rx = UmEntity(max_segment_bits=64, window_size=8), UmEntity(max_segment_bits=64, window_size=8)
    n_sends = 1030  # > SN_MODULUS (1024) -- guarantees a real wraparound
    delivered_all = []
    for i in range(n_sends):
        sdu = _sdu(24, seed=1000 + i)
        for p in tx.transmit(sdu):
            delivered_all.extend(rx.receive(p))
    assert len(delivered_all) == n_sends
    assert tx._next_tx_sn < 1024  # confirms it actually wrapped, not just counted up


def test_non_byte_aligned_sdu_raises():
    tx = UmEntity(max_segment_bits=64)
    with pytest.raises(ValueError, match="not a multiple of 8"):
        tx.transmit(np.zeros(13, dtype="uint8"))


def test_process_alone_returns_empty_list_when_nothing_completes():
    rx = UmEntity(max_segment_bits=16, window_size=8)
    # feed only a MIDDLE segment (sn=1) -- nothing can complete yet
    from spectracuda.mac.pdu import SI_MIDDLE, encode_header

    header = encode_header(pdu_type=0, si=SI_MIDDLE, sn=1, so=16)
    pdu = np.concatenate([header, np.zeros(16, dtype="uint8")])
    assert rx.receive(pdu) == []
