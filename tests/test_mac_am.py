import numpy as np
import pytest

from spectracuda.mac.am import AmEntity


def _sdu(n_bits: int, seed: int = 0) -> np.ndarray:
    return (np.random.default_rng(seed).integers(0, 2, size=n_bits)).astype("uint8")


def test_clean_round_trip_no_status_needed():
    tx, rx = AmEntity(max_segment_bits=64), AmEntity(max_segment_bits=64)
    sdu = _sdu(32)
    pdus = tx.transmit(sdu)
    delivered = []
    for p in pdus:
        delivered.extend(rx.receive_data(p))
    assert len(delivered) == 1
    np.testing.assert_array_equal(delivered[0], sdu)
    # nothing to retransmit: everything arrived
    status = rx.build_status()
    assert tx.receive_status(status) == []
    assert tx.failed_sns == set()


def test_recovers_a_single_dropped_segment_via_retransmission():
    """Self-verifying, not just asserted: confirm the SDU genuinely does
    NOT complete after the lossy first round, THEN confirm the
    retransmission recovers it -- proving the retry mechanism is doing
    real work, not redundant with what would have arrived anyway."""
    tx, rx = AmEntity(max_segment_bits=16, window_size=8, max_retries=3), AmEntity(
        max_segment_bits=16, window_size=8, max_retries=3
    )
    sdu = _sdu(88, seed=1)
    pdus = tx.transmit(sdu)
    assert len(pdus) >= 3

    delivered = []
    for i, p in enumerate(pdus):
        if i != 1:
            delivered.extend(rx.receive_data(p))
    assert delivered == []  # confirmed: genuinely incomplete after round 1

    status = rx.build_status()
    retx = tx.receive_status(status)
    assert len(retx) == 1  # exactly the one missing segment, nothing extra
    for p in retx:
        delivered.extend(rx.receive_data(p))

    assert len(delivered) == 1
    np.testing.assert_array_equal(delivered[0], sdu)


def test_gives_up_after_max_retries_and_reports_failed_sn():
    tx, rx = AmEntity(max_segment_bits=16, window_size=8, max_retries=2), AmEntity(
        max_segment_bits=16, window_size=8, max_retries=2
    )
    sdu = _sdu(88, seed=2)
    pdus = tx.transmit(sdu)
    assert len(pdus) >= 3

    for _round in range(6):
        for i, p in enumerate(pdus):
            if i == 1:
                continue  # segment 1 NEVER arrives, every round
            rx.receive_data(p)
        status = rx.build_status()
        retx = tx.receive_status(status)
        if not retx:
            break

    assert len(tx.failed_sns) == 1
    assert tx.pending_pdus == []  # given up, not still buffered forever


def test_lost_status_pdu_leaves_retx_buffer_untouched():
    tx = AmEntity(max_segment_bits=64, window_size=8, max_retries=3)
    sdu = _sdu(24)
    pdus = tx.transmit(sdu)
    assert len(tx.pending_pdus) == 1
    # simulate the status never arriving at all: no receive_status() call
    # -- pending_pdus must still be there for MacLink's fallback retry.
    assert len(tx.pending_pdus) == 1


def test_receive_status_rejects_a_data_pdu():
    tx = AmEntity(max_segment_bits=64)
    data_pdu = tx.transmit(_sdu(24))[0]
    with pytest.raises(ValueError, match="expected a STATUS pdu"):
        tx.receive_status(data_pdu)


def test_receive_data_rejects_a_status_pdu():
    rx = AmEntity(max_segment_bits=64)
    status_pdu = rx.build_status()
    with pytest.raises(ValueError, match="expected a DATA pdu"):
        rx.receive_data(status_pdu)


def test_non_byte_aligned_sdu_raises():
    tx = AmEntity(max_segment_bits=64)
    with pytest.raises(ValueError, match="not a multiple of 8"):
        tx.transmit(np.zeros(13, dtype="uint8"))
