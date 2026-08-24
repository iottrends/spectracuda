"""The first slice of the two-node redesign (docs/mac.md): one HW unit
that only transmits, one that only receives, EACH owning its own
independently-constructed Ofdm (Mac(mode=..., ofdm_kwargs=...)) -- no
MacLink, no shared Ofdm object, no orchestrator class. Mostly UM mode
(genuinely one-directional); AM's DATA-forward half also works over
send_iq()/receive_iq() (see test_am_mode_data_forward_half_works_over_
send_iq_receive_iq below) -- its STATUS/retransmission round trip needs
a second independent Mac/Ofdm pair for the reverse direction, which is
what the 4-Mac/4-Ofdm bidirectional AM examples are for.
"""
import numpy as np
import pytest

from spectracuda.mac import Mac
from spectracuda.sim import Channel

_PHY_KWARGS = dict(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32,
    crc="crc16", sync="schmidl_cox", cfo="schmidl_cox",
    n_training_symbols=2, backend="numpy",
)


def _bind(mac_a, mac_b):
    """The real 3-call handshake (see docs/mac.md, "Migrate binding...")
    -- mac_a requests, mac_b evaluates against ITS OWN capacity, mac_a
    adopts the result. Asserts success; used by tests where binding
    isn't itself the thing under test."""
    req_iq = mac_a.build_bind_request()
    resp_iq = mac_b.handle_bind_request_iq(req_iq)
    assert resp_iq is not None
    assert mac_a.handle_bind_response_iq(resp_iq)
    assert mac_a.bound and mac_b.bound


def test_two_genuinely_independent_ofdm_objects():
    hw1_tx_mac = Mac(mode="um", ofdm_kwargs=_PHY_KWARGS)
    hw2_rx_mac = Mac(mode="um", ofdm_kwargs=_PHY_KWARGS)
    assert hw1_tx_mac.ofdm is not hw2_rx_mac.ofdm  # never the same object
    assert hw1_tx_mac is not hw2_rx_mac


def test_hw1_tx_to_hw2_rx_clean_channel():
    hw1_tx_mac = Mac(mode="um", ofdm_kwargs=_PHY_KWARGS)
    hw2_rx_mac = Mac(mode="um", ofdm_kwargs=_PHY_KWARGS)
    _bind(hw1_tx_mac, hw2_rx_mac)

    sdu = np.random.default_rng(0).integers(0, 2, size=800).astype("uint8")
    iq_frames = hw1_tx_mac.send_iq(sdu)
    assert len(iq_frames) >= 1  # this config's derived capacity (~55K bits)
    # comfortably fits 800 bits in one PDU -- segmentation-across-multiple-
    # PDUs is already covered by test_mac_um.py's dedicated tests

    delivered = []
    for iq in iq_frames:
        delivered.extend(hw2_rx_mac.receive_iq(iq))

    assert len(delivered) == 1
    np.testing.assert_array_equal(delivered[0], sdu)


def test_hw1_tx_to_hw2_rx_through_a_real_lossy_channel():
    """Same proof, but through spectracuda.sim.Channel -- confirms the
    IQ genuinely crosses a channel transform between the two independent
    objects, not just a direct array handoff."""
    hw1_tx_mac = Mac(mode="um", ofdm_kwargs=_PHY_KWARGS)
    hw2_rx_mac = Mac(mode="um", ofdm_kwargs=_PHY_KWARGS)
    _bind(hw1_tx_mac, hw2_rx_mac)
    channel = Channel(snr_db=25.0, seed=0, backend="numpy")

    sdu = np.random.default_rng(1).integers(0, 2, size=400).astype("uint8")
    delivered = []
    for iq in hw1_tx_mac.send_iq(sdu):
        iq = channel.process(iq)
        delivered.extend(hw2_rx_mac.receive_iq(iq))

    assert len(delivered) == 1
    np.testing.assert_array_equal(delivered[0], sdu)


def test_mismatched_sync_prevents_even_binding():
    """Not a config-match validation (no orchestrator exists at this
    step to put one in, see docs/mac.md) -- confirms a real physical
    mismatch between two independently-configured units genuinely fails,
    not silently works by luck. Caught EARLIER than it used to be, now
    that send_iq() requires a successful bind() first: the bind handshake
    itself travels over the same mismatched PHY (hw1's schmidl_cox
    preamble, hw2's zc sync detector), so it fails to complete before
    send_iq() would even become reachable -- a real, if incidental,
    consequence of the bind gate, not a workaround. (Previously this test
    sent data anyway and observed the decode fail on the receive side;
    that specific path is no longer reachable at all now, which is
    arguably the more honest outcome -- a real radio that can't complete
    a handshake doesn't get to transmit either.)"""
    hw1_tx_mac = Mac(mode="um", ofdm_kwargs={**_PHY_KWARGS, "sync": "schmidl_cox", "cfo": "schmidl_cox"})
    hw2_rx_mac = Mac(mode="um", ofdm_kwargs={**_PHY_KWARGS, "sync": "zc", "cfo": "pilot_based"})

    req_iq = hw1_tx_mac.build_bind_request()
    resp_iq = hw2_rx_mac.handle_bind_request_iq(req_iq)
    assert resp_iq is None  # hw2 never even decoded the request
    assert not hw1_tx_mac.bound
    assert not hw2_rx_mac.bound
    with pytest.raises(ValueError, match="requires a successful bind"):
        hw1_tx_mac.send_iq(np.zeros(8, dtype="uint8"))


def test_send_iq_and_receive_iq_require_ofdm_kwargs():
    plain = Mac(mode="um", max_segment_bits=504)
    with pytest.raises(ValueError, match="requires this Mac to have been constructed with ofdm_kwargs"):
        plain.send_iq(np.zeros(8, dtype="uint8"))
    with pytest.raises(ValueError, match="requires this Mac to have been constructed with ofdm_kwargs"):
        plain.receive_iq(np.zeros(10, dtype="complex64"))


def test_am_mode_data_forward_half_works_over_send_iq_receive_iq():
    """AM's `send_iq()`/`receive_iq()` are no longer blocked (that guard
    was removed once the two-node model made a genuine second Mac/Ofdm
    pair available for the reverse direction) -- the DATA-forward half of
    AM behaves exactly like UM's over the wire, gated by the same bind()
    requirement every mode has. What's deliberately NOT exercised here is
    the STATUS/retransmission round trip: that needs a second independent
    Mac/Ofdm pair for the reverse direction (build_status()/receive_status()
    routed through a DIFFERENT Ofdm than the one DATA travelled over) --
    see the 4-Mac/4-Ofdm bidirectional AM examples for that."""
    hw1_tx_mac = Mac(mode="am", ofdm_kwargs=_PHY_KWARGS)
    hw2_rx_mac = Mac(mode="am", ofdm_kwargs=_PHY_KWARGS)
    _bind(hw1_tx_mac, hw2_rx_mac)

    sdu = np.random.default_rng(4).integers(0, 2, size=600).astype("uint8")
    iq_frames = hw1_tx_mac.send_iq(sdu)
    assert len(iq_frames) >= 1

    delivered = []
    for iq in iq_frames:
        delivered.extend(hw2_rx_mac.receive_iq(iq))

    assert len(delivered) == 1
    np.testing.assert_array_equal(delivered[0], sdu)

    # receive_iq() on a genuinely undecodable frame returns [] for AM too
    # (AmEntity.receive_data() is a list-returning method, same as UM's
    # receive() -- the None-on-failure convention TM/generic modes use
    # would be the wrong shape here).
    garbage = np.zeros(10, dtype="complex64")
    assert hw2_rx_mac.receive_iq(garbage) == []


def test_max_segment_bits_cannot_be_passed_alongside_ofdm_kwargs():
    with pytest.raises(ValueError, match="must not be passed alongside ofdm_kwargs"):
        Mac(mode="um", max_segment_bits=1000, ofdm_kwargs=_PHY_KWARGS)


def test_ofdm_kwargs_requires_crc_enabled():
    with pytest.raises(ValueError, match="requires crc"):
        Mac(mode="um", ofdm_kwargs={**_PHY_KWARGS, "crc": "none"})


def test_max_segment_bits_still_required_without_ofdm_kwargs():
    with pytest.raises(ValueError, match="max_segment_bits is required"):
        Mac(mode="um")


def test_existing_pdu_level_api_unaffected_by_ofdm_kwargs_addition():
    """Regression guard for the exact bug found during implementation:
    Mac.receive_iq() must NOT shadow the __getattr__-delegated PDU-level
    receive()/transmit() that MacLink (and this test) depend on when
    ofdm_kwargs is NOT given."""
    tx = Mac(mode="um", max_segment_bits=504)
    rx = Mac(mode="um", max_segment_bits=504)
    sdu = np.random.default_rng(0).integers(0, 2, size=32).astype("uint8")
    pdus = tx.transmit(sdu)
    delivered = []
    for pdu in pdus:
        delivered.extend(rx.receive(pdu))
    assert len(delivered) == 1
    np.testing.assert_array_equal(delivered[0], sdu)


# --- binding + link-quality reporting, migrated onto Mac (docs/mac.md) ---
# The actual payoff of this migration: MacLink.bind() always evaluates a
# request against its OWN capacity (one shared Ofdm across both roles),
# so it can only ever self-consistently succeed -- test_mac_bind.py's
# evaluate_bind_request() rejection proof was ONLY reachable as a pure
# function before this. Two real Mac(ofdm_kwargs=...) objects have
# genuinely independent self.max_segment_bits, so the tests below are the
# first REJECTION proof through the actual wire mechanism.

def test_real_cross_object_bind_succeeds_with_compatible_capacity():
    hw1 = Mac(mode="um", ofdm_kwargs=_PHY_KWARGS)
    hw2 = Mac(mode="um", ofdm_kwargs=_PHY_KWARGS)
    assert not hw1.bound and not hw2.bound

    req_iq = hw1.build_bind_request()
    resp_iq = hw2.handle_bind_request_iq(req_iq)
    assert resp_iq is not None
    assert hw2.bound  # acceptor already knows its own decision
    assert hw1.handle_bind_response_iq(resp_iq)
    assert hw1.bound and hw2.bound


def test_real_cross_object_bind_genuinely_rejects_a_mismatched_capacity():
    """The proof that was impossible through MacLink: hw1 (qam64, larger
    per-frame capacity when IT transmits/binds) requests more than hw2
    (bpsk, genuinely smaller capacity) can actually support -- hw2's OWN
    evaluate_bind_request() rejects it for real, not self-consistently."""
    big = Mac(mode="um", ofdm_kwargs={**_PHY_KWARGS, "modem": "qam64"})
    small = Mac(mode="um", ofdm_kwargs={**_PHY_KWARGS, "modem": "bpsk"})
    assert big.max_segment_bits > small.max_segment_bits  # genuine asymmetry, not staged

    req_iq = big.build_bind_request()
    resp_iq = small.handle_bind_request_iq(req_iq)
    assert resp_iq is not None  # the REQUEST arrived fine -- rejection is a real decision, not a lost frame
    assert not small.bound  # small genuinely refused it
    assert not big.handle_bind_response_iq(resp_iq)
    assert not big.bound


def test_send_iq_before_bind_raises_after_migration():
    hw1 = Mac(mode="um", ofdm_kwargs=_PHY_KWARGS)
    with pytest.raises(ValueError, match="requires a successful bind"):
        hw1.send_iq(np.zeros(8, dtype="uint8"))


def test_quality_report_round_trip_between_two_real_macs():
    hw1 = Mac(mode="um", ofdm_kwargs=_PHY_KWARGS)
    hw2 = Mac(mode="um", ofdm_kwargs=_PHY_KWARGS)
    _bind(hw1, hw2)

    sdu = np.random.default_rng(3).integers(0, 2, size=200).astype("uint8")
    for iq in hw1.send_iq(sdu):
        hw2.receive_iq(iq)

    report_iq = hw1.build_quality_report()
    decoded = hw2.handle_quality_report_iq(report_iq)
    assert decoded["n_attempts"] > 0
    assert 0.0 <= decoded["delivered_ratio"] <= 1.0
