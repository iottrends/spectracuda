"""Mac(ofdm_kwargs=...) through LDPC (fec0) -- the identical combination
tests/test_mac_block_oriented_fec.py exercises for rs_m8, mirrored here
now that LDPC has its own shortened-codeword support (see
spectracuda/fec/ldpc.py's encode()/decode() docstrings). Before this,
`Mac(ofdm_kwargs=dict(fec="ldpc_..."))`.build_bind_request() raised
ValueError immediately -- LDPC's "exact k_bits multiple only"
requirement meant even the 104-bit bind handshake had no valid size
(see docs/todo.md's #1.x rs_m8-shortening entry, which explicitly
flagged this as "Still broken for `Mac`+LDPC today" at the time).
"""
import numpy as np

from spectracuda.mac import Mac

PHY_KWARGS = dict(
    fft_size=256,
    n_pilot=8,
    n_data=216,
    cp_len=32,
    modem="qpsk",
    fec="ldpc_648_r12",  # fec0, inner -- shortened-codeword support under test
    fec1="none",
    crc="crc16",
    sync="schmidl_cox",
    cfo="schmidl_cox",
    n_training_symbols=2,
    backend="numpy",
)


def _bind(mac_a, mac_b):
    req_iq = mac_a.build_bind_request()
    resp_iq = mac_b.handle_bind_request_iq(req_iq)
    assert resp_iq is not None
    assert mac_a.handle_bind_response_iq(resp_iq)


def test_max_segment_bits_is_real_not_the_degenerate_floor():
    """Pins capacity.py's accepts_partial_block flag now being True for
    LDPC too -- see mac/capacity.py's own module comment."""
    mac = Mac(mode="um", ofdm_kwargs=PHY_KWARGS)
    assert mac.max_segment_bits > 1_000, (
        f"max_segment_bits={mac.max_segment_bits} looks like the old broken "
        f"floor, not a real derived capacity"
    )


def test_bind_handshake_completes():
    """The actual, previously-broken case: the bind handshake is a
    small, fixed 104-bit control PDU -- not a multiple of
    ldpc_648_r12's 324-bit k_bits, so this used to fail inside
    generate_frame() before LDPC's own shortened-codeword support."""
    hw1 = Mac(mode="um", ofdm_kwargs=PHY_KWARGS)
    hw2 = Mac(mode="um", ofdm_kwargs=PHY_KWARGS)
    _bind(hw1, hw2)
    assert hw1.bound and hw2.bound


def test_short_message_round_trips():
    """An ordinary short message, not block-aligned to LDPC's 324-bit
    k_bits either -- the realistic case, not hand-picked to fit."""
    hw1 = Mac(mode="um", ofdm_kwargs=PHY_KWARGS)
    hw2 = Mac(mode="um", ofdm_kwargs=PHY_KWARGS)
    _bind(hw1, hw2)

    sdu = np.random.default_rng(0).integers(0, 2, size=320).astype("uint8")
    delivered = []
    for iq in hw1.send_iq(sdu):
        delivered.extend(hw2.receive_iq(iq))
    assert len(delivered) == 1
    np.testing.assert_array_equal(delivered[0], sdu)


def test_message_longer_than_one_segment_round_trips():
    """Forces real multi-PDU segmentation/reassembly with LDPC in the
    mix -- the last, leftover segment is exactly the "not block-aligned"
    case in practice, not just the bind handshake."""
    hw1 = Mac(mode="um", ofdm_kwargs=PHY_KWARGS)
    hw2 = Mac(mode="um", ofdm_kwargs=PHY_KWARGS)
    _bind(hw1, hw2)

    sdu = np.random.default_rng(1).integers(0, 2, size=hw1.max_segment_bits + 2_000).astype("uint8")
    frames = hw1.send_iq(sdu)
    assert len(frames) >= 2  # genuine segmentation, not a single-PDU no-op

    delivered = []
    for iq in frames:
        delivered.extend(hw2.receive_iq(iq))
    assert len(delivered) == 1
    np.testing.assert_array_equal(delivered[0], sdu)
