"""Mac(ofdm_kwargs=...) through a block-oriented fec0 (rs_m8/LDPC) -- a
combination NO existing MAC test exercises (verified directly: every
existing Mac(ofdm_kwargs=...) test either omits fec= entirely, defaulting
to "none", or uses conv_v27, which is bit-level and has no block-size
constraint to trip over -- see docs/mac.md for the writeup of how this
gap was actually found: while wiring up a real fft=256/16-QAM/
rs_m8+conv_v27 config for a two-process drone-link demo).

Exact config under test throughout this file -- matches liquid-dsp
parity for both FEC stages (rs_m8 as fec0/inner, conv_v27 as fec1/outer,
the CORRECTED ordering from docs/todo.md #1.12), with the interleaver
`unit_bits=8` that same correction requires to actually protect rs_m8:
"""
import numpy as np
import pytest

from spectracuda.mac import Mac

PHY_KWARGS = dict(
    fft_size=256,
    n_pilot=8,
    n_data=216,
    cp_len=32,
    modem="qam16",
    fec="rs_m8",                                  # fec0, inner
    fec1="conv_v27",                               # fec1, outer
    interleaver="block",
    interleaver_kwargs={"unit_bits": 8},
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
    """Pins the capacity.py fix: this used to derive 8 bits (unusable)
    for any block-oriented fec0 routed through Mac(ofdm_kwargs=...) --
    see capacity.py's own module comment for the full root cause."""
    mac = Mac(mode="um", ofdm_kwargs=PHY_KWARGS)
    assert mac.max_segment_bits > 10_000, (
        f"max_segment_bits={mac.max_segment_bits} looks like the old broken "
        f"floor, not a real derived capacity"
    )


def test_bind_handshake_completes():
    """The bind handshake itself is a small, fixed-size control PDU --
    NOT sized to rs_m8's 1784-bit block, so it currently fails inside
    generate_frame() the same way any other non-block-aligned payload
    does (see docs/mac.md's writeup: found via this exact test)."""
    hw1 = Mac(mode="um", ofdm_kwargs=PHY_KWARGS)
    hw2 = Mac(mode="um", ofdm_kwargs=PHY_KWARGS)
    _bind(hw1, hw2)
    assert hw1.bound and hw2.bound


def test_short_message_round_trips():
    """An ordinary short message -- nowhere near max_segment_bits, and
    not block-aligned either. This is the realistic case (a typed line,
    a small telemetry packet), not a hand-picked-to-fit size."""
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
    """Forces real multi-PDU segmentation/reassembly (UM's Segmenter/
    ReassemblyBuffer) with a block-oriented fec0 in the mix -- the last,
    leftover segment is exactly the "not block-aligned" case in
    practice, not just the bind handshake."""
    hw1 = Mac(mode="um", ofdm_kwargs=PHY_KWARGS)
    hw2 = Mac(mode="um", ofdm_kwargs=PHY_KWARGS)
    _bind(hw1, hw2)

    sdu = np.random.default_rng(1).integers(0, 2, size=hw1.max_segment_bits + 8_000).astype("uint8")
    frames = hw1.send_iq(sdu)
    assert len(frames) >= 2  # genuine segmentation, not a single-PDU no-op

    delivered = []
    for iq in frames:
        delivered.extend(hw2.receive_iq(iq))
    assert len(delivered) == 1
    np.testing.assert_array_equal(delivered[0], sdu)
