"""Half-duplex over the SAME two Mac objects, no third/fourth Mac --
see examples/mac_half_duplex_demo.py for the narrated version and
docs/mac.md for the design discussion (this only works symmetrically:
both directions share each unit's one Ofdm, a real, physically-motivated
constraint, not a limitation of this test).
"""
import numpy as np

from spectracuda.mac import Mac

_PHY_KWARGS = dict(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32,
    crc="crc16", sync="schmidl_cox", cfo="schmidl_cox",
    n_training_symbols=2, backend="numpy",
)


def _bind(mac_a, mac_b):
    """The real 3-call handshake (docs/mac.md) -- ONE handshake mutually
    binds both objects (handle_bind_request_iq() sets the acceptor's
    .bound, handle_bind_response_iq() sets the requester's), which is
    why every test below only calls this once even though both
    directions later call send_iq()."""
    req_iq = mac_a.build_bind_request()
    resp_iq = mac_b.handle_bind_request_iq(req_iq)
    assert resp_iq is not None
    assert mac_a.handle_bind_response_iq(resp_iq)
    assert mac_a.bound and mac_b.bound


def test_both_directions_work_with_only_two_mac_objects():
    hw1 = Mac(mode="um", ofdm_kwargs=_PHY_KWARGS)
    hw2 = Mac(mode="um", ofdm_kwargs=_PHY_KWARGS)
    assert hw1.ofdm is not hw2.ofdm  # cross-hardware sharing still eliminated
    _bind(hw1, hw2)

    rng = np.random.default_rng(0)
    sdu_a = rng.integers(0, 2, size=400).astype("uint8")
    sdu_b = rng.integers(0, 2, size=304).astype("uint8")  # different content, different direction

    delivered_a = []
    for iq in hw1.send_iq(sdu_a):
        delivered_a.extend(hw2.receive_iq(iq))
    assert len(delivered_a) == 1
    np.testing.assert_array_equal(delivered_a[0], sdu_a)

    delivered_b = []
    for iq in hw2.send_iq(sdu_b):
        delivered_b.extend(hw1.receive_iq(iq))
    assert len(delivered_b) == 1
    np.testing.assert_array_equal(delivered_b[0], sdu_b)


def test_tx_and_rx_state_are_independent_per_object():
    """The mechanism this whole feature relies on: each Mac's own
    outgoing SN counter only advances when IT transmits, and its own
    reassembly point only advances when IT receives -- the two state
    machines inside one UmEntity never interact, which is exactly why
    one object can safely play both roles."""
    hw1 = Mac(mode="um", ofdm_kwargs=_PHY_KWARGS)
    hw2 = Mac(mode="um", ofdm_kwargs=_PHY_KWARGS)
    _bind(hw1, hw2)

    assert hw1._impl._next_tx_sn == 0
    assert hw1._impl.expected_sn == 0

    sdu = np.random.default_rng(1).integers(0, 2, size=200).astype("uint8")
    for iq in hw1.send_iq(sdu):
        hw2.receive_iq(iq)

    # hw1 TRANSMITTED once -> its own outgoing counter advanced...
    assert hw1._impl._next_tx_sn == 1
    # ...but hw1 has RECEIVED nothing -- its own reassembly point must
    # be untouched by its own transmit activity.
    assert hw1._impl.expected_sn == 0

    # symmetrically on hw2: it only RECEIVED, never transmitted.
    assert hw2._impl.expected_sn == 1
    assert hw2._impl._next_tx_sn == 0


def test_reply_uses_independent_sn_space_from_the_original_direction():
    """hw2's reply doesn't need to know or care what SN hw1 used for its
    own outgoing traffic -- each direction has its own SN sequence,
    starting at 0 independently, the same way two real radios' forward
    and reverse links are independent sequence spaces."""
    hw1 = Mac(mode="um", ofdm_kwargs=_PHY_KWARGS)
    hw2 = Mac(mode="um", ofdm_kwargs=_PHY_KWARGS)
    _bind(hw1, hw2)
    rng = np.random.default_rng(2)

    # hw1 sends several SDUs first, advancing ITS OWN tx SN well past 0
    for _ in range(5):
        sdu = rng.integers(0, 2, size=200).astype("uint8")
        for iq in hw1.send_iq(sdu):
            hw2.receive_iq(iq)
    assert hw1._impl._next_tx_sn == 5

    # hw2's reply must still work correctly, starting from ITS OWN
    # fresh SN=0 -- unaffected by hw1 having already sent 5 PDUs.
    reply = rng.integers(0, 2, size=200).astype("uint8")
    delivered = []
    for iq in hw2.send_iq(reply):
        delivered.extend(hw1.receive_iq(iq))
    assert len(delivered) == 1
    np.testing.assert_array_equal(delivered[0], reply)
