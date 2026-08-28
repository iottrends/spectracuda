"""Mac.receive_iq_batch(): the pooled-Ofdm-replicas, real-multi-core-
decode counterpart to calling receive_iq() once per arrived frame in a
loop (see mac/mac.py's own docstring for why it needs a POOL of Ofdm
replicas, not n threads sharing one Mac's self.ofdm, to be safe at all).

The one thing this test suite exists to prove: receive_iq_batch(),
whatever n_workers it's given, returns EXACTLY what the same frames fed
through receive_iq() one at a time, in order, would have returned --
bit-exact, not "close" -- for every worker count from 1 (the plain
sequential fallback) up through more workers than there are frames.
"""
import numpy as np
import pytest

from spectracuda.mac import Mac

_PHY_KWARGS = dict(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32,
    modem="qpsk", fec="rs_m8", fec1="conv_v27", crc="crc16",
    sync="schmidl_cox", cfo="schmidl_cox",
    channel_estimator="ls", equalizer="mmse",
    backend="numpy",
)


def _bind(mac_a, mac_b):
    req_iq = mac_a.build_bind_request()
    resp_iq = mac_b.handle_bind_request_iq(req_iq)
    assert resp_iq is not None
    assert mac_a.handle_bind_response_iq(resp_iq)
    assert mac_a.bound and mac_b.bound


@pytest.mark.parametrize("sdu_bits,n_workers", [
    (800, 1),      # 1 PDU, n_workers=1 -- plain sequential fallback path
    (800, 2),      # 1 PDU, n_workers=2 -- more workers than frames
    (48000, 1),    # multiple PDUs (this config's capacity is ~24008 raw
    (48000, 2),    #   bits/PDU, see the NEON-kernel benchmarking session
    (48000, 4),    #   this test suite's sibling investigation used),
    (60000, 2),    #   swept across worker counts including > n_pdus
    (60000, 3),
])
def test_receive_iq_batch_matches_sequential_receive_iq(sdu_bits, n_workers):
    tx_mac = Mac(mode="um", ofdm_kwargs=_PHY_KWARGS)
    rx_mac_sequential = Mac(mode="um", ofdm_kwargs=_PHY_KWARGS)
    rx_mac_batch = Mac(mode="um", ofdm_kwargs=_PHY_KWARGS)
    _bind(tx_mac, rx_mac_sequential)
    _bind(tx_mac, rx_mac_batch)

    sdu = np.random.default_rng(0).integers(0, 2, size=sdu_bits).astype("uint8")
    iq_frames = tx_mac.send_iq(sdu)

    sequential_delivered = []
    for iq in iq_frames:
        sequential_delivered.extend(rx_mac_sequential.receive_iq(iq))

    batch_delivered = []
    for result in rx_mac_batch.receive_iq_batch(iq_frames, n_workers=n_workers):
        batch_delivered.extend(result)

    assert len(sequential_delivered) == len(batch_delivered) == 1
    np.testing.assert_array_equal(batch_delivered[0], sdu)
    np.testing.assert_array_equal(batch_delivered[0], sequential_delivered[0])


def test_receive_iq_batch_empty_input():
    rx_mac = Mac(mode="um", ofdm_kwargs=_PHY_KWARGS)
    assert rx_mac.receive_iq_batch([], n_workers=2) == []


def test_receive_iq_batch_uses_independent_ofdm_replicas_not_shared_state():
    """The actual correctness-under-concurrency claim, not just "outputs
    match": each worker must get its OWN Ofdm instance (never self.ofdm
    itself, never two workers sharing one) -- see mac.py's own docstring
    for exactly why sharing would race the native FEC codecs' internal
    buffers."""
    rx_mac = Mac(mode="um", ofdm_kwargs=_PHY_KWARGS)
    replicas = rx_mac._ofdm_replica_pool(3)
    assert len(replicas) == 3
    assert len(set(id(r) for r in replicas)) == 3  # 3 genuinely distinct objects
    assert rx_mac.ofdm not in replicas
    # A smaller later request reuses the same cached prefix, not fresh objects.
    assert rx_mac._ofdm_replica_pool(2) == replicas[:2]
