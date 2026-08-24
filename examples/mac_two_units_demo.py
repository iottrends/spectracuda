"""Two genuinely independent HW units -- HW1 transmit-only, HW2 receive-
only -- each owning its own `Ofdm` (spectracuda/pipeline/ofdm.py), built
via `Mac(mode=..., ofdm_kwargs=...)` (spectracuda/mac/mac.py). No
`MacLink`, no shared `Ofdm` object, no orchestrator class -- HW1 and HW2
never share any object identity; the only thing crossing between them is
the actual IQ data, exactly like two real radios. See docs/mac.md,
"Two-node redesign, step 1" for the full design discussion, and
tests/test_mac_two_units_simple.py for the same scenario as pytest cases.

UM mode only (genuinely one-directional -- AM needs a return status path
back to the sender; see examples/mac_bidirectional_am_batch_demo.py /
mac_bidirectional_am_streaming_demo.py for the full 4-Mac/4-Ofdm picture
that supports).
"""
from __future__ import annotations

import numpy as np

from spectracuda.mac import Mac
from spectracuda.sim import Channel

PHY_KWARGS = dict(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32,
    crc="crc16", sync="schmidl_cox", cfo="schmidl_cox",
    n_training_symbols=2, backend="numpy",
)


def _bind(mac_a, mac_b, verbose=False):
    """The real 3-call handshake -- send_iq() has required a successful
    bind() first ever since binding was migrated onto Mac itself (see
    docs/mac.md)."""
    req_iq = mac_a.build_bind_request()
    resp_iq = mac_b.handle_bind_request_iq(req_iq)
    assert resp_iq is not None
    assert mac_a.handle_bind_response_iq(resp_iq)
    if verbose:
        print(f"bind: both .bound == True (mac_a={mac_a.bound}, mac_b={mac_b.bound})")


def run(seed: int = 0, snr_db: float = 25.0, sdu_bits: int = 800, verbose: bool = True):
    hw1_tx_mac = Mac(mode="um", ofdm_kwargs=PHY_KWARGS)
    hw2_rx_mac = Mac(mode="um", ofdm_kwargs=PHY_KWARGS)
    _bind(hw1_tx_mac, hw2_rx_mac)

    if verbose:
        print(f"hw1_tx_mac.ofdm is hw2_rx_mac.ofdm: {hw1_tx_mac.ofdm is hw2_rx_mac.ofdm}"
              " (must be False -- genuinely independent objects)")
        print(f"derived max_segment_bits (from HW1's Ofdm capacity): {hw1_tx_mac.max_segment_bits}")

    sdu = np.random.default_rng(seed).integers(0, 2, size=sdu_bits).astype("uint8")
    channel = Channel(snr_db=snr_db, seed=seed, backend="numpy")

    if verbose:
        print(f"\nsdu ({sdu.shape[0]} bits):\n{sdu}")
        print(f"\nHW1 transmitting {sdu_bits} bits...")
    iq_frames = hw1_tx_mac.send_iq(sdu)
    if verbose:
        print(f"  -> segmented into {len(iq_frames)} PDU(s), "
              f"each an IQ frame of shape {iq_frames[0].shape}")

    delivered = []
    for i, iq in enumerate(iq_frames):
        if verbose:
            print(f"\ntx iq, frame {i}, BEFORE channel ({iq.shape}):\n{iq}")
        rx_iq = channel.process(iq)  # the only thing crossing between HW1 and HW2
        if verbose:
            print(f"\nrx iq, frame {i}, AFTER channel ({rx_iq.shape}):\n{rx_iq}")
            print(f"\nHW2 receiving frame {i}...")
        out = hw2_rx_mac.receive_iq(rx_iq)
        if verbose:
            print(f"  -> decoded {len(out)} SDU(s) from this frame")
        delivered.extend(out)

    ok = len(delivered) == 1 and np.array_equal(delivered[0], sdu)
    if verbose:
        print(f"\nsdu (tx, {sdu.shape[0]} bits):\n{sdu}")
        if delivered:
            print(f"\nsdu (rx, decoded, {delivered[0].shape[0]} bits):\n{delivered[0]}")
        else:
            print("\nsdu (rx): nothing delivered")
        print(f"\nRound trip correct: {ok}")
    return {
        "independent_objects": hw1_tx_mac.ofdm is not hw2_rx_mac.ofdm,
        "num_pdus": len(iq_frames),
        "num_delivered": len(delivered),
        "correct": ok,
    }


if __name__ == "__main__":
    run()
