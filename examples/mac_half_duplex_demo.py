"""Bidirectional over the SAME two Mac objects (hw1, hw2), no third or
fourth Mac -- but NOT actually constrained to alternate turns the way a
real half-duplex radio is (an earlier version of this file called it
"half duplex" and structured it as direction 1 fully, then direction 2
fully -- that ordering was just how the script happened to be written,
not something the code enforced or needed).

What's actually true, and what this version demonstrates directly rather
than asserting: hw1->hw2 and hw2->hw1 are LOGICALLY INDEPENDENT. Each
Mac's UmEntity keeps its own outgoing SN counter and its own incoming
reassembly buffer as two separate state machines that never touch each
other (see test_mac_half_duplex.py::test_tx_and_rx_state_are_independent_per_object),
so nothing requires one direction to finish before the other starts --
both directions' IQ frames are generated FIRST, then delivered
INTERLEAVED (not "all of direction 1, then all of direction 2"), proving
there's no hidden ordering dependency in the code itself.

Honest caveat, not modeled here: this is baseband IQ simulation, not real
RF. A genuinely simultaneous full-duplex radio sharing ONE antenna/
frequency has a real hardware problem -- its own transmit signal is far
stronger than what it's trying to receive, so true simultaneous tx/rx on
shared spectrum needs active self-interference cancellation (a real,
hard RF engineering problem, still an active research area -- "in-band
full duplex"). hw1.ofdm and hw2.ofdm are independent objects with no
shared-spectrum interference modeled between them, so "parallel" here
means "logically independent, freely interleavable in code," not
"physically simultaneous on the same RF resource."
"""
from __future__ import annotations

import numpy as np

from spectracuda.mac import Mac

PHY_KWARGS = dict(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32,
    crc="crc16", sync="schmidl_cox", cfo="schmidl_cox",
    n_training_symbols=2, backend="numpy",
)


def _bind(mac_a, mac_b):
    """The real 3-call handshake -- send_iq() has required a successful
    bind() first ever since binding was migrated onto Mac itself (see
    docs/mac.md)."""
    req_iq = mac_a.build_bind_request()
    resp_iq = mac_b.handle_bind_request_iq(req_iq)
    assert resp_iq is not None
    assert mac_a.handle_bind_response_iq(resp_iq)


def run(seed: int = 0, verbose: bool = True):
    hw1 = Mac(mode="um", ofdm_kwargs=PHY_KWARGS)
    hw2 = Mac(mode="um", ofdm_kwargs=PHY_KWARGS)
    _bind(hw1, hw2)

    if verbose:
        print(f"hw1.ofdm is hw2.ofdm: {hw1.ofdm is hw2.ofdm} (must be False)")
        print("Only 2 Mac objects exist -- both directions will reuse them.")

    rng = np.random.default_rng(seed)
    sdu_a = rng.integers(0, 2, size=400).astype("uint8")  # hw1 -> hw2
    sdu_b = rng.integers(0, 2, size=304).astype("uint8")  # hw2 -> hw1, different content

    # Both directions' IQ generated FIRST, neither waiting on the other --
    # this is the part that would be impossible to write this way if one
    # direction genuinely depended on the other having gone first.
    if verbose:
        print(f"\nGenerating hw1->hw2 IQ ({sdu_a.shape[0]} bits) "
              f"and hw2->hw1 IQ ({sdu_b.shape[0]} bits) up front, before either is delivered...")
    frames_a = hw1.send_iq(sdu_a)  # hw1 -> hw2
    frames_b = hw2.send_iq(sdu_b)  # hw2 -> hw1
    if verbose:
        print(f"  hw1->hw2: {len(frames_a)} frame(s)   hw2->hw1: {len(frames_b)} frame(s)")

    # Now deliver them INTERLEAVED (b before a, reversed from generation
    # order) -- not "all of a, then all of b" -- to make the independence
    # visible in the delivery order itself, not just in how they were built.
    delivered_a, delivered_b = [], []
    max_len = max(len(frames_a), len(frames_b))
    for i in range(max_len):
        if i < len(frames_b):
            if verbose:
                print(f"delivering hw2->hw1 frame {i}...")
            delivered_b.extend(hw1.receive_iq(frames_b[i]))
        if i < len(frames_a):
            if verbose:
                print(f"delivering hw1->hw2 frame {i}...")
            delivered_a.extend(hw2.receive_iq(frames_a[i]))

    ok_a = len(delivered_a) == 1 and np.array_equal(delivered_a[0], sdu_a)
    ok_b = len(delivered_b) == 1 and np.array_equal(delivered_b[0], sdu_b)

    if verbose:
        print(f"\nsdu_a (hw1->hw2, tx): {sdu_a}")
        print(f"sdu_a (hw1->hw2, rx): {delivered_a[0] if delivered_a else 'nothing delivered'}")
        print(f"hw1->hw2 correct despite interleaved, reversed-order delivery: {ok_a}")
        print(f"\nsdu_b (hw2->hw1, tx): {sdu_b}")
        print(f"sdu_b (hw2->hw1, rx): {delivered_b[0] if delivered_b else 'nothing delivered'}")
        print(f"hw2->hw1 correct despite interleaved, reversed-order delivery: {ok_b}")
        print(f"\nBoth directions correct, delivered interleaved, same 2 objects: {ok_a and ok_b}")

    return {
        "independent_objects": hw1.ofdm is not hw2.ofdm,
        "hw1_to_hw2_correct": ok_a,
        "hw2_to_hw1_correct": ok_b,
    }


if __name__ == "__main__":
    run()
