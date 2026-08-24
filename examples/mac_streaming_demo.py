"""Both earlier MAC demos (examples/mac_two_units_demo.py and
examples/mac_half_duplex_demo.py), rebuilt to receive through
Ofdm.rx_streaming() -- chunked, arbitrarily-aligned IQ, exactly how a
real SDR would hand samples over -- instead of Mac.receive_iq()'s
one-shot rx_process() call on an already-complete frame buffer.

No changes needed anywhere in spectracuda/mac/ to do this: Mac.send_iq()
(TX side, unchanged -- streaming is inherently a receive-side concept,
there's no equivalent complication generating a frame) plus
`<mac_obj>.ofdm.rx_streaming(chunk)` (real chunked PHY decode) plus
`<mac_obj>.receive(bits)` (the SAME PDU-level MAC decode Mac.receive_iq()
already uses internally -- reached via Mac's existing __getattr__
passthrough to its UmEntity, the same public pattern
tests/test_mac_two_units_simple.py's own regression test already
exercises) compose directly.

See docs/todo.md #2.5 / Ofdm.rx_streaming()'s own docstring for the
streaming receiver's design (checked against liquid-dsp's actual
ofdmframesync_execute() state machine before it was built).
"""
from __future__ import annotations

import numpy as np

from spectracuda.mac import Mac

PHY_KWARGS = dict(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32,
    crc="crc16", sync="schmidl_cox", cfo="schmidl_cox",
    n_training_symbols=2, backend="numpy",
)

CHUNK_SIZE = 64  # arbitrary -- rx_streaming() doesn't require any particular
                 # size or alignment; try 37 yourself to see it still works


def _bind(mac_a, mac_b):
    """The real 3-call handshake -- send_iq() has required a successful
    bind() first ever since binding was migrated onto Mac itself (see
    docs/mac.md). Binding itself still goes through rx_process() (batch),
    not rx_streaming() -- it's a one-shot control PDU, not the thing this
    file exists to demonstrate streamed decode of."""
    req_iq = mac_a.build_bind_request()
    resp_iq = mac_b.handle_bind_request_iq(req_iq)
    assert resp_iq is not None
    assert mac_a.handle_bind_response_iq(resp_iq)


def _stream_deliver(iq_frames, rx_mac, verbose, label):
    """Feed each IQ frame through rx_mac.ofdm.rx_streaming() in CHUNK_SIZE
    pieces (not as one complete buffer), collecting whatever MAC-level
    SDU(s) come out. This is the actual mechanism this file exists to
    demonstrate."""
    delivered = []
    for frame_idx, iq in enumerate(iq_frames):
        samples = iq[0]  # (1, N) -> (N,), generate_frame()'s own batch dim
        n_chunks = 0
        for i in range(0, samples.shape[-1], CHUNK_SIZE):
            chunk = samples[i : i + CHUNK_SIZE]
            n_chunks += 1
            result = rx_mac.ofdm.rx_streaming(chunk)
            if result is not None:
                decoded_bits = np.asarray(result["bits"])[0].astype("uint8")
                delivered.extend(rx_mac.receive(decoded_bits))
        if verbose:
            print(f"  {label} frame {frame_idx}: delivered via rx_streaming() "
                  f"after {n_chunks} chunks of {CHUNK_SIZE} samples each")
    return delivered


def run_two_units(seed: int = 0, verbose: bool = True):
    """HW1 transmit-only, HW2 receive-only -- examples/mac_two_units_demo.py,
    but HW2 now decodes via chunked rx_streaming() instead of one-shot
    receive_iq()."""
    if verbose:
        print("=== two_units (streaming) ===")
    hw1 = Mac(mode="um", ofdm_kwargs=PHY_KWARGS)
    hw2 = Mac(mode="um", ofdm_kwargs=PHY_KWARGS)
    _bind(hw1, hw2)
    hw2.ofdm.reset_stream()

    sdu = np.random.default_rng(seed).integers(0, 2, size=800).astype("uint8")
    iq_frames = hw1.send_iq(sdu)

    delivered = _stream_deliver(iq_frames, hw2, verbose, "hw1->hw2")
    ok = len(delivered) == 1 and np.array_equal(delivered[0], sdu)
    if verbose:
        print(f"hw1->hw2 correct (via rx_streaming): {ok}\n")
    return {"correct": ok}


def run_half_duplex(seed: int = 0, verbose: bool = True):
    """Both directions, same two Mac objects (examples/mac_half_duplex_demo.py),
    each side now decoding the other's traffic via its OWN chunked
    rx_streaming() state machine -- hw1.ofdm and hw2.ofdm each maintain
    independent streaming state, same way they maintain independent
    UmEntity SN/reassembly state."""
    if verbose:
        print("=== half_duplex (streaming) ===")
    hw1 = Mac(mode="um", ofdm_kwargs=PHY_KWARGS)
    hw2 = Mac(mode="um", ofdm_kwargs=PHY_KWARGS)
    _bind(hw1, hw2)
    hw1.ofdm.reset_stream()
    hw2.ofdm.reset_stream()

    rng = np.random.default_rng(seed)
    sdu_a = rng.integers(0, 2, size=400).astype("uint8")  # hw1 -> hw2
    sdu_b = rng.integers(0, 2, size=304).astype("uint8")  # hw2 -> hw1

    frames_a = hw1.send_iq(sdu_a)
    frames_b = hw2.send_iq(sdu_b)

    # Deliver interleaved (b before a), same independence proof as the
    # original half-duplex demo -- each side's rx_streaming() state is
    # its own, so delivery order between the two directions still
    # doesn't matter.
    delivered_b = _stream_deliver(frames_b, hw1, verbose, "hw2->hw1")
    delivered_a = _stream_deliver(frames_a, hw2, verbose, "hw1->hw2")

    ok_a = len(delivered_a) == 1 and np.array_equal(delivered_a[0], sdu_a)
    ok_b = len(delivered_b) == 1 and np.array_equal(delivered_b[0], sdu_b)
    if verbose:
        print(f"hw1->hw2 correct (via rx_streaming): {ok_a}")
        print(f"hw2->hw1 correct (via rx_streaming): {ok_b}\n")
    return {"hw1_to_hw2_correct": ok_a, "hw2_to_hw1_correct": ok_b}


if __name__ == "__main__":
    run_two_units()
    run_half_duplex()
