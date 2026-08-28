"""Measures Mac.receive_iq_batch()'s real wall-clock speedup over plain
sequential receive_iq(), across n_workers in {1,2,3,4}, for a batch of
already-arrived IQ frames -- the actual scenario receive_iq_batch is
built for (see its own docstring in mac/mac.py: parallel decode across a
pool of independent Ofdm replicas, one per worker thread, since the
native Viterbi/RS C structs aren't safe to share across concurrently
decoding threads).

Why N_SDUS separate single-PDU SDUs, not one big multi-PDU SDU: Mac's UM
segmentation has a real header-format ceiling -- SO_BITS=16 in mac/pdu.py
caps the segment-offset field at 65535, which in turn caps PDUs/SDU at
floor(65535 / max_segment_bits) + 1. For this script's standard PHY
config that works out to 3 PDUs/SDU (QPSK, ~72024-bit SDU ceiling),
2 PDUs/SDU (QAM16, ~96240-bit ceiling), or exactly 1 PDU/SDU ever
(QAM64 -- max_segment_bits alone already exceeds the ceiling, so QAM64
literally cannot segment a multi-PDU SDU under the current header
format). None of those support a single ~150000-bit SDU. Using several
independent single-PDU SDUs instead sidesteps that bug entirely, keeps
the same total bit count, and is arguably a closer match to
receive_iq_batch's real use case (a burst of independently-arrived
packets) than one artificially large SDU would be anyway.

Same bind-handshake/phy_kwargs setup pattern as
examples/benchmark_x86_stages_v3.py, just driving receive_iq_batch()
instead of the sequential receive_iq() path.
"""
import sys
import time

import numpy as np

from spectracuda.mac import Mac

FFT_SIZE, N_PILOT, N_DATA, CP_LEN = 256, 8, 216, 32
SDU_BITS = 18752           # single-PDU for both modems (< QPSK's 24008 floor), byte-aligned (multiple of 8)
N_SDUS = 8
TOTAL_BITS = SDU_BITS * N_SDUS   # 150016, ~150k
N_ROUNDS = 10
N_WARMUP = 2
WORKER_COUNTS = [1, 2, 3, 4]
_VALID_MODEMS = {"bpsk", "qpsk", "qam16", "qam64", "qam256"}


def build_pair(modem):
    phy_kwargs = dict(
        fft_size=FFT_SIZE, n_pilot=N_PILOT, n_data=N_DATA, cp_len=CP_LEN,
        modem=modem, fec="rs_m8", fec1="conv_v27", crc="crc16",
        sync="schmidl_cox", cfo="schmidl_cox",
        channel_estimator="ls", equalizer="mmse", backend="numpy",
    )
    tx_mac = Mac(mode="um", ofdm_kwargs=phy_kwargs)
    rx_mac = Mac(mode="um", ofdm_kwargs=phy_kwargs)
    req = tx_mac.build_bind_request()
    resp = rx_mac.handle_bind_request_iq(req)
    assert tx_mac.handle_bind_response_iq(resp)
    return tx_mac, rx_mac


def make_frame_batch(tx_mac, rng):
    """N_SDUS independent SDUs -> pooled list of IQ frames + the SDUs
    themselves (for bit-exact verification), in order."""
    sdus = [rng.integers(0, 2, size=SDU_BITS).astype("uint8") for _ in range(N_SDUS)]
    frames = []
    for sdu in sdus:
        frames.extend(tx_mac.send_iq(sdu))
    return frames, sdus


def run_modem(modem):
    print(f"\n{'=' * 60}\n>>> {modem}: {TOTAL_BITS} bits total = {N_SDUS} SDUs x "
          f"{SDU_BITS} bits (1 PDU each)\n{'=' * 60}")
    tx_mac, rx_mac = build_pair(modem)
    rng = np.random.default_rng(0)

    # warmup: build native codec caches + replica pool before timing
    for _ in range(N_WARMUP):
        frames, _ = make_frame_batch(tx_mac, rng)
        rx_mac.receive_iq_batch(frames, n_workers=max(WORKER_COUNTS))

    one_frame_samples = frames[0].shape[-1]
    total_samples = sum(f.shape[-1] for f in frames)
    print(f"{len(frames)} frames/round, {one_frame_samples} IQ samples/frame, "
          f"{total_samples} total samples/round")

    for n_workers in WORKER_COUNTS:
        total_time = 0.0
        n_delivered = 0
        n_bit_exact = 0
        for _ in range(N_ROUNDS):
            frames, sdus = make_frame_batch(tx_mac, rng)
            start = time.perf_counter()
            results = rx_mac.receive_iq_batch(frames, n_workers=n_workers)
            total_time += time.perf_counter() - start
            # 1 frame == 1 SDU here (each SDU is single-PDU), so results[i]
            # lines up directly with sdus[i] -- no cross-matching needed.
            for r, sdu in zip(results, sdus):
                if len(r) == 1:
                    n_delivered += 1
                    if np.array_equal(r[0], sdu):
                        n_bit_exact += 1
        avg_ms = total_time / N_ROUNDS * 1000
        mbps = TOTAL_BITS / avg_ms / 1000
        print(f"  n_workers={n_workers}: {avg_ms:8.4f} ms/round -> {mbps:6.2f} Mbps "
              f"  ({n_delivered}/{N_ROUNDS * N_SDUS} SDUs delivered, {n_bit_exact} bit-exact)")


if __name__ == "__main__":
    modems = [a.lower() for a in sys.argv[1:]] or ["qpsk", "qam16"]
    unknown = [m for m in modems if m not in _VALID_MODEMS]
    if unknown:
        raise SystemExit(f"Unrecognized modem(s) {unknown} -- expected one or more of {sorted(_VALID_MODEMS)}")
    for modem in modems:
        run_modem(modem)
