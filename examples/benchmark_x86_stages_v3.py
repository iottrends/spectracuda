"""v3 of benchmark_x86_stages_v2.py: NO monkey-patching, anywhere.

v1 patched in a standalone Numba prototype (before Numba/native accel was
promoted into the library). v2 patched in a SEPARATE, hand-rolled
libcorrect binding (examples/libcorrect_backend.py) instead of using
spectracuda's own, already-fixed native path (spectracuda/fec/_native.py's
NativeConvolutional/NativeReedSolomon) -- and it turned out that separate
binding doesn't even carry the _DECODE_PAD_PAIRS decode-truncation fix
_native.py has. v2's own "decode check: all correct" only held by
coincidence (rs_m8's byte-granular output always lands conv_v27 on the one
T%8 residue that's safe even unpatched, see that investigation) -- not
because it was exercising the real fix.

That fix lives INSIDE spectracuda now, and activation is, by design,
"FULLY AUTOMATIC and TRANSPARENT, not a new constructor argument or config
flag" (fec/_native.py's own module docstring) -- same story for CRC's
Numba path (fec/crc.py's generate_key() calls numba_generate_key()
directly if available). So this version needs zero patches to exercise
either: Mac(mode=..., ofdm_kwargs=...) is built completely normally, and
whichever backend is actually active on this machine (native C, Numba,
or the pure-Python fallback if neither compiled) is exactly what
send_iq()/receive_iq() use, with nothing swapped in from outside.

Per-stage breakdown, without patching, comes from Python's own
cProfile/pstats reading the REAL call graph instead of wrapping methods:
ConvolutionalCode.decode and ReedSolomonCode.decode are simply different
functions in different files, so their own cumulative time is directly
attributable with no self.scheme bookkeeping needed (unlike v1/v2's
FEC.decode-level patch, which had to dispatch on self.scheme by hand to
tell fec0/fec1 apart). Cumulative time correctly includes whatever that
function calls internally -- a pure-Python loop, or a ctypes call into
native C -- cProfile times wall-clock between a function's own call/
return regardless of what runs underneath it.

The headline TX/RX timing numbers are still measured on a plain,
UNPROFILED pass (cProfile's own per-call overhead is real and would
otherwise inflate them) -- the profiled pass, run separately over the
same real inputs, is used only for the stage-breakdown proportions.

Usage:
    python examples/benchmark_x86_stages_v3.py
"""
from __future__ import annotations

import cProfile
import pstats
import sys
import time

import numpy as np

from spectracuda.fec import _native, _numba_crc
from spectracuda.mac import Mac

FFT_SIZE = 256
N_PILOT = 8
N_DATA = 216
CP_LEN = 32
SDU_BITS = int(sys.argv[1]) if len(sys.argv) > 1 else 24000  # pass a bit count as argv[1] to override
N_ROUNDS = 30
N_WARMUP = 5


def _cumulative_time(stats: pstats.Stats, funcname: str, filename_suffix: str) -> float:
    """Sum of cumulative time (ct), across every call recorded in this one
    profiling pass, for the function named `funcname` defined in a file
    ending in `filename_suffix` -- e.g. ("decode", "viterbi.py") finds
    ConvolutionalCode.decode's own total time, wherever it actually did
    the work (a pure-Python loop, or a call down into the native/Numba
    backend -- either way it's real elapsed time under this function's
    own call/return, cProfile doesn't care what's underneath). Matching
    by (funcname, filename suffix) rather than by object identity is what
    lets this run with ZERO patching: nothing needs to be swapped in to
    observe which real function spent the time."""
    total = 0.0
    for (filename, _lineno, fname), (_cc, _nc, _tt, ct, _callers) in stats.stats.items():
        if fname == funcname and filename.endswith(filename_suffix):
            total += ct
    return total


def run() -> None:
    phy_kwargs = dict(
        fft_size=FFT_SIZE, n_pilot=N_PILOT, n_data=N_DATA, cp_len=CP_LEN,
        modem="qpsk", fec="rs_m8", fec1="conv_v27", crc="crc16",
        sync="schmidl_cox", cfo="schmidl_cox",
        channel_estimator="ls", equalizer="mmse",
        backend="numpy",
    )
    print(f"=== v3 (no monkey-patches -- spectracuda's own transparent native/Numba "
          f"acceleration, whatever's actually active on THIS machine) config: "
          f"fft_size={FFT_SIZE}, n_pilot={N_PILOT}, n_data={N_DATA}, cp_len={CP_LEN}, "
          f"modem=qpsk, fec='rs_m8' (inner), fec1='conv_v27' (outer), crc=crc16, "
          f"sync=schmidl_cox, cfo=schmidl_cox, channel_estimator=ls, equalizer=mmse, "
          f"backend=numpy, sdu_bits={SDU_BITS} ===")
    print(f"    native FEC backend (Viterbi/RS, C, transparent, fec/_native.py): "
          f"{'ACTIVE' if _native.native_available() else 'inactive -- pure-Python fallback'}")
    print(f"    Numba CRC backend (transparent, fec/_numba_crc.py): "
          f"{'ACTIVE' if _numba_crc.numba_available() else 'inactive -- pure-Python fallback'}")

    tx_probe_mac = Mac(mode="um", ofdm_kwargs=phy_kwargs)
    tx_probe_mac.bound = True  # never talks to a real peer -- see v1's own module docstring
    tx_mac = Mac(mode="um", ofdm_kwargs=phy_kwargs)
    rx_mac = Mac(mode="um", ofdm_kwargs=phy_kwargs)

    req = tx_mac.build_bind_request()
    resp = rx_mac.handle_bind_request_iq(req)
    assert tx_mac.handle_bind_response_iq(resp)

    rng = np.random.default_rng(0)
    sdu = rng.integers(0, 2, size=SDU_BITS).astype("uint8")

    ofdm = tx_probe_mac.ofdm
    pdus = tx_probe_mac._impl.transmit(sdu)
    samples_per_symbol = ofdm.fft_size + ofdm.cp_len
    one_frame_iq = ofdm.generate_frame(np.asarray(pdus[0], dtype="uint8")[None, :])
    total_samples = one_frame_iq.shape[-1]
    other_symbols = (total_samples - ofdm.fft_size) / samples_per_symbol
    payload_symbols = other_symbols - ofdm.n_training_symbols - ofdm.num_symbols_header
    print(f"\nSDU: {SDU_BITS} bits -> {len(pdus)} PDU(s), {len(pdus[0])} bits/PDU "
          f"(includes MAC header + FEC/CRC overhead)")
    print(f"One frame: {total_samples} IQ samples = "
          f"1 preamble symbol ({ofdm.fft_size} samples, no CP) + "
          f"{ofdm.n_training_symbols} training + {ofdm.num_symbols_header} header + "
          f"{payload_symbols:.0f} payload OFDM symbols ({samples_per_symbol} samples/symbol each)")

    # -- full TX chain: plain-timed, no profiler active -- this IS the
    # headline number, so it must not carry any profiling overhead --
    for _ in range(N_WARMUP):
        tx_probe_mac.send_iq(sdu)
    start = time.perf_counter()
    for _ in range(N_ROUNDS):
        probe_iq_frames = tx_probe_mac.send_iq(sdu)
    tx_time_per_round = (time.perf_counter() - start) / N_ROUNDS
    n_pdus_per_round = len(probe_iq_frames)
    # Real bug found (and fixed the same way) in v2: tx_time_per_round
    # covers ALL PDUs/frames send_iq() produced for this SDU -- for an
    # SDU spanning >1 PDU (e.g. 30000 bits -> 2 PDUs), dividing the
    # per-frame throughput math by tx_time_per_round directly understates
    # TX Msps by ~n_pdus_per_round x. Must divide by n_pdus_per_round too
    # to get genuine per-FRAME time, matching total_samples (one frame's
    # sample count) below.
    tx_time = tx_time_per_round / n_pdus_per_round
    print(f"\nfull TX chain (send_iq(), {n_pdus_per_round} PDU(s)/SDU): "
          f"{tx_time_per_round * 1000:.4f} ms/SDU ({tx_time * 1000:.4f} ms/frame)")

    # -- TX stage breakdown: a SEPARATE profiled pass over the exact same
    # real send_iq() call -- proportions only, not the headline number
    # (see module docstring) --
    tx_profiler = cProfile.Profile()
    tx_profiler.enable()
    for _ in range(N_ROUNDS):
        tx_probe_mac.send_iq(sdu)
    tx_profiler.disable()
    tx_stats = pstats.Stats(tx_profiler)
    # /N_ROUNDS/n_pdus_per_round, not just /N_ROUNDS -- same per-round-vs-
    # per-frame fix as tx_time above (these functions run once per PDU).
    crc_ms = _cumulative_time(tx_stats, "generate_key", "crc.py") / N_ROUNDS / n_pdus_per_round * 1000
    conv_enc_ms = _cumulative_time(tx_stats, "encode", "viterbi.py") / N_ROUNDS / n_pdus_per_round * 1000
    rs_enc_ms = _cumulative_time(tx_stats, "encode", "reed_solomon.py") / N_ROUNDS / n_pdus_per_round * 1000
    print("TX stage breakdown (profiled pass -- proportions, see module docstring):")
    accounted = 0.0
    for label, ms in [
        ("CRC key generation", crc_ms),
        ("FEC encode -- Viterbi (fec1, outer)", conv_enc_ms),
        ("FEC encode -- Reed-Solomon (fec0, inner)", rs_enc_ms),
    ]:
        accounted += ms
        print(f"  {label:>50}: {ms:.4f} ms")
    print(f"  {'everything else (modem/OFDM/resource-grid/header)':>50}: {tx_time * 1000 - accounted:.4f} ms")

    # -- RX: plain-timed headline pass + decode check. A FRESH batch of
    # frames per pass (tx_mac's SN counter keeps advancing, matching one
    # real receiver's lifetime) -- reusing one fixed batch across passes
    # would make the second pass see only duplicate-SN fast-rejects,
    # not genuine reassembly/delivery work (see v1's own module
    # docstring for why that matters here). --
    for _ in range(N_WARMUP):
        for iq in tx_mac.send_iq(sdu):
            rx_mac.receive_iq(iq)

    headline_rounds = [tx_mac.send_iq(sdu) for _ in range(N_ROUNDS)]
    n_delivered = 0
    n_bit_exact = 0
    n_calls = 0
    start = time.perf_counter()
    for iq_frames in headline_rounds:
        for iq in iq_frames:
            delivered = rx_mac.receive_iq(iq)
            n_delivered += len(delivered)
            n_bit_exact += sum(np.array_equal(d, sdu) for d in delivered)
            n_calls += 1
    rx_total = (time.perf_counter() - start) / n_calls * 1000  # ms/frame, headline

    print(f"\ndecode check: {n_delivered}/{N_ROUNDS} rounds delivered a SDU, "
          f"{n_bit_exact}/{n_delivered} of those bit-exact matches of the original "
          f"({'all correct' if n_bit_exact == N_ROUNDS else 'SOME ROUNDS FAILED -- see below'})")
    if n_bit_exact != N_ROUNDS:
        print("  NOTE: a round not delivering doesn't necessarily mean a decode error -- see v1's")
        print("  module docstring for the ReassemblyBuffer window-eviction explanation.")

    # -- RX stage breakdown: a SEPARATE profiled pass, same real call,
    # fresh frames again -- proportions only, see module docstring --
    profiled_rounds = [tx_mac.send_iq(sdu) for _ in range(N_ROUNDS)]
    rx_profiler = cProfile.Profile()
    rx_profiler.enable()
    n_calls_profiled = 0
    for iq_frames in profiled_rounds:
        for iq in iq_frames:
            rx_mac.receive_iq(iq)
            n_calls_profiled += 1
    rx_profiler.disable()
    rx_stats = pstats.Stats(rx_profiler)

    print(f"\nRX stage breakdown (profiled pass -- proportions, averaged over "
          f"{n_calls_profiled} frames, see module docstring):")
    buckets = [
        ("sync detect + CFO", _cumulative_time(rx_stats, "process", "schmidl_cox.py")
         + _cumulative_time(rx_stats, "correct", "schmidl_cox.py")),
        ("OFDM decode (FFT+CP strip)", _cumulative_time(rx_stats, "process", "fft.py")),
        ("channel estimation + equalization", _cumulative_time(rx_stats, "process", "ls.py")
         + _cumulative_time(rx_stats, "process", "mmse.py")),
        ("FEC decode -- Viterbi (fec1, outer)", _cumulative_time(rx_stats, "decode", "viterbi.py")),
        ("FEC decode -- Reed-Solomon (fec0, inner)", _cumulative_time(rx_stats, "decode", "reed_solomon.py")),
        ("MAC decode", _cumulative_time(rx_stats, "receive", "um.py")),
    ]
    rx_accounted = 0.0
    for label, total_s in buckets:
        ms = total_s / n_calls_profiled * 1000
        rx_accounted += ms
        print(f"  {label:>50}: {ms:.4f} ms")
    print(f"  {'everything else (profiler overhead + unbucketed)':>50}: "
          f"{max(rx_total - rx_accounted, 0.0):.4f} ms")

    # -- headline: what this actually means for sustained throughput,
    # using the PLAIN-TIMED numbers (tx_time/rx_total), never the
    # profiled pass --
    pdu_bits = len(pdus[0])
    tx_mbps = pdu_bits / (tx_time * 1000) / 1000  # bits / ms -> Mbps
    rx_mbps = pdu_bits / rx_total / 1000
    tx_msps = total_samples / (tx_time * 1000) / 1000  # samples / ms -> Msps
    rx_msps = total_samples / rx_total / 1000
    budget_ms = total_samples / 20e6 * 1000  # ms/frame budget for the real target: 20 Msps sample rate
    print(f"\n=== Throughput (single-threaded, one frame at a time) ===")
    print(f"  TX: {tx_time * 1000:.4f} ms/frame -> ~{tx_mbps:.2f} Mbps -> ~{tx_msps:.2f} Msps "
          f"({'OK' if tx_time*1000 <= budget_ms else f'{budget_ms/(tx_time*1000):.2f}x short'} of 20 Msps, budget {budget_ms:.4f} ms)")
    print(f"  RX: {rx_total:.4f} ms/frame -> ~{rx_mbps:.2f} Mbps -> ~{rx_msps:.2f} Msps "
          f"({'OK' if rx_total <= budget_ms else f'{budget_ms/rx_total:.2f}x short'} of 20 Msps, budget {budget_ms:.4f} ms)")
    print(f"  (bottleneck: {'TX' if tx_time*1000 > rx_total else 'RX'})")


if __name__ == "__main__":
    run()
