"""v2 of benchmark_x86_stages.py: BOTH tx and rx routed through
libcorrect (examples/libcorrect_backend.py -- a hand-tuned, SIMD-capable
C library, ctypes-bound, verified true-interop with spectracuda's own
ConvolutionalCode/ReedSolomonCode in examples/prototype_libcorrect.py),
not just rx decode via Numba (that was v1's scope -- see git history).

Why this exists: v1 showed rx decode was NOT the dominant cost once
Numba-JIT'd -- tx encode was, by nearly 10x, because only decode ever
got optimized there. Profiling tx directly (cProfile on Mac.send_iq())
found ConvolutionalCode.encode() and ReedSolomonCode.encode() (both
still pure Python loops, never touched) accounting for ~90% of tx time.
This script fixes BOTH directions with the SAME backend, so tx and rx
are optimized to a comparable degree for the first time.

Same rigor as v1 and the prototypes it built on: correctness is proven
(the "decode check" section) before any timing number is trusted, using
the SAME real Mac/Ofdm send_iq()/receive_iq() data flow throughout, not
synthetic per-stage inputs.

Patches ConvolutionalCode.encode/decode and ReedSolomonCode.encode/
decode at the CLASS level (see libcorrect_backend.py for exactly what
each replaces and what was verified about it) -- this covers every
instance including the throwaway per-call Packetizer rx_process()
rebuilds internally (see v1's own module docstring for why that matters
here too).

Usage:
    python examples/benchmark_x86_stages_v2.py
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict

import numpy as np

from spectracuda.fec.crc import CRC
from spectracuda.fec.fec import FEC
from spectracuda.fec.reed_solomon import ReedSolomonCode
from spectracuda.fec.viterbi import ConvolutionalCode
from spectracuda.mac import Mac

sys.path.insert(0, os.path.dirname(__file__))
from libcorrect_backend import LibcorrectConvolutional, LibcorrectReedSolomon  # noqa: E402
from crc_numba_backend import numba_generate_key  # noqa: E402

FFT_SIZE = 256
N_PILOT = 8
N_DATA = 216
CP_LEN = 32
SDU_BITS = int(sys.argv[1]) if len(sys.argv) > 1 else 24000  # pass a bit count as argv[1] to override
N_ROUNDS = 30
N_WARMUP = 5


def _timed(fn, bucket: str, timings: dict):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        timings[bucket] += time.perf_counter() - start
        return result

    return wrapper


def _install_libcorrect_patch(timings: dict):
    """Replaces ConvolutionalCode/ReedSolomonCode encode+decode at the
    class level with libcorrect-backed implementations (both directions,
    unlike v1 which only replaced decode). FEC.encode/FEC.decode are
    ALSO timed here (bucketed by self.scheme) -- they're thin dispatchers
    (see fec/fec.py) but timing them, not the underlying class methods
    directly, means bit-packing/symbol-splitting overhead for rs_m8's
    multi-block chunking is counted too, not hidden."""
    orig = dict(
        conv_encode=ConvolutionalCode.encode, conv_decode=ConvolutionalCode.decode,
        rs_encode=ReedSolomonCode.encode, rs_decode=ReedSolomonCode.decode,
        fec_encode=FEC.encode, fec_decode=FEC.decode,
        crc_append=CRC.append_key, crc_generate=CRC.generate_key,
    )
    CRC.generate_key = numba_generate_key  # covers append_key (TX) AND check_key (RX) -- both call this

    lc_conv = LibcorrectConvolutional()
    lc_rs = LibcorrectReedSolomon()

    ConvolutionalCode.encode = lambda self, bits: lc_conv.encode(bits)
    ConvolutionalCode.decode = lambda self, bits: lc_conv.decode(bits)
    ReedSolomonCode.encode = lambda self, msg: lc_rs.encode(msg)
    ReedSolomonCode.decode = lambda self, codeword: lc_rs.decode(codeword)

    def patched_fec_encode(self, *args, **kwargs):
        start = time.perf_counter()
        result = orig["fec_encode"](self, *args, **kwargs)
        timings[f"fec_encode[{self.scheme}]"] += time.perf_counter() - start
        return result

    def patched_fec_decode(self, *args, **kwargs):
        start = time.perf_counter()
        result = orig["fec_decode"](self, *args, **kwargs)
        timings[f"fec_decode[{self.scheme}]"] += time.perf_counter() - start
        return result

    FEC.encode = patched_fec_encode
    FEC.decode = patched_fec_decode
    CRC.append_key = _timed(CRC.append_key, "crc_encode", timings)

    def restore():
        ConvolutionalCode.encode = orig["conv_encode"]
        ConvolutionalCode.decode = orig["conv_decode"]
        ReedSolomonCode.encode = orig["rs_encode"]
        ReedSolomonCode.decode = orig["rs_decode"]
        FEC.encode = orig["fec_encode"]
        FEC.decode = orig["fec_decode"]
        CRC.append_key = orig["crc_append"]
        CRC.generate_key = orig["crc_generate"]

    return restore


def _instrument(mac: Mac, timings: dict) -> None:
    ofdm = mac.ofdm
    ofdm.sync.process = _timed(ofdm.sync.process, "sync+cfo", timings)
    ofdm.cfo.process = _timed(ofdm.cfo.process, "sync+cfo", timings)
    ofdm.cfo.correct = _timed(ofdm.cfo.correct, "sync+cfo", timings)
    ofdm.demod.process = _timed(ofdm.demod.process, "ofdm_decode", timings)
    ofdm.channel_estimator.process = _timed(ofdm.channel_estimator.process, "chanest_eq", timings)
    ofdm.equalizer.process = _timed(ofdm.equalizer.process, "chanest_eq", timings)
    mac._impl.receive = _timed(mac._impl.receive, "mac_decode", timings)


def run() -> None:
    phy_kwargs = dict(
        fft_size=FFT_SIZE, n_pilot=N_PILOT, n_data=N_DATA, cp_len=CP_LEN,
        modem="qpsk", fec="rs_m8", fec1="conv_v27", crc="crc16",
        sync="schmidl_cox", cfo="schmidl_cox",
        channel_estimator="ls", equalizer="mmse",
        backend="numpy",
    )
    print(f"=== v2 (libcorrect: both tx encode + rx decode) config: fft_size={FFT_SIZE}, "
          f"n_pilot={N_PILOT}, n_data={N_DATA}, cp_len={CP_LEN}, modem=qpsk, fec='rs_m8' "
          f"(inner), fec1='conv_v27' (outer), crc=crc16, sync=schmidl_cox, cfo=schmidl_cox, "
          f"channel_estimator=ls, equalizer=mmse, backend=numpy, sdu_bits={SDU_BITS} ===")

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

    # -- full TX chain + per-stage TX breakdown --
    tx_timings: dict = defaultdict(float)
    restore_tx_patch = _install_libcorrect_patch(tx_timings)
    try:
        for _ in range(N_WARMUP):
            tx_probe_mac.send_iq(sdu)
        tx_timings.clear()
        start = time.perf_counter()
        for _ in range(N_ROUNDS):
            probe_iq_frames = tx_probe_mac.send_iq(sdu)
        tx_time_per_round = (time.perf_counter() - start) / N_ROUNDS
    finally:
        restore_tx_patch()
    n_pdus_per_round = len(probe_iq_frames)
    # tx_time_per_round covers ALL PDUs/frames send_iq() produced for this
    # SDU (2+ for an SDU that spans multiple PDUs) -- per-FRAME time (what
    # the throughput section below needs, to match one frame's sample
    # count) must divide by n_pdus_per_round too. A real bug found here:
    # an earlier version used tx_time_per_round directly as "per frame,"
    # which understated TX Msps by ~n_pdus_per_round x for any SDU size
    # spanning >1 PDU (e.g. 30000/32000 bits -> 2 PDUs) -- caught because
    # it made TX look WORSE than RX, backwards from every single-PDU run
    # all session, which is what prompted actually checking the math
    # instead of trusting the printed number.
    tx_time = tx_time_per_round / n_pdus_per_round
    print(f"\nfull TX chain (send_iq(), {n_pdus_per_round} PDU(s)/SDU): "
          f"{tx_time_per_round * 1000:.4f} ms/SDU ({tx_time * 1000:.4f} ms/frame)")
    print("TX stage breakdown:")
    accounted = 0.0
    for bucket, label in [
        ("crc_encode", "CRC key generation"),
        ("fec_encode[conv_v27]", "FEC encode -- Viterbi (fec1, outer, libcorrect)"),
        ("fec_encode[rs_m8]", "FEC encode -- Reed-Solomon (fec0, inner, libcorrect)"),
    ]:
        # /N_ROUNDS/n_pdus_per_round, not just /N_ROUNDS -- these timings
        # accumulate once per PDU (CRC/FEC encode run once per frame),
        # same per-round-vs-per-frame fix as tx_time above.
        ms = tx_timings[bucket] / N_ROUNDS / n_pdus_per_round * 1000
        accounted += ms
        print(f"  {label:>50}: {ms:.4f} ms")
    print(f"  {'everything else (modem/OFDM/resource-grid/header)':>50}: {tx_time * 1000 - accounted:.4f} ms")

    # -- RX stages --
    timings: dict = defaultdict(float)
    _instrument(rx_mac, timings)
    restore_rx_patch = _install_libcorrect_patch(timings)
    n_delivered = 0
    n_bit_exact = 0
    try:
        for _ in range(N_WARMUP):
            for iq in tx_mac.send_iq(sdu):
                rx_mac.receive_iq(iq)
        timings.clear()

        n_calls = 0
        for _ in range(N_ROUNDS):
            for iq in tx_mac.send_iq(sdu):
                delivered = rx_mac.receive_iq(iq)
                n_delivered += len(delivered)
                n_bit_exact += sum(np.array_equal(d, sdu) for d in delivered)
                n_calls += 1
    finally:
        restore_rx_patch()

    print(f"\ndecode check: {n_delivered}/{N_ROUNDS} rounds delivered a SDU, "
          f"{n_bit_exact}/{n_delivered} of those bit-exact matches of the original "
          f"({'all correct' if n_bit_exact == N_ROUNDS else 'SOME ROUNDS FAILED -- see below'})")
    if n_bit_exact != N_ROUNDS:
        print("  NOTE: a round not delivering doesn't necessarily mean a decode error -- see v1's")
        print("  module docstring for the ReassemblyBuffer window-eviction explanation.")

    print(f"\nRX stages, per frame decoded (averaged over {n_calls} frames):")
    rx_total = 0.0
    for bucket, label in [
        ("sync+cfo", "sync detect + CFO"),
        ("ofdm_decode", "OFDM decode (FFT+CP strip)"),
        ("chanest_eq", "channel estimation + equalization"),
        ("fec_decode[conv_v27]", "FEC decode -- Viterbi (fec1, outer, libcorrect)"),
        ("fec_decode[rs_m8]", "FEC decode -- Reed-Solomon (fec0, inner, libcorrect)"),
        ("mac_decode", "MAC decode"),
    ]:
        ms = timings[bucket] / n_calls * 1000
        rx_total += ms
        print(f"  {label:>50}: {ms:.4f} ms")

    # -- headline: what this actually means for sustained throughput --
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
