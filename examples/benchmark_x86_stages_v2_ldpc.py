"""LDPC counterpart of benchmark_x86_stages_v2.py: same real Mac
send_iq()/receive_iq() data flow (build_bind_request()/handle_bind_*(),
then a real SDU through the full Mac+Ofdm stack -- not the Ofdm-direct,
no-Mac approach benchmark_x86_stages_ldpc.py/_ldpc_v2.py use, see those
scripts' own module docstrings for why THEY had to work one level down
at the time: LDPC's "exact k_bits multiple only" requirement used to
make even the bind handshake fail. That gap is now closed (LDPC has its
own shortened-codeword support, mirroring rs_m8's -- see fec/ldpc.py's
encode()/decode() docstrings and docs/todo.md), which is what makes
THIS script possible at all), same stopwatch-timed stage-breakdown
technique, same stage-bucket naming convention (`fec_encode[scheme]`/
`fec_decode[scheme]`), same headline throughput format -- literally the
same script with the FEC scheme swapped and the two real, load-bearing
differences that come with it:

1. **No libcorrect-equivalent native backend for LDPC.** v2's whole
   point was patching in libcorrect (a hand-tuned C library) for BOTH
   ConvolutionalCode and ReedSolomonCode, in both directions. LDPC has
   no such external backend integrated into this codebase today -- this
   script times spectracuda's own native numpy min-sum belief-
   propagation decode (fec/ldpc.py) as-is, un-accelerated, same as
   benchmark_x86_stages_ldpc.py already does. (For a real, measured
   alternative-backend comparison, see benchmark_x86_stages_ldpc_aff3ct.py,
   which benchmarks AFF3CT -- a different, standalone C++ tool, not
   something pluggable into this class-patching pattern the way
   libcorrect is.) CRC still gets the same Numba-JIT'd
   `crc_numba_backend.generate_key()` v2 uses -- that acceleration is
   orthogonal to which FEC scheme is chosen.

2. **One FEC stage bucket, not two.** v2's `rs_m8` (fec0/inner) +
   `conv_v27` (fec1/outer) pairing needed two rows in both the TX and RX
   breakdowns. LDPC is used standalone here (`fec1="none"`, matching
   every other LDPC benchmark script in this project) -- so there's a
   single `fec_encode[ldpc_*]`/`fec_decode[ldpc_*]` row instead.

Usage:
    python examples/benchmark_x86_stages_v2_ldpc.py
    python examples/benchmark_x86_stages_v2_ldpc.py 24000
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict

import numpy as np

from spectracuda.fec.crc import CRC
from spectracuda.fec.fec import FEC
from spectracuda.mac import Mac

sys.path.insert(0, os.path.dirname(__file__))
from crc_numba_backend import numba_generate_key  # noqa: E402

FFT_SIZE = 256
N_PILOT = 8
N_DATA = 216
CP_LEN = 32
LDPC_VARIANT = "ldpc_1944_r12"  # same headline variant used throughout this project's other LDPC benchmarks
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


def _install_ldpc_patch(timings: dict):
    """Same idea as v2's _install_libcorrect_patch(): FEC.encode/
    FEC.decode are timed here (bucketed by self.scheme), not
    LDPCCode.encode/decode directly, so the fec.py wrapper's own
    shortened-block chunking overhead (see fec/fec.py's
    _encode_block_level/_decode_block_level) is counted too, not
    hidden -- same reasoning as v2's own comment on this point. No
    ConvolutionalCode/ReedSolomonCode patching here (see module
    docstring, difference 1) -- CRC still gets the same Numba backend
    v2 uses, since that's independent of which FEC scheme is active."""
    orig = dict(
        fec_encode=FEC.encode, fec_decode=FEC.decode,
        crc_append=CRC.append_key, crc_generate=CRC.generate_key,
    )
    CRC.generate_key = numba_generate_key  # covers append_key (TX) AND check_key (RX) -- both call this

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
        modem="qpsk", fec=LDPC_VARIANT, fec1="none", crc="crc16",
        sync="schmidl_cox", cfo="schmidl_cox",
        channel_estimator="ls", equalizer="mmse",
        backend="numpy",
    )
    print(f"=== v2_ldpc (real Mac send_iq()/receive_iq(), native numpy LDPC decode -- see "
          f"module docstring for why there's no libcorrect-equivalent backend here) config: "
          f"fft_size={FFT_SIZE}, n_pilot={N_PILOT}, n_data={N_DATA}, cp_len={CP_LEN}, "
          f"modem=qpsk, fec={LDPC_VARIANT!r} (inner, standalone), fec1='none', crc=crc16, "
          f"sync=schmidl_cox, cfo=schmidl_cox, channel_estimator=ls, equalizer=mmse, "
          f"backend=numpy, sdu_bits={SDU_BITS} ===")

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
    restore_tx_patch = _install_ldpc_patch(tx_timings)
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
    # Same per-round-vs-per-frame fix v2 needed -- see its own module
    # docstring for the real bug this closes (TX looking artificially
    # worse than RX for any SDU spanning >1 PDU).
    tx_time = tx_time_per_round / n_pdus_per_round
    print(f"\nfull TX chain (send_iq(), {n_pdus_per_round} PDU(s)/SDU): "
          f"{tx_time_per_round * 1000:.4f} ms/SDU ({tx_time * 1000:.4f} ms/frame)")
    print("TX stage breakdown:")
    accounted = 0.0
    for bucket, label in [
        ("crc_encode", "CRC key generation"),
        (f"fec_encode[{LDPC_VARIANT}]", "FEC encode -- LDPC (native numpy)"),
    ]:
        ms = tx_timings[bucket] / N_ROUNDS / n_pdus_per_round * 1000
        accounted += ms
        print(f"  {label:>50}: {ms:.4f} ms")
    print(f"  {'everything else (modem/OFDM/resource-grid/header)':>50}: {tx_time * 1000 - accounted:.4f} ms")

    # -- RX stages --
    timings: dict = defaultdict(float)
    _instrument(rx_mac, timings)
    restore_rx_patch = _install_ldpc_patch(timings)
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
        (f"fec_decode[{LDPC_VARIANT}]", "FEC decode -- LDPC (native numpy)"),
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
