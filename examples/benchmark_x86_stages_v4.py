"""v4 of benchmark_x86_stages_v3.py: LDPC instead of rs_m8+conv_v27.

Same real question v3 answers for the two-stage rs_m8(inner)+conv_v27
(outer) FEC this project's drone-link-demo scope requires -- how close
is one frame's TX/RX to the 20 Msps (1.8848ms/frame) budget -- asked
here for LDPC instead, since LDPC's min-sum belief propagation is
structurally GPU-parallel in a way Viterbi/RS are not (see
fec/ldpc.py's own module docstring: "this is the first FEC codec in
this codebase whose iterative decode core is genuinely GPU-parallel
across the batch"). This script is the x86/CPU baseline half of that
question -- see examples/benchmark_ldpc_cuda.py for the GPU half
(meant to run on a CUDA machine this project's own dev box doesn't
have, e.g. Google Colab).

Real, load-bearing difference from v3, found by hitting it directly
(not assumed): Mac + LDPC does not work AT ALL today, not even for
Mac's own bind handshake. LDPC (unlike rs_m8) has no "shortened block"
support -- it requires payload bit counts to be an EXACT multiple of
its own k_bits, a real, already-documented gap (docs/todo.md, fec/
fec.py's encoded_length() docstring: "ldpc_* is NOT included in this --
still requires an exact multiple, a documented, separate gap"). Mac's
own bind-request PDU is a small, FIXED 104-bit message that was never
going to land on an exact multiple of any of the 12 LDPC block sizes,
so `Mac(mode="um", ofdm_kwargs=dict(fec="ldpc_...")).build_bind_request()`
raises ValueError immediately, before any data ever gets sent. So this
script works at the `Ofdm` level directly (generate_frame()/
rx_process()), the same level examples/benchmark_stages_numpy_vs_cupy.py
and tests/test_ofdm_combination_matrix.py already successfully exercise
LDPC through -- no Mac, no bind, no segmentation/reassembly, and
correspondingly no "MAC decode" stage bucket in the breakdown below.

The requested SDU bit count is PADDED UP to the smallest multiple of
the chosen LDPC variant's k_bits that's also byte-aligned (CRC's own
byte-boundary requirement, same as v3/v2's underlying Packetizer) --
reported explicitly (requested vs actual), never silently changed
without saying so.

Same stopwatch-timed stage-breakdown technique as v3's own revision
(see that script's module docstring for why cProfile was replaced): a
plain time.perf_counter() wrapper around spectracuda's real methods,
never a swapped-in implementation, reconciled against a pristine
uninstrumented headline pass rather than trusted as a total on its
own. Same CPU-core-pinning noise reduction as v3
(SPECTRACUDA_BENCH_PIN_CORE).

Usage:
    python examples/benchmark_x86_stages_v4.py
    python examples/benchmark_x86_stages_v4.py 32000
    python examples/benchmark_x86_stages_v4.py qpsk 32000 1/2
    python examples/benchmark_x86_stages_v4.py 32000 qam16 3/4      # order doesn't matter
    python examples/benchmark_x86_stages_v4.py qpsk 32000 1/2 1296  # + override codeword length
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict

import numpy as np

from spectracuda.cfo.schmidl_cox import SchmidlCoxCFO
from spectracuda.channel.ls import LSChannelEstimator
from spectracuda.equalizer.mmse import MMSEEqualizer
from spectracuda.fec.crc import CRC
from spectracuda.fec.ldpc import LDPCCode
from spectracuda.fec.ldpc_tables import BASE_MATRICES
from spectracuda.ofdm.fft import OfdmDemodulator
from spectracuda.pipeline import Ofdm
from spectracuda.sync.schmidl_cox import SchmidlCoxSync

FFT_SIZE = 256
N_PILOT = 8
N_DATA = 216
CP_LEN = 32
N_ROUNDS = 30
N_WARMUP = 5
DEFAULT_PIN_CORE = 2
DEFAULT_CODEWORD_LEN = 1944  # the largest/most-efficient of the 3 IEEE 802.11n lengths
_VALID_MODEMS = {"bpsk", "qpsk", "qam16", "qam64", "qam256"}
_RATE_SUFFIX = {"1/2": "r12", "2/3": "r23", "3/4": "r34", "5/6": "r56"}
_VALID_CODEWORD_LENS = {648, 1296, 1944}


def _parse_args():
    """Classified by shape, same technique v3 uses, extended for a
    third kind of arg (a rate string): digits matching a valid
    codeword length (648/1296/1944) -> codeword length; other digits
    -> SDU bit count; "n/m" -> LDPC rate; else -> modem scheme. Order
    never matters."""
    sdu_bits = 10000  # a safe default that fits one frame across every modem/rate combo below
    modem_scheme = "qpsk"
    rate = "1/2"
    codeword_len = DEFAULT_CODEWORD_LEN
    for arg in sys.argv[1:]:
        if arg in _RATE_SUFFIX:
            rate = arg
        elif arg.isdigit() and int(arg) in _VALID_CODEWORD_LENS:
            codeword_len = int(arg)
        elif arg.isdigit():
            sdu_bits = int(arg)
        elif arg.lower() in _VALID_MODEMS:
            modem_scheme = arg.lower()
        else:
            raise SystemExit(
                f"Unrecognized argument {arg!r} -- expected a bit count (e.g. 32000), "
                f"a modem scheme ({sorted(_VALID_MODEMS)}), an LDPC rate "
                f"({sorted(_RATE_SUFFIX)}), or a codeword length ({sorted(_VALID_CODEWORD_LENS)})"
            )
    variant = f"ldpc_{codeword_len}_r{_RATE_SUFFIX[rate][1:]}"
    return sdu_bits, modem_scheme, rate, variant


SDU_BITS, MODEM_SCHEME, LDPC_RATE, LDPC_VARIANT = _parse_args()


def _pin_to_one_core() -> str:
    """See benchmark_x86_stages_v3.py's own _pin_to_one_core() -- same
    function, same rationale, duplicated rather than imported since
    these are standalone example scripts, not library code."""
    override = os.environ.get("SPECTRACUDA_BENCH_PIN_CORE")
    if override is not None and override.strip().lower() == "off":
        return "disabled (SPECTRACUDA_BENCH_PIN_CORE=off)"
    if not hasattr(os, "sched_setaffinity"):
        return "unavailable (no os.sched_setaffinity on this platform)"
    try:
        core = int(override) if override is not None else DEFAULT_PIN_CORE
        available = os.sched_getaffinity(0)
        if core not in available:
            return f"skipped (core {core} not in this process's available set {sorted(available)})"
        os.sched_setaffinity(0, {core})
        return f"pinned to core {core}"
    except Exception as exc:  # pragma: no cover -- best-effort, never fatal to the benchmark itself
        return f"failed ({exc})"


def _timed(fn, bucket: str, timings: dict):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        timings[bucket] += time.perf_counter() - start
        return result

    return wrapper


def _install_timing_patch(timings: dict):
    """Same real-methods-only stopwatch-wrapping technique as v3's own
    _install_timing_patch() -- see that script's module docstring for
    the full rationale (cProfile was found to disproportionately
    inflate stages built from many small nested calls). No UmEntity
    entry here -- there is no Mac/UmEntity involved at this Ofdm-direct
    level (see this module's own docstring for why)."""
    orig = dict(
        ldpc_encode=LDPCCode.encode, ldpc_decode=LDPCCode.decode,
        crc_generate=CRC.generate_key,
        sync_process=SchmidlCoxSync.process,
        cfo_process=SchmidlCoxCFO.process, cfo_correct=SchmidlCoxCFO.correct,
        ofdm_demod_process=OfdmDemodulator.process,
        ls_process=LSChannelEstimator.process,
        mmse_process=MMSEEqualizer.process,
    )
    LDPCCode.encode = _timed(orig["ldpc_encode"], "ldpc_encode", timings)
    LDPCCode.decode = _timed(orig["ldpc_decode"], "ldpc_decode", timings)
    CRC.generate_key = _timed(orig["crc_generate"], "crc_generate", timings)
    SchmidlCoxSync.process = _timed(orig["sync_process"], "sync_cfo", timings)
    SchmidlCoxCFO.process = _timed(orig["cfo_process"], "sync_cfo", timings)
    SchmidlCoxCFO.correct = _timed(orig["cfo_correct"], "sync_cfo", timings)
    OfdmDemodulator.process = _timed(orig["ofdm_demod_process"], "ofdm_decode", timings)
    LSChannelEstimator.process = _timed(orig["ls_process"], "chanest_eq", timings)
    MMSEEqualizer.process = _timed(orig["mmse_process"], "chanest_eq", timings)

    def restore():
        LDPCCode.encode = orig["ldpc_encode"]
        LDPCCode.decode = orig["ldpc_decode"]
        CRC.generate_key = orig["crc_generate"]
        SchmidlCoxSync.process = orig["sync_process"]
        SchmidlCoxCFO.process = orig["cfo_process"]
        SchmidlCoxCFO.correct = orig["cfo_correct"]
        OfdmDemodulator.process = orig["ofdm_demod_process"]
        LSChannelEstimator.process = orig["ls_process"]
        MMSEEqualizer.process = orig["mmse_process"]

    return restore


def _actual_payload_bits(requested_bits: int, k_bits: int, crc_bits: int) -> int:
    """Smallest payload bit count >= requested_bits such that
    (payload_bits + crc_bits) is an exact multiple of k_bits AND
    payload_bits itself is byte-aligned (CRC's own requirement) -- see
    module docstring. A plain incrementing search, not a closed form:
    cheap (runs once), and correct regardless of k_bits's own
    byte-alignment residue rather than assuming one."""
    n_blocks = max(1, -(-(requested_bits + crc_bits) // k_bits))  # ceil div
    while True:
        payload_bits = n_blocks * k_bits - crc_bits
        if payload_bits >= requested_bits and payload_bits % 8 == 0:
            return payload_bits
        n_blocks += 1


def run() -> None:
    pin_status = _pin_to_one_core()
    phy_kwargs = dict(
        fft_size=FFT_SIZE, n_pilot=N_PILOT, n_data=N_DATA, cp_len=CP_LEN,
        modem=MODEM_SCHEME, fec=LDPC_VARIANT, fec1="none", crc="crc16",
        sync="schmidl_cox", cfo="schmidl_cox",
        channel_estimator="ls", equalizer="mmse",
        backend="numpy",
    )
    print(f"=== v4 (LDPC instead of rs_m8+conv_v27 -- stopwatch-timed stage breakdown, "
          f"Ofdm-direct, no Mac -- see module docstring) config: "
          f"fft_size={FFT_SIZE}, n_pilot={N_PILOT}, n_data={N_DATA}, cp_len={CP_LEN}, "
          f"modem={MODEM_SCHEME}, fec={LDPC_VARIANT!r} (rate {LDPC_RATE}), crc=crc16, "
          f"sync=schmidl_cox, cfo=schmidl_cox, channel_estimator=ls, equalizer=mmse, "
          f"backend=numpy, sdu_bits={SDU_BITS} (requested) ===")
    print(f"    CPU affinity (benchmark-only noise reduction, see v3's module docstring): {pin_status}")

    ofdm = Ofdm(**phy_kwargs)
    k_bits = ofdm.packetizer.fec_codec.k_bits
    n_bits = ofdm.packetizer.fec_codec.n_bits
    crc_bits = ofdm.packetizer.crc_codec.key_length * 8
    payload_bits = _actual_payload_bits(SDU_BITS, k_bits, crc_bits)
    n_ldpc_blocks = (payload_bits + crc_bits) // k_bits

    # There's no Mac layer here to split a too-big SDU across multiple
    # frames (see module docstring) -- everything has to fit in ONE
    # frame's MAX_PAYLOAD_SYMBOLS. Clamp down (not just pad up) rather
    # than let generate_frame() raise a less specific error -- reported
    # explicitly either way, never a silent change.
    max_encoded_bits = ofdm.MAX_PAYLOAD_SYMBOLS * ofdm.bits_per_ofdm_symbol
    max_blocks = max_encoded_bits // n_bits
    if n_ldpc_blocks > max_blocks:
        max_payload_bits = _actual_payload_bits(0, k_bits, crc_bits) if max_blocks == 0 else max_blocks * k_bits - crc_bits
        print(f"    NOTE: {payload_bits} requested payload bits ({n_ldpc_blocks} LDPC blocks) needs "
              f"more OFDM symbols than this frame's MAX_PAYLOAD_SYMBOLS={ofdm.MAX_PAYLOAD_SYMBOLS} "
              f"allows at modem={MODEM_SCHEME} -- there's no Mac layer here to split it across "
              f"multiple frames (see module docstring), so this run is CLAMPED DOWN to the largest "
              f"payload that fits in one frame: {max_payload_bits} bits ({max_blocks} blocks).")
        payload_bits = max_payload_bits
        n_ldpc_blocks = max_blocks
    elif payload_bits != SDU_BITS:
        print(f"    NOTE: {LDPC_VARIANT} has no \"shortened block\" support (unlike rs_m8) -- "
              f"padded requested {SDU_BITS} bits up to {payload_bits} bits "
              f"({n_ldpc_blocks}x {k_bits}-bit LDPC blocks) to land on an exact multiple, "
              f"as this scheme requires.")

    rng = np.random.default_rng(0)
    payload = rng.integers(0, 2, size=payload_bits).astype("uint8")

    one_frame_iq = ofdm.generate_frame(payload[None, :])
    total_samples = one_frame_iq.shape[-1]
    samples_per_symbol = ofdm.fft_size + ofdm.cp_len
    other_symbols = (total_samples - ofdm.fft_size) / samples_per_symbol
    payload_symbols = other_symbols - ofdm.n_training_symbols - ofdm.num_symbols_header
    print(f"\nPayload: {payload_bits} bits -> {n_ldpc_blocks} LDPC codeword(s) of "
          f"{k_bits}->{n_bits} bits each")
    print(f"One frame: {total_samples} IQ samples = "
          f"1 preamble symbol ({ofdm.fft_size} samples, no CP) + "
          f"{ofdm.n_training_symbols} training + {ofdm.num_symbols_header} header + "
          f"{payload_symbols:.0f} payload OFDM symbols ({samples_per_symbol} samples/symbol each)")

    # -- full TX chain: plain-timed, nothing patched --
    for _ in range(N_WARMUP):
        ofdm.generate_frame(payload[None, :])
    start = time.perf_counter()
    for _ in range(N_ROUNDS):
        frame = ofdm.generate_frame(payload[None, :])
    tx_time = (time.perf_counter() - start) / N_ROUNDS
    print(f"\nfull TX chain (generate_frame(), 1 frame): {tx_time * 1000:.4f} ms/frame")

    # -- TX stage breakdown: a SEPARATE stopwatch-instrumented pass --
    tx_timings: dict = defaultdict(float)
    restore_tx = _install_timing_patch(tx_timings)
    try:
        for _ in range(N_ROUNDS):
            ofdm.generate_frame(payload[None, :])
    finally:
        restore_tx()
    crc_ms = tx_timings["crc_generate"] / N_ROUNDS * 1000
    ldpc_enc_ms = tx_timings["ldpc_encode"] / N_ROUNDS * 1000
    print("TX stage breakdown (stopwatch pass -- proportions, see v3's module docstring):")
    accounted = 0.0
    for label, ms in [
        ("CRC key generation", crc_ms),
        ("FEC encode -- LDPC", ldpc_enc_ms),
    ]:
        accounted += ms
        print(f"  {label:>50}: {ms:.4f} ms")
    print(f"  {'everything else (modem/OFDM/resource-grid/header)':>50}: {tx_time * 1000 - accounted:.4f} ms")

    # -- RX: plain-timed headline pass + decode check --
    for _ in range(N_WARMUP):
        ofdm.rx_process(frame)
    start = time.perf_counter()
    n_crc_valid = 0
    n_bit_exact = 0
    for _ in range(N_ROUNDS):
        result = ofdm.rx_process(frame)
        n_crc_valid += int(bool(np.asarray(result["crc_valid"])[0]))
        n_bit_exact += int(np.array_equal(np.asarray(result["bits"])[0], payload))
    rx_total = (time.perf_counter() - start) / N_ROUNDS * 1000  # ms/frame, headline

    print(f"\ndecode check: {n_crc_valid}/{N_ROUNDS} rounds crc_valid, "
          f"{n_bit_exact}/{N_ROUNDS} of those bit-exact matches of the original "
          f"({'all correct' if n_bit_exact == N_ROUNDS else 'SOME ROUNDS FAILED -- see below'})")

    # -- RX stage breakdown: a SEPARATE stopwatch-instrumented pass --
    rx_timings: dict = defaultdict(float)
    restore_rx = _install_timing_patch(rx_timings)
    try:
        for _ in range(N_ROUNDS):
            ofdm.rx_process(frame)
    finally:
        restore_rx()

    print(f"\nRX stage breakdown (stopwatch pass -- proportions, averaged over {N_ROUNDS} frames, "
          f"see v3's module docstring):")
    buckets = [
        ("sync detect + CFO", rx_timings["sync_cfo"]),
        ("OFDM decode (FFT+CP strip)", rx_timings["ofdm_decode"]),
        ("channel estimation + equalization", rx_timings["chanest_eq"]),
        ("FEC decode -- LDPC", rx_timings["ldpc_decode"]),
    ]
    rx_accounted = 0.0
    for label, total_s in buckets:
        ms = total_s / N_ROUNDS * 1000
        rx_accounted += ms
        print(f"  {label:>50}: {ms:.4f} ms")
    print(f"  {'everything else (unbucketed -- header/resource-grid/etc.)':>50}: "
          f"{max(rx_total - rx_accounted, 0.0):.4f} ms")

    # -- headline --
    tx_mbps = payload_bits / (tx_time * 1000) / 1000
    rx_mbps = payload_bits / rx_total / 1000
    tx_msps = total_samples / (tx_time * 1000) / 1000
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
