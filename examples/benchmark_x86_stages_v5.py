"""v5 of benchmark_x86_stages_v4.py: LDPC decode via the validated CUDA
kernel (examples/prototype_ldpc_cuda_kernel.py), on a real GPU, through
the FULL Ofdm pipeline -- not just the isolated FEC-stage Mbps numbers
prototype_ldpc_cuda_kernel.py's own benchmark_speed() already measured.

Real question this answers, that isolated FEC timing can't: does a
GPU-accelerated LDPC decode actually get the WHOLE RX chain (sync+CFO+
FFT+channel-est+equalizer+LDPC decode) under the 20 Msps (1.8848ms/
frame) budget -- not just the FEC stage in isolation. sync/CFO/channel-
est/equalizer still have to run somewhere too; if they're NOT also
GPU-resident and fast, a blazing-fast LDPC decode just uncovers a new
bottleneck sitting right next to it. Known caveat, not fixed here,
documented up front rather than silently assumed away: examples/
benchmark_stages_numpy_vs_cupy.py's own module docstring already flags
that CFO estimation (schmidl_cox) does a per-batch-item Python loop
with a host round-trip per item on EITHER backend -- backend="cupy"
does not automatically make every stage fast.

Same real gap as v4, unrelated to CPU vs GPU: Mac + LDPC still doesn't
work at all (no "shortened block" support, breaks even Mac's own bind
handshake) -- this runs at the Ofdm level directly, same as v4, same
padding/clamping-to-one-frame logic, word for word.

Kernel selection: SPECTRACUDA_LDPC_KERNEL=global (default) or =shared.
prototype_ldpc_cuda_kernel.py's own real Colab results found a genuine
crossover: the shared-memory kernel is BETTER at large batches (2x the
global-memory kernel's throughput by batch=2048, still climbing) but
WORSE at small batches (~8-32 codewords, which is what one real frame
actually produces -- see v4's own measurement) -- so "global" is the
safer default for this single-frame-at-a-time benchmark; pass
SPECTRACUDA_LDPC_KERNEL=shared to compare directly.

GPU timing correctness, a real trap CPU-only v3/v4 never had to handle:
cupy calls are ASYNCHRONOUS -- the CPU-side call returns as soon as the
GPU work is QUEUED, not once it's actually finished. Every plain-timed
pass below calls cupy.cuda.Stream.null.synchronize() before stopping
the clock (same protection examples/benchmark_stages_numpy_vs_cupy.py
and examples/benchmark_ldpc_cuda.py already established) -- skipping
this would silently undercount GPU time and report a fake speedup. The
stage-breakdown pass's _timed() wrapper ALSO syncs after every single
wrapped call (unlike v3/v4's CPU version, which needs no such thing) --
this makes the breakdown pass noticeably slower than the real,
unsynced headline pass, which is fine: proportions are reconciled
against the pristine headline total exactly as v3/v4 already do, so
that extra sync overhead lands in "everything else", not in a wrong
per-stage number.

Usage (on a real CUDA machine, e.g. Colab with a GPU runtime):
    python examples/benchmark_x86_stages_v5.py
    python examples/benchmark_x86_stages_v5.py qpsk 32000 1/2
    SPECTRACUDA_LDPC_KERNEL=shared python examples/benchmark_x86_stages_v5.py 32000
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict

import numpy as np

from spectracuda.backend import cupy_available
from spectracuda.cfo.schmidl_cox import SchmidlCoxCFO
from spectracuda.channel.ls import LSChannelEstimator
from spectracuda.equalizer.mmse import MMSEEqualizer
from spectracuda.fec.crc import CRC
from spectracuda.fec.ldpc import LDPCCode
from spectracuda.ofdm.fft import OfdmDemodulator
from spectracuda.pipeline import Ofdm
from spectracuda.sync.schmidl_cox import SchmidlCoxSync

sys.path.insert(0, os.path.dirname(__file__))
from prototype_ldpc_cuda_kernel import cuda_ldpc_decode, cuda_ldpc_decode_shared  # noqa: E402

FFT_SIZE = 256
N_PILOT = 8
N_DATA = 216
CP_LEN = 32
N_ROUNDS = 30
N_WARMUP = 5
DEFAULT_PIN_CORE = 2
DEFAULT_CODEWORD_LEN = 1944
_VALID_MODEMS = {"bpsk", "qpsk", "qam16", "qam64", "qam256"}
_RATE_SUFFIX = {"1/2": "r12", "2/3": "r23", "3/4": "r34", "5/6": "r56"}
_VALID_CODEWORD_LENS = {648, 1296, 1944}


def _parse_args():
    """Same arg-shape classification as v3/v4 -- see v4's own
    docstring/_parse_args() for the full rationale, unchanged here."""
    sdu_bits = 10000
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
KERNEL_CHOICE = os.environ.get("SPECTRACUDA_LDPC_KERNEL", "global").strip().lower()
if KERNEL_CHOICE not in ("global", "shared"):
    raise SystemExit(f"SPECTRACUDA_LDPC_KERNEL must be 'global' or 'shared', got {KERNEL_CHOICE!r}")
_KERNEL_FN = cuda_ldpc_decode if KERNEL_CHOICE == "global" else cuda_ldpc_decode_shared


def _pin_to_one_core() -> str:
    """See v3's own _pin_to_one_core() -- same function, duplicated
    rather than imported since these are standalone example scripts."""
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
    except Exception as exc:  # pragma: no cover
        return f"failed ({exc})"


def _sync() -> None:
    import cupy

    cupy.cuda.Stream.null.synchronize()


def _timed(fn, bucket: str, timings: dict):
    """Same wrapping technique as v3/v4's own _timed(), PLUS a sync
    after every call -- see module docstring for why that's required
    here and wasn't for the CPU-only versions."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        _sync()
        timings[bucket] += time.perf_counter() - start
        return result

    return wrapper


def _install_timing_patch(timings: dict):
    """Same real-methods-only stopwatch-wrapping technique as v3/v4's
    own _install_timing_patch(). Wraps whatever LDPCCode.decode
    CURRENTLY is -- which by the time this runs is already the CUDA
    kernel (patched once, permanently, in run() below, before this is
    ever called) -- so the "FEC decode -- LDPC" bucket measures the
    real kernel's own cost, not the array-op path."""
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
    """See v4's own _actual_payload_bits() -- identical logic."""
    n_blocks = max(1, -(-(requested_bits + crc_bits) // k_bits))
    while True:
        payload_bits = n_blocks * k_bits - crc_bits
        if payload_bits >= requested_bits and payload_bits % 8 == 0:
            return payload_bits
        n_blocks += 1


def run() -> None:
    if not cupy_available():
        print("No working CuPy/CUDA runtime detected on this machine -- nothing to benchmark. "
              "Run this on a GPU machine (e.g. Colab with a GPU runtime) instead -- see this "
              "script's own module docstring for the exact setup steps.")
        return

    pin_status = _pin_to_one_core()
    phy_kwargs = dict(
        fft_size=FFT_SIZE, n_pilot=N_PILOT, n_data=N_DATA, cp_len=CP_LEN,
        modem=MODEM_SCHEME, fec=LDPC_VARIANT, fec1="none", crc="crc16",
        sync="schmidl_cox", cfo="schmidl_cox",
        channel_estimator="ls", equalizer="mmse",
        backend="cupy",
    )
    print(f"=== v5 (LDPC via the validated CUDA kernel, backend=cupy, Ofdm-direct, no Mac -- "
          f"see module docstring) config: fft_size={FFT_SIZE}, n_pilot={N_PILOT}, n_data={N_DATA}, "
          f"cp_len={CP_LEN}, modem={MODEM_SCHEME}, fec={LDPC_VARIANT!r} (rate {LDPC_RATE}), "
          f"crc=crc16, sync=schmidl_cox, cfo=schmidl_cox, channel_estimator=ls, equalizer=mmse, "
          f"backend=cupy, sdu_bits={SDU_BITS} (requested), ldpc_kernel={KERNEL_CHOICE!r} ===")
    print(f"    CPU affinity (benchmark-only noise reduction, see v3's module docstring): {pin_status}")

    # Patch LDPCCode.decode to the validated CUDA kernel -- ONCE, for the
    # whole run (never restored -- this script exits when done, unlike
    # v3/v4's stage-breakdown patches which are scoped/restored because
    # they wrap a pass the SAME process later needs the original for).
    LDPCCode.decode = _KERNEL_FN

    ofdm = Ofdm(**phy_kwargs)
    k_bits = ofdm.packetizer.fec_codec.k_bits
    n_bits = ofdm.packetizer.fec_codec.n_bits
    crc_bits = ofdm.packetizer.crc_codec.key_length * 8
    payload_bits = _actual_payload_bits(SDU_BITS, k_bits, crc_bits)
    n_ldpc_blocks = (payload_bits + crc_bits) // k_bits

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

    # -- full TX chain: plain-timed, nothing patched -- SYNCED before
    # stopping the clock (see module docstring: cupy is async) --
    for _ in range(N_WARMUP):
        ofdm.generate_frame(payload[None, :])
    _sync()
    start = time.perf_counter()
    for _ in range(N_ROUNDS):
        frame = ofdm.generate_frame(payload[None, :])
    _sync()
    tx_time = (time.perf_counter() - start) / N_ROUNDS
    print(f"\nfull TX chain (generate_frame(), 1 frame): {tx_time * 1000:.4f} ms/frame")

    # -- TX stage breakdown --
    tx_timings: dict = defaultdict(float)
    restore_tx = _install_timing_patch(tx_timings)
    try:
        for _ in range(N_ROUNDS):
            ofdm.generate_frame(payload[None, :])
    finally:
        restore_tx()
    crc_ms = tx_timings["crc_generate"] / N_ROUNDS * 1000
    ldpc_enc_ms = tx_timings["ldpc_encode"] / N_ROUNDS * 1000
    print("TX stage breakdown (stopwatch pass -- proportions, see module docstring re: sync overhead):")
    accounted = 0.0
    for label, ms in [
        ("CRC key generation", crc_ms),
        ("FEC encode -- LDPC", ldpc_enc_ms),
    ]:
        accounted += ms
        print(f"  {label:>50}: {ms:.4f} ms")
    print(f"  {'everything else (modem/OFDM/resource-grid/header)':>50}: {tx_time * 1000 - accounted:.4f} ms")

    # -- RX: plain-timed headline pass + decode check -- SYNCED --
    # cupy does NOT support implicit conversion via np.asarray() (by
    # design, to stop accidental slow/silent device->host copies) --
    # cupy.asnumpy() is this project's own established idiom for that
    # conversion (see fec/fec.py's own _to_host(), reused throughout
    # this codebase), used here instead of the fragile np.asarray(...)
    # + hasattr(..., "get") pattern that shouldn't be trusted blind.
    import cupy

    for _ in range(N_WARMUP):
        ofdm.rx_process(frame)
    _sync()
    start = time.perf_counter()
    n_crc_valid = 0
    n_bit_exact = 0
    for _ in range(N_ROUNDS):
        result = ofdm.rx_process(frame)
        n_crc_valid += int(bool(cupy.asnumpy(result["crc_valid"])[0]))
        n_bit_exact += int(np.array_equal(cupy.asnumpy(result["bits"])[0], payload))
    _sync()
    rx_total = (time.perf_counter() - start) / N_ROUNDS * 1000

    print(f"\ndecode check: {n_crc_valid}/{N_ROUNDS} rounds crc_valid, "
          f"{n_bit_exact}/{N_ROUNDS} of those bit-exact matches of the original "
          f"({'all correct' if n_bit_exact == N_ROUNDS else 'SOME ROUNDS FAILED -- see below'})")
    if n_bit_exact != N_ROUNDS:
        print("  DO NOT TRUST the timing numbers below if this shows failures -- a fast wrong "
              "answer isn't a result. See prototype_ldpc_cuda_kernel.py's own verify_correctness() "
              "for isolated kernel-level debugging.")

    # -- RX stage breakdown --
    rx_timings: dict = defaultdict(float)
    restore_rx = _install_timing_patch(rx_timings)
    try:
        for _ in range(N_ROUNDS):
            ofdm.rx_process(frame)
    finally:
        restore_rx()

    print(f"\nRX stage breakdown (stopwatch pass -- proportions, averaged over {N_ROUNDS} frames, "
          f"see module docstring re: sync overhead):")
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
    print(f"  {'everything else (unbucketed + per-stage sync overhead)':>50}: "
          f"{max(rx_total - rx_accounted, 0.0):.4f} ms")

    # -- headline --
    tx_mbps = payload_bits / (tx_time * 1000) / 1000
    rx_mbps = payload_bits / rx_total / 1000
    tx_msps = total_samples / (tx_time * 1000) / 1000
    rx_msps = total_samples / rx_total / 1000
    budget_ms = total_samples / 20e6 * 1000
    print(f"\n=== Throughput (single-threaded, one frame at a time) ===")
    print(f"  TX: {tx_time * 1000:.4f} ms/frame -> ~{tx_mbps:.2f} Mbps -> ~{tx_msps:.2f} Msps "
          f"({'OK' if tx_time*1000 <= budget_ms else f'{budget_ms/(tx_time*1000):.2f}x short'} of 20 Msps, budget {budget_ms:.4f} ms)")
    print(f"  RX: {rx_total:.4f} ms/frame -> ~{rx_mbps:.2f} Mbps -> ~{rx_msps:.2f} Msps "
          f"({'OK' if rx_total <= budget_ms else f'{budget_ms/rx_total:.2f}x short'} of 20 Msps, budget {budget_ms:.4f} ms)")
    print(f"  (bottleneck: {'TX' if tx_time*1000 > rx_total else 'RX'})")


if __name__ == "__main__":
    run()
