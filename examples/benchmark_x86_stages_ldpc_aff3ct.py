"""v2 of benchmark_x86_stages_ldpc.py: same full OFDM TX/RX pipeline,
but the RX LDPC-decode number comes from AFF3CT
(https://github.com/aff3ct/aff3ct, MIT-licensed, portable x86/ARM
SIMD FEC library, see reference/aff3ct/ and
examples/export_ldpc_qc_for_aff3ct.py), not spectracuda's own native
numpy decode -- that numpy decode is already known to be catastrophically
slow (established in benchmark_x86_stages_ldpc.py's own numbers), so
this script doesn't spend a full N_ROUNDS timed loop re-measuring and
re-printing it. It's used exactly once, as a correctness gate, then
gotten out of the way.

## Why this can't be a true end-to-end AFF3CT-in-the-pipeline run

AFF3CT is a standalone C++ process with its own internal BFER
simulation loop (its own random source, its own encoder, its own AWGN
channel, its own demodulator) -- it has no Python API and no way to
accept LLRs computed by spectracuda's own OfdmDemodulator/
LSChannelEstimator/MMSEEqualizer chain. There is no bridge (not
attempted here -- a real one would mean writing a custom AFF3CT
"Module" in C++ that reads from a named pipe/shared buffer spectracuda
writes into, out of scope for a benchmark script). So this script does
NOT run this frame's actual payload through AFF3CT.

What it DOES do, honestly:
1. Runs the real spectracuda TX chain (generate_frame()) and ONE real
   RX pass (rx_process(), spectracuda's own unpatched numpy LDPC
   decode) purely as a correctness gate -- confirms the frame this
   script built actually round-trips before trusting anything below.
   Not timed in a loop -- one call is enough to prove correctness, and
   its number is not the point of this script.
2. Times the REST of RX (sync/CFO/OFDM-decode/channel-est+equalizer --
   the stages AFF3CT has no equivalent for) over N_ROUNDS, with
   LDPCCode.decode() STUBBED OUT to an instant no-op for this pass
   only (see `_install_fast_stub_decode()`) -- so this loop isn't
   wasted re-measuring the already-known-slow numpy LDPC decode 30
   times just to discard the number.
3. Separately invokes the real AFF3CT binary (subprocess, not
   simulated/estimated) with -F set to this frame's exact
   n_ldpc_blocks, over the exact same LDPC variant/matrix (see
   examples/export_ldpc_qc_for_aff3ct.py), and reads back AFF3CT's own
   measured decode_siho latency for decoding that many codewords
   together.
4. Reports ONE RX total: the real (2) stage times + AFF3CT's real (3)
   decode time. Not a "projection" needing a numpy baseline to swap
   against -- AFF3CT's number simply IS how LDPC decode is measured
   here.

## AFF3CT decoder settings, and why

Same settings verified correct in this session's own interactive
testing (see chat history / examples/export_ldpc_qc_for_aff3ct.py's
own derivation notes): --dec-type BP_FLOODING --dec-implem NMS
--dec-norm 0.75 --dec-ite 50 (matches spectracuda's own min-sum
normalized-alpha=0.75, max_iterations=50 -- see fec/ldpc.py's
_MIN_SUM_ALPHA/_DEFAULT_MAX_ITERATIONS). Syndrome-based early
termination is LEFT ON (no --dec-no-synd) -- that's AFF3CT's actual
default, real-world behavior and its main advantage over spectracuda's
own decode() (which always runs the full max_iterations, no early
exit -- see fec/ldpc.py's decode() docstring). A single AWGN operating
point (2.0 dB Eb/N0, chosen because earlier interactive testing showed
it converges to enough frame errors in a few seconds without either
being noise-free -- which would make every codeword trivially easy --
or needing an impractically long run) is simulated just long enough
(--mnt-max-fe / -e) to get a stable decode_siho average, not a full BER
curve (this script cares about throughput, not error-rate
characterization -- that's a separate, already-answered question, see
the earlier interactive session's own BER sanity check).

## Requires

reference/aff3ct/build/bin/aff3ct-4.7.0 (or aff3ct-<version>) built --
see reference/aerial-cuda-accelerated-ran-NOTES.md's sibling build
notes / this session's own chat history for the exact cmake/make
sequence. This script errors out clearly (not a silent skip) if it
can't find a built binary, since the whole point of THIS script (as
opposed to benchmark_x86_stages_ldpc.py) is the AFF3CT comparison --
override the path via SPECTRACUDA_AFF3CT_BIN if it's built somewhere
else.

Usage: same CLI as benchmark_x86_stages_ldpc.py (identical _parse_args()):
    python examples/benchmark_x86_stages_ldpc_aff3ct.py
    python examples/benchmark_x86_stages_ldpc_aff3ct.py qpsk 32000 1/2
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from spectracuda.cfo.schmidl_cox import SchmidlCoxCFO
from spectracuda.channel.ls import LSChannelEstimator
from spectracuda.equalizer.mmse import MMSEEqualizer
from spectracuda.fec.crc import CRC
from spectracuda.fec.ldpc import LDPCCode
from spectracuda.ofdm.fft import OfdmDemodulator
from spectracuda.pipeline import Ofdm
from spectracuda.sync.schmidl_cox import SchmidlCoxSync

# Sibling script, not a library -- importable because "python
# examples/benchmark_x86_stages_ldpc_aff3ct.py" puts examples/ on sys.path
# automatically. Deliberately IMPORTED rather than duplicated (unlike
# this file's own small helpers below, which follow the established
# "duplicate, don't import" convention for standalone example scripts)
# because export_qc()'s row/col layout + shift-sign convention was
# derived by reading AFF3CT's C++ parser source and corrected against
# real binary behavior -- a second hand-copied version would risk
# silently drifting out of sync with that hard-won fix.
from export_ldpc_qc_for_aff3ct import export_qc

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

# -- AFF3CT settings, see module docstring --
AFF3CT_SNR_EBN0_DB = 2.0
AFF3CT_MAX_FRAME_ERRORS = 5
AFF3CT_TIMEOUT_S = 90
AFF3CT_DEC_NORM = 0.75
AFF3CT_DEC_ITE = 50


def _parse_args():
    """Identical to benchmark_x86_stages_ldpc.py's own _parse_args() --
    duplicated (not imported) for the same standalone-script reason as
    the rest of that file's small helpers."""
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


def _pin_to_one_core() -> str:
    """See benchmark_x86_stages_v3.py's own _pin_to_one_core()."""
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
    """See benchmark_x86_stages_ldpc.py's own _install_timing_patch()."""
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


def _install_fast_stub_decode():
    """Replaces LDPCCode.decode() with an instant no-op for the RX
    stage-breakdown timing pass below -- see module docstring point 2.
    Returns garbage (zeros), not a real decode -- fine, because
    correctness was already proven separately with the REAL decode
    (module docstring point 1) before this stub is ever installed; this
    pass only exists to time the OTHER stages without paying for 30
    more calls into the already-known-slow real BP loop."""
    orig = LDPCCode.decode

    def stub(self, bits, p=0.02, max_iterations=None):
        xp = self.xp
        bits = xp.asarray(bits)
        if bits.ndim == 1:
            bits = bits[None, :]
        n_checks = self.n - self.k
        real_k = bits.shape[-1] - n_checks
        return xp.zeros((bits.shape[0], real_k), dtype="uint8")

    LDPCCode.decode = stub

    def restore():
        LDPCCode.decode = orig

    return restore


def _actual_payload_bits(requested_bits: int, k_bits: int, crc_bits: int) -> int:
    """See benchmark_x86_stages_ldpc.py's own _actual_payload_bits()."""
    n_blocks = max(1, -(-(requested_bits + crc_bits) // k_bits))
    while True:
        payload_bits = n_blocks * k_bits - crc_bits
        if payload_bits >= requested_bits and payload_bits % 8 == 0:
            return payload_bits
        n_blocks += 1


# ---------------------------------------------------------------------------
# AFF3CT integration
# ---------------------------------------------------------------------------

def _find_aff3ct_binary() -> Path:
    override = os.environ.get("SPECTRACUDA_AFF3CT_BIN")
    if override:
        p = Path(override)
        if p.is_file():
            return p
        raise SystemExit(f"SPECTRACUDA_AFF3CT_BIN={override!r} does not exist.")

    repo_root = Path(__file__).resolve().parents[1]
    bin_dir = repo_root / "reference" / "aff3ct" / "build" / "bin"
    if bin_dir.is_dir():
        candidates = sorted(bin_dir.glob("aff3ct-*")) + sorted(bin_dir.glob("aff3ct"))
        for c in candidates:
            if c.is_file() and os.access(c, os.X_OK):
                return c
    raise SystemExit(
        f"No built AFF3CT binary found under {bin_dir} (or $SPECTRACUDA_AFF3CT_BIN).\n"
        "This script's whole point is the AFF3CT comparison, so it errors out here "
        "rather than silently skipping it. Build it first:\n"
        "  cd reference/aff3ct && mkdir -p build && cd build\n"
        "  cmake .. -DCMAKE_BUILD_TYPE=Release -DAFF3CT_COMPILE_EXE=ON\n"
        "  make -j$(nproc) aff3ct-bin\n"
        "(clone reference/aff3ct first -- see examples/export_ldpc_qc_for_aff3ct.py's "
        "module docstring / this project's chat history for the exact clone/submodule steps "
        "if reference/aff3ct/ doesn't exist yet.)"
    )


def _aff3ct_qc_path(variant: str) -> Path:
    out_dir = Path(tempfile.gettempdir()) / "spectracuda_aff3ct_qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{variant}.qc"
    export_qc(variant, str(out_path))  # cheap (one-off table expansion) -- always regenerate, never trust a stale file
    return out_path


# The stats-table row we care about looks like (see this session's own
# interactive testing for the exact real layout, captured once and
# never re-guessed):
#   #            Decoder |       decode_siho | yes |  13 | * || CALLS | TIME | PERC || AVG_Mbps | MIN | MAX || AVG_us | MIN | MAX
_DECODE_ROW_RE = re.compile(r"Decoder\s*\|\s*decode_siho")
# Final SNR-point summary row, e.g.:
#         -1.01 |     2.00 ||      12716 |      ... |       30 | 1.7e-05 | 2.4e-03 ||    2.39 | 00h00'07
_SUMMARY_ROW_RE = re.compile(r"^\s*-?[\d.]+\s*\|\s*-?[\d.]+\s*\|\|")


def _run_aff3ct_decode(bin_path: Path, qc_path: Path, n_blocks: int) -> dict:
    """Runs the real AFF3CT binary (subprocess) decoding n_blocks
    codewords per call (-F n_blocks), early termination ON (its real
    default -- see module docstring), and returns the REAL measured
    per-call latency/throughput plus a BER/FER sanity-check pair parsed
    straight from its own output -- never estimated, never reused from
    a different n_blocks/session."""
    cmd = [
        str(bin_path),
        "-C", "LDPC",
        "--dec-h-path", str(qc_path),
        "--pct-type", "NO",
        "--enc-type", "LDPC_H",
        "--dec-type", "BP_FLOODING",
        "--dec-implem", "NMS",
        "--dec-norm", str(AFF3CT_DEC_NORM),
        "--dec-ite", str(AFF3CT_DEC_ITE),
        "-e", str(AFF3CT_MAX_FRAME_ERRORS),
        "--mdm-type", "BPSK",
        "--chn-type", "AWGN",
        "-m", str(AFF3CT_SNR_EBN0_DB), "-M", str(AFF3CT_SNR_EBN0_DB), "-R", str(AFF3CT_SNR_EBN0_DB),
        "-t", "1", "-F", str(n_blocks),
        "--sim-stats",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=AFF3CT_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"AFF3CT subprocess did not finish within {AFF3CT_TIMEOUT_S}s "
            f"(n_blocks={n_blocks}, cmd={' '.join(cmd)})"
        ) from exc

    lines = result.stdout.splitlines()
    decode_tokens = None
    for line in lines:
        if _DECODE_ROW_RE.search(line):
            decode_tokens = [t.strip() for t in line.split("|") if t.strip()]
    summary_tokens = None
    for line in lines:
        if _SUMMARY_ROW_RE.match(line):
            summary_tokens = [t.strip().rstrip("*").strip() for t in line.split("|") if t.strip()]

    if decode_tokens is None or len(decode_tokens) < 13:
        raise RuntimeError(
            f"Could not parse AFF3CT decode_siho stats row (n_blocks={n_blocks}).\n"
            f"--- stdout tail ---\n{result.stdout[-3000:]}\n"
            f"--- stderr tail ---\n{result.stderr[-1500:]}"
        )
    avg_mbps = float(decode_tokens[9])
    avg_latency_s = float(decode_tokens[12]) / 1e6  # per CALL, i.e. per n_blocks codewords together

    ber = fer = None
    if summary_tokens is not None and len(summary_tokens) >= 7:
        ber = float(summary_tokens[5])
        fer = float(summary_tokens[6])

    return {
        "cmd": cmd,
        "avg_mbps": avg_mbps,
        "avg_latency_s": avg_latency_s,
        "ber": ber,
        "fer": fer,
        "raw_decode_row": decode_tokens,
    }


def run() -> None:
    pin_status = _pin_to_one_core()
    phy_kwargs = dict(
        fft_size=FFT_SIZE, n_pilot=N_PILOT, n_data=N_DATA, cp_len=CP_LEN,
        modem=MODEM_SCHEME, fec=LDPC_VARIANT, fec1="none", crc="crc16",
        sync="schmidl_cox", cfo="schmidl_cox",
        channel_estimator="ls", equalizer="mmse",
        backend="numpy",
    )
    print(f"=== benchmark_x86_stages_ldpc_aff3ct (real Ofdm TX/RX pipeline, LDPC decode via "
          f"AFF3CT -- see module docstring) config: "
          f"fft_size={FFT_SIZE}, n_pilot={N_PILOT}, n_data={N_DATA}, cp_len={CP_LEN}, "
          f"modem={MODEM_SCHEME}, fec={LDPC_VARIANT!r} (rate {LDPC_RATE}), crc=crc16, "
          f"sdu_bits={SDU_BITS} (requested) ===")
    print(f"    CPU affinity: {pin_status}")

    aff3ct_bin = _find_aff3ct_binary()
    print(f"    AFF3CT binary: {aff3ct_bin}")

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
        print(f"    NOTE: clamped down to the largest payload that fits in one frame: "
              f"{max_payload_bits} bits ({max_blocks} blocks) -- see benchmark_x86_stages_ldpc.py's "
              f"module docstring for why (no Mac layer at this level).")
        payload_bits = max_payload_bits
        n_ldpc_blocks = max_blocks
    elif payload_bits != SDU_BITS:
        print(f"    NOTE: padded requested {SDU_BITS} bits up to {payload_bits} bits "
              f"({n_ldpc_blocks}x {k_bits}-bit LDPC blocks).")

    rng = np.random.default_rng(0)
    payload = rng.integers(0, 2, size=payload_bits).astype("uint8")
    frame = ofdm.generate_frame(payload[None, :])
    print(f"\nPayload: {payload_bits} bits -> {n_ldpc_blocks} LDPC codeword(s) of {k_bits}->{n_bits} bits each")

    # -- full TX chain --
    for _ in range(N_WARMUP):
        ofdm.generate_frame(payload[None, :])
    start = time.perf_counter()
    for _ in range(N_ROUNDS):
        frame = ofdm.generate_frame(payload[None, :])
    tx_time = (time.perf_counter() - start) / N_ROUNDS
    print(f"\nfull TX chain (generate_frame(), 1 frame): {tx_time * 1000:.4f} ms/frame")

    # -- correctness gate: ONE real RX pass, real (unpatched) numpy LDPC
    # decode -- not timed in a loop, see module docstring point 1 --
    result = ofdm.rx_process(frame)
    bit_exact = np.array_equal(np.asarray(result["bits"])[0], payload)
    print(f"decode check: {'bit-exact match' if bit_exact else 'MISMATCH -- see below'} "
          f"(real spectracuda numpy decode, one pass, correctness gate only)")
    if not bit_exact:
        raise SystemExit("Correctness gate failed -- refusing to report timing for a broken frame.")

    # -- RX stage breakdown: sync/CFO/OFDM-decode/chanest+eq only, LDPC
    # decode stubbed to a no-op for this pass -- see module docstring point 2 --
    rx_timings: dict = defaultdict(float)
    restore_timing = _install_timing_patch(rx_timings)
    restore_stub = _install_fast_stub_decode()
    try:
        for _ in range(N_ROUNDS):
            ofdm.rx_process(frame)
    finally:
        restore_stub()
        restore_timing()
    breakdown_buckets = [
        ("sync detect + CFO", rx_timings["sync_cfo"] / N_ROUNDS * 1000),
        ("OFDM decode (FFT+CP strip)", rx_timings["ofdm_decode"] / N_ROUNDS * 1000),
        ("channel estimation + equalization", rx_timings["chanest_eq"] / N_ROUNDS * 1000),
    ]
    non_ldpc_rx_ms = sum(ms for _, ms in breakdown_buckets)

    # -- real AFF3CT measurement, same variant, same n_ldpc_blocks --
    print(f"\n=== AFF3CT: decoding {n_ldpc_blocks} codeword(s) of {LDPC_VARIANT} "
          f"(-F {n_ldpc_blocks}, {AFF3CT_SNR_EBN0_DB} dB Eb/N0, early termination ON, "
          f"single core) ===")
    qc_path = _aff3ct_qc_path(LDPC_VARIANT)
    aff3ct_result = _run_aff3ct_decode(aff3ct_bin, qc_path, n_ldpc_blocks)
    aff3ct_decode_ms = aff3ct_result["avg_latency_s"] * 1000
    print(f"  measured decode_siho: {aff3ct_decode_ms:.4f} ms for all {n_ldpc_blocks} codeword(s) "
          f"together ({aff3ct_result['avg_mbps']:.3f} Mb/s)")
    if aff3ct_result["ber"] is not None:
        print(f"  sanity check (AFF3CT's own BFER sim, its own random payload -- NOT this frame's "
              f"payload, see module docstring): BER={aff3ct_result['ber']:.3e} FER={aff3ct_result['fer']:.3e} "
              f"-- {'sane (nonzero-but-low, a real working decoder)' if 0 < aff3ct_result['ber'] < 1e-2 else 'CHECK THIS -- unexpected'}")

    rx_total_ms = non_ldpc_rx_ms + aff3ct_decode_ms
    print(f"\nRX stage breakdown (LDPC decode via AFF3CT):")
    for label, ms in breakdown_buckets:
        print(f"  {label:>50}: {ms:.4f} ms")
    print(f"  {'FEC decode -- LDPC (AFF3CT)':>50}: {aff3ct_decode_ms:.4f} ms")
    print(f"  {'RX total':>50}: {rx_total_ms:.4f} ms")

    total_samples = frame.shape[-1]
    budget_ms = total_samples / 20e6 * 1000
    tx_mbps = payload_bits / (tx_time * 1000) / 1000
    rx_mbps = payload_bits / rx_total_ms / 1000
    print(f"\n=== Throughput (single-threaded, one frame at a time; budget {budget_ms:.4f} ms/frame for 20 Msps) ===")
    print(f"  TX: {tx_time*1000:.4f} ms/frame -> ~{tx_mbps:.2f} Mbps "
          f"({'OK' if tx_time*1000 <= budget_ms else f'{budget_ms/(tx_time*1000):.2f}x short'})")
    print(f"  RX: {rx_total_ms:.4f} ms/frame -> ~{rx_mbps:.2f} Mbps "
          f"({'OK' if rx_total_ms <= budget_ms else f'{budget_ms/rx_total_ms:.2f}x short'})")


if __name__ == "__main__":
    run()
