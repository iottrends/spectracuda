"""x86 (backend="numpy") timing: the full TX chain, and individual RX
stages -- sync+CFO, OFDM decode (FFT+CP strip), channel estimation +
equalization, FEC decode (both stages), and MAC decode -- for
fft_size=256 with BOTH Viterbi and Reed-Solomon active in the SAME
concatenated chain: fec="rs_m8" (fec0/inner) + fec1="conv_v27"
(fec1/outer), the corrected assignment from docs/todo.md #1.2/#1.12
(Viterbi faces the channel, RS cleans up its bursty residual -- Viterbi
must be fec1, decoded first, not fec0).

Real data flow throughout, not synthetic per-stage inputs (an earlier,
more elaborate benchmark script hand-built synthetic inputs per stage
and got several of their shapes wrong). Every timed stage here is
instrumented in place while mac.send_iq()/receive_iq() run completely
normally end to end, so every stage sees exactly the real,
correctly-shaped data it would in production.

One real wrinkle, worth stating rather than hiding: Ofdm.rx_process()
rebuilds a FRESH Packetizer internally on every call (see ofdm.py's
"freshly-built Packetizer(header_fields[...])" -- it must resolve fec/
fec1/crc from the just-decoded header, never assume its own
construction-time values, matching this project's "resolve from the
wire, not from self" principle throughout). That throwaway packetizer's
FEC objects are NOT the same instances as ofdm.packetizer's -- so FEC
timing here patches FEC.decode at the CLASS level (every instance,
including throwaway ones), bucketed by each call's own self.scheme,
rather than patching one specific instance and silently timing zero.

Runs with Numba only for the Viterbi (conv_v27) stage -- not the old
pure-NumPy path side by side with it (see
prototype_viterbi_numba.py/git history for that comparison and its
~100x number). The patched FEC.decode() below genuinely calls the
Numba-JIT decoder as conv_v27's real decode path for this run, not a
shadow timing next to the original.

Usage:
    python examples/benchmark_x86_stages.py
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict

import numpy as np

from spectracuda.fec.fec import FEC
from spectracuda.fec.reed_solomon import ReedSolomonCode
from spectracuda.mac import Mac

sys.path.insert(0, os.path.dirname(__file__))
from prototype_viterbi_numba import _numba_decode as _numba_viterbi_decode  # noqa: E402 -- ~100x, see that script
from prototype_rs_numba import _numba_decode_full_contract as _numba_rs_decode  # noqa: E402 -- ~10-12x, handles shortened blocks too

FFT_SIZE = 256
N_PILOT = 8
N_DATA = 216
CP_LEN = 32
SDU_BITS = 4000
N_ROUNDS = 30
N_WARMUP = 5


def _timed(fn, bucket: str, timings: dict):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        timings[bucket] += time.perf_counter() - start
        return result

    return wrapper


def _install_fec_class_patch(timings: dict):
    """Patches FEC.decode at the CLASS level -- catches every FEC
    instance's decode() call, including the throwaway packetizer's (see
    module docstring), bucketed by that instance's own self.scheme so
    fec0 (rs_m8) and fec1 (conv_v27) are reported separately. Returns a
    restore function.

    For conv_v27, the REAL decode path used here IS the Numba-JIT one
    (prototype_viterbi_numba.py) -- not a side computation alongside the
    old pure-numpy path, that path isn't run here at all (its
    ~100x-slower number is already known from the earlier comparison
    run; this run is "with numba only", per what was asked).

    For rs_m8, FEC.decode() itself isn't patched to bypass anything --
    it still calls the real _decode_symbol_level() (which does the real
    bit/symbol packing and multi-block splitting for a message that
    isn't an exact multiple of K=223 symbols, see fec.py). Instead
    ReedSolomonCode.decode() is ALSO patched at the class level, one
    level deeper, so every block (full-length or the shortened leftover
    one) goes through the Numba full-contract decode
    (prototype_rs_numba.py's _numba_decode_full_contract(), verified
    against exactly this shortened-block case before being trusted
    here) instead of the original per-codeword algorithm. The existing
    fec_decode[rs_m8] timing bucket picks up the faster time
    automatically, since it already times the whole FEC.decode() call
    that calls down into this.

    decode() must still raise ValueError on a genuinely undecodable
    input, same contract the original had (the caller -- Packetizer,
    ultimately Mac._rx_one_frame() -- catches that and treats the frame
    as lost, not a crash) -- both Numba paths do."""
    orig_fec_decode = FEC.decode
    orig_rs_decode = ReedSolomonCode.decode

    def patched_fec_decode(self, *args, **kwargs):
        start = time.perf_counter()
        if self.scheme == "conv_v27":
            bits = args[0]
            result = _numba_viterbi_decode(self._impl, bits)
        else:
            result = orig_fec_decode(self, *args, **kwargs)
        timings[f"fec_decode[{self.scheme}]"] += time.perf_counter() - start
        return result

    def patched_rs_decode(self, codeword, *args, **kwargs):
        # x86-only, matching the rest of this script -- _numba_rs_decode
        # always assumes plain numpy input, it does NOT replicate
        # ReedSolomonCode.decode()'s own cupy.asnumpy() host-conversion
        # (this whole benchmark only ever runs backend="numpy").
        return _numba_rs_decode(codeword)

    FEC.decode = patched_fec_decode
    ReedSolomonCode.decode = patched_rs_decode

    def restore():
        FEC.decode = orig_fec_decode
        ReedSolomonCode.decode = orig_rs_decode

    return restore


def _instrument(mac: Mac, timings: dict) -> None:
    """Instance-level patches for the stages that ARE persistent self.
    attributes (sync/cfo/demod/channel_estimator/equalizer -- unlike the
    packetizer/FEC objects, these are built once in Ofdm.__init__ and
    reused directly, never rebuilt per rx_process() call, since they
    don't depend on header-resolved scheme choices)."""
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
    print(f"=== config: fft_size={FFT_SIZE}, n_pilot={N_PILOT}, n_data={N_DATA}, "
          f"cp_len={CP_LEN}, modem=qpsk, fec='rs_m8' (inner), "
          f"fec1='conv_v27' (outer), crc=crc16, sync=schmidl_cox, "
          f"cfo=schmidl_cox, channel_estimator=ls, equalizer=mmse, "
          f"backend=numpy, sdu_bits={SDU_BITS} ===")

    # tx_probe_mac is a throwaway, used ONLY for the "what's being sent"
    # introspection and the "full TX chain" timing loop below -- neither
    # of those PDUs is ever forwarded to any receiver. tx_mac/rx_mac
    # (built separately) stay a clean, PAIRED, contiguous-SN pair used
    # ONLY for the RX-stage section further down. Keeping these separate
    # is a real fix, not just tidiness: reusing one Mac for both left
    # gaps in the SN sequence tx_mac's real receiver (rx_mac) never saw
    # (every un-forwarded send_iq() call here still bumps the SN
    # counter), which triggered ReassemblyBuffer's window-eviction logic
    # to flush multiple pending items in one receive_iq() call (35
    # delivered from 30 calls, caught by checking n_delivered against
    # N_ROUNDS directly rather than assuming 1:1).
    tx_probe_mac = Mac(mode="um", ofdm_kwargs=phy_kwargs)
    # tx_probe_mac never talks to a real peer (its PDUs are never
    # forwarded anywhere -- see comment above), so there's no real bind
    # handshake to run; send_iq() only checks self.bound, set directly.
    tx_probe_mac.bound = True
    tx_mac = Mac(mode="um", ofdm_kwargs=phy_kwargs)
    rx_mac = Mac(mode="um", ofdm_kwargs=phy_kwargs)

    # Bind handshake -- one-time, not part of the steady-state tx/rx chain
    # this script measures, so not timed.
    req = tx_mac.build_bind_request()
    resp = rx_mac.handle_bind_request_iq(req)
    assert tx_mac.handle_bind_response_iq(resp)

    rng = np.random.default_rng(0)
    sdu = rng.integers(0, 2, size=SDU_BITS).astype("uint8")

    # -- what's actually being sent, in real units, not just a timing
    # number with no context: PDU size, and the OFDM-symbol breakdown of
    # one frame (derived from the real generated IQ length, not assumed
    # -- preamble has no CP so isn't samples_per_symbol-sized like the
    # rest, see Ofdm's own docstring) --
    ofdm = tx_probe_mac.ofdm
    pdus = tx_probe_mac._impl.transmit(sdu)
    samples_per_symbol = ofdm.fft_size + ofdm.cp_len
    one_frame_iq = ofdm.generate_frame(np.asarray(pdus[0], dtype="uint8")[None, :])
    total_samples = one_frame_iq.shape[-1]
    other_symbols = (total_samples - ofdm.fft_size) / samples_per_symbol  # preamble excluded (no CP)
    payload_symbols = other_symbols - ofdm.n_training_symbols - ofdm.num_symbols_header
    print(f"\nSDU: {SDU_BITS} bits -> {len(pdus)} PDU(s), {len(pdus[0])} bits/PDU "
          f"(includes MAC header + FEC/CRC overhead)")
    print(f"One frame: {total_samples} IQ samples = "
          f"1 preamble symbol ({ofdm.fft_size} samples, no CP) + "
          f"{ofdm.n_training_symbols} training + {ofdm.num_symbols_header} header + "
          f"{payload_symbols:.0f} payload OFDM symbols "
          f"({samples_per_symbol} samples/symbol each)")

    # -- full TX chain: segmentation + FEC/CRC/interleave + modem +
    # resource-grid + IFFT/CP + preamble/training, everything send_iq()
    # actually does -- timed on tx_probe_mac, never forwarded anywhere --
    for _ in range(N_WARMUP):
        tx_probe_mac.send_iq(sdu)
    start = time.perf_counter()
    for _ in range(N_ROUNDS):
        probe_iq_frames = tx_probe_mac.send_iq(sdu)
    tx_time = (time.perf_counter() - start) / N_ROUNDS
    print(f"\nfull TX chain (send_iq(), {len(probe_iq_frames)} PDU(s)/SDU): {tx_time * 1000:.4f} ms")

    # -- individual RX stages, instrumented on the REAL receive_iq() call.
    # Each round generates a FRESH iq_frames (a real, un-timed send_iq()
    # call on tx_mac, not instrumented) so rx_mac sees a genuinely new SN
    # every time -- reusing one fixed iq_frames across every round would
    # make every call after the first a duplicate-SN reject in
    # ReassemblyBuffer (mac.receive_iq() correctly returns 0 SDUs for a
    # repeat -- checked directly before writing this), which would make
    # "MAC decode" time the cheap duplicate fast-path almost every call
    # instead of genuine reassembly/delivery work. --
    timings: dict = defaultdict(float)
    _instrument(rx_mac, timings)
    restore_fec_patch = _install_fec_class_patch(timings)
    n_delivered = 0
    n_bit_exact = 0
    try:
        for _ in range(N_WARMUP):
            for iq in tx_mac.send_iq(sdu):
                rx_mac.receive_iq(iq)
        timings.clear()  # drop warm-up timing, keep only the timed rounds below

        n_calls = 0
        for _ in range(N_ROUNDS):
            for iq in tx_mac.send_iq(sdu):
                delivered = rx_mac.receive_iq(iq)
                n_delivered += len(delivered)
                n_bit_exact += sum(np.array_equal(d, sdu) for d in delivered)
                n_calls += 1
    finally:
        restore_fec_patch()

    print(f"\ndecode check: {n_delivered}/{N_ROUNDS} rounds delivered a SDU, "
          f"{n_bit_exact}/{n_delivered} of those bit-exact matches of the original "
          f"({'all correct' if n_bit_exact == N_ROUNDS else 'SOME ROUNDS FAILED -- see below'})")
    if n_bit_exact != N_ROUNDS:
        print("  NOTE: a round not delivering doesn't necessarily mean a decode")
        print("  error -- rx_mac's ReassemblyBuffer is a real, bounded, stateful")
        print("  window across all rounds (same object reused throughout this")
        print("  script, matching one real receiver's lifetime), so this can also")
        print("  reflect its own real give-up/eviction behavior, not just failure.")
        print("  A delivered-but-NOT-bit-exact SDU, however, IS a real decode bug.")

    print(f"\nRX stages, per frame decoded (averaged over {n_calls} frames):")
    for bucket, label in [
        ("sync+cfo", "sync detect + CFO"),
        ("ofdm_decode", "OFDM decode (FFT+CP strip)"),
        ("chanest_eq", "channel estimation + equalization"),
        ("fec_decode[conv_v27]", "FEC decode -- Viterbi (fec1, outer, Numba-JIT)"),
        ("fec_decode[rs_m8]", "FEC decode -- Reed-Solomon (fec0, inner, Numba-JIT)"),
        ("mac_decode", "MAC decode"),
    ]:
        print(f"  {label:>40}: {timings[bucket] / n_calls * 1000:.4f} ms")


if __name__ == "__main__":
    run()
