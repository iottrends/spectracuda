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

Usage:
    python examples/benchmark_x86_stages.py
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

from spectracuda.fec.fec import FEC
from spectracuda.mac import Mac

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
    restore function."""
    orig_decode = FEC.decode

    def patched_decode(self, *args, **kwargs):
        start = time.perf_counter()
        result = orig_decode(self, *args, **kwargs)
        timings[f"fec_decode[{self.scheme}]"] += time.perf_counter() - start
        return result

    FEC.decode = patched_decode
    return lambda: setattr(FEC, "decode", orig_decode)


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

    tx_mac = Mac(mode="um", ofdm_kwargs=phy_kwargs)
    rx_mac = Mac(mode="um", ofdm_kwargs=phy_kwargs)

    # Bind handshake -- one-time, not part of the steady-state tx/rx chain
    # this script measures, so not timed.
    req = tx_mac.build_bind_request()
    resp = rx_mac.handle_bind_request_iq(req)
    assert tx_mac.handle_bind_response_iq(resp)

    rng = np.random.default_rng(0)
    sdu = rng.integers(0, 2, size=SDU_BITS).astype("uint8")

    # -- full TX chain: segmentation + FEC/CRC/interleave + modem +
    # resource-grid + IFFT/CP + preamble/training, everything send_iq()
    # actually does --
    for _ in range(N_WARMUP):
        tx_mac.send_iq(sdu)
    start = time.perf_counter()
    for _ in range(N_ROUNDS):
        iq_frames = tx_mac.send_iq(sdu)
    tx_time = (time.perf_counter() - start) / N_ROUNDS
    print(f"\nfull TX chain (send_iq(), {len(iq_frames)} PDU(s)/SDU): {tx_time * 1000:.4f} ms")

    # -- individual RX stages, instrumented on the REAL receive_iq() call --
    timings: dict = defaultdict(float)
    _instrument(rx_mac, timings)
    restore_fec_patch = _install_fec_class_patch(timings)
    try:
        for _ in range(N_WARMUP):
            for iq in iq_frames:
                rx_mac.receive_iq(iq)
        timings.clear()  # drop warm-up timing, keep only the timed rounds below

        n_calls = 0
        for _ in range(N_ROUNDS):
            for iq in iq_frames:
                rx_mac.receive_iq(iq)
                n_calls += 1
    finally:
        restore_fec_patch()

    print(f"RX stages, per frame decoded (averaged over {n_calls} frames):")
    for bucket, label in [
        ("sync+cfo", "sync detect + CFO"),
        ("ofdm_decode", "OFDM decode (FFT+CP strip)"),
        ("chanest_eq", "channel estimation + equalization"),
        ("fec_decode[conv_v27]", "FEC decode -- Viterbi (fec1, outer)"),
        ("fec_decode[rs_m8]", "FEC decode -- Reed-Solomon (fec0, inner)"),
        ("mac_decode", "MAC decode"),
    ]:
        print(f"  {label:>40}: {timings[bucket] / n_calls * 1000:.4f} ms")


if __name__ == "__main__":
    run()
