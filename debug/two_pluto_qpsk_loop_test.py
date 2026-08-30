"""Same known-good config as two_pluto_qpsk_bind_test.py (see that
file's own docstring for the full proven-settings rationale), but as a
tight LOOP within one process instead of re-spawning a fresh process
(and re-opening both Pluto contexts, ~1-2s of overhead each time) per
trial. Opens both Pluto handles ONCE, then repeatedly: build a fresh
random SDU, TX it, capture, decode via Mac.receive_iq(), tally
success/fail, sleep to pace ~100ms between trials, print a running
line. Reports a final success rate over N trials -- this project's own
"measure it for real, don't extrapolate from 1-2 runs" rule, extended
to real hardware reliability characterization, not just a single
speed number.

Usage:
    python3 debug/two_pluto_qpsk_loop_test.py \\
        --uri-tx ip:192.168.2.1 --uri-rx ip:192.168.3.1 --n 30 --period 0.1
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from spectracuda.mac import Mac  # noqa: E402


def _import_adi():
    try:
        return __import__("adi")
    except ImportError:
        sys.exit("error: pyadi-iio not installed. Run 'pip install pyadi-iio'.")


# Two known-good PHY configs, selected via --fft. 256 is
# two_pluto_qpsk_bind_test.py's own proven config (see that file's
# docstring). 64 matches examples/drone_air_unit.py/drone_ground_unit.py's
# own PHY_KWARGS (the actual drone-link config, smaller fft = shorter
# OFDM symbols = lower per-symbol latency, at the cost of fewer data
# subcarriers per symbol -- see that file if this drifts out of sync).
PHY_KWARGS_BY_FFT = {
    256: dict(
        fft_size=256, n_pilot=8, n_data=216, cp_len=32, modem="qpsk",
        fec="rs_m8", fec1="conv_v27", crc="crc16",
        sync="schmidl_cox", cfo="schmidl_cox",
        channel_estimator="ls", equalizer="mmse",
        backend="numpy",
    ),
    64: dict(
        fft_size=64, n_pilot=6, n_data=44, cp_len=16, modem="qpsk",
        fec="rs_m8", fec1="conv_v27",
        interleaver="block", interleaver_kwargs={"unit_bits": 8},
        crc="crc16", sync="schmidl_cox", cfo="schmidl_cox",
        n_training_symbols=2,  # matches drone_air_unit.py's own proven fft=64
        # config -- this preset originally left it at the library default of
        # 1, unlike the 256 preset's implicit 1 which is fine at that fft
        # size. At fft=64 the sync preamble is only 64 samples (schmidl_cox's
        # preamble is exactly fft_size samples -- see sync/schmidl_cox.py),
        # a 4x-smaller correlation window than fft=256's, so it needs the
        # extra averaging n_training_symbols=2 buys for channel estimation
        # to not compound the sync-side noise penalty. See this session's
        # own findings for the measured 256-vs-64 comparison this explains.
        channel_estimator="ls", equalizer="mmse",
        backend="numpy",
    ),
}
SDU_BITS = 2000


def _normalize_from_adc(samples: np.ndarray) -> np.ndarray:
    """Verbatim copy of ~/work/ofdm-hls/sim/pluto_loopback.py's own
    normalize_from_adc() -- see examples/drone_tui/pluto.py's own copy
    of this same function for why it's duplicated, not imported."""
    a = np.abs(samples)
    if a.size == 0:
        return samples
    ref = np.percentile(a, 95)
    if ref < 1e-9:
        return samples
    return samples * (0.5 / ref)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uri-tx", required=True)
    ap.add_argument("--uri-rx", required=True)
    ap.add_argument("--freq", type=float, default=2.425e9)
    ap.add_argument("--rate", type=float, default=5e6)
    ap.add_argument("--tx-gain", type=float, default=0.0)
    ap.add_argument("--rx-gain", type=float, default=60.0)
    ap.add_argument("--agc", default="manual")
    ap.add_argument("--rx-samples", type=int, default=40_000, help="smaller than the one-shot script's 200k -- see module docstring on why a huge buffer dilutes a short burst")
    ap.add_argument("--n", type=int, default=30, help="number of trials")
    ap.add_argument("--period", type=float, default=0.1, help="seconds between trial starts (best-effort -- a slow trial just runs back-to-back with the next)")
    ap.add_argument("--settle", type=float, default=0.15, help="seconds to let the tx buffer swap take effect before treating a capture as current -- achieved by DRAINING rx.rx() calls (each one paces itself to real time), never by a blind time.sleep() disconnected from the radio (see the loop body's own comment for why that desyncs captures by ~1 trial)")
    ap.add_argument("--flush", type=int, default=3, help="EXTRA stale rx.rx() calls to drain on top of the settle-scaled amount (see the loop body) -- see capture_with_pluto()")
    ap.add_argument("--fft", type=int, default=256, choices=sorted(PHY_KWARGS_BY_FFT), help="selects one of the known-good PHY_KWARGS presets above")
    ap.add_argument("--lead-zeros", type=int, default=None, help="default: 2*(fft_size+cp_len) for the selected --fft")
    ap.add_argument("--trail-zeros", type=int, default=None, help="default: 2*(fft_size+cp_len) for the selected --fft")
    args = ap.parse_args()

    PHY_KWARGS = PHY_KWARGS_BY_FFT[args.fft]
    if args.lead_zeros is None:
        args.lead_zeros = 2 * (PHY_KWARGS["fft_size"] + PHY_KWARGS["cp_len"])
    if args.trail_zeros is None:
        args.trail_zeros = 2 * (PHY_KWARGS["fft_size"] + PHY_KWARGS["cp_len"])

    adi = _import_adi()
    rf_bw = int(max(args.rate * 1.25, 5e6))

    tx = adi.Pluto(uri=args.uri_tx)
    tx.sample_rate = int(args.rate)
    tx.tx_lo = int(args.freq)
    tx.tx_rf_bandwidth = rf_bw
    tx.tx_hardwaregain_chan0 = float(args.tx_gain)
    tx.tx_cyclic_buffer = True  # MUST be cyclic (matches capture_with_pluto()) -- the capture
    # window only opens after settle+flush (tens of ms later); a true
    # one-shot burst (~1ms long) would have already finished by then,
    # and the "real" capture would just grab silence/noise. An earlier
    # version of this script set this False and got a suspicious,
    # perfectly consistent noise-floor-only dBFS across every trial --
    # that was this bug, not a real link-quality finding.

    rx = adi.Pluto(uri=args.uri_rx)
    rx.sample_rate = int(args.rate)
    rx.rx_lo = int(args.freq)
    rx.rx_rf_bandwidth = rf_bw
    rx.gain_control_mode_chan0 = args.agc
    if args.agc == "manual":
        rx.rx_hardwaregain_chan0 = float(args.rx_gain)
    rx.rx_buffer_size = int(args.rx_samples)

    print(f"[loop] uri_tx={args.uri_tx} uri_rx={args.uri_rx} freq={args.freq/1e9:.3f}GHz "
          f"rate={args.rate/1e6:.1f}MSPS tx_gain={args.tx_gain:+.1f}dB rx_gain={args.rx_gain:+.1f}dB "
          f"({args.agc}) rx_samples={args.rx_samples} n={args.n} period={args.period}s fft={args.fft}")

    n_ok = 0
    n_fail = 0
    rng = np.random.default_rng()
    for i in range(args.n):
        t_start = time.perf_counter()

        # A FRESH Mac (and thus a fresh, SN-0 ReassemblyBuffer) every
        # trial -- deliberately, not hoisted above the loop. This session
        # found that one shared Mac across all N trials makes a single
        # missed PHY decode permanently desync the reassembly window's
        # expected_sn for the rest of the run (window_size=32 > a short
        # loop's trial count, so it never hits the give-up/resync path),
        # which then reports FAIL on every later trial regardless of
        # that trial's own real RF/PHY outcome. Each trial here is a
        # genuinely independent one-shot test (matches
        # two_pluto_qpsk_bind_test.py's own per-invocation Mac), so each
        # trial's Mac must be independent too -- see this session's own
        # notes for the exact expected_sn trace that pinned this down.
        mac = Mac(mode="um", ofdm_kwargs=PHY_KWARGS)
        mac.bound = True

        sdu = rng.integers(0, 2, size=SDU_BITS).astype("uint8")
        frames = mac.send_iq(sdu)
        if len(frames) != 1:
            print(f"  [{i:3d}] SKIP -- SDU_BITS produced {len(frames)} frames, expected 1")
            continue
        iq_frame = np.asarray(frames[0])[0].astype(np.complex64)
        burst = np.concatenate([
            np.zeros(args.lead_zeros, dtype=np.complex64),
            iq_frame,
            np.zeros(args.trail_zeros, dtype=np.complex64),
        ])
        peak = np.max(np.abs(burst))
        scaled = burst * (2 ** 13 / peak) if peak > 1e-12 else burst * 0

        tx.tx(scaled.astype(np.complex64))
        # NOT a blind time.sleep() before draining -- Pluto keeps
        # sampling continuously in the background the whole time
        # regardless of whether this process is reading, so a sleep
        # here just lets an unread backlog build up that a small fixed
        # --flush count can't catch up on afterward (found this session:
        # settle=0.15s with flush=2 was STILL decoding the PREVIOUS
        # trial's content, ~1 trial stale, every time). Each rx.rx()
        # call itself already blocks for exactly one buffer's worth of
        # real time (rx_buffer_size/rate), so draining continuously via
        # rx.rx() calls alone -- no separate sleep -- both provides the
        # settle time AND guarantees we never fall behind real-time.
        buf_duration_s = args.rx_samples / args.rate
        n_flush = max(1, int(args.settle / buf_duration_s)) + args.flush
        for _ in range(n_flush):
            rx.rx()
        rx_samples = np.asarray(rx.rx(), dtype="complex64")
        tx.tx_destroy_buffer()
        rx_scaled = _normalize_from_adc(rx_samples)

        try:
            delivered = mac.receive_iq(rx_scaled[None, :].astype(np.complex64))
        except Exception as e:
            print(f"  [{i:3d}] EXC {type(e).__name__}: {e}")
            n_fail += 1
            delivered = []

        if delivered and np.array_equal(np.asarray(delivered[0]), sdu):
            n_ok += 1
            status = "OK"
        else:
            n_fail += 1
            status = "FAIL"
        # Same formula as ~/work/ofdm-hls/sim/pluto_loopback.py's own
        # estimate_rx_dbfs() (fs=2**15 reference) -- not imported here
        # since only this one function is needed, duplicated to match.
        p_lin = float(np.mean(np.abs(rx_samples) ** 2))
        rssi_dbfs = 10 * np.log10(p_lin / (2**15 * 2**15) + 1e-300)
        print(f"  [{i:3d}] {status}  rx_dbfs={rssi_dbfs:6.1f}  running: {n_ok}/{i+1} ok "
              f"({100*n_ok/(i+1):.0f}%)")

        elapsed = time.perf_counter() - t_start
        remaining = args.period - elapsed
        if remaining > 0:
            time.sleep(remaining)

    print(f"\n=== FINAL: {n_ok}/{args.n} succeeded ({100*n_ok/args.n:.1f}%), {n_fail} failed ===")


if __name__ == "__main__":
    main()
