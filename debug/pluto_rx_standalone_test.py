"""RX-only, no TX at all -- isolates whether rx.rx() itself is slow, or
whether it's slow only when a TX thread/connection is concurrently
active (see debug/pluto_ofdm_streaming_test.py, which measured rx.rx()
at median 30ms/max 100ms with TX running concurrently, vs under 1ms in
an earlier fully-isolated sweep with no TX at all). Runs for --seconds
(default 10s), same read+decode loop body, then prints the same timing
breakdown.

Usage:
    python3 debug/pluto_rx_standalone_test.py --uri-rx ip:192.168.3.1 --seconds 10
"""
import argparse
import sys
import time

import numpy as np
import adi

sys.path.insert(0, "/home/abhi/work/spectracuda")
from spectracuda.pipeline import Ofdm

PHY_KWARGS = dict(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32, modem="qpsk",
    fec="rs_m8", fec1="conv_v27", crc="crc16",
    sync="schmidl_cox", cfo="schmidl_cox",
    channel_estimator="ls", equalizer="mmse",
    backend="numpy",
)
RX_SAMPLES = 100_000  # read big (amortize the per-call daemon overhead the
# online sources + pluto_air_unit.py's own comment both point at), then loop
# around and feed 2048-sample pieces to rx_streaming() (see below) -- NOT a
# single big rx_streaming() call, which would violate its 2048-sample
# SEEKING-state search cap for this fft=256 config.
STREAM_CHUNK = 2048
BUDGET_US = RX_SAMPLES / 5e6 * 1e6  # 20000us at 5Msps

ap = argparse.ArgumentParser()
ap.add_argument("--uri-rx", required=True)
ap.add_argument("--freq", type=float, default=2.425e9)
ap.add_argument("--rate", type=float, default=5e6)
ap.add_argument("--rx-gain", type=float, default=60.0)
ap.add_argument("--seconds", type=float, default=10.0)
args = ap.parse_args()

rf_bw = int(max(args.rate * 1.25, 5e6))

rx = adi.Pluto(uri=args.uri_rx)  # NO tx object opened at all in this script
rx.sample_rate = int(args.rate)
rx.rx_lo = int(args.freq)
rx.rx_rf_bandwidth = rf_bw
rx.gain_control_mode_chan0 = "manual"
rx.rx_hardwaregain_chan0 = args.rx_gain
rx.rx_buffer_size = RX_SAMPLES

ofdm_rx = Ofdm(**PHY_KWARGS)
ofdm_rx.reset_stream()

for _ in range(10):  # PySDR's own recommended flush before real measurement
    rx.rx()

rx_call_us = []
decode_us = []
n_decoded = 0
n_over_budget = 0

print(f"[rx-standalone] uri_rx={args.uri_rx} rate={args.rate/1e6:.1f}Msps rx_gain={args.rx_gain:+.1f}dB "
      f"rx_samples={RX_SAMPLES} stream_chunk={STREAM_CHUNK} running for {args.seconds:.1f}s ...")

t_end = time.perf_counter() + args.seconds
n_calls = 0
while time.perf_counter() < t_end:
    t1 = time.perf_counter()
    raw = rx.rx()
    t_after_rx = time.perf_counter()
    buf = np.asarray(raw, dtype="complex64")
    # Loop the one big 100k read out into STREAM_CHUNK-sized pieces for
    # rx_streaming() -- NOT one big call (would violate its 2048-sample cap).
    results = []
    for start in range(0, len(buf), STREAM_CHUNK):
        piece = buf[start:start + STREAM_CHUNK]
        results.append(ofdm_rx.rx_streaming(piece[None, :]))
    t2 = time.perf_counter()

    n_calls += 1
    rx_call_us.append((t_after_rx - t1) * 1e6)
    decode_us.append((t2 - t_after_rx) * 1e6)
    if (t2 - t1) * 1e6 > BUDGET_US:
        n_over_budget += 1
    for result in results:
        if result is not None:
            n_decoded += 1

rx_arr = np.array(rx_call_us)
dec_arr = np.array(decode_us)
total_samples_seen = n_calls * RX_SAMPLES
coverage_pct = 100 * total_samples_seen / (args.seconds * args.rate)

print(f"\nloop iterations: {n_calls}  over {BUDGET_US:.0f}us budget: {n_over_budget} "
      f"({100 * n_over_budget / max(n_calls, 1):.2f}%)")
print(f"  rx.rx() alone:     min={rx_arr.min():8.1f}us  median={np.median(rx_arr):8.1f}us  "
      f"mean={rx_arr.mean():8.1f}us  max={rx_arr.max():8.1f}us")
print(f"  rx_streaming() x2: min={dec_arr.min():8.1f}us  median={np.median(dec_arr):8.1f}us  "
      f"mean={dec_arr.mean():8.1f}us  max={dec_arr.max():8.1f}us")
print(f"  samples actually looked at: {total_samples_seen} of {int(args.seconds*args.rate)} "
      f"({coverage_pct:.2f}% real-time coverage)")
print(f"  frames completed decode (no TX active, so expected ~0): {n_decoded}")
