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
    # Root cause of this script's own [SLOW] rx_streaming() stalls (up to
    # ~1.6s, no TX active at all): a false sync detection decodes noise
    # into a random-but-registered fec0, which (12 of 17 registered
    # schemes are LDPC) usually lands on an LDPC variant, and
    # constructing it (GF(2) matrix inversion) is what actually stalls.
    # strict_fec_check=True rejects any decoded fec0/fec1 that isn't
    # this receiver's own configured rs_m8/conv_v27 BEFORE constructing
    # a codec for it -- see Ofdm.__init__'s own comment.
    strict_fec_check=True,
)
RX_SAMPLES = 100_000  # read big (amortize the per-call daemon overhead the
# online sources + pluto_air_unit.py's own comment both point at), then loop
# around and feed 2048-sample pieces to rx_streaming() (see below) -- NOT a
# single big rx_streaming() call, which would violate its 2048-sample
# SEEKING-state search cap for this fft=256 config.
STREAM_CHUNK = 2048

ap = argparse.ArgumentParser()
ap.add_argument("--uri-rx", required=True)
ap.add_argument("--freq", type=float, default=2.425e9)
ap.add_argument("--rate", type=float, default=5e6)
ap.add_argument("--rx-gain", type=float, default=60.0)
ap.add_argument("--seconds", type=float, default=10.0)
args = ap.parse_args()

BUDGET_US = RX_SAMPLES / args.rate * 1e6  # e.g. 20000us at 5Msps, 25000us at 4Msps

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

SLOW_ITER_THRESHOLD_US = 2 * BUDGET_US  # flag+diagnose any iteration this far over budget

t_end = time.perf_counter() + args.seconds
n_calls = 0
while time.perf_counter() < t_end:
    t1 = time.perf_counter()
    raw = rx.rx()
    t_after_rx = time.perf_counter()
    buf = np.asarray(raw, dtype="complex64")
    # Loop the one big 100k read out into STREAM_CHUNK-sized pieces for
    # rx_streaming() -- NOT one big call (would violate its 2048-sample cap).
    # Per-sub-call timing kept alongside (worst_sub_us/idx) purely to
    # localize a slow outer iteration down to one of the ~49 sub-calls,
    # for the SLOW-iteration diagnostic below -- not part of the normal
    # rx_call_us/decode_us stats.
    results = []
    n_sub_calls = 0
    worst_sub_us = 0.0
    worst_sub_idx = -1
    for start in range(0, len(buf), STREAM_CHUNK):
        piece = buf[start:start + STREAM_CHUNK]
        t_sub0 = time.perf_counter()
        results.append(ofdm_rx.rx_streaming(piece[None, :]))
        sub_us = (time.perf_counter() - t_sub0) * 1e6
        if sub_us > worst_sub_us:
            worst_sub_us = sub_us
            worst_sub_idx = n_sub_calls
        n_sub_calls += 1
    t2 = time.perf_counter()

    decode_total_us = (t2 - t_after_rx) * 1e6
    if decode_total_us > SLOW_ITER_THRESHOLD_US:
        print(f"  [SLOW] iteration #{n_calls} (0-indexed): decode_total={decode_total_us:.0f}us "
              f"(budget={BUDGET_US:.0f}us) -- worst sub-call #{worst_sub_idx}/{n_sub_calls} "
              f"took {worst_sub_us:.0f}us"
              + (" <-- FIRST iteration, looks like a cold-start cost" if n_calls == 0 else ""))

    n_calls += 1
    rx_call_us.append((t_after_rx - t1) * 1e6)
    decode_us.append(decode_total_us)
    if (t2 - t1) * 1e6 > BUDGET_US:
        n_over_budget += 1
    for result in results:
        if result is not None:
            n_decoded += 1

rx_arr = np.array(rx_call_us)
dec_arr = np.array(decode_us)
total_samples_seen = n_calls * RX_SAMPLES
coverage_pct = 100 * total_samples_seen / (args.seconds * args.rate)
# Fixed per-read count: RX_SAMPLES/STREAM_CHUNK rounded up (last piece is a
# shorter remainder), same every iteration since rx.rx() always hands back
# exactly RX_SAMPLES samples.
sub_calls_per_read = -(-RX_SAMPLES // STREAM_CHUNK)
total_sub_calls = n_calls * sub_calls_per_read

print(f"\n=== SUMMARY over {args.seconds:.1f}s ===")
print(f"rx.rx() calls done (each a {RX_SAMPLES}-sample / 100k read): {n_calls}")
print(f"  time per rx.rx() call:      min={rx_arr.min():8.1f}us  median={np.median(rx_arr):8.1f}us  "
      f"mean={rx_arr.mean():8.1f}us  max={rx_arr.max():8.1f}us")
print(f"rx_streaming() sub-calls per 100k read: {sub_calls_per_read}  "
      f"({RX_SAMPLES} samples / {STREAM_CHUNK}-sample chunks, rounded up)")
print(f"  time per 100k read's worth of rx_streaming() calls: "
      f"min={dec_arr.min():8.1f}us  median={np.median(dec_arr):8.1f}us  "
      f"mean={dec_arr.mean():8.1f}us  max={dec_arr.max():8.1f}us")
print(f"Total rx_streaming() sub-calls over the full run: "
      f"{n_calls} x {sub_calls_per_read} = {total_sub_calls}")
print(f"Iterations over the {BUDGET_US:.0f}us per-read budget: {n_over_budget} "
      f"({100 * n_over_budget / max(n_calls, 1):.2f}%)")
print(f"Samples actually looked at: {total_samples_seen} of {int(args.seconds*args.rate)} "
      f"({coverage_pct:.2f}% real-time coverage)")
print(f"Frames completed decode (no TX active, so expected ~0): {n_decoded}")
