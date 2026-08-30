"""Minimal Ofdm-only (no Mac) rx_streaming() test over real Pluto hardware.
Two threads: TX sends one-shot bursts on a period; RX+decode is merged in
one loop (read -> scale -> rx_streaming() x2 -> check), timed per iteration
against the ~820us a 4096-sample read should take at 5Msps.

Usage:
    python3 debug/pluto_ofdm_streaming_test.py --uri-tx ip:192.168.2.1 --uri-rx ip:192.168.3.1 --n 30 --period 0.1
"""
import argparse
import sys
import threading
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
PAYLOAD_BITS = 2000
RX_SAMPLES = 4096
STREAM_CHUNK = 2048  # cap for fft=256 is 2048 (STREAM_SEARCH_WINDOW_SYMBOLS*fft_size)
BUDGET_US = RX_SAMPLES / 5e6 * 1e6  # 820us at 5Msps

ap = argparse.ArgumentParser()
ap.add_argument("--uri-tx", required=True)
ap.add_argument("--uri-rx", required=True)
ap.add_argument("--freq", type=float, default=2.425e9)
ap.add_argument("--rate", type=float, default=5e6)
ap.add_argument("--tx-gain", type=float, default=0.0)
ap.add_argument("--rx-gain", type=float, default=60.0)
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--period", type=float, default=0.1)
args = ap.parse_args()

rf_bw = int(max(args.rate * 1.25, 5e6))

tx = adi.Pluto(uri=args.uri_tx)
tx.sample_rate = int(args.rate)
tx.tx_lo = int(args.freq)
tx.tx_rf_bandwidth = rf_bw
tx.tx_hardwaregain_chan0 = args.tx_gain

rx = adi.Pluto(uri=args.uri_rx)
rx.sample_rate = int(args.rate)
rx.rx_lo = int(args.freq)
rx.rx_rf_bandwidth = rf_bw
rx.gain_control_mode_chan0 = "manual"
rx.rx_hardwaregain_chan0 = args.rx_gain
rx.rx_buffer_size = RX_SAMPLES

ofdm_tx = Ofdm(**PHY_KWARGS)
ofdm_rx = Ofdm(**PHY_KWARGS)
ofdm_rx.reset_stream()

payload_bits = np.random.default_rng(0).integers(0, 2, size=PAYLOAD_BITS).astype("uint8")
iq_frame = np.asarray(ofdm_tx.generate_frame(payload_bits))[0].astype(np.complex64)
lead = trail = 2 * (PHY_KWARGS["fft_size"] + PHY_KWARGS["cp_len"])
burst = np.concatenate([np.zeros(lead, dtype=np.complex64), iq_frame, np.zeros(trail, dtype=np.complex64)])
peak = np.max(np.abs(burst))
scaled_burst = (burst * (2 ** 13 / peak)).astype(np.complex64)

# One-time calibration: cyclic capture of a DIFFERENT payload (so a stale
# leftover read is distinguishable from a real decode), derive FIXED_SCALE.
cal_payload = np.random.default_rng(999).integers(0, 2, size=PAYLOAD_BITS).astype("uint8")
cal_frame = np.asarray(ofdm_tx.generate_frame(cal_payload))[0].astype(np.complex64)
cal_burst = np.concatenate([np.zeros(lead, dtype=np.complex64), cal_frame, np.zeros(trail, dtype=np.complex64)])
cal_peak = np.max(np.abs(cal_burst))
cal_scaled = (cal_burst * (2 ** 13 / cal_peak)).astype(np.complex64)

rx.rx_buffer_size = 100_000
tx.tx_cyclic_buffer = True
tx.tx(cal_scaled)
time.sleep(0.05)
for _ in range(2):
    rx.rx()
cal_capture = np.asarray(rx.rx(), dtype="complex64")
tx.tx_destroy_buffer()
tx.tx_cyclic_buffer = False
ref = float(np.percentile(np.abs(cal_capture), 95))
FIXED_SCALE = 0.5 / ref
print(f"FIXED_SCALE={FIXED_SCALE:.5f}")

rx.rx_buffer_size = RX_SAMPLES
for _ in range(5):  # drain leftover backlog from the 100k calibration capture
    rx.rx()

stop = threading.Event()
lock = threading.Lock()
n_ok = n_decoded = n_calls = n_over_budget = 0


rx_call_us = []   # time inside rx.rx() alone, one entry per loop pass
decode_us = []    # time inside the two rx_streaming() calls alone, one entry per loop pass


def rx_decode_loop():
    global n_ok, n_decoded, n_calls, n_over_budget
    while not stop.is_set():
        t1 = time.perf_counter()
        raw = rx.rx()
        t_after_rx = time.perf_counter()
        buf = np.asarray(raw, dtype="complex64") * FIXED_SCALE
        result1 = ofdm_rx.rx_streaming(buf[:STREAM_CHUNK][None, :])
        result2 = ofdm_rx.rx_streaming(buf[STREAM_CHUNK:][None, :])
        t2 = time.perf_counter()
        with lock:
            n_calls += 1
            rx_call_us.append((t_after_rx - t1) * 1e6)
            decode_us.append((t2 - t_after_rx) * 1e6)
            if (t2 - t1) * 1e6 > BUDGET_US:
                n_over_budget += 1
            for result in (result1, result2):
                if result is not None:
                    n_decoded += 1
                    decoded = np.asarray(result.get("bits")).reshape(-1)[:PAYLOAD_BITS]
                    crc_ok = bool(np.asarray(result.get("crc_valid"))[0])
                    ok = crc_ok and np.array_equal(decoded, payload_bits)
                    if ok:
                        n_ok += 1
                    print(f"  [decode] crc_valid={crc_ok} bit_match={ok}  n_ok={n_ok} n_decoded={n_decoded}")


def tx_loop():
    for i in range(args.n):
        tx.tx(scaled_burst)
        print(f"[{i:3d}] TX sent")
        time.sleep(args.period)


t_rx = threading.Thread(target=rx_decode_loop, daemon=True)
t_rx.start()
tx_loop()
time.sleep(0.5)
stop.set()
t_rx.join(timeout=5.0)

rx_arr = np.array(rx_call_us)
dec_arr = np.array(decode_us)
print(f"\nloop iterations: {n_calls}  over {BUDGET_US:.0f}us budget: {n_over_budget} "
      f"({100 * n_over_budget / max(n_calls, 1):.2f}%)")
if len(rx_arr):
    print(f"  rx.rx() alone:     min={rx_arr.min():8.1f}us  median={np.median(rx_arr):8.1f}us  "
          f"mean={rx_arr.mean():8.1f}us  max={rx_arr.max():8.1f}us")
    print(f"  rx_streaming() x2: min={dec_arr.min():8.1f}us  median={np.median(dec_arr):8.1f}us  "
          f"mean={dec_arr.mean():8.1f}us  max={dec_arr.max():8.1f}us")
print(f"FINAL: {args.n} trials, {n_decoded} decoded, {n_ok} ok ({100 * n_ok / args.n:.1f}%)")
