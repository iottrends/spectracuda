"""First real-hardware test of spectracuda's own TX/RX chain (Mac.send_iq()
/ Ofdm.rx_process()) through an actual Pluto -- not the ofdm-hls/sim
project's separate reference chain (~/work/ofdm-hls/sim/ofdm_reference.py),
which was only ever used to prove the Pluto RF path itself works (BIST +
RF loopback, last week).

Reuses that proven hardware I/O layer DIRECTLY (imported, not re-derived --
`configure_radio()`/`capture_with_pluto()`/`normalize_for_dac()`/
`normalize_from_adc()` from pluto_loopback.py, verbatim), since re-guessing
pyadi-iio gain/buffer/cyclic-TX settings blind would risk breaking what
already works. Only the TX-burst-generation and RX-decode calls are
swapped: spectracuda's Mac.send_iq()/Ofdm.rx_process() instead of
ofdm_reference.generate()/decode_full().

Smallest possible step, matching this project's own "correctness before
trusting a number" rule, extended to real hardware: --mode pluto-bist
first (AD9363 internal digital loopback, RF front-end bypassed entirely --
exactly what was already validated last week) before --mode pluto-rf or
--mode two-pluto. One frame, one round trip, bit-exact check -- nothing
about throughput/Msps is measured here, that's a separate, later step
once a single frame demonstrably survives real hardware at all.

Requires: pyadi-iio installed, and ~/work/ofdm-hls/sim on this machine
(only for its proven pluto_loopback.py helpers -- change SIM_DIR below if
it lives elsewhere).

Usage:
    # BIST first (no RF front-end, matches what's already proven):
    python3 examples/pluto_spectracuda_loopback.py --mode pluto-bist --uri ip:192.168.2.1

    # Once that passes, RF self-loopback:
    python3 examples/pluto_spectracuda_loopback.py --mode pluto-rf --uri ip:192.168.2.1 \\
        --freq 2.4e9 --rate 20e6 --tx-gain -30 --rx-gain 40

    # Two Plutos, real OTA link:
    python3 examples/pluto_spectracuda_loopback.py --mode two-pluto \\
        --uri-tx ip:192.168.2.1 --uri-rx ip:192.168.3.1 --freq 2.4e9 --rate 20e6
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from spectracuda.mac import Mac  # noqa: E402

SIM_DIR = os.path.expanduser("~/work/ofdm-hls/sim")
if not os.path.isdir(SIM_DIR):
    sys.exit(
        f"error: {SIM_DIR} not found -- this script reuses pluto_loopback.py's "
        f"already-proven hardware I/O helpers from there. Edit SIM_DIR at the "
        f"top of this file if your ofdm-hls checkout lives somewhere else."
    )
sys.path.insert(0, SIM_DIR)
from pluto_loopback import (  # noqa: E402
    capture_with_pluto, configure_radio, estimate_rx_dbfs, normalize_for_dac,
    normalize_from_adc, rx_antenna_dbm, tx_power_dbm,
)

# Matches drone_air_unit.py's PHY_KWARGS (fft=256, rs_m8+conv_v27) -- the
# project's own target config -- minus the interleaver (not needed for a
# single-frame proof) and pinned to backend="numpy" (no GPU needed for one
# frame). qpsk here, not qam16, to keep the FIRST hardware round-trip as
# simple/robust as possible -- bump to qam16 once this passes.
PHY_KWARGS = dict(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32, modem="qpsk",
    fec="rs_m8", fec1="conv_v27", crc="crc16",
    sync="schmidl_cox", cfo="schmidl_cox",
    channel_estimator="ls", equalizer="mmse",
    backend="numpy",
)

SDU_BITS = 2000  # small and deliberate: must fit in exactly ONE PDU/frame
                  # (send_iq() would otherwise return multiple frames, and
                  # this script's whole point is proving ONE frame survives
                  # real hardware, not multi-frame reassembly yet)


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--mode", choices=["pluto-bist", "pluto-rf", "two-pluto"], default="pluto-bist")
    ap.add_argument("--uri", default="ip:192.168.2.1")
    ap.add_argument("--uri-tx", default=None)
    ap.add_argument("--uri-rx", default=None)
    ap.add_argument("--freq", type=float, default=2.4e9)
    ap.add_argument("--rate", type=float, default=20e6)
    ap.add_argument("--tx-gain", type=float, default=-30.0)
    ap.add_argument("--rx-gain", type=float, default=40.0)
    ap.add_argument("--agc", choices=["manual", "slow_attack", "fast_attack", "hybrid"], default="manual")
    ap.add_argument("--rx-samples", type=int, default=200_000)
    ap.add_argument("--settle", type=float, default=0.05)
    ap.add_argument("--flush", type=int, default=2)
    ap.add_argument("--lead-zeros", type=int, default=2 * (PHY_KWARGS["fft_size"] + PHY_KWARGS["cp_len"]))
    ap.add_argument("--trail-zeros", type=int, default=2 * (PHY_KWARGS["fft_size"] + PHY_KWARGS["cp_len"]))
    args = ap.parse_args()

    mac = Mac(mode="um", ofdm_kwargs=PHY_KWARGS)
    mac.bound = True  # never binds to a real peer here -- see benchmark_x86_stages_v2.py's own precedent
    rng = np.random.default_rng(0)
    sdu = rng.integers(0, 2, size=SDU_BITS).astype("uint8")

    frames = mac.send_iq(sdu)
    if len(frames) != 1:
        sys.exit(
            f"error: SDU_BITS={SDU_BITS} produced {len(frames)} frames, expected exactly 1 "
            f"-- reduce SDU_BITS so it fits in one PDU before running this script"
        )
    iq_frame = np.asarray(frames[0])[0]  # (1, N) -> (N,), generate_frame()'s own batch dim
    print(f"[spectracuda] TX frame: {len(iq_frame)} samples ({len(iq_frame) / args.rate * 1e3:.2f} ms @ {args.rate/1e6:.1f} MSPS)")

    burst = np.concatenate([
        np.zeros(args.lead_zeros, dtype=np.complex64),
        iq_frame.astype(np.complex64),
        np.zeros(args.trail_zeros, dtype=np.complex64),
    ])

    print(f"[pluto] mode={args.mode}  freq={args.freq/1e9:.3f} GHz  rate={args.rate/1e6:.2f} MSPS  "
          f"tx_gain={args.tx_gain:+.1f} dB  rx_gain={args.rx_gain:+.1f} dB ({args.agc})")

    rx_samples = capture_with_pluto(args, burst)
    rssi_dbfs = estimate_rx_dbfs(rx_samples)
    rssi_dbm = rx_antenna_dbm(rssi_dbfs, args.rx_gain)
    p_tx_dbm = tx_power_dbm(args.tx_gain)
    print(f"[pluto] RX captured: {len(rx_samples)} samples")
    print(f"[link]  TX power ~ {p_tx_dbm:+.1f} dBm   RX dBFS = {rssi_dbfs:+.1f}   RX dBm ~ {rssi_dbm:+.1f}")
    if rssi_dbfs > -3:
        print("[pluto] WARN: RX appears to be clipping -- back off TX or RX gain")
    elif rssi_dbfs < -50:
        print("[pluto] WARN: RX very weak -- consider increasing TX or RX gain")

    # Same cross-correlation alignment technique as pluto_loopback.py's own
    # capture_with_pluto() caller -- the cyclic-TX capture lands at a random
    # phase, so find the burst's start via matched-filter correlation against
    # a known chunk of the burst we transmitted, rather than assuming offset 0.
    template = burst[:5000].astype(np.complex128)
    corr = np.abs(np.correlate(rx_samples.astype(np.complex128), template, mode="valid"))
    max_valid_offset = len(rx_samples) - len(burst)
    offset = int(np.argmax(corr[: max_valid_offset + 1])) if max_valid_offset >= 0 else 0
    rx_one = rx_samples[offset : offset + len(burst)]
    if len(rx_one) < len(burst):
        print(f"[pluto] WARN: RX capture ended mid-burst ({len(rx_one)}/{len(burst)} samples) -- increase --rx-samples")

    if args.mode == "pluto-bist":
        rx_scaled = rx_one * (16.0 / (2 ** 13))  # deterministic BIST tap-point scale, matches pluto_loopback.py
    else:
        rx_scaled = normalize_from_adc(rx_one)

    rx_unpadded = rx_scaled[args.lead_zeros : args.lead_zeros + len(iq_frame)]
    print(f"[pluto] aligned at offset={offset}  peak |rx| = {np.max(np.abs(rx_unpadded)):.4f}")

    # mac.receive_iq(), NOT ofdm.rx_process() directly -- rx_process()
    # returns the raw PDU bits (32-bit MAC header + SDU, see
    # spectracuda/mac/pdu.py's HEADER_LEN_BITS), and receive_iq() is the
    # real API that strips that header and does the actual PDU->SDU
    # extraction. An earlier version of this script called rx_process()
    # directly and hand-sliced [:SDU_BITS] from bit 0 -- comparing the
    # header+payload against the payload alone, which produced a ~49%
    # bit-mismatch rate that looked like a real decode failure (EVM was
    # actually ~0 and CRC was valid) until caught by a digital dry-run
    # before ever touching real hardware.
    print("[spectracuda] decoding via Mac.receive_iq() ...")
    delivered = mac.receive_iq(rx_unpadded[None, :].astype(np.complex64))

    if not delivered:
        print("[spectracuda] FAIL: no SDU delivered (sync miss, FEC/CRC failure, or reassembly gap)")
        sys.exit(3)
    decoded_sdu = np.asarray(delivered[0])
    bit_exact = np.array_equal(decoded_sdu, sdu)
    print(f"[spectracuda] bit_exact={bit_exact}")
    if bit_exact:
        print("[spectracuda] SUCCESS: real Pluto hardware round-trip, bit-exact match")
    else:
        n_err = int(np.sum(decoded_sdu != sdu)) if len(decoded_sdu) == len(sdu) else -1
        print(f"[spectracuda] FAIL: decoded SDU mismatch (bit errors: {n_err if n_err >= 0 else 'length mismatch'})")
        sys.exit(5)


if __name__ == "__main__":
    main()
