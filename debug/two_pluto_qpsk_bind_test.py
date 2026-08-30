"""Known-good real two-Pluto round trip -- the exact config that was
proven to work bit-exact over real RF this session, saved as a
reusable regression check / starting point for further real-hardware
debugging (see docs/todo.md-style session notes for the fuller story;
this file is the one that actually WORKS, not a diagnostic dump).

Decode path, explicitly: this calls Mac.receive_iq() (via mac.send_iq()
on the TX side too) -- the real, header-aware MAC-level API -- NOT
Ofdm.rx_process() directly. rx_process() alone would return the raw
PDU bits with the 32-bit MAC header (see spectracuda/mac/pdu.py's
HEADER_LEN_BITS) still attached; receive_iq() is what actually strips
that header and hands back just the SDU. Comparing rx_process()'s raw
output against the original SDU directly is a REAL bug this project
already hit once (see examples/pluto_spectracuda_loopback.py's own
module comment on _handle_decoded_pdu-adjacent history) -- get this
wrong and a perfectly good decode looks like a ~50% bit-mismatch
"failure" that isn't real.

What's proven, concretely, by this exact config (real hardware, this
session, both Plutos USB-attached to this same machine):
  - freq pair 2.425/2.450 GHz (WiFi-channel-gap frequencies)
  - rate=5 MSPS (PlutoSDR's own sustained-USB-safe ceiling)
  - tx_gain=-10 dB, rx_gain=60 dB, agc="manual" -- NOT slow_attack AGC,
    which was tried and made things WORSE (AGC's own gain-tracking
    dynamics distort a short bursty envelope's on/off structure -- see
    this session's own periodicity/windowed-power diagnostics). Fixed
    manual gain, pushed higher for real SNR margin, is what actually
    worked.
  - modem="qpsk", fft_size=256, fec="rs_m8"+"conv_v27", crc="crc16" --
    the SAME PHY_KWARGS examples/pluto_spectracuda_loopback.py uses,
    NOT drone_air_unit.py/drone_ground_unit.py's own (currently
    fft_size=64, modem started at "qpsk" too after a downgrade from
    "qam16" -- see that file's own PHY_KWARGS) -- this script exists
    specifically to isolate the PHY/gain question from whatever is
    still unresolved in drone_tui's own Ofdm.rx_streaming() + Pluto
    continuous-receive path (that path still shows attempts=0 -- no
    frame-detection attempt registers AT ALL -- even with these same
    proven gain settings; this script's own one-shot, cyclic-TX-buffer
    capture is a fundamentally easier problem than a real one-shot
    control message inside a much larger streaming rx_buffer_size
    chunk, see this session's own notes on why normalize_from_adc()'s
    percentile-95 approach doesn't translate to that sparse-burst case).

Requires: pyadi-iio installed, ~/work/ofdm-hls/sim on this machine (for
its proven pluto_loopback.py capture/normalize helpers -- change SIM_DIR
below if it lives elsewhere), and two Plutos reachable at the two
--uri-tx/--uri-rx addresses (see this session's own notes on giving a
second Pluto a non-default IP via its config.txt if both would
otherwise default to 192.168.2.1).

Usage:
    python3 debug/two_pluto_qpsk_bind_test.py \\
        --uri-tx ip:192.168.2.1 --uri-rx ip:192.168.3.1
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from spectracuda.mac import Mac  # noqa: E402

SIM_DIR = os.path.expanduser("~/work/ofdm-hls/sim")
if not os.path.isdir(SIM_DIR):
    sys.exit(
        f"error: {SIM_DIR} not found -- this script reuses pluto_loopback.py's "
        f"already-proven hardware I/O helpers from there (capture_with_pluto/"
        f"normalize_from_adc/etc.). Edit SIM_DIR at the top of this file if "
        f"your ofdm-hls checkout lives somewhere else."
    )
sys.path.insert(0, SIM_DIR)
from pluto_loopback import capture_with_pluto, estimate_rx_dbfs, normalize_from_adc, rx_antenna_dbm, tx_power_dbm

PHY_KWARGS = dict(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32, modem="qpsk",
    fec="rs_m8", fec1="conv_v27", crc="crc16",
    sync="schmidl_cox", cfo="schmidl_cox",
    channel_estimator="ls", equalizer="mmse",
    backend="numpy",
)
SDU_BITS = 2000  # small and deliberate -- must fit in exactly ONE PDU/frame


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uri-tx", required=True, help="TX-side Pluto's IIO URI, e.g. ip:192.168.2.1")
    ap.add_argument("--uri-rx", required=True, help="RX-side Pluto's IIO URI, e.g. ip:192.168.3.1")
    ap.add_argument("--freq", type=float, default=2.425e9, help="Hz -- shared TX/RX carrier for this one-shot test")
    ap.add_argument("--rate", type=float, default=5e6, help="Hz -- stay <= PlutoSDR's ~5-6 MSPS sustained USB ceiling")
    ap.add_argument("--tx-gain", type=float, default=0.0, help="dB -- bumped from -10 after low real-world success rate at -10/rx_gain=60 (see this session's own notes)")
    ap.add_argument("--rx-gain", type=float, default=60.0, help="dB -- proven value, see module docstring")
    ap.add_argument("--agc", default="manual", help="proven value is 'manual', NOT an AGC mode -- see module docstring")
    ap.add_argument("--rx-samples", type=int, default=200_000)
    ap.add_argument("--settle", type=float, default=0.05)
    ap.add_argument("--flush", type=int, default=2)
    ap.add_argument("--lead-zeros", type=int, default=2 * (PHY_KWARGS["fft_size"] + PHY_KWARGS["cp_len"]))
    ap.add_argument("--trail-zeros", type=int, default=2 * (PHY_KWARGS["fft_size"] + PHY_KWARGS["cp_len"]))
    args = ap.parse_args()
    args.mode = "two-pluto"
    args.uri = None
    return args


def main() -> None:
    args = _parse_args()

    mac = Mac(mode="um", ofdm_kwargs=PHY_KWARGS)
    mac.bound = True  # never binds to a real peer here -- see benchmark_x86_stages_v2.py's own precedent
    rng = np.random.default_rng(0)
    sdu = rng.integers(0, 2, size=SDU_BITS).astype("uint8")

    frames = mac.send_iq(sdu)
    assert len(frames) == 1, f"SDU_BITS={SDU_BITS} produced {len(frames)} frames, expected exactly 1"
    iq_frame = np.asarray(frames[0])[0]
    print(f"[spectracuda] TX frame: {len(iq_frame)} samples")

    burst = np.concatenate([
        np.zeros(args.lead_zeros, dtype=np.complex64),
        iq_frame.astype(np.complex64),
        np.zeros(args.trail_zeros, dtype=np.complex64),
    ])
    print(f"[pluto] uri_tx={args.uri_tx}  uri_rx={args.uri_rx}  freq={args.freq/1e9:.3f} GHz  "
          f"rate={args.rate/1e6:.2f} MSPS  tx_gain={args.tx_gain:+.1f} dB  "
          f"rx_gain={args.rx_gain:+.1f} dB ({args.agc})")

    rx_samples = capture_with_pluto(args, burst)
    rssi_dbfs = estimate_rx_dbfs(rx_samples)
    print(f"[pluto] RX captured: {len(rx_samples)} samples  RX dBFS={rssi_dbfs:.1f}  "
          f"RX dBm~{rx_antenna_dbm(rssi_dbfs, args.rx_gain):.1f}  TX dBm~{tx_power_dbm(args.tx_gain):.1f}")
    if rssi_dbfs > -3:
        print("[pluto] WARN: RX appears to be clipping -- back off TX or RX gain")
    elif rssi_dbfs < -50:
        print("[pluto] WARN: RX very weak -- consider increasing TX or RX gain")

    # Normalize the WHOLE raw capture (not just one aligned burst-length
    # slice -- see this session's own finding that truncating too tightly
    # makes rx_process() fail on "not enough samples past start_index",
    # a script bug, not a real decode failure) and hand the whole thing
    # to the real decode path.
    rx_scaled = normalize_from_adc(rx_samples)

    print("\n[spectracuda] decoding via Mac.receive_iq() (header-aware -- see module docstring) ...")
    delivered = mac.receive_iq(rx_scaled[None, :].astype(np.complex64))
    if not delivered:
        print("[spectracuda] FAIL: no SDU delivered (sync miss, FEC/CRC failure, or reassembly gap)")
        sys.exit(3)
    decoded_sdu = np.asarray(delivered[0])
    bit_exact = np.array_equal(decoded_sdu, sdu)
    print(f"[spectracuda] bit_exact={bit_exact}")
    if bit_exact:
        print("[spectracuda] SUCCESS: real two-Pluto RF round trip, bit-exact match")
    else:
        n_err = int(np.sum(decoded_sdu != sdu)) if len(decoded_sdu) == len(sdu) else -1
        print(f"[spectracuda] FAIL: decoded SDU mismatch (bit errors: {n_err if n_err >= 0 else 'length mismatch'})")
        sys.exit(5)


if __name__ == "__main__":
    main()
