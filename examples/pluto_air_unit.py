"""Air unit over a REAL Pluto -- internally calls drone_air_unit.py's own
Mac/bind/watchdog/quality-report logic UNCHANGED (imported directly, not
duplicated), with only the ZMQ transport swapped for pluto_channel.py's
real-hardware one. See pluto_channel.py's module docstring for the one
real behavioral difference this requires (drone_air_unit._send_chunks is
monkeypatched to send a whole frame as one tx() burst, not small looped
sends -- everything else is drone_air_unit.py's own code, verbatim).

FDD: this air unit's --tx-freq must equal the ground unit's --rx-freq,
and this air unit's --rx-freq must equal the ground unit's --tx-freq
(two independent RF channels, same idea as the two TCP ports the ZMQ
version used).

First bring-up should still go through examples/pluto_spectracuda_
loopback.py's --mode pluto-bist / pluto-rf on EACH Pluto individually
before running this two-node link -- this script assumes the hardware
path itself already works (per your own BIST/RF-loopback testing), it
does not re-verify that.

Usage:
    python3 examples/pluto_air_unit.py --uri ip:192.168.2.1 \\
        --tx-freq 2.41e9 --rx-freq 2.40e9 --rate 5e6 --tx-gain -10
"""
from __future__ import annotations

import argparse
import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import drone_air_unit as air  # noqa: E402
from pluto_channel import PlutoChannel  # noqa: E402
from spectracuda.mac import Mac  # noqa: E402

# rx_streaming() has ~40-150us of near-fixed per-call overhead (measured
# directly), almost independent of chunk size -- CHUNK_SIZE=64 (drone_air_
# unit.py's own constant) sustains only ~0.4 Msps, far below any real
# Pluto rate; a real mocked-hardware dry run of this exact link hung on
# this before it was found. 2048 sustains well over 20 Msps at the same
# per-call overhead, comfortably above the ~5-6 Msps PlutoSDR USB ceiling.
# Deliberately NOT air.CHUNK_SIZE -- that constant is fine for the ZMQ demo
# (never needed real throughput) but would silently strangle a real link.
PLUTO_STREAM_CHUNK_SIZE = 2048


def _pluto_send_frame(push_socket, iq_frame, push_lock):
    """Replaces air._send_chunks -- see pluto_channel.py's module
    docstring for why real RF needs the whole frame in ONE tx() call,
    not chunked. Same push_lock protection as the original (still real:
    multiple threads -- stdin sender, quality reporter, bind-reply --
    must not step on each other's tx() calls either)."""
    samples = np.asarray(iq_frame)[0]  # (1, N) -> (N,), generate_frame()'s own batch dim
    with push_lock:
        push_socket.send(samples.astype("complex64").tobytes())


def run(args) -> None:
    air._send_chunks = _pluto_send_frame  # see module docstring -- the one real behavior change

    channel = PlutoChannel(
        uri=args.uri, tx_freq=args.tx_freq, rx_freq=args.rx_freq, rate=args.rate,
        tx_gain=args.tx_gain, rx_gain=args.rx_gain, agc=args.agc,
        rx_buffer_size=args.rx_buffer_size, chunk_size=PLUTO_STREAM_CHUNK_SIZE,
    )

    air_mac = Mac(mode="um", ofdm_kwargs=air.PHY_KWARGS)
    air_mac.ofdm.reset_stream()
    print(f"[air] Pluto ready: tx_freq={args.tx_freq/1e9:.3f}GHz rx_freq={args.rx_freq/1e9:.3f}GHz "
          f"rate={args.rate/1e6:.2f}MSPS; max_segment_bits={air_mac.max_segment_bits}")
    print("[air] waiting for ground to bind...")

    bound_event = threading.Event()
    heartbeat = {"received_count": 0}
    push_lock = threading.Lock()

    threading.Thread(target=air._stdin_send_loop, args=("air", air_mac, channel.tx_socket, bound_event, push_lock), daemon=True).start()
    threading.Thread(target=air._quality_report_loop, args=("air", air_mac, channel.tx_socket, bound_event, push_lock), daemon=True).start()
    threading.Thread(target=air._heartbeat_watchdog_loop, args=("air", air_mac, bound_event, heartbeat), daemon=True).start()

    while True:
        bits = air._recv_one_chunk_and_stream_decode(channel.rx_socket, air_mac)
        if bits is not None:
            air._handle_decoded_pdu("air", air_mac, bits, channel.tx_socket, heartbeat, push_lock)
            if air_mac.bound and not bound_event.is_set():
                print("[air] bound -- type a message and press enter to send it")
                bound_event.set()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uri", default="ip:192.168.2.1")
    ap.add_argument("--tx-freq", type=float, default=2.41e9, help="Hz -- must equal ground's --rx-freq")
    ap.add_argument("--rx-freq", type=float, default=2.40e9, help="Hz -- must equal ground's --tx-freq")
    ap.add_argument("--rate", type=float, default=5e6, help="Hz -- stay <= PlutoSDR's ~5-6 MSPS sustained USB ceiling")
    ap.add_argument("--tx-gain", type=float, default=-10.0)
    ap.add_argument("--rx-gain", type=float, default=40.0)
    ap.add_argument("--agc", choices=["manual", "slow_attack", "fast_attack", "hybrid"], default="manual")
    ap.add_argument("--rx-buffer-size", type=int, default=200_000)
    run(ap.parse_args())
