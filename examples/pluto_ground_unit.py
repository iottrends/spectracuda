"""Ground unit over a REAL Pluto -- see pluto_air_unit.py's module
docstring for the shared design (internally calls drone_ground_unit.py's
own Mac/bind/watchdog/quality-report logic unchanged, only _send_chunks
is monkeypatched for real RF).

FDD: this ground unit's --tx-freq must equal the air unit's --rx-freq,
and this ground unit's --rx-freq must equal the air unit's --tx-freq.

Usage:
    python3 examples/pluto_ground_unit.py --uri ip:192.168.3.1 \\
        --tx-freq 2.40e9 --rx-freq 2.41e9 --rate 5e6 --tx-gain -10
"""
from __future__ import annotations

import argparse
import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import drone_ground_unit as ground  # noqa: E402
from pluto_channel import PlutoChannel  # noqa: E402
from spectracuda.mac import Mac  # noqa: E402

# See pluto_air_unit.py's identical constant for the full writeup --
# CHUNK_SIZE=64 (ground.CHUNK_SIZE, the ZMQ demo's own constant) sustains
# only ~0.4 Msps against rx_streaming()'s measured per-call overhead;
# 2048 sustains well over 20 Msps at the same overhead.
PLUTO_STREAM_CHUNK_SIZE = 2048


def _pluto_send_frame(push_socket, iq_frame, push_lock):
    """See pluto_air_unit.py's identical function for the full writeup."""
    samples = np.asarray(iq_frame)[0]
    with push_lock:
        push_socket.send(samples.astype("complex64").tobytes())


def run(args) -> None:
    ground._send_chunks = _pluto_send_frame

    channel = PlutoChannel(
        uri=args.uri, tx_freq=args.tx_freq, rx_freq=args.rx_freq, rate=args.rate,
        tx_gain=args.tx_gain, rx_gain=args.rx_gain, agc=args.agc,
        rx_buffer_size=args.rx_buffer_size, chunk_size=PLUTO_STREAM_CHUNK_SIZE,
    )

    ground_mac = Mac(mode="um", ofdm_kwargs=ground.PHY_KWARGS)
    ground_mac.ofdm.reset_stream()
    print(f"[ground] Pluto ready: tx_freq={args.tx_freq/1e9:.3f}GHz rx_freq={args.rx_freq/1e9:.3f}GHz "
          f"rate={args.rate/1e6:.2f}MSPS; max_segment_bits={ground_mac.max_segment_bits}")

    bound_event = threading.Event()
    heartbeat = {"received_count": 0}
    push_lock = threading.Lock()

    threading.Thread(target=ground._stdin_send_loop, args=("ground", ground_mac, channel.tx_socket, bound_event, push_lock), daemon=True).start()
    threading.Thread(target=ground._quality_report_loop, args=("ground", ground_mac, channel.tx_socket, bound_event, push_lock), daemon=True).start()
    threading.Thread(
        target=ground._heartbeat_watchdog_loop,
        args=("ground", ground_mac, channel.tx_socket, push_lock, bound_event, heartbeat), daemon=True,
    ).start()

    print("[ground] sending bind request...")
    ground._send_bind_request(ground_mac, channel.tx_socket, push_lock)

    while True:
        bits = ground._recv_one_chunk_and_stream_decode(channel.rx_socket, ground_mac)
        if bits is not None:
            ground._handle_decoded_pdu("ground", ground_mac, bits, channel.tx_socket, heartbeat, push_lock)
            if ground_mac.bound and not bound_event.is_set():
                print("[ground] bound -- type a message and press enter to send it")
                bound_event.set()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uri", default="ip:192.168.2.1")
    ap.add_argument("--tx-freq", type=float, default=2.40e9, help="Hz -- must equal air's --rx-freq")
    ap.add_argument("--rx-freq", type=float, default=2.41e9, help="Hz -- must equal air's --tx-freq")
    ap.add_argument("--rate", type=float, default=5e6, help="Hz -- stay <= PlutoSDR's ~5-6 MSPS sustained USB ceiling")
    ap.add_argument("--tx-gain", type=float, default=-10.0)
    ap.add_argument("--rx-gain", type=float, default=40.0)
    ap.add_argument("--agc", choices=["manual", "slow_attack", "fast_attack", "hybrid"], default="manual")
    ap.add_argument("--rx-buffer-size", type=int, default=200_000)
    run(ap.parse_args())
