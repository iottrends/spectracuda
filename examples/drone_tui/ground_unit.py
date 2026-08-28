"""Ground unit with the same live curses dashboard as air_unit.py in
this folder -- mirrors that file's design exactly (see its module
docstring for the full reuse rationale), reusing drone_ground_unit.py's
own Mac/bind/watchdog/quality-report logic unchanged. The one asymmetry,
carried over from drone_ground_unit.py itself: ground is the bind
INITIATOR and binds both ZeroMQ sockets (fixed, known address); air
connects out to it.

Run against air_unit.py in this same folder, over the ZMQ loopback
(default) or a real Pluto:
    python examples/drone_tui/ground_unit.py [--bind-ip *]
    python examples/drone_tui/air_unit.py

    python examples/drone_tui/ground_unit.py --transport pluto --uri ip:192.168.3.1 --tx-freq 2.40e9 --rx-freq 2.41e9
    python examples/drone_tui/air_unit.py    --transport pluto --uri ip:192.168.2.1 --tx-freq 2.41e9 --rx-freq 2.40e9

See air_unit.py's module docstring for the full --transport pluto
design writeup (pluto.py owns the radio, this file only gains a flag
and a small local adapter).

Ctrl-C to quit (curses.wrapper restores the terminal first).
"""
from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # examples/ -- sibling of this folder
import drone_ground_unit as ground  # noqa: E402
from spectracuda.mac import Mac  # noqa: E402
from spectracuda.mac.pdu import HEADER_LEN_BITS, TYPE_DATA, TYPE_LINK_QUALITY, decode_header  # noqa: E402
from spectracuda.mac.quality import decode_quality_report  # noqa: E402

from adaptive_mcs import MCS_TABLE, McsController  # noqa: E402  -- same folder
from commands import CommandDispatcher  # noqa: E402  -- same folder
from dashboard import Dashboard  # noqa: E402  -- same folder
from stats import RateTracker  # noqa: E402  -- same folder

LOCAL_QUALITY_POLL_S = 0.2  # see air_unit.py's identical constant/loop for the full rationale


class _PlutoSocket:
    """See air_unit.py's identical class for the full writeup."""

    def __init__(self, handle) -> None:
        self._handle = handle

    def send(self, raw_bytes: bytes) -> None:
        from pluto import pluto_tx

        pluto_tx(self._handle, np.frombuffer(raw_bytes, dtype="complex64"))

    def recv(self) -> bytes:
        from pluto import pluto_rx

        return pluto_rx(self._handle).tobytes()


def _pluto_send_frame(push_socket, iq_frame, push_lock) -> None:
    """See air_unit.py's identical function for the full writeup."""
    samples = np.asarray(iq_frame)[0]
    with push_lock:
        push_socket.send(samples.astype("complex64").tobytes())


def _setup_transport(args):
    """See air_unit.py's identical function for the full writeup. Ground
    BINDS both ZMQ sockets (fixed address) rather than connecting out --
    the one asymmetry vs air_unit.py's version, carried over from
    drone_ground_unit.py itself."""
    if args.transport == "pluto":
        from pluto import pluto_init

        handle = pluto_init(
            uri=args.uri, tx_freq=args.tx_freq, rx_freq=args.rx_freq, rate=args.rate,
            tx_gain=args.tx_gain, rx_gain=args.rx_gain, agc=args.agc, rx_buffer_size=args.rx_buffer_size,
        )
        ground._send_chunks = _pluto_send_frame
        return _PlutoSocket(handle), _PlutoSocket(handle)

    import zmq

    ctx = zmq.Context()
    push = ctx.socket(zmq.PUSH)
    push.bind(f"tcp://{args.bind_ip}:{ground.GROUND_TO_AIR_PORT}")
    pull = ctx.socket(zmq.PULL)
    pull.bind(f"tcp://{args.bind_ip}:{ground.AIR_TO_GROUND_PORT}")
    return push, pull


def _receive_loop(mac, pull, push, push_lock, heartbeat, bound_event, dashboard, rx_rate, mcs) -> None:
    while True:
        bits = ground._recv_one_chunk_and_stream_decode(pull, mac)
        if bits is None:
            continue
        header = decode_header(bits[:HEADER_LEN_BITS])
        if header["pdu_type"] == TYPE_DATA:
            rx_rate.record(len(bits))  # DATA only -- see stats.RateTracker's docstring for why heartbeats are excluded
        elif header["pdu_type"] == TYPE_LINK_QUALITY:
            report = decode_quality_report(bits)
            dashboard.set_peer_quality(report)
            if mcs is not None:
                # See air_unit.py's identical branch for the full writeup.
                new_modem = mcs.on_quality_report(report)
                if new_modem is not None:
                    with push_lock:
                        mac.set_tx_scheme(modem=new_modem)
                    dashboard.set_tx_scheme(new_modem)
                    dashboard.log(f"[ground] adaptive MCS: switched to {new_modem}")
        ground._handle_decoded_pdu("ground", mac, bits, push, heartbeat, push_lock)
        if mac.bound and not bound_event.is_set():
            dashboard.log("[ground] bound -- type a message and press enter to send it")
            bound_event.set()


def _send_message(mac, push, push_lock, dashboard, tx_rate, text: str, quiet: bool = False) -> None:
    """See air_unit.py's identical function for the full writeup, incl.
    why mac.send_iq() itself is inside `with push_lock:` here (a real
    segfault, traced to a native-encoder race between this and
    ground._quality_report_loop's own generate_frame() calls)."""
    bits = np.unpackbits(np.frombuffer(text.encode("utf-8"), dtype="uint8"))
    with push_lock:
        for iq_frame in mac.send_iq(bits):
            ground._send_chunks(push, iq_frame, push_lock)
    tx_rate.record(len(bits))
    if not quiet:
        dashboard.log(f"[ground] sent: {text!r}")


def _make_traffic_send_fn(mac, push, bound_event, push_lock, dashboard, tx_rate):
    """See air_unit.py's identical function for the full writeup."""

    def send_fn(text: str) -> bool:
        if not bound_event.is_set():
            return False
        _send_message(mac, push, push_lock, dashboard, tx_rate, text, quiet=True)
        return True

    return send_fn


def _make_sender(mac, push, bound_event, push_lock, dashboard, dispatcher, tx_rate):
    def on_submit(line: str) -> None:
        line = line.strip()
        if not line:
            return
        if dispatcher.is_command(line):
            dispatcher.dispatch(line)
            return
        if not bound_event.is_set():
            dashboard.log(f"[ground] not bound yet -- dropped: {line!r}")
            return
        _send_message(mac, push, push_lock, dashboard, tx_rate, line)

    return on_submit


def _local_quality_poll_loop(mac, dashboard, rx_rate, tx_rate, interval_s=LOCAL_QUALITY_POLL_S) -> None:
    """See air_unit.py's identical function for the full writeup."""
    while True:
        time.sleep(interval_s)
        dashboard.set_local_quality(mac.quality.report_dict())
        dashboard.set_bound(mac.bound)
        dashboard.set_rx_rate(rx_rate.rate_bps())
        dashboard.set_tx_rate(tx_rate.rate_bps())


def run(args) -> None:
    dashboard = Dashboard("ground")
    ground.print = dashboard.log  # route drone_ground_unit.py's own prints into the log pane -- see air_unit.py's module docstring

    push, pull = _setup_transport(args)

    ground_mac = Mac(mode="um", ofdm_kwargs=ground.PHY_KWARGS)
    ground_mac.ofdm.reset_stream()
    if args.transport == "pluto":
        dashboard.log(f"[ground] Pluto ready: tx_freq={args.tx_freq/1e9:.3f}GHz rx_freq={args.rx_freq/1e9:.3f}GHz "
                      f"rate={args.rate/1e6:.2f}MSPS; max_segment_bits={ground_mac.max_segment_bits}")
    else:
        dashboard.log(
            f"[ground] bound to {args.bind_ip}:{ground.GROUND_TO_AIR_PORT}/{ground.AIR_TO_GROUND_PORT}; "
            f"max_segment_bits={ground_mac.max_segment_bits}"
        )

    bound_event = threading.Event()
    heartbeat = {"received_count": 0}
    push_lock = threading.RLock()  # RLock, not Lock -- see _send_message()'s docstring
    rx_rate = RateTracker()
    tx_rate = RateTracker()

    mcs = None
    if args.adaptive_mcs:
        # See air_unit.py's identical block for why start_index is derived
        # from PHY_KWARGS's own modem, not McsController's own default.
        mcs = McsController(start_index=MCS_TABLE.index(ground.PHY_KWARGS["modem"]))
        dashboard.set_tx_scheme(mcs.current_modem)
        dashboard.log(f"[ground] adaptive MCS enabled, starting at {mcs.current_modem}")

    threading.Thread(target=ground._quality_report_loop, args=("ground", ground_mac, push, bound_event, push_lock), daemon=True).start()
    threading.Thread(
        target=ground._heartbeat_watchdog_loop,
        args=("ground", ground_mac, push, push_lock, bound_event, heartbeat), daemon=True,
    ).start()
    threading.Thread(
        target=_receive_loop, args=(ground_mac, pull, push, push_lock, heartbeat, bound_event, dashboard, rx_rate, mcs), daemon=True
    ).start()
    threading.Thread(target=_local_quality_poll_loop, args=(ground_mac, dashboard, rx_rate, tx_rate), daemon=True).start()

    # Ground initiates the ONE bind exchange this link needs -- see
    # drone_ground_unit.py's own module docstring for why one exchange is
    # enough for both sides. Re-triggered later by _heartbeat_watchdog_loop
    # if the link ever goes quiet -- same helper, same code path.
    dashboard.log("[ground] sending bind request...")
    ground._send_bind_request(ground_mac, push, push_lock)

    dispatcher = CommandDispatcher(
        "ground", _make_traffic_send_fn(ground_mac, push, bound_event, push_lock, dashboard, tx_rate), dashboard.log
    )
    dashboard.log("[ground] type a message to send it, or /help for commands (e.g. /traffic start rate=5 size=64)")
    dashboard.run(_make_sender(ground_mac, push, bound_event, push_lock, dashboard, dispatcher, tx_rate))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transport", choices=["zmq", "pluto"], default="zmq")
    parser.add_argument("--bind-ip", default="*", help="[zmq] address to bind on (default: * = all interfaces)")
    parser.add_argument("--uri", default="ip:192.168.3.1", help="[pluto] device URI")
    parser.add_argument("--tx-freq", type=float, default=2.40e9, help="[pluto] Hz -- must equal air's --rx-freq")
    parser.add_argument("--rx-freq", type=float, default=2.41e9, help="[pluto] Hz -- must equal air's --tx-freq")
    parser.add_argument("--rate", type=float, default=5e6, help="[pluto] Hz -- stay <= PlutoSDR's ~5-6 MSPS sustained USB ceiling")
    parser.add_argument("--tx-gain", type=float, default=-10.0, help="[pluto]")
    parser.add_argument("--rx-gain", type=float, default=40.0, help="[pluto]")
    parser.add_argument("--agc", choices=["manual", "slow_attack", "fast_attack", "hybrid"], default="manual", help="[pluto]")
    parser.add_argument(
        "--adaptive-mcs", dest="adaptive_mcs", action="store_true", default=True,
        help="adjust our own outgoing modem scheme based on the peer's LINK_QUALITY reports (default: on)",
    )
    parser.add_argument("--no-adaptive-mcs", dest="adaptive_mcs", action="store_false", help="disable adaptive MCS")
    parser.add_argument("--rx-buffer-size", type=int, default=200_000, help="[pluto]")
    args = parser.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        pass  # curses.wrapper has already restored the terminal by the time this is reached
