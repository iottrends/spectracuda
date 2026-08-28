"""Air unit with a live curses dashboard -- same protocol, PHY config,
Mac object, and bind/watchdog/quality-report logic as
examples/drone_air_unit.py, reused UNCHANGED via direct import. Same
reuse pattern examples/pluto_air_unit.py already established (swapping
drone_air_unit.py's transport out from under it via a module-level
monkeypatch, without duplicating any of its protocol logic) -- here the
thing being swapped is presentation, not transport: drone_air_unit.py's
bare `print()`-to-stdout event log becomes Dashboard.log() (see
`air.print = dashboard.log` below), and its blocking `sys.stdin`
readline sender is replaced by dashboard.py's non-blocking curses input
line. Everything else -- the Mac object, PHY_KWARGS, the bind handshake,
_quality_report_loop, _heartbeat_watchdog_loop, _recv_one_chunk_and_
stream_decode, _handle_decoded_pdu, _send_chunks -- is drone_air_unit.py's
own code, called directly, not copied.

The one place this file duplicates a single decode_header() call rather
than threading a callback through drone_air_unit.py: _receive_loop below
peeks at each decoded PDU's type itself (cheap -- pure bit indexing, no
DSP) so it can hand LINK_QUALITY reports to the dashboard's structured
"peer quality" panel, then still calls air._handle_decoded_pdu()
unchanged for the actual dispatch (which decodes the header a second
time internally) -- a redundant decode, not redundant logic.

Run against ground_unit.py in this same folder, over the ZMQ loopback
(default) or a real Pluto:
    python examples/drone_tui/ground_unit.py
    python examples/drone_tui/air_unit.py [--ground-ip 127.0.0.1]

    python examples/drone_tui/ground_unit.py --transport pluto --uri ip:192.168.3.1 --tx-freq 2.40e9 --rx-freq 2.41e9
    python examples/drone_tui/air_unit.py    --transport pluto --uri ip:192.168.2.1 --tx-freq 2.41e9 --rx-freq 2.40e9

--transport pluto: all radio-specific code lives in pluto.py (this
folder) -- pluto_init() configures the device once, pluto_tx()/
pluto_rx() replace the ZMQ push/pull calls everywhere below. See
pluto.py's own module docstring for why it's simpler than examples/
pluto_channel.py's PlutoChannel (no background thread/queue needed --
this file already has its own single receive loop calling pluto_rx()
synchronously). `_PlutoSocket` just below is a ~5-line local adapter
so drone_air_unit.py's reused _send_chunks()/_recv_one_chunk_and_
stream_decode() (which expect ZMQ-shaped .send(bytes)/.recv()->bytes)
can be pointed at pluto.py's plain functions without pluto.py itself
needing to know about that interface -- pluto.py only ever exposes
pluto_init/pluto_tx/pluto_rx, by design.

Ctrl-C to quit (curses.wrapper restores the terminal first).
"""
from __future__ import annotations

import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # examples/ -- sibling of this folder
import drone_air_unit as air  # noqa: E402
from spectracuda.mac import Mac  # noqa: E402
from spectracuda.mac.pdu import HEADER_LEN_BITS, TYPE_DATA, TYPE_LINK_QUALITY, decode_header  # noqa: E402
from spectracuda.mac.quality import decode_quality_report  # noqa: E402

from adaptive_mcs import MCS_TABLE, McsController  # noqa: E402  -- same folder
from commands import CommandDispatcher  # noqa: E402  -- same folder
from dashboard import Dashboard  # noqa: E402  -- same folder, examples/ isn't on sys.path for this one
from stats import RateTracker  # noqa: E402  -- same folder

LOCAL_QUALITY_POLL_S = 0.2  # how often the dashboard's own "local RX quality"/rate panels refresh


class _PlutoSocket:
    """See module docstring -- duck-types just enough of ZMQ's socket
    interface (.send(bytes)/.recv()->bytes) for drone_air_unit.py's
    _send_chunks()/_recv_one_chunk_and_stream_decode() to keep working
    unchanged against pluto.py's plain pluto_tx()/pluto_rx() functions.
    Lives here, not in pluto.py -- pluto.py owns the radio, this is
    transport glue local to this file."""

    def __init__(self, handle) -> None:
        self._handle = handle

    def send(self, raw_bytes: bytes) -> None:
        from pluto import pluto_tx  # local import -- only needed when --transport pluto is actually used

        pluto_tx(self._handle, np.frombuffer(raw_bytes, dtype="complex64"))

    def recv(self) -> bytes:
        from pluto import pluto_rx  # local import -- see send()'s comment

        return pluto_rx(self._handle).tobytes()


def _pluto_send_frame(push_socket, iq_frame, push_lock) -> None:
    """Replaces air._send_chunks when --transport pluto -- one whole
    frame, one push_socket.send() (== one pluto_tx() call), never
    chunked (see pluto.py's module docstring for why real RF needs the
    whole frame in one burst). Same push_lock protection as the ZMQ
    path's _send_chunks()."""
    samples = np.asarray(iq_frame)[0]
    with push_lock:
        push_socket.send(samples.astype("complex64").tobytes())


def _setup_transport(args):
    """-> (push, pull) socket-like objects (.send(bytes)/.recv()->bytes),
    ZMQ or Pluto depending on args.transport. Also monkeypatches
    air._send_chunks for the Pluto case -- see _pluto_send_frame()'s
    docstring for why that one function specifically needs different
    behavior for real RF (matches examples/pluto_air_unit.py's own
    monkeypatch, same reasoning)."""
    if args.transport == "pluto":
        from pluto import pluto_init

        handle = pluto_init(
            uri=args.uri, tx_freq=args.tx_freq, rx_freq=args.rx_freq, rate=args.rate,
            tx_gain=args.tx_gain, rx_gain=args.rx_gain, agc=args.agc, rx_buffer_size=args.rx_buffer_size,
        )
        air._send_chunks = _pluto_send_frame
        return _PlutoSocket(handle), _PlutoSocket(handle)

    import zmq

    ctx = zmq.Context()
    pull = ctx.socket(zmq.PULL)
    pull.connect(f"tcp://{args.ground_ip}:{air.GROUND_TO_AIR_PORT}")
    push = ctx.socket(zmq.PUSH)
    push.connect(f"tcp://{args.ground_ip}:{air.AIR_TO_GROUND_PORT}")
    return push, pull


def _receive_loop(mac, pull, push, push_lock, heartbeat, bound_event, dashboard, rx_rate, mcs) -> None:
    while True:
        bits = air._recv_one_chunk_and_stream_decode(pull, mac)
        if bits is None:
            continue
        header = decode_header(bits[:HEADER_LEN_BITS])
        if header["pdu_type"] == TYPE_DATA:
            rx_rate.record(len(bits))  # DATA only -- see stats.RateTracker's docstring for why heartbeats are excluded
        elif header["pdu_type"] == TYPE_LINK_QUALITY:
            report = decode_quality_report(bits)
            dashboard.set_peer_quality(report)
            if mcs is not None:
                # This is the PEER's report of how OUR own outgoing frames
                # have been landing -- exactly the signal adaptive MCS needs
                # to adjust OUR transmit scheme (see adaptive_mcs.py's own
                # docstring for why: the peer, not us, can see our loss/
                # success rate). set_tx_scheme() mutates mac.ofdm's modem/
                # packetizer -- same push_lock as every other generate_frame()
                # caller in this file, since a concurrent send could be
                # mid-generate_frame() against those same attributes.
                new_modem = mcs.on_quality_report(report)
                if new_modem is not None:
                    with push_lock:
                        mac.set_tx_scheme(modem=new_modem)
                    dashboard.set_tx_scheme(new_modem)
                    dashboard.log(f"[air] adaptive MCS: switched to {new_modem}")
        air._handle_decoded_pdu("air", mac, bits, push, heartbeat, push_lock)
        if mac.bound and not bound_event.is_set():
            dashboard.log("[air] bound -- type a message and press enter to send it")
            bound_event.set()


def _send_message(mac, push, push_lock, dashboard, tx_rate, text: str, quiet: bool = False) -> None:
    """The one real send path -- shared by manual dashboard input
    (on_submit, below) and TrafficGenerator's synthetic load (commands.py/
    traffic.py), so both go through mac.send_iq()/air._send_chunks()
    exactly once, not two copies of the same three lines.

    `mac.send_iq(bits)` itself -- not just air._send_chunks() -- is
    inside `with push_lock:` here: a real segfault was hit running a
    sustained ~100 pkt/s /traffic run concurrently with
    air._quality_report_loop's own generate_frame() calls, both driving
    the native Viterbi/RS C encoder on the SAME shared mac.ofdm instance
    from different threads with no serialization between them. See
    drone_air_unit.py's _send_chunks() docstring for the full writeup --
    push_lock is a threading.RLock() here for the same reason (see
    run(), below): air._send_chunks() re-acquires it internally."""
    bits = np.unpackbits(np.frombuffer(text.encode("utf-8"), dtype="uint8"))
    with push_lock:
        for iq_frame in mac.send_iq(bits):
            air._send_chunks(push, iq_frame, push_lock)
    tx_rate.record(len(bits))
    if not quiet:
        dashboard.log(f"[air] sent: {text!r}")


def _make_traffic_send_fn(mac, push, bound_event, push_lock, dashboard, tx_rate):
    """Passed to TrafficGenerator as its send_fn -- returns False (a
    drop, not an error) instead of sending while unbound, same gating
    on_submit applies to a manually typed line, just without logging
    every single drop (TrafficGenerator already logs a periodic
    sent/dropped summary -- see traffic.py)."""

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
            dashboard.log(f"[air] not bound yet -- dropped: {line!r}")
            return
        _send_message(mac, push, push_lock, dashboard, tx_rate, line)

    return on_submit


def _local_quality_poll_loop(mac, dashboard, rx_rate, tx_rate, interval_s=LOCAL_QUALITY_POLL_S) -> None:
    """Not fed by the receive loop's own per-frame observations directly
    (those already flow into mac.quality via _recv_one_chunk_and_stream_
    decode's mac.quality.observe() call, unchanged) -- this just polls
    the resulting running aggregate on a timer so the dashboard's "local
    RX quality" panel updates smoothly even between arrivals, and so
    dashboard.set_bound() still reflects a watchdog-triggered rebind
    (mac.bound flipping False) even when nothing is currently arriving
    to receive. Also where rx_rate/tx_rate's windowed rate_bps() gets
    sampled into the dashboard -- record() happens at actual send/receive
    time (see _send_message/_receive_loop), this just reads the result
    on the same cadence as everything else here."""
    while True:
        time.sleep(interval_s)
        dashboard.set_local_quality(mac.quality.report_dict())
        dashboard.set_bound(mac.bound)
        dashboard.set_rx_rate(rx_rate.rate_bps())
        dashboard.set_tx_rate(tx_rate.rate_bps())


def run(args) -> None:
    dashboard = Dashboard("air")
    air.print = dashboard.log  # route drone_air_unit.py's own prints (bind/quality/watchdog messages) into the log pane -- see module docstring

    push, pull = _setup_transport(args)

    air_mac = Mac(mode="um", ofdm_kwargs=air.PHY_KWARGS)
    air_mac.ofdm.reset_stream()
    if args.transport == "pluto":
        dashboard.log(f"[air] Pluto ready: tx_freq={args.tx_freq/1e9:.3f}GHz rx_freq={args.rx_freq/1e9:.3f}GHz "
                      f"rate={args.rate/1e6:.2f}MSPS; max_segment_bits={air_mac.max_segment_bits}")
    else:
        dashboard.log(f"[air] connected to ground at {args.ground_ip}; max_segment_bits={air_mac.max_segment_bits}")
    dashboard.log("[air] waiting for ground to bind...")

    bound_event = threading.Event()
    heartbeat = {"received_count": 0}
    push_lock = threading.RLock()  # RLock, not Lock -- see _send_message()'s docstring
    rx_rate = RateTracker()
    tx_rate = RateTracker()

    mcs = None
    if args.adaptive_mcs:
        # start_index matches air.PHY_KWARGS's own modem -- so the
        # controller's notion of "current" agrees with air_mac.ofdm's
        # ACTUAL starting scheme from the very first LINK_QUALITY report,
        # rather than defaulting to McsController's own start_index=1
        # ("qpsk") regardless of what this PHY was actually configured for.
        mcs = McsController(start_index=MCS_TABLE.index(air.PHY_KWARGS["modem"]))
        dashboard.set_tx_scheme(mcs.current_modem)
        dashboard.log(f"[air] adaptive MCS enabled, starting at {mcs.current_modem}")

    threading.Thread(target=air._quality_report_loop, args=("air", air_mac, push, bound_event, push_lock), daemon=True).start()
    threading.Thread(target=air._heartbeat_watchdog_loop, args=("air", air_mac, bound_event, heartbeat), daemon=True).start()
    threading.Thread(
        target=_receive_loop, args=(air_mac, pull, push, push_lock, heartbeat, bound_event, dashboard, rx_rate, mcs), daemon=True
    ).start()
    threading.Thread(target=_local_quality_poll_loop, args=(air_mac, dashboard, rx_rate, tx_rate), daemon=True).start()

    dispatcher = CommandDispatcher(
        "air", _make_traffic_send_fn(air_mac, push, bound_event, push_lock, dashboard, tx_rate), dashboard.log
    )
    dashboard.log("[air] type a message to send it, or /help for commands (e.g. /traffic start rate=5 size=64)")
    dashboard.run(_make_sender(air_mac, push, bound_event, push_lock, dashboard, dispatcher, tx_rate))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transport", choices=["zmq", "pluto"], default="zmq")
    parser.add_argument("--ground-ip", default="127.0.0.1", help="[zmq] ground unit's address (default: 127.0.0.1 for local testing)")
    parser.add_argument("--uri", default="ip:192.168.2.1", help="[pluto] device URI")
    parser.add_argument("--tx-freq", type=float, default=2.41e9, help="[pluto] Hz -- must equal ground's --rx-freq")
    parser.add_argument("--rx-freq", type=float, default=2.40e9, help="[pluto] Hz -- must equal ground's --tx-freq")
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
