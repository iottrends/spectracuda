"""Air unit -- one of two independently runnable processes (the other is
drone_ground_unit.py, not yet built) meant to run on two separate
machines. Connects out to a known ground-station address; ground binds,
air connects -- matching how a real ground control station (fixed,
known IP) and a mobile air unit actually relate.

Model, settled after simplifying the earlier design (see docs/mac.md):
- ONE Mac object on each side (not four) -- this Mac is used for BOTH
  sending its own traffic and decoding the peer's, exactly the pattern
  examples/mac_half_duplex_demo.py already proved works. UM mode, no
  AM/STATUS complexity.
- Exactly TWO ZeroMQ channels total: ground_to_air (ground PUSH -> air
  PULL) and air_to_ground (air PUSH -> ground PULL). Nothing else.
- The bind handshake is NOT a separate mechanism -- it's just more
  messages on these same two channels. Ground initiates
  (build_bind_request()); air evaluates it and replies
  (handle_bind_request_iq() -- which ALSO sets air's OWN .bound as a
  side effect of evaluating the request, so one exchange sets both
  sides' .bound, same as mac_half_duplex_demo.py's own _bind() helper).
- Every arrived frame -- bind request, bind response, or ordinary data
  -- goes through the exact same path: Ofdm.rx_streaming() (chunked,
  exercising the real streaming receiver, not batch rx_process()), then
  a peek at the decoded header's pdu_type to decide what to do with it.
  Mac.receive_iq() can't be used here since it assumes every arrived
  frame is DATA -- same reason the bidirectional-AM streaming demo
  needed its own manual dispatch.

Config: fft=256, 16-QAM, rs_m8(fec0)+conv_v27(fec1)+block interleaver --
the exact config that needed the shortened-Reed-Solomon fix (see
docs/mac.md) to even complete a bind handshake at all.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import zmq

from spectracuda.backend import default_backend
from spectracuda.mac import Mac
from spectracuda.mac.bind import (
    decode_bind_request,
    decode_bind_response,
    encode_bind_response,
    evaluate_bind_request,
)
from spectracuda.mac.pdu import (
    HEADER_LEN_BITS,
    TYPE_BIND_REQUEST,
    TYPE_BIND_RESPONSE,
    TYPE_DATA,
    TYPE_LINK_QUALITY,
    decode_header,
)
from spectracuda.mac.quality import decode_quality_report, encode_quality_report


def _to_host(ofdm, arr):
    """arr may genuinely be a cupy.ndarray (whenever ofdm.backend ==
    "cupy") -- plain np.asarray() raises on that (CuPy disallows
    implicit conversion). Same real bug/fix as Mac._rx_one_frame()
    (spectracuda/mac/mac.py) -- caught on a real Colab Tesla T4 run
    (2026-08-25)."""
    if ofdm.backend == "cupy":
        import cupy

        return cupy.asnumpy(arr)
    return np.asarray(arr)

PHY_KWARGS = dict(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32, modem="qam16",
    fec="rs_m8", fec1="conv_v27",
    interleaver="block", interleaver_kwargs={"unit_bits": 8},
    crc="crc16", sync="schmidl_cox", cfo="schmidl_cox",
    n_training_symbols=2, backend=default_backend(),  # "cupy" if a working CUDA runtime is present, else "numpy"
)

GROUND_TO_AIR_PORT = 5555  # ground PUSH -> air PULL (also carries the bind request)
AIR_TO_GROUND_PORT = 5556  # air PUSH -> ground PULL (also carries the bind response)

CHUNK_SIZE = 64  # arbitrary -- rx_streaming() doesn't require any particular size/alignment
LINK_QUALITY_INTERVAL_S = 0.1  # 100ms -- DSP-derived only (see module docstring), not a hardware poll rate
HEARTBEAT_MISS_LIMIT = 5  # 5 consecutive missed LINK_QUALITY reports (~500ms of silence) -> link presumed down
# NOTE: air never initiates a bind (ground always does, see module
# docstring) -- so unlike drone_ground_unit.py, there's no
# _send_bind_request() here. Air's watchdog only clears its own .bound
# and waits for a fresh BIND_REQUEST to arrive, same as it did at startup.


def _send_chunks(push_socket, iq_frame, push_lock):
    """One generate_frame() output -> a sequence of CHUNK_SIZE-sample
    ZeroMQ messages, exactly the same chunking mac_streaming_demo.py's
    _stream_deliver() does in-process, just pushed over a real socket
    instead of handed directly to rx_streaming() in a for-loop.

    `push_lock` is mandatory, not optional -- a real, genuinely observed
    bug found while testing the heartbeat watchdog below: THREE threads
    (stdin-sender, quality-reporter, and the watchdog's re-bind) can all
    call this function on the SAME push socket. Without serializing each
    call's ENTIRE multi-chunk frame, two threads' chunks can interleave
    on the wire (thread A's chunk 1, thread B's chunk 1, thread A's
    chunk 2, ...) -- rx_streaming() on the receiving end has no way to
    tell that apart from one badly corrupted frame, since it just
    expects one continuous stream of a single frame's chunks at a time.
    Symptom actually observed: after a real kill-and-restart-air test,
    the link kept re-triggering the watchdog's "link presumed down"
    path every ~500ms indefinitely instead of stabilizing, even though
    air was genuinely alive and sending -- traced to exactly this race
    (the quality-reporter and the watchdog's re-bind send landing at
    close to the same moment, corrupting both).

    A SECOND, more severe race in the same family, found later (via a
    sustained ~100 pkt/s synthetic-traffic run, examples/drone_tui/
    traffic.py, segfaulting the process after a few thousand frames):
    `push_lock` here only ever protected the SOCKET WRITE, not the
    `mac.ofdm.generate_frame()` call that produces `iq_frame` in the
    first place. Every caller below (`_handle_decoded_pdu`'s bind-
    response branch, `_stdin_send_loop`, `_quality_report_loop`) calls
    generate_frame() on the SAME shared `mac.ofdm` instance from its own
    thread -- and generate_frame() drives the native Viterbi/RS C
    encoder structs (ctypes, GIL released for the call's duration,
    exactly like the native DECODE path documented in Mac.
    receive_iq_batch()'s own module docstring), which are NOT safe for
    two threads to be inside concurrently. `push_lock` being held only
    around `_send_chunks()` left a window where thread A could be mid-
    generate_frame() while thread B entered it too, corrupting both --
    silent most of the time, a segfault once unlucky enough. FIX: every
    caller below now holds `push_lock` around generate_frame() AND
    _send_chunks() together, as one critical section -- which is why
    `push_lock` must be a threading.RLock() (see run(), below), not a
    plain Lock: _send_chunks() re-acquires it internally, and a plain
    Lock would deadlock on that nested acquisition from the same
    thread."""
    samples = np.asarray(iq_frame)[0]  # (1, N) -> (N,), generate_frame()'s own batch dim
    with push_lock:
        for i in range(0, samples.shape[-1], CHUNK_SIZE):
            push_socket.send(samples[i : i + CHUNK_SIZE].tobytes())


def _recv_one_chunk_and_stream_decode(pull_socket, mac):
    """Blocks for exactly one incoming chunk, feeds it to
    mac.ofdm.rx_streaming(). Returns decoded raw PDU bits the instant a
    complete frame finishes (which pdu_type it is isn't known yet --
    see _handle_decoded_pdu() below), else None (still accumulating, OR
    the frame that just completed failed its CRC -- see below).

    Also feeds mac.quality (LinkQualityTracker) from every completed
    frame, success or failure -- a real gap fixed here, not present when
    this function was first written: without this, build_quality_report()
    would always report on an empty tracker (n_attempts=0), since nothing
    was ever calling .observe(). Same rssi_db/evm/delivered accounting
    Mac._rx_one_frame() uses internally for the batch (rx_process())
    path -- this is that same logic, written out for the streaming path,
    which doesn't go through Mac._rx_one_frame() at all."""
    raw = pull_socket.recv()
    chunk = np.frombuffer(raw, dtype="complex64")
    result = mac.ofdm.rx_streaming(chunk)
    if result is None:
        return None
    crc_valid = result["crc_valid"]
    delivered = crc_valid is None or bool(_to_host(mac.ofdm, crc_valid)[0])
    evm = result["evm"]
    mac.quality.observe(
        rssi_db=float(_to_host(mac.ofdm, result["rssi_db"])[0]),
        evm=None if evm is None else float(_to_host(mac.ofdm, evm)[0]),
        delivered=delivered,
    )
    if not delivered:
        # A real bug this also fixes: the original version returned decoded
        # bits unconditionally, even when crc_valid was False -- handing a
        # corrupted-but-structurally-decoded frame to _handle_decoded_pdu()
        # as if it were trustworthy. Now it's dropped, same as every other
        # streaming demo in this repo already does.
        return None
    return _to_host(mac.ofdm, result["bits"])[0].astype("uint8")


def _handle_decoded_pdu(label, mac, bits, reply_push_socket, heartbeat, push_lock):
    """One decoded PDU, of whatever type it turns out to be, dispatched
    by peeking at its header -- the same "decode first, THEN find out
    what kind of message it was" pattern the bidirectional-AM streaming
    demo already established, needed here for the same reason: bind and
    data share the same two wire channels.

    `heartbeat` is a single-key dict ({"received_count": int}), shared
    with _heartbeat_watchdog_loop() below -- incremented here every time
    a LINK_QUALITY pdu genuinely arrives FROM THE PEER, which is the
    signal the watchdog watches for silence on. A dict (not a plain int)
    specifically so it's a mutable object multiple threads can share
    without needing a `nonlocal`/global -- same reasoning `bound_event`
    already uses a shared object for."""
    header = decode_header(bits[:HEADER_LEN_BITS])

    if header["pdu_type"] == TYPE_BIND_REQUEST:
        request = decode_bind_request(bits)
        decision = evaluate_bind_request(request, local_max_segment_bits=mac.max_segment_bits)
        mac.bound = decision["accepted"]  # the real accept/reject decision, evaluated locally
        response_pdu = encode_bind_response(decision)
        with push_lock:  # generate_frame() + _send_chunks() as one critical section -- see _send_chunks()'s docstring
            response_iq = mac.ofdm.generate_frame(response_pdu[None, :])
            _send_chunks(reply_push_socket, response_iq, push_lock)
        print(f"[{label}] BIND_REQUEST received, evaluated -- bound={mac.bound}")

    elif header["pdu_type"] == TYPE_BIND_RESPONSE:
        response = decode_bind_response(bits)
        mac.bound = response["accepted"]
        print(f"[{label}] BIND_RESPONSE received -- bound={mac.bound}")

    elif header["pdu_type"] == TYPE_DATA:
        for sdu_bits in mac.receive(bits):  # UM's own reassembly -- may be empty (segment pending)
            text = np.packbits(sdu_bits).tobytes().decode("utf-8", errors="replace")
            print(f"[{label}] received: {text!r}")

    elif header["pdu_type"] == TYPE_LINK_QUALITY:
        heartbeat["received_count"] += 1
        report = decode_quality_report(bits)
        print(
            f"[{label}] LINK_QUALITY from peer: rssi={report['mean_rssi_db']:.1f}dB "
            f"evm={report['mean_evm']:.4f} delivered={report['delivered_ratio']:.1%} "
            f"({report['n_delivered']}/{report['n_attempts']})"
        )


def _stdin_send_loop(label, mac, push_socket, bound_event, push_lock):
    """Runs in its own thread (stdin.readline() blocks) -- sends whatever
    the operator types, one line -> one send_iq() SDU (UM segments/
    reassembles automatically if a line is longer than one PDU's
    max_segment_bits -- nothing special needed here for that).

    bound_event.wait() is called INSIDE the loop, once per line, not
    once before it -- a real bug in the first version of this function:
    checking it only at startup meant that once unblocked, this loop
    would stay unblocked forever, even after the watchdog below clears
    bound_event on a detected link failure -- send_iq() would then just
    raise on the next line typed. Re-checking per line means a lost link
    genuinely pauses sending until the watchdog's re-bind completes."""
    print(f"[{label}] waiting to bind...")
    import sys
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        bound_event.wait()
        bits = np.unpackbits(np.frombuffer(line.encode("utf-8"), dtype="uint8"))
        with push_lock:  # see _send_chunks()'s docstring -- generate_frame() needs the lock too, not just the socket write
            for iq_frame in mac.send_iq(bits):
                _send_chunks(push_socket, iq_frame, push_lock)
        print(f"[{label}] sent: {line!r}")


def _quality_report_loop(label, mac, push_socket, bound_event, push_lock, interval_s=LINK_QUALITY_INTERVAL_S):
    """Runs in its own thread -- every interval_s (default 100ms), pushes
    mac's own current LinkQualityTracker snapshot to the peer, chunked
    over the same channel as everything else. bound_event.wait() is
    checked every tick (not once at startup) for the same reason
    _stdin_send_loop()'s does -- see its docstring."""
    while True:
        time.sleep(interval_s)
        bound_event.wait()
        report_pdu = encode_quality_report(mac.quality.report_dict())
        with push_lock:  # see _send_chunks()'s docstring -- generate_frame() needs the lock too, not just the socket write
            report_iq = mac.ofdm.generate_frame(report_pdu[None, :])
            _send_chunks(push_socket, report_iq, push_lock)


def _heartbeat_watchdog_loop(label, mac, bound_event, heartbeat, interval_s=LINK_QUALITY_INTERVAL_S):
    """Runs in its own thread -- watches heartbeat["received_count"]
    (incremented in _handle_decoded_pdu() whenever a LINK_QUALITY pdu
    genuinely arrives from the peer). If it hasn't advanced for
    HEARTBEAT_MISS_LIMIT consecutive checks (default 5 x 100ms = ~500ms
    of silence), treats the link as down: marks mac.bound False and
    clears bound_event, which pauses both send loops above until a fresh
    bind completes. Air never initiates its own bind (see module
    docstring) -- it just waits for a new BIND_REQUEST to arrive, same
    as it did at startup; ground's own watchdog is the one that actually
    re-sends the request once IT notices air has gone quiet."""
    bound_event.wait()  # don't start watching until the first bind ever completes
    last_seen = heartbeat["received_count"]
    misses = 0
    while True:
        time.sleep(interval_s)
        if not mac.bound:
            bound_event.wait()  # already down (e.g. ground's watchdog is mid-rebind) -- wait it out
            last_seen = heartbeat["received_count"]
            misses = 0
            continue
        if heartbeat["received_count"] == last_seen:
            misses += 1
        else:
            misses = 0
            last_seen = heartbeat["received_count"]
        if misses >= HEARTBEAT_MISS_LIMIT:
            print(f"[{label}] {HEARTBEAT_MISS_LIMIT} consecutive LINK_QUALITY reports missed "
                  f"-- link presumed down, waiting for a fresh bind")
            mac.bound = False
            bound_event.clear()
            misses = 0


def run(ground_ip: str = "127.0.0.1", verbose: bool = True):
    ctx = zmq.Context()
    pull = ctx.socket(zmq.PULL)
    pull.connect(f"tcp://{ground_ip}:{GROUND_TO_AIR_PORT}")
    push = ctx.socket(zmq.PUSH)
    push.connect(f"tcp://{ground_ip}:{AIR_TO_GROUND_PORT}")

    air_mac = Mac(mode="um", ofdm_kwargs=PHY_KWARGS)
    air_mac.ofdm.reset_stream()
    if verbose:
        print(f"[air] connected to ground at {ground_ip}; max_segment_bits={air_mac.max_segment_bits}")
        print("[air] waiting for ground to bind...")

    bound_event = threading.Event()
    heartbeat = {"received_count": 0}
    push_lock = threading.RLock()  # see _send_chunks()'s docstring -- required, not optional; RLock (not Lock) because callers now nest their own generate_frame()+_send_chunks() critical section around _send_chunks()'s own internal acquisition

    sender = threading.Thread(target=_stdin_send_loop, args=("air", air_mac, push, bound_event, push_lock), daemon=True)
    sender.start()
    reporter = threading.Thread(
        target=_quality_report_loop, args=("air", air_mac, push, bound_event, push_lock), daemon=True
    )
    reporter.start()
    watchdog = threading.Thread(
        target=_heartbeat_watchdog_loop, args=("air", air_mac, bound_event, heartbeat), daemon=True
    )
    watchdog.start()

    while True:
        bits = _recv_one_chunk_and_stream_decode(pull, air_mac)
        if bits is not None:
            _handle_decoded_pdu("air", air_mac, bits, push, heartbeat, push_lock)
            if air_mac.bound and not bound_event.is_set():
                print("[air] bound -- type a message and press enter to send it")
                bound_event.set()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-ip", default="127.0.0.1", help="ground unit's address (default: 127.0.0.1 for local testing)")
    args = parser.parse_args()
    run(ground_ip=args.ground_ip)
