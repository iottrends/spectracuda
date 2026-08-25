"""Ground unit -- the other of the two independently runnable processes
(see drone_air_unit.py, which has the full design writeup). This is the
stable, known-address side (a real ground control station has a fixed
IP; the air unit is the one that's mobile and connects out to it) --
ground BINDS both ZeroMQ sockets, air CONNECTS to `--bind-ip` (or
whatever address ground is actually reachable at).

Same model as drone_air_unit.py, mirrored:
- ONE Mac object (ground_mac), used for both sending and receiving.
- Same two channels: ground_to_air (ground PUSH -> air PULL) and
  air_to_ground (ground PULL <- air PUSH).
- Ground is the one that INITIATES the bind handshake here (an arbitrary
  but consistent choice -- air's handle_bind_request_iq() evaluates it
  and sets ITS OWN .bound as a side effect, so this one exchange sets
  both sides' .bound, same as air's own writeup explains).

Config must match drone_air_unit.py's PHY_KWARGS exactly -- both sides
of a real link always do; there is deliberately no shared config object
between them (see docs/mac.md's "we dont share any config between the
transmitter and receiver" design principle) -- this is two independent
copies of the same numbers, not one shared source.
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
    encode_bind_request,
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
LINK_QUALITY_INTERVAL_S = 0.1  # 100ms -- DSP-derived only (see drone_air_unit.py's module docstring)
HEARTBEAT_MISS_LIMIT = 5  # 5 consecutive missed LINK_QUALITY reports (~500ms of silence) -> link presumed down


def _send_chunks(push_socket, iq_frame, push_lock):
    """See drone_air_unit.py's identical helper for why `push_lock` is
    mandatory -- a real concurrency bug (chunks from concurrent senders
    interleaving on the wire and corrupting rx_streaming() on the far
    end) found while testing the heartbeat watchdog below."""
    samples = np.asarray(iq_frame)[0]
    with push_lock:
        for i in range(0, samples.shape[-1], CHUNK_SIZE):
            push_socket.send(samples[i : i + CHUNK_SIZE].tobytes())


def _send_bind_request(mac, push_socket, push_lock):
    """Encodes+sends a fresh BIND_REQUEST -- factored out so
    _heartbeat_watchdog_loop() can re-trigger the exact same handshake
    used at startup (below, in run()), not a separate/different path.
    Unlike drone_air_unit.py, ground genuinely needs this: ground is the
    bind INITIATOR (see module docstring), so ground's own watchdog is
    the one that actually re-establishes the link after a detected
    failure -- air's watchdog only clears state and waits."""
    request_pdu = encode_bind_request(mac.mode, mac.max_segment_bits, mac._window_size, mac._max_retries)
    request_iq = mac.ofdm.generate_frame(request_pdu[None, :])
    _send_chunks(push_socket, request_iq, push_lock)


def _recv_one_chunk_and_stream_decode(pull_socket, mac):
    """See drone_air_unit.py's identical function for the full writeup
    -- also feeds mac.quality (LinkQualityTracker) from every completed
    frame and drops anything that fails CRC, both real fixes made when
    link-quality reporting was added (see docs/mac.md)."""
    raw = pull_socket.recv()
    chunk = np.frombuffer(raw, dtype="complex64")
    result = mac.ofdm.rx_streaming(chunk)
    if result is None:
        return None
    crc_valid = result["crc_valid"]
    delivered = crc_valid is None or bool(np.asarray(crc_valid)[0])
    evm = result["evm"]
    mac.quality.observe(
        rssi_db=float(np.asarray(result["rssi_db"])[0]),
        evm=None if evm is None else float(np.asarray(evm)[0]),
        delivered=delivered,
    )
    if not delivered:
        return None
    return np.asarray(result["bits"])[0].astype("uint8")


def _handle_decoded_pdu(label, mac, bits, reply_push_socket, heartbeat, push_lock):
    """See drone_air_unit.py's identical function for the full writeup
    on `heartbeat`."""
    header = decode_header(bits[:HEADER_LEN_BITS])

    if header["pdu_type"] == TYPE_BIND_REQUEST:
        request = decode_bind_request(bits)
        decision = evaluate_bind_request(request, local_max_segment_bits=mac.max_segment_bits)
        mac.bound = decision["accepted"]
        response_pdu = encode_bind_response(decision)
        response_iq = mac.ofdm.generate_frame(response_pdu[None, :])
        _send_chunks(reply_push_socket, response_iq, push_lock)
        print(f"[{label}] BIND_REQUEST received, evaluated -- bound={mac.bound}")

    elif header["pdu_type"] == TYPE_BIND_RESPONSE:
        response = decode_bind_response(bits)
        mac.bound = response["accepted"]
        print(f"[{label}] BIND_RESPONSE received -- bound={mac.bound}")

    elif header["pdu_type"] == TYPE_DATA:
        for sdu_bits in mac.receive(bits):
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
    """See drone_air_unit.py's identical function for why bound_event.wait()
    is called per-line, inside the loop, not once at the top."""
    print(f"[{label}] waiting to bind...")
    import sys
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        bound_event.wait()
        bits = np.unpackbits(np.frombuffer(line.encode("utf-8"), dtype="uint8"))
        for iq_frame in mac.send_iq(bits):
            _send_chunks(push_socket, iq_frame, push_lock)
        print(f"[{label}] sent: {line!r}")


def _quality_report_loop(label, mac, push_socket, bound_event, push_lock, interval_s=LINK_QUALITY_INTERVAL_S):
    """See drone_air_unit.py's identical function for the full writeup."""
    while True:
        time.sleep(interval_s)
        bound_event.wait()
        report_pdu = encode_quality_report(mac.quality.report_dict())
        report_iq = mac.ofdm.generate_frame(report_pdu[None, :])
        _send_chunks(push_socket, report_iq, push_lock)


def _heartbeat_watchdog_loop(label, mac, push_socket, push_lock, bound_event, heartbeat, interval_s=LINK_QUALITY_INTERVAL_S):
    """See drone_air_unit.py's identical function for the full writeup.
    The one real difference: ground IS the bind initiator, so ground's
    watchdog doesn't just clear state and wait -- it actually re-sends a
    fresh BIND_REQUEST once it detects the peer has gone quiet."""
    bound_event.wait()
    last_seen = heartbeat["received_count"]
    misses = 0
    while True:
        time.sleep(interval_s)
        if not mac.bound:
            bound_event.wait()
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
                  f"-- link presumed down, re-binding")
            mac.bound = False
            bound_event.clear()
            misses = 0
            _send_bind_request(mac, push_socket, push_lock)


def run(bind_ip: str = "*", verbose: bool = True):
    ctx = zmq.Context()
    push = ctx.socket(zmq.PUSH)
    push.bind(f"tcp://{bind_ip}:{GROUND_TO_AIR_PORT}")
    pull = ctx.socket(zmq.PULL)
    pull.bind(f"tcp://{bind_ip}:{AIR_TO_GROUND_PORT}")

    ground_mac = Mac(mode="um", ofdm_kwargs=PHY_KWARGS)
    ground_mac.ofdm.reset_stream()
    if verbose:
        print(f"[ground] bound to {bind_ip}:{GROUND_TO_AIR_PORT}/{AIR_TO_GROUND_PORT}; "
              f"max_segment_bits={ground_mac.max_segment_bits}")

    bound_event = threading.Event()
    heartbeat = {"received_count": 0}
    push_lock = threading.Lock()  # see _send_chunks()'s docstring -- required, not optional

    sender = threading.Thread(
        target=_stdin_send_loop, args=("ground", ground_mac, push, bound_event, push_lock), daemon=True
    )
    sender.start()
    reporter = threading.Thread(
        target=_quality_report_loop, args=("ground", ground_mac, push, bound_event, push_lock), daemon=True
    )
    reporter.start()
    watchdog = threading.Thread(
        target=_heartbeat_watchdog_loop,
        args=("ground", ground_mac, push, push_lock, bound_event, heartbeat), daemon=True
    )
    watchdog.start()

    # Ground initiates the ONE bind exchange this link needs -- see
    # module docstring for why one exchange is enough for both sides.
    # Re-triggered later by _heartbeat_watchdog_loop() if the link ever
    # goes quiet -- same helper, same code path, not a separate one.
    if verbose:
        print("[ground] sending bind request...")
    _send_bind_request(ground_mac, push, push_lock)

    while True:
        bits = _recv_one_chunk_and_stream_decode(pull, ground_mac)
        if bits is not None:
            _handle_decoded_pdu("ground", ground_mac, bits, push, heartbeat, push_lock)
            if ground_mac.bound and not bound_event.is_set():
                print("[ground] bound -- type a message and press enter to send it")
                bound_event.set()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind-ip", default="*", help="address to bind on (default: * = all interfaces)")
    args = parser.parse_args()
    run(bind_ip=args.bind_ip)
