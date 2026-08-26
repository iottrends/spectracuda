"""Real-Pluto transport that duck-types ZMQ's PUSH/PULL socket interface
(.send(bytes)/.recv()->bytes) so drone_air_unit.py/drone_ground_unit.py's
existing, proven logic (Mac, bind handshake, rx_streaming() dispatch,
heartbeat watchdog, link-quality reporting) can be reused UNCHANGED --
only the transport underneath is swapped, via pluto_air_unit.py/
pluto_ground_unit.py importing those modules and monkeypatching just
their _send_chunks() (see this module's own docstring below for why that
one function specifically needs different behavior for real RF, not just
a different socket).

One physical Pluto per node (matching "2 plutos, 2 different Pi5s"), each
running full-duplex (TX and RX simultaneously, both AD9363-native) --
NOT the two-pluto self-test mode from pluto_loopback.py (which used two
Plutos to isolate CFO/SRO on ONE end of a link, not to BE the two ends of
a real link). FDD (frequency-division duplex): air transmits on
--tx-freq, listens on --rx-freq; ground's --tx-freq must equal air's
--rx-freq and vice versa -- exactly the same "two independent channels"
idea GROUND_TO_AIR_PORT/AIR_TO_GROUND_PORT already used, just RF
frequencies instead of TCP ports.

Reuses pluto_loopback.py's ALREADY-PROVEN gain/bandwidth/AGC configuration
pattern (see its own configure_radio()) rather than re-deriving it --
only added what that function didn't need: full-duplex (tx_lo != rx_lo
simultaneously on one device) and continuous (not burst/cyclic) TX/RX.
"""
from __future__ import annotations

import queue
import sys
import threading

import numpy as np


def _import_adi():
    try:
        return __import__("adi")
    except ImportError:
        sys.exit("error: pyadi-iio not installed. Run 'pip install pyadi-iio'.")


class PlutoChannel:
    """One physical Pluto, full-duplex, FDD. Presents TWO duck-typed
    socket-like objects (.tx_socket, .rx_socket) so existing ZMQ-shaped
    code (_send_chunks(push_socket, ...), _recv_one_chunk_and_stream_
    decode(pull_socket, ...)) works against it with zero changes to
    THEIR code -- only _send_chunks() itself gets monkeypatched at the
    call site (see pluto_air_unit.py/pluto_ground_unit.py), because a
    real RF burst needs to go out as ONE continuous tx() call per frame,
    not N separate small ones (see module docstring)."""

    def __init__(self, uri: str, tx_freq: float, rx_freq: float, rate: float,
                 tx_gain: float, rx_gain: float, agc: str, rx_buffer_size: int,
                 chunk_size: int) -> None:
        adi = _import_adi()
        self.chunk_size = chunk_size
        self.rate = rate

        self.sdr = adi.Pluto(uri=uri)
        rf_bw = int(max(rate * 1.25, 5e6))  # matches pluto_loopback.py's configure_radio()
        self.sdr.sample_rate = int(rate)
        self.sdr.tx_lo = int(tx_freq)
        self.sdr.tx_rf_bandwidth = rf_bw
        self.sdr.tx_hardwaregain_chan0 = float(tx_gain)
        self.sdr.tx_cyclic_buffer = False  # one-shot per tx() call, not a repeating burst -- see send_frame()
        self.sdr.rx_lo = int(rx_freq)
        self.sdr.rx_rf_bandwidth = rf_bw
        self.sdr.gain_control_mode_chan0 = agc
        if agc == "manual":
            self.sdr.rx_hardwaregain_chan0 = float(rx_gain)
        self.sdr.rx_buffer_size = int(rx_buffer_size)

        self._tx_lock = threading.Lock()  # serializes concurrent send_frame() calls onto one tx()
        self._chunk_queue: queue.Queue = queue.Queue(maxsize=1024)
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()

        self.tx_socket = _PlutoTxSocket(self)
        self.rx_socket = _PlutoRxSocket(self)

    def send_frame(self, iq_frame: np.ndarray) -> None:
        """The WHOLE frame in ONE tx() call -- real hardware, real RF
        timing: splitting this into small separate calls (like the ZMQ
        chunking this replaces) would introduce real gaps between calls
        (USB round-trip + DMA setup each time), breaking the phase
        continuity sync/CFO estimation depends on. peak=2**13 matches
        pluto_loopback.py's own normalize_for_dac() (12 dB below full
        scale, room for OFDM PAPR)."""
        samples = np.asarray(iq_frame)
        if samples.ndim == 2:
            samples = samples[0]  # generate_frame()'s own (1, N) batch dim
        m = np.max(np.abs(samples))
        scaled = samples * (2 ** 13 / m) if m > 1e-12 else samples * 0
        with self._tx_lock:
            self.sdr.tx(scaled.astype(np.complex64))

    def _rx_loop(self) -> None:
        """Continuously pulls big buffers off the ADC and slices them
        into chunk_size pieces on an internal queue -- this is what lets
        rx_socket.recv() duck-type ZMQ PULL's "block for exactly one
        chunk" semantics on top of a real radio that doesn't have
        message framing at all, just a continuous sample stream (which
        is exactly what rx_streaming() is designed to consume -- no
        chunk-boundary assumptions on its input, see its own docstring
        in spectracuda/pipeline/ofdm.py)."""
        while True:
            buf = np.asarray(self.sdr.rx(), dtype="complex64")
            for i in range(0, len(buf), self.chunk_size):
                chunk = buf[i : i + self.chunk_size]
                if len(chunk) == 0:
                    continue
                # Never block the live ADC read behind a slow consumer --
                # a real bug found via a mocked-hardware dry run (see this
                # project's commit history): a plain blocking .put() on a
                # bounded queue stalls THIS thread the instant the queue
                # fills, which stops draining the radio entirely until the
                # consumer catches up -- exactly backwards for a live
                # receiver, which must always keep sampling and should
                # drop STALE backlog, not fresh incoming samples, under
                # sustained backpressure (rx_streaming() falling behind
                # real-time, e.g. while decoding a large frame).
                try:
                    self._chunk_queue.put_nowait(chunk.tobytes())
                except queue.Full:
                    try:
                        self._chunk_queue.get_nowait()  # drop the oldest queued chunk
                    except queue.Empty:
                        pass
                    self._chunk_queue.put_nowait(chunk.tobytes())


class _PlutoTxSocket:
    """Duck-types ZMQ PUSH's .send(bytes) -- but see send_frame()'s
    docstring: this does NOT chunk. It expects to be called with an
    ENTIRE frame's raw bytes at once (which is what the monkeypatched
    _send_chunks() in pluto_air_unit.py/pluto_ground_unit.py does), not
    ZMQ-style repeated small .send() calls."""

    def __init__(self, channel: PlutoChannel) -> None:
        self._channel = channel

    def send(self, raw_bytes: bytes) -> None:
        samples = np.frombuffer(raw_bytes, dtype="complex64")
        self._channel.send_frame(samples)


class _PlutoRxSocket:
    """Duck-types ZMQ PULL's .recv() -> bytes (blocks for exactly one
    chunk, in order) -- backed by PlutoChannel's continuous background
    rx thread. Nothing else about _recv_one_chunk_and_stream_decode()
    needs to change to use this."""

    def __init__(self, channel: PlutoChannel) -> None:
        self._channel = channel

    def recv(self) -> bytes:
        return self._channel._chunk_queue.get()
