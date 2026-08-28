"""Everything PlutoSDR-specific lives in this one file. air_unit.py/
ground_unit.py stay almost untouched -- they gain a --transport flag and
call pluto_init()/pluto_tx()/pluto_rx() at the exact points they'd
otherwise use a ZMQ socket's connect/send/recv, nothing else about their
own logic (Mac, bind handshake, dashboard, /traffic commands) changes.

Deliberately simpler than examples/pluto_channel.py's PlutoChannel: no
background RX thread, no bounded drop-oldest queue. That machinery
existed there specifically to duck-type ZMQ PULL's "block for exactly
one small chunk" semantics on top of a radio that has no message framing
at all. Here, air_unit.py/ground_unit.py call pluto_rx() directly,
synchronously, from their own single receive loop -- so pluto_rx() can
just be sdr.rx() itself: one whole rx_buffer_size-sample buffer per
call, the same plain call/response style pysdr.org's own PlutoSDR
examples use. mac.ofdm.rx_streaming() places no requirement on the size
or alignment of what it's fed (see its own docstring in spectracuda/
pipeline/ofdm.py) -- a bigger buffer per call just means coarser
latency granularity, not a correctness difference.

TX side, similarly: pluto_tx() sends one WHOLE OFDM frame (exactly what
Ofdm.generate_frame() produced) in a single tx() call -- never chunked.
Splitting a frame into smaller separate tx() calls would introduce real
gaps (USB round-trip + DMA setup each time) that break the phase
continuity sync/CFO estimation depends on -- same reasoning
pluto_channel.py's own send_frame() already documented.

Radio configuration (FDD tx_lo/rx_lo, rf_bw, gain/AGC, DAC scaling) is
copied from pluto_channel.py's already-hardware-proven PlutoChannel.
__init__/send_frame() rather than re-derived -- only the chunking/
threading is dropped, not the actual radio settings.
"""
from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np


def _import_adi() -> Any:
    try:
        return __import__("adi")
    except ImportError:
        sys.exit("error: pyadi-iio not installed. Run 'pip install pyadi-iio'.")


@dataclass
class PlutoHandle:
    sdr: Any
    tx_lock: threading.Lock  # serializes concurrent pluto_tx() calls onto the one physical radio's tx()


def pluto_init(
    uri: str, tx_freq: float, rx_freq: float, rate: float,
    tx_gain: float, rx_gain: float, agc: str, rx_buffer_size: int,
) -> PlutoHandle:
    """Configures one physical Pluto for full-duplex FDD (this node
    transmits on tx_freq, listens on rx_freq -- the peer's tx_freq/
    rx_freq must be swapped relative to this one, same as the two
    ZMQ ports being two independent directions). Call once at startup;
    the returned handle is what every pluto_tx()/pluto_rx() call takes."""
    adi = _import_adi()
    sdr = adi.Pluto(uri=uri)
    rf_bw = int(max(rate * 1.25, 5e6))  # matches pluto_channel.py's own configure_radio() pattern
    sdr.sample_rate = int(rate)
    sdr.tx_lo = int(tx_freq)
    sdr.tx_rf_bandwidth = rf_bw
    sdr.tx_hardwaregain_chan0 = float(tx_gain)
    sdr.tx_cyclic_buffer = False  # one-shot per tx() call, not a repeating burst
    sdr.rx_lo = int(rx_freq)
    sdr.rx_rf_bandwidth = rf_bw
    sdr.gain_control_mode_chan0 = agc
    if agc == "manual":
        sdr.rx_hardwaregain_chan0 = float(rx_gain)
    sdr.rx_buffer_size = int(rx_buffer_size)
    return PlutoHandle(sdr=sdr, tx_lock=threading.Lock())


def pluto_tx(handle: PlutoHandle, iq_frame: np.ndarray) -> None:
    """One whole OFDM frame, one tx() call -- no chunking (see module
    docstring). peak=2**13 (12dB below full scale, room for OFDM PAPR)
    matches pluto_channel.py's/pluto_loopback.py's own DAC scaling."""
    samples = np.asarray(iq_frame)
    if samples.ndim == 2:
        samples = samples[0]  # generate_frame()'s own (1, N) batch dim
    peak = np.max(np.abs(samples))
    scaled = samples * (2 ** 13 / peak) if peak > 1e-12 else samples * 0
    with handle.tx_lock:
        handle.sdr.tx(scaled.astype(np.complex64))


def pluto_rx(handle: PlutoHandle) -> np.ndarray:
    """One whole rx_buffer_size-sample buffer, straight off the ADC --
    blocks until the Pluto has that many samples ready. No slicing, no
    queue (see module docstring): air_unit.py/ground_unit.py hand this
    straight to mac.ofdm.rx_streaming()."""
    return np.asarray(handle.sdr.rx(), dtype="complex64")
