"""Framing layer: header codec, CRC+FEC packetizer, and packet stats --
the pieces that turn decoded OFDM symbols into a validated payload,
independent of the OFDM modem engine itself (liquid-dsp's equivalent
separation: `packetizer` knows nothing about `ofdmflexframegen`/
`ofdmflexframesync`; those call INTO it, not the reverse).

`Ofdm` (spectracuda/pipeline/ofdm.py) owns one `HeaderCodec` and one
`Packetizer` instance and delegates to them, rather than doing this
logic inline -- see docs/todo.md #1.1 for the gap this closes ("you
can't reuse 'decode a framed packet' logic outside Ofdm itself").
"""
from .header import HeaderCodec
from .packetizer import Packetizer
from .stats import compute_evm, compute_rssi_db

__all__ = ["HeaderCodec", "Packetizer", "compute_evm", "compute_rssi_db"]
