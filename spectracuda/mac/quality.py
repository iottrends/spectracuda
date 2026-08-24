"""Link-quality reporting: aggregated RSSI/EVM/delivery-ratio stats,
tracked locally by LinkQualityTracker and exchanged over the air as a
LINK_QUALITY pdu.

Building on Ofdm.rx_process()'s existing per-frame `rssi_db`/`evm`
readouts (docs/todo.md #1.1) rather than inventing new PHY-level metrics
-- this module just aggregates what the PHY already reports, over
multiple frames/rounds, and gives it a wire format so it can be reported
to a peer, closer to liquid-dsp's `ofdmflexframesync_get_framedatastats()`
concept but as an actual EXCHANGED message rather than only a local
readout.

Honesty note, same spirit as RSSI already being documented as "relative,
not calibrated dBm" (framing/stats.py): `delivered_ratio` here is used as
the BER-TREND proxy, NOT a true bit-error-rate -- a receiver has no
ground-truth transmitted bits to compute real BER against. It's the
fraction of PHY-frame attempts that arrived usably (frame_found AND, if
crc-checked, crc_valid), which trends the same direction as BER (more
errors -> more frame/CRC failures -> lower delivered_ratio) without being
the same quantity.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .pdu import SI_FULL, TYPE_LINK_QUALITY, decode_header, encode_header, HEADER_LEN_BITS

_U16_MAX = (1 << 16) - 1
_RSSI_SCALE = 100  # int16 fixed-point, 0.01 dB resolution
_EVM_SCALE = 10000  # uint16 fixed-point, 0.0001 resolution (evm is normally 0..~2)


class LinkQualityTracker:
    """Running aggregate over PHY-frame attempts (any pdu type -- every
    physical frame is informative about link quality, not just DATA
    ones, see session.py's _phy_round()). Local-only bookkeeping, no
    Ofdm/PHY dependency itself -- callers feed it observations via
    observe()."""

    def __init__(self) -> None:
        self.n_attempts = 0
        self.n_delivered = 0
        self._rssi_sum = 0.0
        self._evm_sum = 0.0
        self._evm_count = 0  # evm is only meaningful when a frame was found

    def observe(self, rssi_db: float, evm: Optional[float], delivered: bool) -> None:
        self.n_attempts += 1
        self.n_delivered += int(delivered)
        self._rssi_sum += float(rssi_db)
        if evm is not None:
            self._evm_sum += float(evm)
            self._evm_count += 1

    @property
    def delivered_ratio(self) -> float:
        return self.n_delivered / self.n_attempts if self.n_attempts else 0.0

    @property
    def mean_rssi_db(self) -> float:
        return self._rssi_sum / self.n_attempts if self.n_attempts else 0.0

    @property
    def mean_evm(self) -> float:
        return self._evm_sum / self._evm_count if self._evm_count else 0.0

    def report_dict(self) -> Dict[str, Any]:
        return {
            "n_attempts": self.n_attempts,
            "n_delivered": self.n_delivered,
            "mean_rssi_db": self.mean_rssi_db,
            "mean_evm": self.mean_evm,
        }


def _u16(v: int) -> np.ndarray:
    if not (0 <= v <= _U16_MAX):
        raise ValueError(f"value {v} does not fit in 2 bytes")
    return np.unpackbits(np.array([(v >> 8) & 0xFF, v & 0xFF], dtype="uint8"))


def _i16(v: int) -> np.ndarray:
    """Signed 16-bit, two's complement (RSSI in dB is commonly negative)."""
    if not (-(1 << 15) <= v <= (1 << 15) - 1):
        raise ValueError(f"value {v} does not fit in a signed 2-byte field")
    return _u16(v & 0xFFFF)


def _read_u16(bits: np.ndarray, byte_offset: int) -> int:
    packed = np.packbits(bits[byte_offset * 8 : byte_offset * 8 + 16])
    return (int(packed[0]) << 8) | int(packed[1])


def _read_i16(bits: np.ndarray, byte_offset: int) -> int:
    v = _read_u16(bits, byte_offset)
    return v - (1 << 16) if v >= (1 << 15) else v


def encode_quality_report(stats: Dict[str, Any]) -> np.ndarray:
    """stats: a LinkQualityTracker.report_dict()-shaped dict -> HEADER
    (32 bits, TYPE_LINK_QUALITY) + 8-byte payload. n_attempts/n_delivered
    are clamped to 65535 (a session running longer than that many PHY
    rounds is far beyond this project's test scale; clamping rather than
    raising keeps a long-running report from becoming un-sendable)."""
    n_attempts = min(int(stats["n_attempts"]), _U16_MAX)
    n_delivered = min(int(stats["n_delivered"]), _U16_MAX)
    rssi_fixed = int(round(stats["mean_rssi_db"] * _RSSI_SCALE))
    rssi_fixed = max(-(1 << 15), min((1 << 15) - 1, rssi_fixed))
    evm_fixed = int(round(stats["mean_evm"] * _EVM_SCALE))
    evm_fixed = max(0, min(_U16_MAX, evm_fixed))

    header = encode_header(pdu_type=TYPE_LINK_QUALITY, si=SI_FULL, sn=0, so=0)
    payload = np.concatenate([_u16(n_attempts), _u16(n_delivered), _i16(rssi_fixed), _u16(evm_fixed)])
    return np.concatenate([header, payload])


def decode_quality_report(pdu_bits: np.ndarray) -> Dict[str, Any]:
    pdu_bits = np.asarray(pdu_bits, dtype="uint8")
    header = decode_header(pdu_bits[:HEADER_LEN_BITS])
    if header["pdu_type"] != TYPE_LINK_QUALITY:
        raise ValueError(f"expected a LINK_QUALITY pdu, got pdu_type={header['pdu_type']}")
    payload = pdu_bits[HEADER_LEN_BITS:]
    n_attempts = _read_u16(payload, 0)
    n_delivered = _read_u16(payload, 2)
    mean_rssi_db = _read_i16(payload, 4) / _RSSI_SCALE
    mean_evm = _read_u16(payload, 6) / _EVM_SCALE
    return {
        "n_attempts": n_attempts,
        "n_delivered": n_delivered,
        "delivered_ratio": n_delivered / n_attempts if n_attempts else 0.0,
        "mean_rssi_db": mean_rssi_db,
        "mean_evm": mean_evm,
    }
