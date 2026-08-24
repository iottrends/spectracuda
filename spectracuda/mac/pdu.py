"""PduHeader: the fixed 32-bit header shared by every UM/AM/control PDU
type (TM has no header at all -- see tm.py).

This project's own header format, NOT derived from a 3GPP spec table (see
docs/mac.md -- "simplified/custom, behavior-only" was an explicit choice,
same footing as e.g. LSChannelEstimator/ZFEqualizer: no attempt to match
real TS 36.322/38.322 RLC PDU bit layouts). What IS carried over from the
real spec, deliberately, is the *purpose* of each field:

    TYPE (3 bits): DATA(0), STATUS(1) -- AM's per-PDU ACK/NACK bitmap
                   (am.py) -- BIND_REQUEST(2)/BIND_RESPONSE(3) -- the
                   binding handshake (bind.py) -- LINK_QUALITY(4) --
                   aggregated RSSI/EVM/delivery-ratio reporting
                   (quality.py). Widened from 1 to 3 bits when
                   BIND/LINK_QUALITY were added (see docs/mac.md) --
                   DATA=0/STATUS=1 keep their original values, so this
                   was purely additive, not a renumbering.
    SI   (2 bits): segmentation indicator -- FULL (0, an unsegmented SDU),
                   FIRST (1), MIDDLE (2), LAST (3) segment of a larger SDU.
                   Unused (0) for control PDU types (STATUS/BIND/
                   LINK_QUALITY) -- they don't segment.
    SN   (10 bits): sequence number, modulo 1024, for DATA pdus.
                   Deliberately kept as real modulo arithmetic (see
                   sn_precedes()) rather than a naive `a < b` comparison
                   -- a naive comparison silently breaks the moment SN
                   wraps around, which WILL happen in any test that runs
                   long enough to matter. For a STATUS pdu, this field is
                   repurposed as the status report's window base_sn (see
                   am.py); unused (0) for BIND/LINK_QUALITY.
    SO   (16 bits): segment offset, in BITS (not bytes) from the start of
                   the original SDU -- bit-oriented to match this
                   project's existing bit-oriented convention throughout
                   (Modem/FEC/HeaderCodec all count bits, not bytes; see
                   pipeline/ofdm.py's own header docstring for the same
                   reasoning). Bounds a single SDU to 65535 bits (~8KB) of
                   representable offset. Unused (0) for control types.
    RESERVED (1 bit): zero, unused -- rounds the header to a clean 32
                   bits (4 bytes) rather than a 31-bit oddity, matching
                   the byte-alignment convention HeaderCodec/Packetizer
                   already use for their own bit<->byte packing helpers.

Total: 32 bits, MSB-first within the field order above (TYPE, SI, SN, SO,
RESERVED), packed/unpacked the same way HeaderCodec does its own 112-bit
header (np.packbits/np.unpackbits over a plain bit array). Every PDU type
reuses this SAME header rather than each control type inventing its own
format -- consistent with STATUS's original "repurpose the shared header"
approach, now extended to BIND/LINK_QUALITY too; type-specific payload
content (if any) follows immediately after these 32 bits (see bind.py/
quality.py/am.py for each type's own payload layout).
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

TYPE_DATA = 0
TYPE_STATUS = 1
TYPE_BIND_REQUEST = 2
TYPE_BIND_RESPONSE = 3
TYPE_LINK_QUALITY = 4
_MAX_TYPE = 7  # 3 bits

SI_FULL = 0
SI_FIRST = 1
SI_MIDDLE = 2
SI_LAST = 3

SN_BITS = 10
SN_MODULUS = 1 << SN_BITS  # 1024
SO_BITS = 16
TYPE_BITS = 3
HEADER_LEN_BITS = TYPE_BITS + 2 + SN_BITS + SO_BITS + 1  # 32


def sn_add(sn: int, delta: int) -> int:
    """Sequence number arithmetic is always modulo SN_MODULUS -- a plain
    `sn + delta` silently produces an out-of-range value once traffic runs
    past SN 1023."""
    return (sn + delta) % SN_MODULUS


def sn_precedes(a: int, b: int) -> bool:
    """True if `a` comes strictly before `b` in sequence-number order,
    correctly handling wraparound: compares the forward distance from `a`
    to `b` against half the SN space (the standard sliding-window-safe
    comparison -- SN_MODULUS/2 = 512 is this project's window-safety
    margin, generously larger than the actual reassembly/retransmission
    window sizes UM/AM use, so ambiguity never arises in practice here)."""
    return 0 < ((b - a) % SN_MODULUS) < (SN_MODULUS // 2)


def encode_header(*, pdu_type: int, si: int, sn: int, so: int) -> np.ndarray:
    """-> (HEADER_LEN_BITS,) uint8 bit array, MSB-first per field."""
    if not (0 <= pdu_type <= _MAX_TYPE):
        raise ValueError(f"pdu_type must be 0-{_MAX_TYPE}, got {pdu_type}")
    if not (0 <= si <= 3):
        raise ValueError(f"si must be 0-3, got {si}")
    if not (0 <= sn < SN_MODULUS):
        raise ValueError(f"sn must be 0-{SN_MODULUS - 1}, got {sn}")
    if not (0 <= so < (1 << SO_BITS)):
        raise ValueError(f"so must be 0-{(1 << SO_BITS) - 1}, got {so}")

    def _bits(value: int, width: int) -> np.ndarray:
        return np.array([(value >> i) & 1 for i in range(width - 1, -1, -1)], dtype="uint8")

    return np.concatenate([
        _bits(pdu_type, TYPE_BITS),
        _bits(si, 2),
        _bits(sn, SN_BITS),
        _bits(so, SO_BITS),
        np.zeros(1, dtype="uint8"),  # reserved
    ])


def decode_header(bits: np.ndarray) -> Dict[str, Any]:
    """Inverse of encode_header(). bits: (HEADER_LEN_BITS,) uint8 -> dict
    with pdu_type/si/sn/so. Raises ValueError if bits has the wrong
    length (a caller error, not corruption -- header length is fixed and
    known, unlike the OFDM header's over-the-air decode)."""
    bits = np.asarray(bits, dtype="uint8")
    if bits.shape[-1] != HEADER_LEN_BITS:
        raise ValueError(f"expected {HEADER_LEN_BITS} header bits, got {bits.shape[-1]}")

    def _value(offset: int, width: int) -> int:
        v = 0
        for b in bits[offset : offset + width]:
            v = (v << 1) | int(b)
        return v

    return {
        "pdu_type": _value(0, TYPE_BITS),
        "si": _value(TYPE_BITS, 2),
        "sn": _value(TYPE_BITS + 2, SN_BITS),
        "so": _value(TYPE_BITS + 2 + SN_BITS, SO_BITS),
    }
