"""Binding handshake: a lightweight BIND_REQUEST/BIND_RESPONSE exchange
agreeing on session parameters (mode, max_segment_bits, window_size,
max_retries) before data flows.

No liquid-dsp or 3GPP precedent claimed -- this project's own design,
same "simplified/custom, behavior-only" footing as pdu.py's header
format. Real behavioral point, not just plumbing: `MacLink.send()` (see
session.py) refuses to run before `bind()` has succeeded, so a session
genuinely cannot exchange data without an explicit handshake step first.

Payloads are BYTE-ALIGNED, whole-byte fields (not tightly bit-packed like
the DATA header) -- deliberate, since these are rare, at-most-once-per-
session control messages, not bandwidth-sensitive the way a header sent
on every single data PDU is; simplicity wins over a few extra bits of
overhead here.

`evaluate_bind_request()` is a pure function (no Ofdm/PHY/Block
dependency) precisely so it can be tested directly with two
independently-chosen configs, proving real mismatch-rejection -- not just
self-consistent "both sides agree because they were built identically"
plumbing (see docs/mac.md and tests/test_mac_bind.py). It rejects rather
than silently clamping/renegotiating a too-large `max_segment_bits`
request -- "fail loud," the same convention this codebase's FEC/LDPC
code already follows for its own capacity/correctability limits.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .pdu import (
    HEADER_LEN_BITS,
    SI_FULL,
    TYPE_BIND_REQUEST,
    TYPE_BIND_RESPONSE,
    decode_header,
    encode_header,
)

_MODE_CODES = {"tm": 0, "um": 1, "am": 2}
_MODE_NAMES = {v: k for k, v in _MODE_CODES.items()}

REASON_NONE = 0
REASON_UNKNOWN_MODE = 1
REASON_SEGMENT_TOO_LARGE = 2
_REASON_NAMES = {
    REASON_NONE: "none",
    REASON_UNKNOWN_MODE: "unknown_mode",
    REASON_SEGMENT_TOO_LARGE: "segment_too_large",
}

_U8_MAX = (1 << 8) - 1
_U32_MAX = (1 << 32) - 1


def _u8(v: int) -> np.ndarray:
    if not (0 <= v <= _U8_MAX):
        raise ValueError(f"value {v} does not fit in 1 byte")
    return np.unpackbits(np.array([v], dtype="uint8"))


def _u32(v: int) -> np.ndarray:
    # max_segment_bits genuinely needs 4 bytes, not 2 -- a u16 (max 65535)
    # was the original design here, and it silently worked in every test
    # written against MacLink (which only ever self-evaluates its OWN,
    # coincidentally-small-scale-tested capacity) until a genuine
    # two-object test at fft_size=256/modem="qam64" derived a REAL
    # max_segment_bits of 165840 -- comfortably over 65535. Caught by
    # that real cross-object test, not found by inspection.
    if not (0 <= v <= _U32_MAX):
        raise ValueError(f"value {v} does not fit in 4 bytes")
    return np.unpackbits(
        np.array([(v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF], dtype="uint8")
    )


def _read_u8(bits: np.ndarray, byte_offset: int) -> int:
    return int(np.packbits(bits[byte_offset * 8 : byte_offset * 8 + 8])[0])


def _read_u32(bits: np.ndarray, byte_offset: int) -> int:
    b0 = _read_u8(bits, byte_offset)
    b1 = _read_u8(bits, byte_offset + 1)
    b2 = _read_u8(bits, byte_offset + 2)
    b3 = _read_u8(bits, byte_offset + 3)
    return (b0 << 24) | (b1 << 16) | (b2 << 8) | b3


def encode_bind_request(mode: str, max_segment_bits: int, window_size: int, max_retries: int) -> np.ndarray:
    """-> HEADER (32 bits, TYPE_BIND_REQUEST) + 7-byte payload."""
    if mode not in _MODE_CODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {sorted(_MODE_CODES)}")
    header = encode_header(pdu_type=TYPE_BIND_REQUEST, si=SI_FULL, sn=0, so=0)
    payload = np.concatenate([
        _u8(_MODE_CODES[mode]),
        _u32(max_segment_bits),
        _u8(window_size),
        _u8(max_retries),
    ])
    return np.concatenate([header, payload])


def decode_bind_request(pdu_bits: np.ndarray) -> Dict[str, Any]:
    pdu_bits = np.asarray(pdu_bits, dtype="uint8")
    header = decode_header(pdu_bits[:HEADER_LEN_BITS])
    if header["pdu_type"] != TYPE_BIND_REQUEST:
        raise ValueError(f"expected a BIND_REQUEST pdu, got pdu_type={header['pdu_type']}")
    payload = pdu_bits[HEADER_LEN_BITS:]
    mode_code = _read_u8(payload, 0)
    if mode_code not in _MODE_NAMES:
        raise ValueError(f"decoded mode code {mode_code} is not a known mode -- likely corruption")
    return {
        "mode": _MODE_NAMES[mode_code],
        "max_segment_bits": _read_u32(payload, 1),
        "window_size": _read_u8(payload, 5),
        "max_retries": _read_u8(payload, 6),
    }


def evaluate_bind_request(request: Dict[str, Any], local_max_segment_bits: int) -> Dict[str, Any]:
    """The real accept/reject decision, as a pure function independent of
    any PHY/session machinery (see module docstring for why).

    Accepts iff `request["mode"]` is a known mode AND
    `request["max_segment_bits"] <= local_max_segment_bits` -- rejects
    (does NOT silently clamp down to what's actually supported) if the
    requester asked for more capacity than this side can provide, so a
    real capacity mismatch is caught explicitly rather than producing a
    session that later fails in some more confusing way.

    Returns {"accepted": bool, "reason": str, "mode", "max_segment_bits",
    "window_size", "max_retries"} -- on accept, the params to actually
    use (echoed back from the request); on reject, the request's own
    params are still echoed (for the caller's/log's benefit) but MUST
    NOT be used to start a session."""
    mode = request.get("mode")
    if mode not in _MODE_CODES:
        return {**request, "accepted": False, "reason": _REASON_NAMES[REASON_UNKNOWN_MODE]}
    if request["max_segment_bits"] > local_max_segment_bits:
        return {**request, "accepted": False, "reason": _REASON_NAMES[REASON_SEGMENT_TOO_LARGE]}
    return {**request, "accepted": True, "reason": _REASON_NAMES[REASON_NONE]}


def encode_bind_response(decision: Dict[str, Any]) -> np.ndarray:
    """-> HEADER (32 bits, TYPE_BIND_RESPONSE) + 9-byte payload, from an
    evaluate_bind_request()-shaped dict."""
    reason_code = {v: k for k, v in _REASON_NAMES.items()}[decision["reason"]]
    mode_code = _MODE_CODES.get(decision.get("mode"), 0)
    header = encode_header(pdu_type=TYPE_BIND_RESPONSE, si=SI_FULL, sn=0, so=0)
    payload = np.concatenate([
        _u8(1 if decision["accepted"] else 0),
        _u8(reason_code),
        _u8(mode_code),
        _u32(decision.get("max_segment_bits", 0)),
        _u8(decision.get("window_size", 0)),
        _u8(decision.get("max_retries", 0)),
    ])
    return np.concatenate([header, payload])


def decode_bind_response(pdu_bits: np.ndarray) -> Dict[str, Any]:
    pdu_bits = np.asarray(pdu_bits, dtype="uint8")
    header = decode_header(pdu_bits[:HEADER_LEN_BITS])
    if header["pdu_type"] != TYPE_BIND_RESPONSE:
        raise ValueError(f"expected a BIND_RESPONSE pdu, got pdu_type={header['pdu_type']}")
    payload = pdu_bits[HEADER_LEN_BITS:]
    accepted = bool(_read_u8(payload, 0))
    reason_code = _read_u8(payload, 1)
    if reason_code not in _REASON_NAMES:
        raise ValueError(f"decoded reason code {reason_code} is not known -- likely corruption")
    mode_code = _read_u8(payload, 2)
    if mode_code not in _MODE_NAMES:
        raise ValueError(f"decoded mode code {mode_code} is not a known mode -- likely corruption")
    return {
        "accepted": accepted,
        "reason": _REASON_NAMES[reason_code],
        "mode": _MODE_NAMES[mode_code],
        "max_segment_bits": _read_u32(payload, 3),
        "window_size": _read_u8(payload, 7),
        "max_retries": _read_u8(payload, 8),
    }
