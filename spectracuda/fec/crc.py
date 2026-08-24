"""CRC: CRC(scheme) -- one class, scheme-name string, mirrors
liquid-dsp's crc_generate_key()/crc_append_key()/crc_check_key() family
(src/fec/src/crc.c) directly (same "one class + scheme string" pattern
as Modem/FEC).

Unlike FEC's conv_v27/rs_m8 (which wrap Phil Karn's external `libfec`
library with no fallback -- see fec/viterbi.py, fec/reed_solomon.py),
liquid-dsp's CRC module is entirely self-contained C with no external
dependency. That makes this a genuine byte-exact PORT, not a from-
scratch reimplementation of a standard algorithm -- and it's been
verified against liquid-dsp's own values, not just assumed:

  - liquid's crc32 turns out to be bit-for-bit identical to the
    well-known IEEE-802.3/zlib CRC-32 (confirmed: `zlib.crc32(bytes(range(256)))
    == 0x29058c73`, liquid's own crc32_testvector autotest expects the
    same 0x29058c73 for the same 256-byte input).
  - liquid's crc8/crc16/crc24 are NOT the standard named variants
    (CRC-8/ROHC, CRC-16/ARC, etc.) despite using their textbook
    polynomials (0x07, 0x8005, 0x5D6DCB) -- because `crc.c` computes all
    four widths through the *same* 32-bit `unsigned int` register
    (`key8=~0`, `key16=~0`, ... all start as 0xFFFFFFFF, not 0xFF/0xFFFF/
    0xFFFFFF), the "extra" high bits above each scheme's true width bleed
    into the result via the right-shift for every message length, not
    just short ones (confirmed empirically: a true 8-bit-register CRC-8
    disagrees with liquid's crc8 even at 256 bytes of input). So this
    module reproduces liquid's actual 32-bit-register behavior exactly,
    not the "obvious" narrower-register textbook formula, which would be
    wire-INcompatible with real liquid-dsp despite using the "right"
    polynomial.

Verified byte-for-byte against all four of crc.c's own autotest vectors
(src/fec/tests/crc_autotest.c, data=bytes(range(256))):
  crc8_testvector:  0x53
  crc16_testvector: 0x6fc6
  crc24_testvector: 0x10c59b
  crc32_testvector: 0x29058c73
(see tests/test_crc.py's test_matches_liquid_dsp_testvectors).

Schemes: "none", "checksum" (8-bit two's-complement byte sum -- liquid's
LIQUID_CRC_CHECKSUM), "crc8", "crc16", "crc24", "crc32" -- everything in
liquid's crc_scheme enum except LIQUID_CRC_UNKNOWN (an error sentinel,
not a constructible scheme).

liquid-dsp's own usage (src/fec/src/packetizer.c): append the key to the
RAW message *before* FEC-encoding, and validate it *after* FEC-decoding
-- the CRC is what actually tells you whether a "successfully" completed
FEC decode produced correct data (Viterbi in particular has no other way
to signal a bad decode -- it always returns *a* path, right or wrong).
`Ofdm` wires this class in following that exact order (see
pipeline/ofdm.py's docstring and generate_frame/rx_process).

Byte-oriented, like liquid-dsp's own `unsigned char *` interface --
callers pass whole bytes, not bits (`Ofdm` does its own bit<->byte
packing around this, the same way it already does for the header).

Batch-shape contract: generate_key(msg) takes (n_batch, n) uint8 bytes
-> (n_batch,) uint32 keys. append_key(msg) takes (n_batch, n) ->
(n_batch, n+key_length) (key appended MSB-first, matching
crc_append_key). check_key(msg_with_key) takes (n_batch, n+key_length)
-> (n_batch,) bool (matches crc_check_key/crc_validate_message: never
raises on mismatch, just reports validity -- see this module's
docstring and pipeline/ofdm.py for why that's a deliberate design
choice, not an oversight).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..block import Block

_REG_MASK = np.uint64(0xFFFFFFFF)

_KEY_LENGTH_BYTES = {"none": 0, "checksum": 1, "crc8": 1, "crc16": 2, "crc24": 3, "crc32": 4}
_WIDTH_MASK = {
    "none": np.uint64(0),
    "checksum": np.uint64(0xFF),
    "crc8": np.uint64(0xFF),
    "crc16": np.uint64(0xFFFF),
    "crc24": np.uint64(0xFFFFFF),
    "crc32": np.uint64(0xFFFFFFFF),
}
# Original (non-reflected) polynomials, exactly crc.c's CRC8_POLY/CRC16_POLY/
# CRC24_POLY/CRC32_POLY.
_POLY_ORIG = {"crc8": 0x07, "crc16": 0x8005, "crc24": 0x5D6DCB, "crc32": 0x04C11DB7}
_POLY_WIDTH_BYTES = {"crc8": 1, "crc16": 2, "crc24": 3, "crc32": 4}


def _reverse_byte(b: int) -> int:
    r = 0
    for i in range(8):
        r |= ((b >> i) & 1) << (7 - i)
    return r


_REVERSE_BYTE_TABLE = np.array([_reverse_byte(b) for b in range(256)], dtype=np.uint64)


def _reverse_word(x: int, n_bytes: int) -> int:
    """Matches liquid_reverse_uint16/24/32 exactly: each byte's bits are
    reversed AND the byte order is reversed -- together a full
    n_bytes*8-bit reversal (see byte_utilities.c)."""
    out = 0
    for i in range(n_bytes):
        byte = (x >> (8 * i)) & 0xFF
        out |= _reverse_byte(byte) << (8 * (n_bytes - 1 - i))
    return out


def _build_table(scheme: str) -> np.ndarray:
    """256-entry reflected-CRC lookup table, built through the SAME
    32-bit register crc.c actually uses (not a register sized to the
    scheme's own width) -- see module docstring for why that distinction
    is load-bearing here, not cosmetic."""
    width_bytes = _POLY_WIDTH_BYTES[scheme]
    poly_orig = _POLY_ORIG[scheme]
    poly = _reverse_byte(poly_orig) if width_bytes == 1 else _reverse_word(poly_orig, width_bytes)
    poly = np.uint64(poly)
    table = np.arange(256, dtype=np.uint64)
    for _ in range(8):
        mask = np.where(table & np.uint64(1), _REG_MASK, np.uint64(0))
        table = ((table >> np.uint64(1)) ^ (poly & mask)) & _REG_MASK
    return table


_TABLES = {scheme: _build_table(scheme) for scheme in _POLY_ORIG}

SCHEMES = ("none", "checksum", "crc8", "crc16", "crc24", "crc32")


class CRC(Block):
    def __init__(self, scheme: str, *, backend=None) -> None:
        super().__init__(backend=backend)
        if scheme not in SCHEMES:
            raise ValueError(f"Unknown CRC scheme {scheme!r}; expected one of {SCHEMES}")
        self.scheme = scheme
        self.key_length = _KEY_LENGTH_BYTES[scheme]  # bytes
        self.batch_shape_doc = (
            f"generate_key(msg): (n_batch, n) uint8 bytes -> (n_batch,) uint32 key. "
            f"append_key(msg): (n_batch, n) -> (n_batch, n+{self.key_length}) uint8. "
            f"check_key(msg_with_key): (n_batch, n+{self.key_length}) -> (n_batch,) bool."
        )

    def _to_host_bytes(self, arr: Any) -> np.ndarray:
        if self.backend == "cupy":
            import cupy

            arr = cupy.asnumpy(arr)
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr.astype(np.uint8)

    def generate_key(self, msg_bytes: Any) -> np.ndarray:
        """msg_bytes: (n_batch, n) uint8 -> (n_batch,) key values (as
        Python/numpy ints, width depending on scheme -- 0 for "none")."""
        msg = self._to_host_bytes(msg_bytes)
        n_batch = msg.shape[0]

        if self.scheme == "none":
            return np.zeros(n_batch, dtype=np.uint64)

        if self.scheme == "checksum":
            # matches checksum_generate_key(): 8-bit two's-complement of
            # the byte sum.
            total = msg.astype(np.int64).sum(axis=1) & 0xFF
            key = ((~total) + 1) & 0xFF
            return key.astype(np.uint64)

        table = _TABLES[self.scheme]
        width_mask = _WIDTH_MASK[self.scheme]
        key = np.full(n_batch, _REG_MASK, dtype=np.uint64)
        n = msg.shape[1]
        for i in range(n):
            b = msg[:, i].astype(np.uint64)
            idx = (key ^ b) & np.uint64(0xFF)
            key = ((key >> np.uint64(8)) ^ table[idx]) & _REG_MASK
        return (~key) & _REG_MASK & width_mask

    def append_key(self, msg_bytes: Any) -> np.ndarray:
        """(n_batch, n) -> (n_batch, n+key_length), key bytes appended
        MSB-first (matches crc_append_key). No-op (returns msg
        unchanged) for scheme="none"."""
        msg = self._to_host_bytes(msg_bytes)
        if self.scheme == "none":
            return msg
        key = self.generate_key(msg)
        n_batch = msg.shape[0]
        key_bytes = np.empty((n_batch, self.key_length), dtype=np.uint8)
        for i in range(self.key_length):
            shift = (self.key_length - i - 1) * 8
            key_bytes[:, i] = (key >> np.uint64(shift)) & np.uint64(0xFF)
        return np.concatenate([msg, key_bytes], axis=-1)

    def check_key(self, msg_with_key_bytes: Any) -> np.ndarray:
        """(n_batch, n+key_length) -> (n_batch,) bool: splits off the
        trailing key_length bytes as the received key and checks it
        against a freshly computed key over the remaining bytes (matches
        crc_check_key/crc_validate_message). Always True for
        scheme="none" (matches liquid: LIQUID_CRC_NONE validates
        unconditionally)."""
        msg_with_key = self._to_host_bytes(msg_with_key_bytes)
        if self.scheme == "none":
            return np.ones(msg_with_key.shape[0], dtype=bool)
        if msg_with_key.shape[-1] < self.key_length:
            raise ValueError(
                f"message of {msg_with_key.shape[-1]} bytes is shorter than "
                f"{self.scheme!r}'s key_length={self.key_length} bytes"
            )
        msg = msg_with_key[:, : -self.key_length]
        received_key_bytes = msg_with_key[:, -self.key_length :]
        received_key = np.zeros(msg_with_key.shape[0], dtype=np.uint64)
        for i in range(self.key_length):
            received_key = (received_key << np.uint64(8)) | received_key_bytes[:, i].astype(np.uint64)
        expected_key = self.generate_key(msg)
        return expected_key == received_key

    def process(self, batch: Any, **kwargs: Any) -> Any:
        """Alias for append_key() -- required to satisfy Block's
        abstract process() contract, matching the same pattern used by
        Modem/FEC. Call check_key() explicitly for the inverse
        direction."""
        return self.append_key(batch)
