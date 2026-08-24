"""HeaderCodec: encode/decode the 112-bit liquid-dsp-style OFDM frame
header (protocol_version, payload_len_bits, mod_scheme, crc0, fec0,
fec1, user_data) to/from a scrambled bit array.

Extracted out of `Ofdm` itself: this is a pure bit-level codec (no
OFDM/IQ/subcarrier dependency at all) -- liquid-dsp keeps the
equivalent logic (`ofdmflexframegen_encode_header`/`_decode_header`)
folded into its OFDM framing internals, but spectracuda separates it out
so "encode/decode a frame header" is reusable/testable independently of
any OFDM machinery (see docs/todo.md #1.1 for the motivating gap: this
used to live as ~50 lines inline inside `Ofdm._encode_header_bits`/
`_decode_header_bits`).

`Ofdm` still owns turning these 112 bits into/from actual OFDM symbols
(spreading them across the header's dedicated symbol(s) for frequency
diversity, BPSK-modulating, scattering onto subcarriers) -- that part
genuinely IS OFDM-specific (see `pipeline/ofdm.py`'s
`_build_header_symbols`/`_decode_header_symbols`, which call into this
module's `encode_bits`/`decode_bits` for the bit-level part only).
`Ofdm` keeps thin `_encode_header_bits`/`_decode_header_bits` wrapper
methods that just delegate here, so existing call sites/tests keep
working unchanged.

Field-code tables (`MOD_SCHEME_CODES`, `CRC_SCHEME_CODES`,
`FEC_SCHEME_CODES` and their `_NAMES` reverses) live here too, not in
`pipeline/ofdm.py` -- they're header-wire-format concerns, not OFDM
ones. `FEC_SCHEME_CODES` codes match liquid-dsp's own numbering for
mod_scheme/crc where a liquid-dsp precedent exists; the 12 LDPC
variants (a deliberate scope expansion beyond liquid-dsp parity -- see
spectracuda.fec's module docstring) are assigned codes 3-14 in sorted
order, deterministic/auto-updating if ldpc_tables.BASE_MATRICES ever
gains more variants.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..fec.ldpc_tables import BASE_MATRICES as _LDPC_BASE_MATRICES

MOD_SCHEME_CODES = {"bpsk": 0, "qpsk": 1, "qam16": 2, "qam64": 3, "qam256": 4}
MOD_SCHEME_NAMES = {v: k for k, v in MOD_SCHEME_CODES.items()}

# crc0 codes match liquid-dsp's own crc_scheme enum values exactly
# (LIQUID_CRC_UNKNOWN=0 is deliberately unrepresented here -- a decoded
# 0 raises ValueError, not NotImplementedError, since it's an invalid/
# reserved code rather than a real-but-unsupported one).
CRC_SCHEME_CODES = {"none": 1, "checksum": 2, "crc8": 3, "crc16": 4, "crc24": 5, "crc32": 6}
CRC_SCHEME_NAMES = {v: k for k, v in CRC_SCHEME_CODES.items()}

# fec0 is a 5-bit field (0-31); "none"/conv_v27/rs_m8 take 0-2, and the
# 12 IEEE 802.11n LDPC variants take 3-14. 17 codes used of 31.
FEC_SCHEME_CODES = {"none": 0, "conv_v27": 1, "rs_m8": 2}
for _i, _variant in enumerate(sorted(_LDPC_BASE_MATRICES)):
    FEC_SCHEME_CODES[_variant] = 3 + _i
del _i, _variant
FEC_SCHEME_NAMES = {v: k for k, v in FEC_SCHEME_CODES.items()}


class HeaderCodec:
    """112-bit header <-> field-dict codec. Stateless besides its fixed
    scrambling mask (see pipeline/ofdm.py's class docstring for why
    scrambling exists: unscrambled, mostly-repeated header content
    constructively interferes into a massive time-domain PAPR spike --
    a real bug found during development, not a defensive-only measure).

        byte 0:      protocol/version                 8 bits
        bytes 1-2:   payload length, in BITS           16 bits
        byte 3:      mod_scheme (payload's modulation)  8 bits
        byte 4:      crc_type(3b) + fec0(5b)            8 bits
        byte 5:      fec1                                8 bits
        bytes 6-13:  user-defined data (8 bytes)        64 bits
                                                        --------
                                                        112 bits
    """

    HEADER_LEN_BITS = 112
    #: Bumped if the wire format changes; decoded and currently
    #: unchecked beyond being read back in the header dict (no
    #: compatibility logic yet -- a real protocol would reject
    #: mismatched versions).
    PROTOCOL_VERSION = 1

    def __init__(self, scramble_seed: int = 42) -> None:
        self.scramble_seed = scramble_seed
        self._scramble_mask = np.random.default_rng(scramble_seed).integers(
            0, 2, size=self.HEADER_LEN_BITS
        ).astype("uint8")

    def encode_bits(
        self,
        payload_len_bits: int,
        mod_scheme: str,
        fec0: str,
        user_data: Optional[bytes],
        crc0: str = "none",
        fec1: str = "none",
    ) -> np.ndarray:
        """Build the 112-bit (14-byte) header content, then scramble it
        with the fixed mask. Returns a plain-numpy uint8 bit array,
        MSB-first. fec0 is the INNER code, fec1 the OUTER code (matching
        liquid-dsp's own packetizer_create() convention -- see
        spectracuda/framing/packetizer.py's module docstring for the
        verified encode/decode ordering); fec1 defaults to "none"
        (single-stage FEC, unchanged from before this parameter existed)."""
        if not (0 <= payload_len_bits < 2 ** 16):
            raise ValueError(
                f"payload_len_bits={payload_len_bits} doesn't fit in the "
                f"header's 16-bit field (max 65535)"
            )
        if mod_scheme not in MOD_SCHEME_CODES:
            raise ValueError(
                f"mod_scheme={mod_scheme!r} has no header code; supported: "
                f"{sorted(MOD_SCHEME_CODES)}"
            )
        if fec0 not in FEC_SCHEME_CODES:
            raise ValueError(
                f"fec0={fec0!r} has no header code; supported: {sorted(FEC_SCHEME_CODES)}"
            )
        if fec1 not in FEC_SCHEME_CODES:
            raise ValueError(
                f"fec1={fec1!r} has no header code; supported: {sorted(FEC_SCHEME_CODES)}"
            )
        if crc0 not in CRC_SCHEME_CODES:
            raise ValueError(
                f"crc0={crc0!r} has no header code; supported: {sorted(CRC_SCHEME_CODES)}"
            )
        if user_data is None:
            user_data = bytes(8)
        else:
            user_data = bytes(user_data)
            if len(user_data) != 8:
                raise ValueError(f"user_data must be exactly 8 bytes, got {len(user_data)}")

        header_bytes = bytearray(14)
        header_bytes[0] = self.PROTOCOL_VERSION
        header_bytes[1] = (payload_len_bits >> 8) & 0xFF
        header_bytes[2] = payload_len_bits & 0xFF
        header_bytes[3] = MOD_SCHEME_CODES[mod_scheme]
        header_bytes[4] = ((CRC_SCHEME_CODES[crc0] & 0x07) << 5) | (FEC_SCHEME_CODES[fec0] & 0x1F)
        header_bytes[5] = FEC_SCHEME_CODES[fec1] & 0x1F
        header_bytes[6:14] = user_data

        bits = np.unpackbits(np.frombuffer(bytes(header_bytes), dtype=np.uint8))  # 112 bits, MSB-first
        return bits ^ self._scramble_mask

    def decode_bits(self, bits: np.ndarray) -> Dict[str, Any]:
        """Inverse of encode_bits. Raises ValueError if the decoded crc,
        fec0, or fec1 code isn't a known scheme (likely header
        corruption -- LIQUID_CRC_UNKNOWN=0 is deliberately unrepresented
        in CRC_SCHEME_NAMES for exactly this reason)."""
        unscrambled = np.asarray(bits, dtype="uint8") ^ self._scramble_mask
        header_bytes = np.packbits(unscrambled).tobytes()

        protocol_version = header_bytes[0]
        payload_len_bits = (header_bytes[1] << 8) | header_bytes[2]
        mod_scheme_code = header_bytes[3]
        crc_code = (header_bytes[4] >> 5) & 0x07
        fec0_code = header_bytes[4] & 0x1F
        fec1_code = header_bytes[5] & 0x1F
        user_data = bytes(header_bytes[6:14])

        if mod_scheme_code not in MOD_SCHEME_NAMES:
            raise ValueError(
                f"decoded mod_scheme code {mod_scheme_code} is not a known "
                f"scheme -- likely header corruption"
            )
        if crc_code not in CRC_SCHEME_NAMES:
            raise ValueError(
                f"decoded CRC scheme code {crc_code} is not a known scheme "
                f"-- likely header corruption"
            )
        if fec0_code not in FEC_SCHEME_NAMES:
            raise ValueError(
                f"decoded FEC0 scheme code {fec0_code} is not a known "
                f"scheme -- likely header corruption"
            )
        if fec1_code not in FEC_SCHEME_NAMES:
            raise ValueError(
                f"decoded FEC1 scheme code {fec1_code} is not a known "
                f"scheme -- likely header corruption"
            )

        return {
            "protocol_version": int(protocol_version),
            "payload_len_bits": int(payload_len_bits),
            "mod_scheme": MOD_SCHEME_NAMES[mod_scheme_code],
            "crc": CRC_SCHEME_NAMES[crc_code],
            "fec0": FEC_SCHEME_NAMES[fec0_code],
            "fec1": FEC_SCHEME_NAMES[fec1_code],
            "user_data": user_data,
        }
