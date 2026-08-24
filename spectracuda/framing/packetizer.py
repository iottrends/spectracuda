"""Packetizer: CRC + (up to two-stage) FEC composition as one reusable
object, mirroring liquid-dsp's own `packetizer` (src/fec/src/
packetizer.c) -- chains CRC and FEC into one encode/decode object,
independent of the OFDM modem engine, exactly the separation liquid-dsp
itself keeps (`packetizer_encode`/`packetizer_decode` know nothing about
`ofdmflexframegen`/`ofdmflexframesync`; those call INTO packetizer, not
the other way around).

Extracted out of `Ofdm.generate_frame()`/`rx_process()`, where this
exact CRC-then-FEC (encode) / FEC-then-CRC (decode) logic used to live
inline -- see docs/todo.md #1.1 for the motivating gap ("you can't
reuse 'decode a framed packet' logic outside Ofdm itself"). `Ofdm` now
owns one `Packetizer` instance for its OWN transmit-side choice of crc/
fec0/fec1 (`self.packetizer`), and constructs a second, throwaway
`Packetizer` per `rx_process()` call from whatever the DECODED header
says (matching the same "resolve from the wire, not from self"
principle already established for `mod_scheme`/`fec0`/`fec1`/`crc` --
see pipeline/ofdm.py's class docstring).

Two-stage FEC (`fec`=fec0=inner, `fec1`=outer -- see docs/todo.md #1.2):
liquid-dsp's own `packetizer_create()` docs are explicit about which is
which ("_fec0: inner forward error-correction code", "_fec1: outer
forward error-correction code"), and its `plan[0].fs = fec0`,
`plan[1].fs = fec1` with `packetizer_encode()` running `plan[0]` (fec0)
THEN `plan[1]` (fec1), while `packetizer_decode()` runs them in REVERSE
(`plan[1]`/fec1 first, `plan[0]`/fec0 last) -- verified directly against
`reference/liquid-dsp/src/fec/src/packetizer.c`, not assumed from the
"inner/outer" naming alone. So:

    encode: raw bits -> CRC append -> fec0 (inner) encode -> fec1 (outer)
            encode -> wire bits
    decode: wire bits -> fec1 (outer) decode -> fec0 (inner) decode ->
            CRC strip+check -> raw bits

This is the standard concatenated-code convention -- matching e.g.
DVB-S's RS+convolutional pairing means mapping RS to fec0 and the
convolutional code to fec1 (`Packetizer(fec="rs_m8", fec1="conv_v27")`),
NOT the reverse: DVB applies RS to the raw data FIRST (matching fec0,
encoded first here) and the convolutional code SECOND, closest to the
channel (matching fec1, decoded FIRST here) -- so THIS codebase's
fec0="inner" is the code applied closest to the raw message, and
fec1="outer" is the code applied closest to the channel/decoded first,
regardless of which real-world scheme name goes in which slot. (An
earlier draft of this docstring, and one session explanation, had this
backwards -- caught only once a real conv_v27+rs_m8 demonstration test
was actually built, not before; see docs/todo.md #1.12 for the full
correction.) fec1 defaults to "none" (single-stage, the only mode that
existed before this), so this is purely additive, not a breaking change
to the single-stage API.

Order (single-stage, unchanged): append the CRC key to the raw message
first, THEN FEC-encode the result. CRC mismatches are never raised as
exceptions (matches liquid's `crc_validate_message` -- see
`spectracuda.fec.CRC`'s module docstring for why); FEC failures (e.g. an
uncorrectable codeword, at EITHER stage) DO raise `ValueError`,
propagated from the underlying `FEC.decode()` -- with the stage
(fec1/outer vs fec0/inner) named in the re-raised message, since a
bare "FEC decode failed" wouldn't say which of the two stages actually
gave up.

Interleaving (`interleaver=`, see docs/todo.md #1.12): sits BETWEEN
fec0's encoded output and fec1's input -- `encode(): ... -> fec0 encode
-> interleave -> fec1 encode -> wire bits`, `decode(): wire bits ->
fec1 decode -> DEINTERLEAVE -> fec0 decode -> ...`. Only meaningful with
a real fec1 configured (that's the whole point: spreading fec1's
residual errors -- Viterbi's bursty traceback failures, specifically,
when Viterbi is the fec1/outer scheme, e.g. `fec="rs_m8",
fec1="conv_v27"` -- across fec0's error-correction window before fec0
sees them, since fec1 is decoded FIRST and fec0 LAST); the constructor
raises `ValueError` if `interleaver != "none"` with `fec1 == "none"`,
rather than silently accepting a no-op configuration.
`interleaver=` picks a scheme name registered under
`spectracuda.interleaver` (`"block"`/`"permutation"`/`"convolutional"`/
`"liquid"` -- see that package's own docstring for what each one is);
`interleaver_kwargs` forwards algorithm-specific tuning (`rows=`,
`seed=`, `branches=`/`base_delay=`, `depth=`) to whichever class gets
resolved.

Deliberate, explicit EXCEPTION to this codebase's usual "resolve from
the wire, not from self" rule (the rule mod_scheme/crc/fec0/fec1 all
follow -- see pipeline/ofdm.py's class docstring): interleaver choice
is NOT signaled anywhere in the header, the same way liquid-dsp itself
never signals its own interleaver's depth/parameters over the air
either -- both ends of a real link have to already agree on it
out-of-band, exactly like this project's own preamble_seed/
training_seed. So `Ofdm.rx_process()` builds its throwaway `Packetizer`
using the RECEIVING object's OWN `self.interleaver`/
`self.interleaver_kwargs`, not anything decoded from the frame -- a
receiver misconfigured with the wrong interleaver will fail to decode,
the same way it would with a genuinely mismatched preamble.

n_bits varies per call (whichever of fec0/fec1 sits on the interleaver's
input side -- e.g. `conv_v27`, a streaming code that accepts any k --
can produce a different encoded length for different raw payload
sizes), so interleaver instances are built lazily and cached by n_bits
(`self._interleaver_cache`), not built once at construction time the
way crc_codec/fec_codec/fec1_codec are.

Batch-shape contract: encode(bits) takes (n_batch, k) raw bits ->
(n_batch, n) encoded bits. decode(bits) takes (n_batch, n) encoded bits
-> {"bits": (n_batch, k) decoded raw bits, "crc_valid": (n_batch,) bool
array or None if crc="none"}; may raise ValueError if fec="none" or
either FEC stage can't correct enough errors.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..block import Block
from ..fec import CRC, FEC
from ..registry import resolve

# Importing this registers "block"/"permutation"/"convolutional"/"liquid"
# into the interleaver registry as a side effect (same pattern used
# throughout this codebase for sync/cfo/channel_estimator/equalizer --
# see e.g. pipeline/ofdm.py's own registration imports).
from .. import interleaver as _interleaver  # noqa: F401


class Packetizer(Block):
    """Parameters
    ----------
    crc:
        "none" or one of spectracuda.fec.CRC's schemes ("checksum",
        "crc8", "crc16", "crc24", "crc32").
    fec:
        "none" or one of spectracuda.fec.FEC's schemes ("conv_v27",
        "rs_m8", or any "ldpc_<n>_r<rate>" variant) -- the INNER code
        (liquid-dsp's fec0), applied directly to the CRC-appended raw
        message.
    fec1:
        "none" (default) or a second FEC scheme -- the OUTER code
        (liquid-dsp's fec1), applied around fec0's own output. Rarely
        needed (concatenated coding is a real but fairly specialized
        technique); defaults to "none" so single-stage use (the only
        mode that existed before this parameter) is unaffected.
    interleaver:
        "none" (default) or one of spectracuda.interleaver's schemes
        ("block", "permutation", "convolutional", "liquid"). Only valid
        when fec1 != "none" (see module docstring).
    interleaver_kwargs:
        Optional dict forwarded to whichever interleaver class gets
        resolved (e.g. {"rows": 8} for "block", {"seed": 99} for
        "permutation", {"branches": 6, "base_delay": 4} for
        "convolutional", {"depth": 2} for "liquid").
    """

    def __init__(
        self,
        crc: str = "none",
        fec: str = "none",
        fec1: str = "none",
        *,
        interleaver: str = "none",
        interleaver_kwargs: Optional[Dict[str, Any]] = None,
        backend=None,
    ) -> None:
        super().__init__(backend=backend)
        if interleaver != "none" and fec1 == "none":
            raise ValueError(
                f"interleaver={interleaver!r} requires fec1 != 'none' -- "
                f"interleaving only makes sense between two FEC stages "
                f"(spreading fec0's residual errors before fec1 sees them); "
                f"with no fec1 there's nothing downstream for it to protect "
                f"(see docs/todo.md #1.12)"
            )
        self.crc = crc
        self.fec = fec
        self.fec1 = fec1
        self.interleaver = interleaver
        self.interleaver_kwargs = interleaver_kwargs or {}
        self._interleaver_cache: Dict[int, Block] = {}
        self.crc_codec = None if crc == "none" else CRC(crc, backend=backend)
        self.fec_codec = None if fec == "none" else FEC(fec, backend=backend)
        self.fec1_codec = None if fec1 == "none" else FEC(fec1, backend=backend)
        self.crc_key_length_bytes = self.crc_codec.key_length if self.crc_codec is not None else 0
        self.batch_shape_doc = (
            "encode(bits): (n_batch, k) raw bits -> (n_batch, n) encoded bits "
            "(CRC append -> fec0/inner encode -> [interleave] -> fec1/outer "
            "encode). decode(bits): (n_batch, n) encoded bits -> "
            "{'bits': (n_batch, k), 'crc_valid': (n_batch,) bool array or None} "
            "(fec1/outer decode -> [deinterleave] -> fec0/inner decode -> "
            "CRC strip+check)."
        )

    def _get_interleaver(self, n_bits: int) -> Block:
        if n_bits not in self._interleaver_cache:
            self._interleaver_cache[n_bits] = resolve(
                "interleaver", self.interleaver, n_bits=n_bits, backend=self.backend, **self.interleaver_kwargs
            )
        return self._interleaver_cache[n_bits]

    def _to_host(self, arr: Any) -> np.ndarray:
        if self.backend == "cupy":
            import cupy

            return cupy.asnumpy(arr)
        return np.asarray(arr)

    def _bits_to_bytes(self, bits: Any) -> np.ndarray:
        """(n_batch, n_bits) -> (n_batch, n_bits//8) uint8, MSB-first --
        host-side, tiny metadata work (same rationale as Ofdm's own
        _to_host: this is framing bookkeeping, not bulk DSP)."""
        host_bits = self._to_host(bits).astype("uint8")
        return np.packbits(host_bits, axis=-1)

    def _bytes_to_bits(self, byte_arr: np.ndarray) -> Any:
        """Inverse of _bits_to_bytes, returned on self.xp."""
        bits = np.unpackbits(byte_arr, axis=-1)
        return self.xp.asarray(bits)

    def encoded_length(self, raw_bit_count: int) -> int:
        """Bit count encode() produces for a given raw (pre-CRC, pre-FEC)
        bit count -- used by callers (e.g. Ofdm) that need to know how
        many bits/symbols to expect before actually decoding."""
        if self.crc_codec is not None and raw_bit_count % 8 != 0:
            raise ValueError(
                f"raw_bit_count={raw_bit_count} is not a multiple of 8 -- "
                f"crc={self.crc!r} operates on whole bytes"
            )
        pre_fec_bit_count = raw_bit_count + self.crc_key_length_bytes * 8
        after_fec0 = (
            self.fec_codec.encoded_length(pre_fec_bit_count)
            if self.fec_codec is not None
            else pre_fec_bit_count
        )
        if self.fec1_codec is not None:
            return self.fec1_codec.encoded_length(after_fec0)
        return after_fec0

    def encode(self, bits: Any) -> Any:
        xp = self.xp
        bits = xp.asarray(bits)
        if bits.ndim == 1:
            bits = bits[None, :]
        raw_bit_count = bits.shape[-1]

        if self.crc_codec is not None:
            if raw_bit_count % 8 != 0:
                raise ValueError(
                    f"payload bit count {raw_bit_count} is not a multiple of 8 -- "
                    f"crc={self.crc!r} operates on whole bytes (matching liquid-"
                    f"dsp's own byte-oriented CRC/packetizer); pad the raw "
                    f"payload to a byte boundary yourself for now"
                )
            payload_bytes = self._bits_to_bytes(bits)
            bits_with_crc = self._bytes_to_bits(self.crc_codec.append_key(payload_bytes))
        else:
            bits_with_crc = bits

        # fec0 (inner) first, THEN fec1 (outer) -- see module docstring
        # for the verified liquid-dsp packetizer_encode order.
        after_fec0 = self.fec_codec.encode(bits_with_crc) if self.fec_codec is not None else bits_with_crc

        if self.interleaver != "none":
            after_fec0 = self._get_interleaver(after_fec0.shape[-1]).encode(after_fec0)

        return self.fec1_codec.encode(after_fec0) if self.fec1_codec is not None else after_fec0

    def decode(self, bits: Any) -> Dict[str, Any]:
        """Self-contained: unlike Ofdm's own MAX_PAYLOAD_SYMBOLS/expected-
        length bookkeeping (which needs the decoded header's
        payload_len_bits to know how many OFDM symbols to gather in the
        first place), decode() here only needs the already-gathered
        encoded bits -- FEC.decoded_length() derives each stage's input
        bit count from its own output length alone, and
        crc_key_length_bytes is fixed by this object's own crc= choice."""
        xp = self.xp
        bits = xp.asarray(bits)
        if bits.ndim == 1:
            bits = bits[None, :]

        # fec1 (outer) decoded FIRST, THEN fec0 (inner) -- the reverse of
        # encode(), exactly liquid-dsp's packetizer_decode order (see
        # module docstring). Stage named in a re-raised ValueError so a
        # decode failure says WHICH stage gave up, not just "FEC failed".
        if self.fec1_codec is not None:
            try:
                after_fec1 = self.fec1_codec.decode(bits)
            except ValueError as exc:
                raise ValueError(f"fec1 (outer, {self.fec1!r}) decode failed: {exc}") from exc
        else:
            after_fec1 = bits

        if self.interleaver != "none":
            after_fec1 = self._get_interleaver(after_fec1.shape[-1]).decode(after_fec1)

        if self.fec_codec is not None:
            try:
                bits_with_crc = self.fec_codec.decode(after_fec1)
            except ValueError as exc:
                raise ValueError(f"fec0 (inner, {self.fec!r}) decode failed: {exc}") from exc
        else:
            bits_with_crc = after_fec1

        if self.crc_codec is not None:
            bytes_with_crc = self._bits_to_bytes(bits_with_crc)
            crc_valid = self.crc_codec.check_key(bytes_with_crc)
            raw_bytes = bytes_with_crc[:, : -self.crc_key_length_bytes]
            raw_bits = self._bytes_to_bits(raw_bytes)
        else:
            crc_valid = None
            raw_bits = bits_with_crc

        return {"bits": raw_bits, "crc_valid": crc_valid}

    def process(self, batch: Any, **kwargs: Any) -> Any:
        """Alias for encode() -- required to satisfy Block's abstract
        process() contract, matching the same pattern used by Modem/
        FEC/CRC. Call decode() explicitly for the inverse direction."""
        return self.encode(batch)
