"""FEC: FEC(scheme) -- one class, scheme-name string, mirrors
liquid-dsp's fec_create(scheme) directly (same pattern as Modem(scheme)).

liquid-dsp-parity scope: "conv_v27" (rate-1/2, K=7 convolutional/
Viterbi) and "rs_m8" (RS(255,223) over GF(256)) -- exactly liquid-dsp's
own LIQUID_FEC_CONV_V27 and LIQUID_FEC_RS_M8. Neither is actually ported
from liquid-dsp -- see fec/viterbi.py's and fec/reed_solomon.py's module
docstrings: liquid-dsp wraps Phil Karn's external `libfec` C library for
BOTH, with no fallback when it's absent. Both implementations here are
from-scratch, using the same well-documented standard algorithms/
parameters liquid-dsp's underlying library would use.

Deliberate scope expansion BEYOND liquid-dsp parity: the 12-variant
IEEE 802.11n QC-LDPC family ("ldpc_648_r12" ... "ldpc_1944_r56", see
fec/ldpc.py/fec/ldpc_tables.py). liquid-dsp has no LDPC at all
(docs/liquid-dsp-api-inventory.md and docs/todo.md both listed this as
a "confirmed non-gap" for that reason) -- added anyway, the same
reasoning pattern already used for LSChannelEstimator/ZFEqualizer/
MMSEEqualizer (no liquid-dsp precedent -> design from a standard
reference instead of deferring).

Uniform bits-in/bits-out interface regardless of scheme, matching
Modem's convention. Three distinct underlying shapes get folded into
that one interface:
  - `ConvolutionalCode`: a streaming code, bit-by-bit, accepts any
    length k directly.
  - `ReedSolomonCode`: GF(256) byte SYMBOLS in fixed 223-symbol blocks
    (its own natural unit) -- this class packs/unpacks bits<->bytes for
    "rs_m8" (`_SYMBOL_LEVEL_SCHEMES`).
  - `LDPCCode`: fixed-size bit BLOCKS (like RS's block-ness, but
    already operating in raw bits, not byte symbols -- no packbits/
    unpackbits step needed, just a reshape) for all 12 "ldpc_*"
    variants (`_BLOCK_BIT_SCHEMES`).
Both block-shaped kinds transparently chunk longer payloads into
multiple codewords by folding extra blocks into the underlying code's
own batch dimension (encoding/decoding them all in one call, then
reshaping back). Neither actually requires the payload to be an exact
multiple of `k_bits` -- both accept N full blocks plus, if anything's
left over, one "shortened" leftover block covering exactly that
remainder (see `ReedSolomonCode`'s and `LDPCCode`'s own encode()/
decode() docstrings for the technique, `_encode_symbol_level`/
`_encode_block_level` below for the chunking) -- `FEC.
accepts_partial_block` is `True` for both.

Batch-shape contract: encode(bits) takes (n_batch, k) bits ->
(n_batch, n) bits; decode(bits, **kwargs) is the inverse (may raise
ValueError if a codeword has more errors than the scheme can correct --
see the underlying class's docstring for what "more" means for that
scheme; LDPC's decode() also accepts `p=`/`max_iterations=` kwargs,
forwarded through). k/n depend on the scheme and, for "rs_m8"/"ldpc_*",
on how many blocks the payload spans -- use encoded_length()/
decoded_length() to compute the exact sizes rather than hardcoding a
formula per scheme (this is what `Ofdm` does internally).
"""
from __future__ import annotations

import functools
from typing import Any

import numpy as np

from ..block import Block
from .ldpc import LDPCCode
from .ldpc_tables import BASE_MATRICES
from .reed_solomon import ReedSolomonCode
from .viterbi import ConvolutionalCode

_SCHEMES = {
    "conv_v27": ConvolutionalCode,
    "rs_m8": ReedSolomonCode,
}
for _variant in BASE_MATRICES:
    _SCHEMES[_variant] = functools.partial(LDPCCode, variant=_variant)
del _variant

_SYMBOL_LEVEL_SCHEMES = {"rs_m8"}  # these operate on 8-bit symbols internally, not raw bits
_BLOCK_BIT_SCHEMES = set(BASE_MATRICES)  # fixed-size bit blocks, already raw-bit (no byte-packing)


class FEC(Block):
    def __init__(self, scheme: str, *, backend=None) -> None:
        super().__init__(backend=backend)
        if scheme not in _SCHEMES:
            raise ValueError(f"Unknown FEC scheme {scheme!r}; expected one of {sorted(_SCHEMES)}")
        self.scheme = scheme
        self._impl = _SCHEMES[scheme](backend=backend)
        # k_bits alone doesn't say whether a non-multiple length is
        # actually usable (rs_m8: yes, via shortening -- see
        # encoded_length()'s docstring; ldpc_*: no, still exact-multiple
        # only, a documented separate gap) -- callers like
        # mac/capacity.py need this to know which search strategy is
        # even correct to run.
        self.accepts_partial_block = scheme in _SYMBOL_LEVEL_SCHEMES or scheme in _BLOCK_BIT_SCHEMES

        if scheme in _SYMBOL_LEVEL_SCHEMES:
            self.k_bits = self._impl.k * 8
            self.n_bits = self._impl.n * 8
            self.batch_shape_doc = (
                f"encode: (n_batch, m*{self.k_bits}) bits -> (n_batch, m*{self.n_bits}) "
                f"bits for any m>=1 blocks (packed 8 bits/symbol internally "
                f"for {scheme!r}). decode: the inverse; raises ValueError if "
                f"any block has more than t={self._impl.t} symbol errors."
            )
        elif scheme in _BLOCK_BIT_SCHEMES:
            self.k_bits = self._impl.k
            self.n_bits = self._impl.n
            self.batch_shape_doc = (
                f"encode: (n_batch, m*{self.k_bits} + r) bits for any m>=0 full "
                f"blocks plus an optional shortened leftover r (0 < r < "
                f"{self.k_bits}, see ldpc.py's own shortened-codeword docstring) "
                f"-> (n_batch, m*{self.n_bits} + (r + n_checks)) bits ({scheme!r}'s "
                f"own block size -- no byte-packing needed, LDPC already operates "
                f"on raw bits). decode: the inverse; raises ValueError if BP "
                f"doesn't converge to a zero-syndrome codeword within "
                f"max_iterations."
            )
        else:
            self.tail_bits = self._impl.tail_bits
            self.batch_shape_doc = self._impl.batch_shape_doc

    def _to_host(self, arr: Any) -> np.ndarray:
        if self.backend == "cupy":
            import cupy

            return cupy.asnumpy(arr)
        return np.asarray(arr)

    def encoded_length(self, k: int) -> int:
        """Bit count encode() produces for a given raw bit count k.

        rs_m8 ("shortened" -- see reed_solomon.py's encode() docstring)
        and ldpc_* (`_BLOCK_BIT_SCHEMES`, "shortened" -- see ldpc.py's
        own encode() docstring, same technique): k no longer has to be
        an exact multiple of k_bits for EITHER kind. It's N full k_bits
        blocks plus, if there's anything left over, ONE shortened block
        covering exactly that leftover (never padded up to a full
        block). For rs_m8: real_k=leftover/8 symbols in, real_k+nroots
        symbols (leftover + 8*nroots bits) out. For ldpc_*: real_k=
        leftover bits in, real_k+n_checks bits out (n_checks = n_bits -
        k_bits). This was a real bug fix for rs_m8, not a relaxation for
        its own sake: this exact-multiple requirement is what made
        Mac(ofdm_kwargs=dict(fec="rs_m8")) derive an unusable 8-bit
        max_segment_bits, and made every non-block-sized message
        (starting with the bind handshake itself) fail outright -- see
        docs/mac.md's writeup. ldpc_* hit the identical wall (confirmed
        directly, same symptom, same root cause) until LDPC's own
        shortened-codeword support closed it the same way."""
        if self.scheme in _SYMBOL_LEVEL_SCHEMES:
            n_full_blocks, leftover = divmod(k, self.k_bits)
            length = n_full_blocks * self.n_bits
            if leftover > 0:
                length += leftover + 8 * self._impl.nroots
            return length
        if self.scheme in _BLOCK_BIT_SCHEMES:
            n_full_blocks, leftover = divmod(k, self.k_bits)
            length = n_full_blocks * self.n_bits
            if leftover > 0:
                length += leftover + (self.n_bits - self.k_bits)  # + n_checks
            return length
        return 2 * (k + self.tail_bits)  # conv_v27: rate 1/2 + zero-tail

    def decoded_length(self, n: int) -> int:
        """Inverse of encoded_length(): raw bit count decode() produces
        for a given encoded bit count n. rs_m8/ldpc_*: see
        encoded_length()'s docstring -- the shortened leftover block (if
        any) is always strictly smaller than one full block's encoded
        size, so `n // n_bits` unambiguously recovers the same N full
        blocks encoded_length() started from; the remainder is exactly
        that leftover block's own encoded size."""
        if self.scheme in _SYMBOL_LEVEL_SCHEMES:
            n_full_blocks, leftover_enc = divmod(n, self.n_bits)
            length = n_full_blocks * self.k_bits
            if leftover_enc > 0:
                length += leftover_enc - 8 * self._impl.nroots
            return length
        if self.scheme in _BLOCK_BIT_SCHEMES:
            n_full_blocks, leftover_enc = divmod(n, self.n_bits)
            length = n_full_blocks * self.k_bits
            if leftover_enc > 0:
                length += leftover_enc - (self.n_bits - self.k_bits)  # - n_checks
            return length
        if n % 2 != 0:
            raise ValueError(f"n={n} is not even (conv_v27 is rate 1/2)")
        k = n // 2 - self.tail_bits
        if k < 0:
            raise ValueError(f"n={n} is too short to contain the {self.tail_bits}-bit tail")
        return k

    def _pack_bits_to_symbols(self, bits: Any, expected_bits: int) -> np.ndarray:
        """bits: (n_batch, m*expected_bits) for any m>=1 blocks ->
        (n_batch*m, expected_bits//8) symbols, folding extra blocks into
        the batch dimension so ReedSolomonCode's own batched encode/
        decode handles them all in one call."""
        host_bits = self._to_host(bits).astype("uint8")
        if host_bits.ndim == 1:
            host_bits = host_bits[None, :]
        n_batch, total_bits = host_bits.shape
        if total_bits % expected_bits != 0:
            raise ValueError(
                f"bit count {total_bits} is not a multiple of {expected_bits} "
                f"({self.scheme}'s block size)"
            )
        n_blocks = total_bits // expected_bits
        reshaped = host_bits.reshape(n_batch * n_blocks, expected_bits)
        return np.packbits(reshaped, axis=-1), n_batch, n_blocks

    def _unpack_symbols_to_bits(self, symbols: Any, n_batch: int, n_blocks: int) -> np.ndarray:
        host_symbols = self._to_host(symbols).astype("uint8")
        bits = np.unpackbits(host_symbols, axis=-1)
        return bits.reshape(n_batch, n_blocks * bits.shape[-1])

    def _pack_bits_to_blocks(self, bits: Any, expected_bits: int):
        """bits: (n_batch, m*expected_bits) for any m>=1 blocks ->
        (n_batch*m, expected_bits), folding extra blocks into the batch
        dimension -- same chunking idea as _pack_bits_to_symbols, but no
        byte-packing step: LDPC (unlike rs_m8) already operates directly
        on raw bits, so this stays a pure reshape on self.xp, no host
        round-trip needed."""
        xp = self.xp
        bits = xp.asarray(bits)
        if bits.ndim == 1:
            bits = bits[None, :]
        n_batch, total_bits = bits.shape
        if total_bits % expected_bits != 0:
            raise ValueError(
                f"bit count {total_bits} is not a multiple of {expected_bits} "
                f"({self.scheme}'s block size)"
            )
        n_blocks = total_bits // expected_bits
        return bits.reshape(n_batch * n_blocks, expected_bits), n_batch, n_blocks

    def _unpack_blocks_to_bits(self, blocks: Any, n_batch: int, n_blocks: int) -> Any:
        xp = self.xp
        blocks = xp.asarray(blocks)
        return blocks.reshape(n_batch, n_blocks * blocks.shape[-1])

    def _encode_block_level(self, bits: Any) -> Any:
        """ldpc_* encode: N full k_bits blocks (existing batched path,
        unchanged) + at most one shortened leftover block (ldpc.py's own
        shortened-codeword support) -- mirrors _encode_symbol_level()
        above exactly, minus the byte-packing step (LDPC already
        operates on raw bits, no packbits/unpackbits needed)."""
        xp = self.xp
        bits = xp.asarray(bits)
        if bits.ndim == 1:
            bits = bits[None, :]
        n_batch, total_bits = bits.shape
        n_full_blocks, leftover_bits = divmod(total_bits, self.k_bits)

        parts = []
        if n_full_blocks > 0:
            full_bits = bits[:, : n_full_blocks * self.k_bits]
            blocks, nb, nblk = self._pack_bits_to_blocks(full_bits, self.k_bits)
            encoded_blocks = self._impl.encode(blocks)
            parts.append(self._unpack_blocks_to_bits(encoded_blocks, nb, nblk))
        if leftover_bits > 0:
            leftover = bits[:, n_full_blocks * self.k_bits :]
            encoded_leftover = self._impl.encode(leftover)  # (n_batch, leftover_bits + n_checks)
            parts.append(encoded_leftover)
        encoded = parts[0] if len(parts) == 1 else xp.concatenate(parts, axis=-1)
        return encoded

    def _decode_block_level(self, bits: Any, **kwargs: Any) -> Any:
        """Inverse of _encode_block_level() -- see its docstring.
        **kwargs (p=, max_iterations=) forwarded to LDPCCode.decode()
        for both the full-block batch and the shortened leftover, same
        as the un-shortened path already did."""
        xp = self.xp
        bits = xp.asarray(bits)
        if bits.ndim == 1:
            bits = bits[None, :]
        n_batch, total_bits = bits.shape
        n_full_blocks, leftover_enc_bits = divmod(total_bits, self.n_bits)

        parts = []
        if n_full_blocks > 0:
            full_bits = bits[:, : n_full_blocks * self.n_bits]
            blocks, nb, nblk = self._pack_bits_to_blocks(full_bits, self.n_bits)
            decoded_blocks = self._impl.decode(blocks, **kwargs)  # may raise ValueError
            parts.append(self._unpack_blocks_to_bits(decoded_blocks, nb, nblk))
        if leftover_enc_bits > 0:
            leftover = bits[:, n_full_blocks * self.n_bits :]
            decoded_leftover = self._impl.decode(leftover, **kwargs)  # may raise ValueError; real_k inferred from leftover's own length
            parts.append(decoded_leftover)
        decoded = parts[0] if len(parts) == 1 else xp.concatenate(parts, axis=-1)
        return decoded

    def _encode_symbol_level(self, bits: Any) -> Any:
        """rs_m8 encode: N full k_bits blocks (existing batched path,
        unchanged) + at most one shortened leftover block (reed_solomon.py's
        new shortened-code support, Step 1) -- see encoded_length()'s
        docstring for why this is split this way, not padded."""
        xp = self.xp
        host_bits = self._to_host(bits).astype("uint8")
        if host_bits.ndim == 1:
            host_bits = host_bits[None, :]
        n_batch, total_bits = host_bits.shape
        n_full_blocks, leftover_bits = divmod(total_bits, self.k_bits)
        if leftover_bits % 8 != 0:
            # np.packbits below would otherwise SILENTLY zero-pad this up
            # to a byte boundary rather than raise -- a real bug caught
            # here (not by inspection): encode() would succeed, but
            # decode() has no way to know the true length was 100 bits
            # and not 104, so the round trip silently comes back wrong,
            # not loud. Same "fail loud on a genuinely bad shape"
            # convention this codebase already uses elsewhere (e.g.
            # Packetizer.encode()'s own CRC byte-alignment check).
            raise ValueError(
                f"leftover bit count {leftover_bits} (after {n_full_blocks} full "
                f"{self.k_bits}-bit blocks) is not a multiple of 8 -- rs_m8 packs "
                f"whole bytes into symbols, a non-byte-aligned leftover can't be "
                f"represented"
            )

        parts = []
        if n_full_blocks > 0:
            full_bits = host_bits[:, : n_full_blocks * self.k_bits]
            symbols, nb, nblk = self._pack_bits_to_symbols(full_bits, self.k_bits)
            encoded_symbols = self._impl.encode(symbols)
            parts.append(self._unpack_symbols_to_bits(encoded_symbols, nb, nblk))
        if leftover_bits > 0:
            leftover_symbols = np.packbits(host_bits[:, n_full_blocks * self.k_bits :], axis=-1)
            encoded_leftover = self._impl.encode(leftover_symbols)
            parts.append(np.unpackbits(encoded_leftover, axis=-1))
        encoded = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=-1)
        return xp.asarray(encoded)

    def _decode_symbol_level(self, bits: Any) -> Any:
        """Inverse of _encode_symbol_level() -- see its docstring."""
        xp = self.xp
        host_bits = self._to_host(bits).astype("uint8")
        if host_bits.ndim == 1:
            host_bits = host_bits[None, :]
        n_batch, total_bits = host_bits.shape
        n_full_blocks, leftover_enc_bits = divmod(total_bits, self.n_bits)
        if leftover_enc_bits % 8 != 0:
            # Same real bug as _encode_symbol_level()'s check -- see its
            # comment. A leftover that's byte-aligned but still too short
            # to contain a real payload (<= 8*nroots bits, i.e. real_k
            # would be <= 0) is caught separately, downstream, by
            # ReedSolomonCode.decode()'s own real_k range check.
            raise ValueError(
                f"leftover encoded bit count {leftover_enc_bits} (after "
                f"{n_full_blocks} full {self.n_bits}-bit blocks) is not a "
                f"multiple of 8 -- not a genuine rs_m8-encoded stream"
            )

        parts = []
        if n_full_blocks > 0:
            full_bits = host_bits[:, : n_full_blocks * self.n_bits]
            symbols, nb, nblk = self._pack_bits_to_symbols(full_bits, self.n_bits)
            decoded_symbols = self._impl.decode(symbols)  # may raise ValueError
            parts.append(self._unpack_symbols_to_bits(decoded_symbols, nb, nblk))
        if leftover_enc_bits > 0:
            leftover_symbols = np.packbits(host_bits[:, n_full_blocks * self.n_bits :], axis=-1)
            decoded_leftover = self._impl.decode(leftover_symbols)  # may raise ValueError
            parts.append(np.unpackbits(decoded_leftover, axis=-1))
        decoded = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=-1)
        return xp.asarray(decoded)

    def encode(self, bits: Any) -> Any:
        xp = self.xp
        if self.scheme in _SYMBOL_LEVEL_SCHEMES:
            return self._encode_symbol_level(bits)
        if self.scheme in _BLOCK_BIT_SCHEMES:
            return self._encode_block_level(bits)
        return self._impl.encode(bits)

    def decode(self, bits: Any, **kwargs: Any) -> Any:
        xp = self.xp
        if self.scheme in _SYMBOL_LEVEL_SCHEMES:
            return self._decode_symbol_level(bits)
        if self.scheme in _BLOCK_BIT_SCHEMES:
            return self._decode_block_level(bits, **kwargs)
        return self._impl.decode(bits)

    def process(self, batch: Any, **kwargs: Any) -> Any:
        """Alias for encode() -- required to satisfy Block's abstract
        process() contract, matching the same pattern used by Modem.
        Call decode() explicitly for the inverse direction."""
        return self.encode(batch)
