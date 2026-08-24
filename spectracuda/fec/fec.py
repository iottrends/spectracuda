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
reshaping back) -- callers never need to know the block size divides
their payload, only that it must (`FEC(scheme)` accepts any bit count
that's a multiple of `k_bits`).

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
                f"encode: (n_batch, m*{self.k_bits}) bits -> (n_batch, m*{self.n_bits}) "
                f"bits for any m>=1 blocks ({scheme!r}'s own block size -- no byte-"
                f"packing needed, LDPC already operates on raw bits). decode: the "
                f"inverse; raises ValueError if BP doesn't converge to a zero-"
                f"syndrome codeword within max_iterations."
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
        """Bit count encode() produces for a given raw bit count k."""
        if self.scheme in _SYMBOL_LEVEL_SCHEMES or self.scheme in _BLOCK_BIT_SCHEMES:
            if k % self.k_bits != 0:
                raise ValueError(
                    f"k={k} is not a multiple of {self.scheme}'s block size "
                    f"({self.k_bits} bits)"
                )
            return (k // self.k_bits) * self.n_bits
        return 2 * (k + self.tail_bits)  # conv_v27: rate 1/2 + zero-tail

    def decoded_length(self, n: int) -> int:
        """Inverse of encoded_length(): raw bit count decode() produces
        for a given encoded bit count n."""
        if self.scheme in _SYMBOL_LEVEL_SCHEMES or self.scheme in _BLOCK_BIT_SCHEMES:
            if n % self.n_bits != 0:
                raise ValueError(
                    f"n={n} is not a multiple of {self.scheme}'s codeword "
                    f"size ({self.n_bits} bits)"
                )
            return (n // self.n_bits) * self.k_bits
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

    def encode(self, bits: Any) -> Any:
        xp = self.xp
        if self.scheme in _SYMBOL_LEVEL_SCHEMES:
            symbols, n_batch, n_blocks = self._pack_bits_to_symbols(bits, self.k_bits)
            encoded_symbols = self._impl.encode(symbols)
            return xp.asarray(self._unpack_symbols_to_bits(encoded_symbols, n_batch, n_blocks))
        if self.scheme in _BLOCK_BIT_SCHEMES:
            blocks, n_batch, n_blocks = self._pack_bits_to_blocks(bits, self.k_bits)
            encoded_blocks = self._impl.encode(blocks)
            return self._unpack_blocks_to_bits(encoded_blocks, n_batch, n_blocks)
        return self._impl.encode(bits)

    def decode(self, bits: Any, **kwargs: Any) -> Any:
        xp = self.xp
        if self.scheme in _SYMBOL_LEVEL_SCHEMES:
            symbols, n_batch, n_blocks = self._pack_bits_to_symbols(bits, self.n_bits)
            decoded_symbols = self._impl.decode(symbols)  # may raise ValueError
            return xp.asarray(self._unpack_symbols_to_bits(decoded_symbols, n_batch, n_blocks))
        if self.scheme in _BLOCK_BIT_SCHEMES:
            blocks, n_batch, n_blocks = self._pack_bits_to_blocks(bits, self.n_bits)
            decoded_blocks = self._impl.decode(blocks, **kwargs)  # may raise ValueError
            return self._unpack_blocks_to_bits(decoded_blocks, n_batch, n_blocks)
        return self._impl.decode(bits)

    def process(self, batch: Any, **kwargs: Any) -> Any:
        """Alias for encode() -- required to satisfy Block's abstract
        process() contract, matching the same pattern used by Modem.
        Call decode() explicitly for the inverse direction."""
        return self.encode(batch)
