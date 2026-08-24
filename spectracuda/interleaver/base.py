"""_PermutationInterleaverBase: shared machinery for all four interleaver
strategies (see docs/todo.md #1.12) -- every one of them reduces to
"compute a fixed permutation of UNITS once, then gather/inverse-gather
on encode/decode," so the actual algorithm difference between block/
permutation/convolutional is entirely contained in each subclass's
`_compute_unit_permutation()`, not in how encode/decode work.
(`LiquidInterleaver` doesn't use this base class's unit mechanism --
its mixed bit/byte granularity across passes is intrinsic to faithfully
reproducing liquid-dsp's own algorithm, not something this generic
"uniform unit size" abstraction can express -- see its own docstring.)

This is the same "gather-only, no scatter" philosophy already used for
LDPC's belief-propagation message passing (spectracuda/fec/ldpc.py) --
here it's simpler still: one static index array per direction
(`self._perm` for encode, `self._inverse_perm` for decode, related by
`np.argsort`, the standard technique for inverting a permutation array),
computed once at construction and reused for every call.

`unit_bits` (default 1 = permute individual bits), found to matter for
real correctness, not just a convenience knob: a bit-level permutation
genuinely SPREADS a contiguous run of wrong bits (verified directly --
see tests/test_interleaver.py's burst-spreading test), but when the
downstream code is BYTE-oriented (Reed-Solomon), spreading at bit
granularity can make things WORSE, not better -- confirmed empirically
during development, not assumed: a ~50-bit contiguous Viterbi decode-
error burst that stayed within ~7-8 bytes (and so trivially fit RS's
t=16 budget) UNINTERLEAVED, after being scattered by a BIT-level
permutation, touched ~50 DIFFERENT bytes instead -- turning an easily-
correctable burst into one that broke RS's per-codeword budget on BOTH
resulting codewords. Byte-granularity interleaving (`unit_bits=8`) of
the exact same scenario left only 2 and 4 byte errors per codeword,
comfortably under budget. This matches liquid-dsp's own design choice
(`interleaver.c`'s pass 1 swaps whole BYTES, not bits -- see
interleaver/liquid.py's docstring) -- callers protecting a byte-
oriented code (e.g. `Packetizer(fec1="conv_v27", fec="rs_m8",
interleaver="block", interleaver_kwargs={"unit_bits": 8})`) should set
`unit_bits=8` (or a multiple of it), not rely on the bit-level default.

Subclasses must call `self._build_permutation()` at the end of their own
`__init__`, AFTER setting whatever algorithm-specific config attributes
`_compute_unit_permutation()` needs (rows, seed, branches, depth, ...) --
`_PermutationInterleaverBase.__init__` itself only sets up `n_bits`/
`unit_bits`/`backend`/`xp`, deliberately not calling
`_compute_unit_permutation()` on its own, since that needs subclass
state that doesn't exist yet at that point in `__init__` order.

`_build_permutation()` defensively re-verifies every computed UNIT
permutation is a genuine bijection over `range(n_units)` before
expanding it to bit granularity and trusting it (same "verify, don't
just derive and assume" standard already applied throughout this
codebase -- e.g. LDPC's parity-submatrix full-rank check) -- a subclass
bug that produces a non-bijective "permutation" fails loudly at
construction, not silently at decode time with corrupted data.

Batch-shape contract: encode(bits)/decode(bits) take/return
(n_batch, n_bits) bits -- a fixed-size permutation, not a variable-
length transform; n_bits is set once at construction.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..block import Block


class _PermutationInterleaverBase(Block):
    def __init__(self, n_bits: int, *, unit_bits: int = 1, backend=None) -> None:
        super().__init__(backend=backend)
        if n_bits < 1:
            raise ValueError(f"n_bits={n_bits} must be >= 1")
        if unit_bits < 1:
            raise ValueError(f"unit_bits={unit_bits} must be >= 1")
        if n_bits % unit_bits != 0:
            raise ValueError(
                f"n_bits={n_bits} must be a multiple of unit_bits={unit_bits} "
                f"(each unit is moved as one indivisible block -- see module "
                f"docstring for why unit_bits=8 matters when protecting a "
                f"byte-oriented downstream code like rs_m8)"
            )
        self.n_bits = n_bits
        self.unit_bits = unit_bits
        self.n_units = n_bits // unit_bits
        self.batch_shape_doc = (
            f"encode/decode: (n_batch, {n_bits}) bits <-> (n_batch, {n_bits}) "
            f"interleaved bits (fixed permutation of {self.n_units} "
            f"{unit_bits}-bit units, exact inverse)."
        )

    def _compute_unit_permutation(self, n_units: int) -> np.ndarray:
        """Subclasses override: return a length-n_units array that's a
        permutation of range(n_units) (each index 0..n_units-1 appears
        exactly once) -- perm[i] = which ORIGINAL unit ends up at output
        position i. Each unit is unit_bits wide and always moves as one
        block (see module docstring)."""
        raise NotImplementedError

    def _build_permutation(self) -> None:
        unit_perm = np.asarray(self._compute_unit_permutation(self.n_units), dtype=np.int64)
        if unit_perm.shape != (self.n_units,) or sorted(unit_perm.tolist()) != list(range(self.n_units)):
            raise ValueError(
                f"{type(self).__name__}._compute_unit_permutation(n_units={self.n_units}) "
                f"did not return a valid permutation -- this is an implementation "
                f"bug in the interleaver algorithm, not a data/config problem"
            )
        # Expand the unit-level permutation to bit granularity: unit u's
        # unit_bits bits always move together, in order.
        bit_perm = (
            unit_perm[:, None] * self.unit_bits + np.arange(self.unit_bits, dtype=np.int64)[None, :]
        ).reshape(-1)
        self._perm = self.xp.asarray(bit_perm)
        self._inverse_perm = self.xp.asarray(np.argsort(bit_perm))

    def encode(self, bits: Any) -> Any:
        xp = self.xp
        bits = xp.asarray(bits)
        if bits.ndim == 1:
            bits = bits[None, :]
        if bits.shape[-1] != self.n_bits:
            raise ValueError(f"expected {self.n_bits} bits, got {bits.shape[-1]}")
        return bits[:, self._perm]

    def decode(self, bits: Any) -> Any:
        xp = self.xp
        bits = xp.asarray(bits)
        if bits.ndim == 1:
            bits = bits[None, :]
        if bits.shape[-1] != self.n_bits:
            raise ValueError(f"expected {self.n_bits} bits, got {bits.shape[-1]}")
        return bits[:, self._inverse_perm]

    def process(self, batch: Any, **kwargs: Any) -> Any:
        """Alias for encode() -- required to satisfy Block's abstract
        process() contract, matching every other scheme class in this
        codebase. Call decode() explicitly for the inverse direction."""
        return self.encode(batch)
