"""LiquidInterleaver: a verified port of liquid-dsp's actual interleaver
algorithm (`reference/liquid-dsp/src/fec/src/interleaver.c`) -- read
directly, not assumed from general "telecom interleaver" knowledge, and
it turned out to be neither of the two textbook designs
(BlockInterleaver/ConvolutionalInterleaver) this module's siblings
implement. It's liquid-dsp's own bespoke, multi-pass design:

  - Grid dimensions M = 1 + floor(sqrt(n_bytes)), N sized so M*N >= n_bytes
    (`interleaver_create`).
  - `depth` passes (default 4, matching liquid's own default):
    - Pass 1 (`interleaver_permute`): swaps WHOLE bytes between a
      position i (first half of the array) and a computed partner j,
      found by walking a stateful (m, n) counter across the M x N grid
      geometry (m increments each candidate, wrapping to advance n every
      M candidates), rejecting any j landing in the second half.
    - Passes 2-4 (`interleaver_permute_mask`): the SAME index-walk, but
      with N offset by +2/+4/+8 per pass, and restricted to swapping only
      the BIT positions selected by a fixed mask (0x0f, then 0x55, then
      0x33) between the paired bytes -- i.e. progressively finer-grained,
      sub-byte mixing on top of pass 1's whole-byte shuffle.
  - Decode runs the same passes in exact REVERSE order (depth 4 down to
    1) -- valid because each pass is a set of pairwise swaps (a
    self-inverse operation on its own), but the passes use different
    per-pass N offsets, so they must be undone in the opposite sequence
    they were applied in, not just re-run forwards.

Represented here as a single composed BIT-level permutation array (this
class's `_compute_unit_permutation`, with unit_bits fixed at 1 -- see
below), matching the same
gather/inverse-gather machinery every other interleaver in this package
uses, rather than re-running the C code's imperative multi-pass swap
loop at encode/decode time. This required simulating the algorithm at
BIT granularity from the start (the masked passes swap individual bits
within byte pairs, not whole bytes, so a pure byte-level permutation
can't represent passes 2-4 exactly) -- verified two independent ways
before being trusted, not assumed correct from a single translation
pass:
  1. `decode(encode(x)) == x` for real byte data, running the literal
     ported swap-loop algorithm (not the permutation-array form).
  2. The DERIVED bit-level permutation array, applied as a plain gather
     to bit-unpacked data, produces byte-for-byte IDENTICAL output to
     running the literal swap-loop algorithm directly on the same data.
Both checks passed for every block size tried (8 to 255 bytes) during
development; see tests/test_interleaver.py for the standing versions of
both checks.

Byte-oriented, matching liquid-dsp's own `unsigned char *` interface --
n_bits must be a multiple of 8.

Batch-shape contract: see _PermutationInterleaverBase.
"""
from __future__ import annotations

from typing import Any, List

import numpy as np

from ..registry import register
from .base import _PermutationInterleaverBase


def _grid_dims(n_bytes: int):
    M = 1 + int(np.floor(np.sqrt(n_bytes)))
    N = n_bytes // M
    while n_bytes >= M * N:
        N += 1
    return M, N


def _index_walk(n_bytes: int, M: int, N: int):
    """Generator reproducing the exact stateful (m, n) counter walk
    shared by interleaver_permute()/interleaver_permute_mask() -- yields
    n2=n_bytes//2 valid (i, j) pairs with j < n2, in the same order the
    C code produces them (state persists across every candidate,
    accepted or rejected, throughout the whole walk).

    n_bytes (not n2) is the required input: the C code's starting value
    `n = _n/3` uses the FULL byte count (_n), not half of it -- an
    earlier version of this function tried to reconstruct it from n2
    alone via `(2*n2)//3`, which silently diverges from the true
    `n_bytes//3` whenever n_bytes is an ODD multiple of 3 (confirmed:
    39, 45, and -- critically -- 255, RS(255,223)'s own codeword size,
    all mismatch). Caught by direct comparison against n_bytes//3 before
    this class was trusted, not assumed correct from the refactor."""
    n2 = n_bytes // 2
    m = 0
    n = n_bytes // 3  # matches C's `n = _n/3` exactly
    for i in range(n2):
        while True:
            j = m * N + n
            m += 1
            if m == M:
                n = (n + 1) % N
                m = 0
            if j < n2:
                break
        yield i, j


def _permute_labels_wholebyte(labels: List[List[int]], n_bytes: int, M: int, N: int) -> List[List[int]]:
    labels = [group[:] for group in labels]
    for i, j in _index_walk(n_bytes, M, N):
        labels[2 * j + 1], labels[2 * i + 0] = labels[2 * i + 0], labels[2 * j + 1]
    return labels


def _permute_labels_mask(labels: List[List[int]], n_bytes: int, M: int, N: int, mask: int) -> List[List[int]]:
    labels = [group[:] for group in labels]
    bit_positions = [k for k in range(8) if (mask >> (7 - k)) & 1]
    for i, j in _index_walk(n_bytes, M, N):
        for k in bit_positions:
            labels[2 * i + 0][k], labels[2 * j + 1][k] = labels[2 * j + 1][k], labels[2 * i + 0][k]
    return labels


@register("interleaver", "liquid")
class LiquidInterleaver(_PermutationInterleaverBase):
    """Parameters
    ----------
    n_bits:
        Fixed block size this interleaver operates on; must be a
        multiple of 8 (liquid-dsp's algorithm is byte-oriented).
    depth:
        Number of permutation passes, matching liquid-dsp's own
        `interleaver_set_depth()` (default 4, liquid's own default;
        valid range 0-4 -- liquid only defines masks/offsets for that
        many passes).
    """

    def __init__(self, n_bits: int, *, depth: int = 4, backend=None, **kwargs: Any) -> None:
        if n_bits % 8 != 0:
            raise ValueError(
                f"n_bits={n_bits} must be a multiple of 8 -- liquid-dsp's "
                f"interleaver algorithm is byte-oriented"
            )
        if not (0 <= depth <= 4):
            raise ValueError(f"depth={depth} must be in [0, 4] -- matches liquid-dsp's own valid range")
        # unit_bits=1 fixed (not exposed): this algorithm's granularity
        # is intrinsically mixed (whole-byte pass 1, sub-byte masked
        # passes 2-4), which the base class's uniform "unit_bits" knob
        # can't express -- see module docstring.
        super().__init__(n_bits, unit_bits=1, backend=backend)
        self.depth = depth
        self._build_permutation()

    def _compute_unit_permutation(self, n_bits: int) -> np.ndarray:
        n_bytes = n_bits // 8
        M, N = _grid_dims(n_bytes)
        labels = [[8 * byte_i + bit_i for bit_i in range(8)] for byte_i in range(n_bytes)]

        depth = self.depth
        if depth > 0:
            labels = _permute_labels_wholebyte(labels, n_bytes, M, N)
        if depth > 1:
            labels = _permute_labels_mask(labels, n_bytes, M, N + 2, 0x0F)
        if depth > 2:
            labels = _permute_labels_mask(labels, n_bytes, M, N + 4, 0x55)
        if depth > 3:
            labels = _permute_labels_mask(labels, n_bytes, M, N + 8, 0x33)

        return np.array([bit for group in labels for bit in group], dtype=np.int64)
