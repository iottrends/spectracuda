"""PermutationInterleaver: one fixed pseudo-random shuffle table,
generated once from a fixed seed at construction, applied as a gather
on encode and the inverse gather on decode. The same "deterministic,
fixed-seed randomization -- not secret, not data-dependent, just not
constant/predictable content" technique already used elsewhere in this
codebase for the header's own scramble mask and generate_frame()'s
payload-padding filler bits (see pipeline/ofdm.py's class docstring and
framing/header.py's HeaderCodec).

Unlike BlockInterleaver's fixed M-row burst-spreading radius, a random
permutation can scatter a contiguous burst arbitrarily across the whole
block, not just within one column-read's worth of separation -- a real
trade-off (no fixed, predictable minimum spreading distance the way a
block interleaver's M guarantees one, but no fixed-shape limitation
either). See docs/todo.md #1.12 for the full comparison against the
other three interleaver strategies, and base.py's module docstring for
`unit_bits` (permute whole bytes, not individual bits, when protecting
a byte-oriented code -- found to matter for real correctness, not just
a convenience knob).

Batch-shape contract: see _PermutationInterleaverBase.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..registry import register
from .base import _PermutationInterleaverBase


@register("interleaver", "permutation")
class PermutationInterleaver(_PermutationInterleaverBase):
    """Parameters
    ----------
    n_bits:
        Fixed block size this interleaver operates on.
    seed:
        Seed for the fixed pseudo-random permutation -- same seed +
        same n_units always produces the same table (both tx and rx
        must agree on this, exactly like preamble_seed/training_seed
        elsewhere in Ofdm -- interleaver choice is a local
        configuration detail, never signaled over the wire; see
        framing/packetizer.py's module docstring).
    unit_bits:
        Size of each indivisible permuted block, in bits. Default 1
        (permute individual bits). Set to 8 (or a multiple) when
        protecting a byte-oriented downstream code (e.g. rs_m8) -- see
        base.py's module docstring.
    """

    def __init__(self, n_bits: int, *, seed: int = 1234, unit_bits: int = 1, backend=None, **kwargs: Any) -> None:
        super().__init__(n_bits, unit_bits=unit_bits, backend=backend)
        self.seed = seed
        self._build_permutation()

    def _compute_unit_permutation(self, n_units: int) -> np.ndarray:
        return np.random.default_rng(self.seed).permutation(n_units)
