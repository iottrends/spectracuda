"""BlockInterleaver: the textbook matrix interleaver -- write data into
an M x N grid row-by-row, read it back out column-by-column (the
inverse does the reverse mapping). A burst of up to M consecutive
UNITS in the OUTPUT stream comes from M different rows, i.e. M units
that were far apart in the ORIGINAL stream -- the whole point of
interleaving: turn a localized burst into scattered individual errors
after de-interleaving, small enough for a downstream code (e.g.
Reed-Solomon) to correct per-codeword. See docs/todo.md #1.12 for the
full write-up (why this exists, how it compares to the other three
interleaver strategies, and why it's the RECOMMENDED default for this
codebase specifically: pure reshape + transpose, trivially vectorizes
across the batch on self.xp, no per-item Python loop, no state -- the
same array-op style as every other block in this project) -- and see
base.py's module docstring for `unit_bits` (permute whole bytes, not
individual bits, when protecting a byte-oriented code -- found to
matter for real correctness, not just a convenience knob).

Row count M defaults to liquid-dsp's own dimensioning rule (`1 +
floor(sqrt(n_units))`, see interleaver/liquid.py's module docstring)
purely as a reasonable, tested default -- NOT a claim this class
reproduces liquid-dsp's actual permutation (it doesn't; see
interleaver/liquid.py for the class that does). N is sized to
`ceil(n_units / M)` so the grid always fully covers n_units (with
M*N possibly slightly larger than n_units; the excess "virtual" cells
are never real unit positions and are dropped from the read-out order,
not zero-padded into the output).

Batch-shape contract: see _PermutationInterleaverBase.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..registry import register
from .base import _PermutationInterleaverBase


@register("interleaver", "block")
class BlockInterleaver(_PermutationInterleaverBase):
    """Parameters
    ----------
    n_bits:
        Fixed block size this interleaver operates on.
    rows:
        Number of rows (M) in the write/read grid, in UNITS (see
        unit_bits). Defaults to `1 + floor(sqrt(n_units))` (liquid-dsp's
        own dimensioning rule, used here only as a sensible default --
        see module docstring).
    unit_bits:
        Size of each indivisible permuted block, in bits. Default 1
        (permute individual bits). Set to 8 (or a multiple) when
        protecting a byte-oriented downstream code (e.g. rs_m8) -- see
        base.py's module docstring for why bit-granularity can actively
        HURT a byte-oriented code rather than help it.
    """

    def __init__(
        self, n_bits: int, *, rows: Optional[int] = None, unit_bits: int = 1, backend=None, **kwargs: Any
    ) -> None:
        super().__init__(n_bits, unit_bits=unit_bits, backend=backend)
        self.rows = rows
        self._build_permutation()

    def _compute_unit_permutation(self, n_units: int) -> np.ndarray:
        M = self.rows if self.rows is not None else 1 + int(np.floor(np.sqrt(n_units)))
        if M < 1:
            raise ValueError(f"rows={self.rows} must be >= 1")
        N = -(-n_units // M)  # ceil(n_units / M)
        grid = np.arange(M * N).reshape(M, N)  # write order: row-major
        read_order = grid.T.reshape(-1)  # read order: column-major
        return read_order[read_order < n_units]  # drop virtual (padding) cells
