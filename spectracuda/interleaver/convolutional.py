"""ConvolutionalInterleaver: a finite-block adaptation of the classic
Forney/Ramsey-type convolutional interleaver that CCSDS deep-space
telemetry and DVB-S actually specify -- I parallel "branches", branch i
delayed relative to branch 0 by a per-branch amount, spreading bursts
across the reconstructed stream. See docs/todo.md #1.12 for the full
comparison against the other three interleaver strategies, and base.py's
module docstring for `unit_bits` (permute whole bytes, not individual
bits, when protecting a byte-oriented code -- found to matter for real
correctness, not just a convenience knob).

Deliberate, explicit deviation from the real (CCSDS/DVB) algorithm, not
a claimed reproduction of it: the genuine Forney interleaver is a
CONTINUOUS STREAMING construct -- each branch is an unbounded FIFO whose
depth grows without limit across the whole transmission, with an
explicit startup transient (the first several output slots are
undefined/flushed) that only makes sense when data flows across many
frames, not one self-contained block. This project already made the
equivalent "adapt a streaming liquid-dsp construct to a batch/one-frame-
at-a-time model rather than port the state machine" call once before
(see ZadoffChuSync's module docstring re: qdetector.c). Here: split the
n_units positions across I branches by `position % I`, then apply a
per-branch CIRCULAR shift of `(branch_index * base_delay) % branch_length`
to that branch's own index list (instead of an ever-growing linear FIFO
delay), and round-robin-merge the shifted per-branch lists back
together. This is a genuine, verified bijection over one finite block
(each branch's own list is trivially still a permutation of itself after
a circular shift; round-robin-merging disjoint permutations covering the
whole set is still a permutation of the whole set) with the same
qualitative property real convolutional interleaving exists for
(consecutive input positions land in DIFFERENT branches, and different
branches carry different shift amounts, so a contiguous run of input
positions is spread across the output) -- not a bit-exact reproduction
of any published standard's specific frame-boundary/flush behavior.

Batch-shape contract: see _PermutationInterleaverBase.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..registry import register
from .base import _PermutationInterleaverBase


@register("interleaver", "convolutional")
class ConvolutionalInterleaver(_PermutationInterleaverBase):
    """Parameters
    ----------
    n_bits:
        Fixed block size this interleaver operates on.
    branches:
        Number of parallel branches (I). Default 4.
    base_delay:
        Per-branch circular-shift increment (B) -- branch i is shifted
        by `(i * base_delay) % branch_length`. Default 1 (always valid
        for any n_units/branches combination); tune upward relative to
        the expected burst length / downstream FEC block size for a
        specific deployment -- this class doesn't assume a particular
        external standard's parameters (see module docstring).
    unit_bits:
        Size of each indivisible permuted block, in bits. Default 1
        (permute individual bits). Set to 8 (or a multiple) when
        protecting a byte-oriented downstream code (e.g. rs_m8) -- see
        base.py's module docstring.
    """

    def __init__(
        self,
        n_bits: int,
        *,
        branches: int = 4,
        base_delay: int = 1,
        unit_bits: int = 1,
        backend=None,
        **kwargs: Any,
    ) -> None:
        super().__init__(n_bits, unit_bits=unit_bits, backend=backend)
        if branches < 1:
            raise ValueError(f"branches={branches} must be >= 1")
        self.branches = branches
        self.base_delay = base_delay
        self._build_permutation()

    def _compute_unit_permutation(self, n_units: int) -> np.ndarray:
        I = self.branches
        B = self.base_delay
        branch_lists = [[] for _ in range(I)]
        for pos in range(n_units):
            branch_lists[pos % I].append(pos)

        shifted = []
        for i in range(I):
            lst = branch_lists[i]
            L = len(lst)
            if L == 0:
                shifted.append([])
                continue
            shift = (i * B) % L
            shifted.append(lst[shift:] + lst[:shift])

        perm = []
        idx = [0] * I
        filled = 0
        while filled < n_units:
            for i in range(I):
                if idx[i] < len(shifted[i]):
                    perm.append(shifted[i][idx[i]])
                    idx[i] += 1
                    filled += 1
        return np.array(perm, dtype=np.int64)
