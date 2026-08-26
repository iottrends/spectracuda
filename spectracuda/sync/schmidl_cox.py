"""SchmidlCoxSync: Schmidl & Cox (1997) self-correlation preamble
detector for OFDM -- timing acquisition from a preamble made of two
identical time-domain halves.

liquid-dsp doesn't expose this as its own reusable block; the algorithm
is extracted from the S0/S1 preamble handling buried inside
reference/liquid-dsp/src/framing/src/ofdmframesync.c (see
docs/liquid-dsp-api-inventory.md) and re-implemented here as an
independent, swappable `sync` strategy.

Candidate start offsets are searched as a batch (vectorized prefix-sum
correlation across every offset at once), not a sequential sliding-window
loop -- this is the Phase 2 design insight from docs/architecture.md: GPU
batching applies to sync by framing it as "many candidate windows
evaluated at once," not "one window slid forward one sample at a time."

CFO is deliberately NOT estimated here -- see cfo/schmidl_cox.py
(SchmidlCoxCFO), kept as its own strategy class independent of which
`sync` block found the start index (docs/architecture.md, "CFO
placement").

Deliberate deviation from the textbook 1997 formula: the original paper
normalizes by R(d) = sum |r[d+m+L]|^2 (second-half energy only). That
denominator can collapse toward zero whenever a candidate window
straddles the boundary between the preamble and a silent/low-energy
region (e.g. right after the preamble ends), producing spurious,
unbounded metric spikes far from the true peak -- confirmed empirically
during development (a candidate at such a boundary produced a "metric"
of ~77, when the true peak is bounded at 1.0 in the noiseless case).
This implementation instead uses the symmetric energy
R(d) = (sum |r[d+m]|^2 + sum |r[d+m+L]|^2) / 2 (both halves), a
refinement used in later timing-recovery literature, which requires
*both* windows to have low energy for the denominator to vanish and is
exact in the noiseless matched case just like the original formula.

Batch-shape contract: process(rx) takes (n_batch, n_samples) complex rx
-> dict with 'start_index' (n_batch,) int and 'metric' (n_batch,) float
(peak normalized timing-metric value, in [0, 1], useful as a detection
confidence/threshold signal).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..block import Block
from ..registry import register


@register("sync", "schmidl_cox")
class SchmidlCoxSync(Block):
    """Parameters
    ----------
    fft_size:
        OFDM FFT size, M. The preamble is one OFDM-symbol-length (M
        samples, no CP) block built from energy on every other
        subcarrier, which makes its two M/2-sample halves identical.
    """

    def __init__(self, fft_size: int, *, backend=None) -> None:
        super().__init__(backend=backend)
        if fft_size % 2 != 0:
            raise ValueError("fft_size must be even for Schmidl-Cox (needs two equal halves)")
        self.fft_size = fft_size
        self.half_len = fft_size // 2
        self.batch_shape_doc = (
            "(n_batch, n_samples) complex rx in -> dict with 'start_index' "
            "(n_batch,) int and 'metric' (n_batch,) float out"
        )

    def generate_preamble(self, pn_symbols: Any = None, seed: int = 0) -> Any:
        """Build the time-domain Schmidl-Cox preamble (no CP -- callers
        add it, e.g. via OfdmModulator, matching how it's transmitted):
        a PN sequence on even subcarriers, zero on odd subcarriers, which
        produces two identical M/2-sample halves after IFFT."""
        xp = self.xp
        n_even = self.fft_size // 2
        if pn_symbols is None:
            from ..modem import Modem

            rng = np.random.default_rng(seed)
            bits = rng.integers(0, 2, size=2 * n_even).astype("uint8")
            pn_symbols = Modem("qpsk", backend=self.backend).modulate(bits.reshape(1, -1))[0]
        freq = xp.zeros(self.fft_size, dtype="complex64")
        freq[0::2] = xp.asarray(pn_symbols)
        return xp.fft.ifft(freq)

    def process(self, rx: Any, **kwargs: Any) -> Any:
        xp = self.xp
        rx = xp.asarray(rx)
        L = self.half_len
        n_samples = rx.shape[-1]
        if n_samples < 2 * L:
            raise ValueError(f"need at least {2 * L} samples, got {n_samples}")

        # a[n] = conj(r[n]) * r[n+L]; b1[n] = |r[n]|^2, b2[n] = |r[n+L]|^2.
        # P(d)/R(d) are windowed sums over m=0..L-1, computed for every
        # candidate d at once via prefix sums (no per-offset Python loop).
        # R(d) symmetrizes over BOTH halves' energy -- see module
        # docstring for why the textbook second-half-only R(d) is unsafe.
        a = xp.conj(rx[:, :-L]) * rx[:, L:]
        b1 = xp.abs(rx[:, :-L]) ** 2
        b2 = xp.abs(rx[:, L:]) ** 2
        n_batch = rx.shape[0]

        def _cumsum_with_leading_zero(x):
            zero = xp.zeros((n_batch, 1), dtype=x.dtype)
            return xp.concatenate([zero, xp.cumsum(x, axis=-1)], axis=-1)

        a_cum = _cumsum_with_leading_zero(a)
        b1_cum = _cumsum_with_leading_zero(b1)
        b2_cum = _cumsum_with_leading_zero(b2)

        n_candidates = n_samples - 2 * L + 1
        # Plain contiguous slices, not a fancy-index gather: the windowed-
        # sum-difference at every candidate d is a_cum[d+L]-a_cum[d] for
        # d=0..n_candidates-1, i.e. a_cum[L:L+n_candidates] - a_cum[:n_candidates]
        # element-for-element -- mathematically identical to indexing with
        # `idx = arange(n_candidates)` (confirmed bit-exact before trusting
        # this), but numpy's fancy indexing can't tell an arange index is
        # just a contiguous range and pays a per-element gather cost for it
        # regardless -- measured ~5.7x slower than the equivalent slice on
        # a real frame-length buffer, and this is sync's own dominant cost.
        p = a_cum[:, L : L + n_candidates] - a_cum[:, :n_candidates]
        r1 = b1_cum[:, L : L + n_candidates] - b1_cum[:, :n_candidates]
        r2 = b2_cum[:, L : L + n_candidates] - b2_cum[:, :n_candidates]
        r = 0.5 * (r1 + r2)
        metric = xp.abs(p) ** 2 / (r ** 2 + 1e-12)

        start_index = xp.argmax(metric, axis=-1)
        batch_idx = xp.arange(n_batch)
        peak_metric = metric[batch_idx, start_index]
        return {"start_index": start_index, "metric": peak_metric}
