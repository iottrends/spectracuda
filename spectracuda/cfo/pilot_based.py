"""PilotBasedCFO: carrier-frequency-offset estimation from the phase
slope of known pilot subcarriers across REPEATED known OFDM symbols
(the training symbol(s) `Ofdm` already sends -- see pipeline/ofdm.py's
docstring, "the SAME training symbol is repeated n_training_symbols
times"), rather than from a preamble's self-correlation structure
(SchmidlCoxCFO).

No liquid-dsp precedent to port (liquid-dsp folds CFO tracking into
`ofdmframesync`/`symtrack` internals, not a standalone reusable object
-- see docs/todo.md, §1.4); designed from the standard pilot-phase-slope
reference technique instead, the same "from reference, not from
liquid-dsp source" footing as LSChannelEstimator.

Why this exists as a SEPARATE strategy from SchmidlCoxCFO, not a
variant of it: SchmidlCoxCFO's estimate depends entirely on the
preamble having Schmidl & Cox's specific two-identical-halves time-
domain structure (it directly re-reads raw samples at start_index and
assumes the first/second fft_size//2-sample halves are equal up to a
CFO phase ramp). A `sync=` strategy whose preamble does NOT have that
shape -- ZadoffChuSync's Zadoff-Chu preamble is one *contiguous* known
sequence, not two repeated halves -- makes SchmidlCoxCFO's estimate
meaningless (confirmed empirically during development: pairing them
produced a garbage ~-0.97 "CFO" out of an identity/no-CFO channel,
which then corrupted the whole frame once applied). PilotBasedCFO has
no such dependency: it only needs *some* known, repeated OFDM-symbol-
shaped content after the preamble, at whatever pilot subcarriers the
grid defines -- it's the correct pairing for ZadoffChuSync (or any
future `sync=` strategy whose preamble isn't Schmidl-Cox-shaped), the
same way SchmidlCoxCFO is the correct pairing for SchmidlCoxSync.

Algorithm: for two repeated known OFDM symbols separated by
`symbol_period = cp_len + fft_size` samples (the corresponding-position
samples, i.e. same offset within each symbol's own post-CP portion),
`rx[k]` at pilot subcarrier k in the LATER symbol equals `rx[k]` in the
EARLIER one times `exp(j*2*pi*eps*symbol_period/fft_size)` -- the
channel response and known pilot value are identical in both symbols
(same repeated content, same short-term-static channel) and cancel
exactly in the ratio, leaving only the CFO-induced phase term. Averaged
over every pilot subcarrier and every consecutive repeat-pair (summing
the complex ratio before taking its angle, not averaging angles
directly, to stay well-behaved near the wrap boundary under noise) for
noise reduction, then converted to the same "fraction of subcarrier
spacing" convention SchmidlCoxCFO uses: `eps_hat = angle(ratio_sum) *
fft_size / (2*pi*symbol_period)`.

Deliberate trade-off, called out rather than hidden: `symbol_period` is
necessarily larger than Schmidl-Cox's fft_size//2 half-preamble spacing
(it includes a full cp_len), so this estimator's UNAMBIGUOUS range is
narrower: |eps| must stay below `0.5 * fft_size / symbol_period` (< 0.5,
vs Schmidl-Cox's full (-1, 1]) to avoid phase wraparound -- a standard,
well-known trade-off of larger-separation phase-slope estimators (finer
resolution, narrower unambiguous range), not a bug.

Requires n_repeats >= 2 (need at least two occurrences of the known
symbol to form one phase-slope measurement) -- `Ofdm` passes
`n_repeats=n_training_symbols`, so `n_training_symbols=1` (Ofdm's
allowed minimum) is NOT enough for this particular cfo strategy; it
raises a clear ValueError rather than silently returning a meaningless
number.

Real noise-sensitivity trade-off, found empirically rather than assumed
away: this estimate is combined from only `n_pilot` complex samples per
repeat, far fewer than SchmidlCoxCFO's correlation across an entire
half-preamble's worth of time samples (fft_size//2). At a small grid's
usual pilot count (e.g. n_pilot=6, as several of this project's own
tests use) and ordinary OFDM test SNRs (20-25 dB), the estimate can be
dominated by whichever pilot subcarrier happens to sit in a deep
multipath fade for that particular channel realization, occasionally
producing a badly wrong estimate that then corrupts an otherwise-
decodable frame once "corrected" -- confirmed directly (not assumed) by
sweeping AWGN seeds against a fixed multipath+CFO scenario: most
realizations gave a usable estimate, but some gave one off by 2-5x or
worse. Larger n_pilot and/or higher SNR narrow this considerably (also
confirmed empirically -- see tests/test_ofdm_class.py's
test_sync_zc_under_real_multipath_and_awgn_channel, which needs
n_pilot=32 and snr_db=40 to reliably decode cleanly, well above what
the SchmidlCoxCFO-paired tests in this suite need). Not a bug: it's the
standard, expected cost of a phase-slope estimate over few tones vs a
correlation over many time samples -- users of this strategy at low
pilot counts/SNR should expect looser CFO tracking than SchmidlCoxCFO
provides.

Batch-shape contract: process(rx, start_index) takes (n_batch,
n_samples) complex rx and (n_batch,) int start indices -> (n_batch,)
float cfo estimate (fraction of subcarrier spacing, valid range
+-0.5*fft_size/symbol_period). correct(rx, cfo_estimate): identical
convention/implementation to SchmidlCoxCFO.correct.
"""
from __future__ import annotations

from typing import Any

from ..block import Block
from ..registry import register


@register("cfo", "pilot_based")
class PilotBasedCFO(Block):
    """Parameters
    ----------
    fft_size:
        OFDM FFT size.
    cp_len:
        Cyclic prefix length of the repeated known OFDM symbol(s) --
        NOT the preamble (which has no CP; see `sync=`'s own docstring).
    pilot_indices:
        Subcarrier indices of the pilot tones (e.g. ResourceGrid.pilot_indices).
    tx_pilots:
        Known transmitted pilot values at `pilot_indices`. Accepted for
        interface/documentation parity with LSChannelEstimator's
        constructor and future extension, but NOT used in the phase-
        slope computation itself: since the same known content repeats
        across every one of the `n_repeats` symbols, the pilot value
        (and the channel response) cancels exactly in the ratio between
        repeats -- see module docstring's algorithm section.
    n_repeats:
        Number of repeated known OFDM symbols available right after the
        preamble (matches `Ofdm`'s `n_training_symbols`). Must be >= 2.
    """

    def __init__(
        self, fft_size: int, cp_len: int, pilot_indices, tx_pilots, n_repeats: int = 2, *, backend=None,
        **kwargs: Any,
    ) -> None:
        super().__init__(backend=backend)
        if fft_size < 2:
            raise ValueError("fft_size must be >= 2")
        self.fft_size = fft_size
        self.cp_len = cp_len
        self.symbol_period = cp_len + fft_size
        self.pilot_indices = self.xp.asarray(pilot_indices)
        self.tx_pilots = self.xp.asarray(tx_pilots)  # unused in the math -- see docstring
        self.n_repeats = n_repeats
        self.batch_shape_doc = (
            "(n_batch, n_samples) complex rx + (n_batch,) int start_index "
            "in -> (n_batch,) float cfo estimate out (fraction of "
            "subcarrier spacing)"
        )

    def process(self, rx: Any, start_index: Any = None, **kwargs: Any) -> Any:
        if start_index is None:
            raise ValueError("PilotBasedCFO.process requires start_index=")
        if self.n_repeats < 2:
            raise ValueError(
                f"n_repeats={self.n_repeats} < 2 -- a phase-slope CFO estimate "
                f"needs at least two repeated known OFDM symbols (Ofdm's "
                f"n_training_symbols must be >= 2 to pair with cfo='pilot_based')"
            )
        xp = self.xp
        rx = xp.asarray(rx)
        L = self.fft_size
        cp = self.cp_len
        period = self.symbol_period
        n_batch = rx.shape[0]

        cfo = xp.empty((n_batch,), dtype="float64")
        for b in range(n_batch):
            d = int(start_index[b]) + L  # preamble has no CP -- see sync docstring
            needed = d + (self.n_repeats - 1) * period + cp + L
            if needed > rx.shape[-1]:
                raise ValueError(
                    f"start_index {start_index[b]} + preamble + "
                    f"{self.n_repeats} repeated symbols exceeds available "
                    f"samples ({rx.shape[-1]}) for batch item {b}"
                )
            rx_pilots = xp.empty((self.n_repeats, self.pilot_indices.shape[0]), dtype="complex64")
            for i in range(self.n_repeats):
                sym_start = d + i * period + cp  # skip this repeat's own CP
                symbol_time = rx[b, sym_start : sym_start + L]
                symbol_freq = xp.fft.fft(symbol_time).astype("complex64")
                rx_pilots[i] = symbol_freq[self.pilot_indices]

            ratio_sum = xp.sum(rx_pilots[1:] * xp.conj(rx_pilots[:-1]))
            dphi = float(xp.angle(ratio_sum))
            cfo[b] = dphi * L / (2 * float(xp.pi) * period)
        return cfo

    def correct(self, rx: Any, cfo_estimate: Any) -> Any:
        """Identical convention/implementation to SchmidlCoxCFO.correct
        -- both strategies produce a CFO estimate in the same "fraction
        of subcarrier spacing" units, so the correction step doesn't
        need to know which strategy produced it."""
        xp = self.xp
        rx = xp.asarray(rx)
        cfo_estimate = xp.asarray(cfo_estimate)
        n = xp.arange(rx.shape[-1])
        phase = xp.exp(-1j * 2 * xp.pi * cfo_estimate[:, None] * n[None, :] / self.fft_size)
        return rx * phase.astype(rx.dtype if xp.iscomplexobj(rx) else "complex64")
