"""ZadoffChuSync: matched-filter preamble detector using a Zadoff-Chu
(CAZAC) sequence as a KNOWN reference waveform -- the sync strategy
architecturally closest to a genuine liquid-dsp port, per docs/todo.md
(`qdetector_create_linear`, reference/liquid-dsp/src/framing/src/
qdetector.proto.c: a generalized "correlate against a known template
sequence" frame detector, of which a ZC sequence is one common choice
of template).

What's actually ported vs re-derived: qdetector.c is a *streaming* state
machine (seek/align states, one sample fed at a time, an explicit
integer-bin CFO-offset search swept per candidate window) built for
liquid-dsp's execute()-one-sample-at-a-time API -- none of that
translates directly to this project's batch-whole-buffer model (see
docs/architecture.md, "Batch-shape contract"). What DOES carry over
directly is the core idea: cross-correlate the received signal against
a known time-domain template via FFT (the same correlation-theorem
technique qdetector.c itself uses internally, frame-by-frame), which is
reimplemented here as one vectorized FFT-based matched filter over the
WHOLE batch at once, not a per-sample loop. Verified against a brute-
force double-loop reference correlation on a synthetic signal before
being trusted (see tests/test_sync_zadoff_chu.py).

The Zadoff-Chu sequence itself (the template) is the standard CAZAC
(constant-amplitude zero-autocorrelation) formula used throughout
LTE/5G (PRACH preambles, PSS/SSS) -- not liquid-dsp source (liquid-dsp
doesn't special-case ZC sequences; qdetector accepts an arbitrary
template), a from-reference implementation like Viterbi/Reed-Solomon:

    x_u[n] = exp(-j*pi*u*n*(n+1)/N)   N odd
    x_u[n] = exp(-j*pi*u*n^2/N)       N even

for root index u with gcd(u, N)=1 (required for the sequence's zero-
cyclic-autocorrelation-at-nonzero-lag property to hold).

Deliberate simplification vs qdetector.c, called out rather than
hidden: no integer-bin CFO-offset search across candidate windows (the
`for offset=-range..range` sweep in qdetector.c, which trades extra
compute for detection robustness under large uncorrected CFO). Matches
this project's existing `sync` ⊥ `cfo` decoupling (SchmidlCoxSync also
does not estimate/search CFO -- see docs/architecture.md, "CFO
placement"): this class assumes the CFO present at detection time is
small enough not to catastrophically decorrelate the matched filter
(true for the CFO ranges this project's own Channel/CFO tests use), and
leaves CFO estimation/correction entirely to the paired `cfo=` strategy,
same as the Schmidl-Cox pair.

Batch-shape contract: process(rx) takes (n_batch, n_samples) complex rx
-> dict with 'start_index' (n_batch,) int and 'metric' (n_batch,) float
(normalized matched-filter correlation, in [0, 1], 1.0 at a noiseless
exact match -- same bounded-metric convention as SchmidlCoxSync, so the
two are interchangeable behind `sync=`).
"""
from __future__ import annotations

import math
from typing import Any

from ..block import Block
from ..registry import register


@register("sync", "zc")
class ZadoffChuSync(Block):
    """Parameters
    ----------
    fft_size:
        Preamble length, in samples -- matches `Ofdm`'s hard assumption
        that any `sync=` preamble is exactly one `fft_size`-sample block
        with no CP (see `pipeline/ofdm.py`'s `rx_process`:
        `pos = start_index + self.fft_size`).
    root:
        Zadoff-Chu root index u. Must satisfy gcd(root, fft_size) == 1
        (required for the sequence's ideal autocorrelation property).
        Default 1 always satisfies this for any fft_size.
    """

    def __init__(self, fft_size: int, *, root: int = 1, backend=None) -> None:
        super().__init__(backend=backend)
        if fft_size < 2:
            raise ValueError("fft_size must be >= 2")
        if math.gcd(root, fft_size) != 1:
            raise ValueError(
                f"root={root} is not coprime with fft_size={fft_size} -- "
                f"the Zadoff-Chu zero-autocorrelation property requires "
                f"gcd(root, fft_size) == 1"
            )
        self.fft_size = fft_size
        self.root = root
        self.batch_shape_doc = (
            "(n_batch, n_samples) complex rx in -> dict with 'start_index' "
            "(n_batch,) int and 'metric' (n_batch,) float out"
        )
        self._template = self.generate_preamble()
        self._template_energy = float(self.xp.sum(self.xp.abs(self._template) ** 2).real)

    def generate_preamble(self, root: Any = None, seed: int = 0) -> Any:
        """Build the time-domain Zadoff-Chu preamble (no CP -- see class
        docstring). `seed` is accepted-but-unused: kept only so this
        method is call-compatible with SchmidlCoxSync.generate_preamble
        (which `Ofdm.__init__` calls as `generate_preamble(seed=...)`);
        unlike a PN-sequence preamble, a ZC sequence is fully determined
        by `root` (a constructor-time choice), not random content."""
        xp = self.xp
        u = self.root if root is None else root
        n = xp.arange(self.fft_size, dtype="float64")
        if self.fft_size % 2 == 0:
            phase = -xp.pi * u * (n ** 2) / self.fft_size
        else:
            phase = -xp.pi * u * (n * (n + 1)) / self.fft_size
        return xp.exp(1j * phase).astype("complex64")

    def process(self, rx: Any, **kwargs: Any) -> Any:
        xp = self.xp
        rx = xp.asarray(rx)
        L = self.fft_size
        n_batch, n_samples = rx.shape
        if n_samples < L:
            raise ValueError(f"need at least {L} samples, got {n_samples}")
        n_candidates = n_samples - L + 1

        # Matched-filter cross-correlation y[d] = sum_n conj(template[n]) *
        # rx[d+n], computed for every candidate d at once via one FFT-based
        # linear convolution per batch item (verified against a brute-force
        # reference before being trusted -- see module docstring/tests):
        # y[d] == conv(rx, conj(template[::-1]))[L-1+d].
        kernel = xp.conj(self._template[::-1])
        conv_len = n_samples + L - 1
        n_fft = 1
        while n_fft < conv_len:
            n_fft *= 2
        rx_freq = xp.fft.fft(rx, n=n_fft, axis=-1)
        kernel_freq = xp.fft.fft(kernel, n=n_fft)
        full_conv = xp.fft.ifft(rx_freq * kernel_freq[None, :], axis=-1)
        y = full_conv[:, L - 1 : L - 1 + n_candidates]

        # Sliding-window rx energy, via cumulative sum (same O(n) exact
        # technique SchmidlCoxSync uses for its own R(d) denominator) --
        # avoids an O(n_candidates) Python loop.
        energy = xp.abs(rx) ** 2
        zero = xp.zeros((n_batch, 1), dtype=energy.dtype)
        energy_cum = xp.concatenate([zero, xp.cumsum(energy, axis=-1)], axis=-1)
        idx = xp.arange(n_candidates)
        energy_window = energy_cum[:, idx + L] - energy_cum[:, idx]

        # Epsilon tied to this buffer's OWN mean energy (a signal-scale
        # proxy), not a bare constant -- a fixed epsilon like 1e-12
        # produces a spurious argmax at any candidate window that's
        # exactly/near-zero energy (e.g. the all-silence tail after the
        # preamble): there, |y|^2 and energy_window are both only
        # floating-point round-off noise from the FFT (~1e-6 absolute,
        # independent of the actual signal scale elsewhere in the
        # buffer), so their ratio is an unstable near-0/near-0 division
        # that can exceed a genuine matched peak's metric of ~1.0
        # (confirmed empirically: observed 2.9 at such a window before
        # this fix -- see tests/test_sync_zadoff_chu.py's
        # test_batch_with_different_offsets_per_item, which originally
        # failed this way). Scaling epsilon by the buffer's own mean
        # energy keeps it negligible at a genuine peak (energy_window
        # there is far above the buffer mean) while swamping the
        # near-zero-energy false case regardless of the input's overall
        # amplitude scale.
        energy_floor = xp.mean(energy, axis=-1, keepdims=True)
        eps = 1e-6 * energy_floor * self._template_energy + 1e-20
        metric = xp.abs(y) ** 2 / (energy_window * self._template_energy + eps)

        start_index = xp.argmax(metric, axis=-1)
        batch_idx = xp.arange(n_batch)
        peak_metric = metric[batch_idx, start_index]
        return {"start_index": start_index, "metric": peak_metric}
