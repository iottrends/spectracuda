"""SchmidlCoxCFO: coarse carrier-frequency-offset estimate from the
Schmidl & Cox (1997) self-correlation of an OFDM preamble's two identical
halves, evaluated at an already-detected start index.

Kept as its own swappable strategy, independent of which `sync` block
produced that start index (docs/architecture.md, "CFO placement") --
this class only needs (rx, start_index), so it pairs with
SchmidlCoxSync's own output just as readily as with any other sync
method that can hand back a start index, and it does not reuse
SchmidlCoxSync's batched multi-candidate search: at estimation time
there's exactly one window to evaluate, so this recomputes P/R directly
at that single position per batch item.

Batch-shape contract: process(rx, start_index) takes (n_batch, n_samples)
complex rx and (n_batch,) int start indices -> (n_batch,) float cfo
estimate, normalized as a fraction of subcarrier spacing (angle(P)/pi,
valid range (-1, 1]). correct(rx, cfo_estimate) applies the estimated
per-batch-item constant phase-rotation correction across the given
samples.
"""
from __future__ import annotations

from typing import Any

from ..block import Block
from ..registry import register


@register("cfo", "schmidl_cox")
class SchmidlCoxCFO(Block):
    def __init__(self, fft_size: int, *, backend=None, **kwargs: Any) -> None:
        """**kwargs is a deliberate sink, not an oversight: `Ofdm.__init__`
        resolves `cfo=` through one shared default_kwargs dict that also
        has to satisfy PilotBasedCFO's differently-shaped constructor
        (cp_len/pilot_indices/tx_pilots/n_repeats -- see its docstring);
        this class only needs fft_size, so it silently ignores the rest,
        the same registry pattern already used for channel_estimator's
        shared kwargs dict."""
        super().__init__(backend=backend)
        if fft_size % 2 != 0:
            raise ValueError("fft_size must be even for Schmidl-Cox CFO")
        self.fft_size = fft_size
        self.half_len = fft_size // 2
        self.batch_shape_doc = (
            "(n_batch, n_samples) complex rx + (n_batch,) int start_index "
            "in -> (n_batch,) float cfo estimate out (fraction of "
            "subcarrier spacing)"
        )

    def process(self, rx: Any, start_index: Any = None, **kwargs: Any) -> Any:
        if start_index is None:
            raise ValueError("SchmidlCoxCFO.process requires start_index=")
        xp = self.xp
        rx = xp.asarray(rx)
        L = self.half_len
        n_batch = rx.shape[0]

        cfo = xp.empty((n_batch,), dtype="float64")
        for b in range(n_batch):
            d = int(start_index[b])
            if d + 2 * L > rx.shape[-1]:
                raise ValueError(
                    f"start_index {d} + 2*fft_size//2 exceeds available "
                    f"samples ({rx.shape[-1]}) for batch item {b}"
                )
            first_half = rx[b, d : d + L]
            second_half = rx[b, d + L : d + 2 * L]
            p = xp.sum(xp.conj(first_half) * second_half)
            cfo[b] = float(xp.angle(p) / xp.pi)
        return cfo

    def correct(self, rx: Any, cfo_estimate: Any) -> Any:
        """Apply the per-batch-item estimated CFO as a phase de-rotation
        across the whole given signal (constant-CFO assumption over the
        observed window -- standard for a coarse pre-FFT correction).

        Built from cos()/sin() in float32 rather than a single complex
        xp.exp() call: numerically identical (exp(i*theta) IS
        cos(theta)+i*sin(theta), not an approximation of it), but
        measured ~3x faster here -- xp.exp() on a complex array upcasts
        to complex128 and runs the generic complex-transcendental path,
        while real-valued float32 cos/sin hit numpy's faster elementwise
        loops. This runs over the WHOLE received frame's samples on
        every single RX call, so it was a real, measured per-frame cost
        (~1ms on a ~38k-sample frame), not a micro-optimization."""
        xp = self.xp
        rx = xp.asarray(rx)
        cfo_estimate = xp.asarray(cfo_estimate).astype("float32")
        n = xp.arange(rx.shape[-1], dtype="float32")
        angle = (-2 * xp.pi * cfo_estimate[:, None] * n[None, :] / self.fft_size).astype("float32")
        phase = (xp.cos(angle) + 1j * xp.sin(angle)).astype(
            rx.dtype if xp.iscomplexobj(rx) else "complex64"
        )
        return rx * phase
