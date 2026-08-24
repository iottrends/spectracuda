"""Channel: reusable impairment simulator, modeled on liquid-dsp's
channel_cccf (docs/liquid-dsp-api-inventory.md lists its four
impairments: add_awgn, add_carrier_offset, add_multipath, add_shadowing).
Covers the first three -- shadowing (correlated log-normal fading) is a
documented future addition, not built here.

liquid-dsp builds a channel incrementally and applies it with one call:

    channel_cccf channel = channel_cccf_create();
    channel_cccf_add_awgn(channel, noise_floor, SNRdB);
    channel_cccf_add_carrier_offset(channel, dphi, 0.0f);
    channel_cccf_execute_block(channel, buf, buf_len, buf);

This class takes the same impairments as constructor parameters instead
of a stateful add_*() builder -- matches spectracuda's
everything-in-the-constructor convention (established by Ofdm) rather
than reproducing liquid-dsp's builder idiom -- and applies them all with
one process() call. Not OFDM-specific; works on any complex time-domain
samples. This consolidates impairment code that used to be hand-rolled
inline in examples/ofdm_256_*.py.

Batch-shape contract: process(tx_iq) takes (n_batch, n_samples) complex
-> (n_batch, n_samples) complex (same shape). Multipath is applied as a
full convolution, then truncated back to the input length -- the
physically-realistic channel tail beyond that point is discarded,
matching what a frame's cyclic prefix/guard interval is meant to absorb.
Pad your input with trailing zeros yourself (as the examples do) if you
need to keep that tail instead.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..block import Block


class Channel(Block):
    """Parameters
    ----------
    snr_db:
        AWGN signal-to-noise ratio in dB, measured against the actual
        input signal's own power (per batch item) -- None disables noise.
    multipath_taps:
        Complex FIR tap array (length = number of channel taps), or None
        to disable multipath. Use `Channel.random_multipath_taps(...)`
        for a quick random Rayleigh-ish channel.
    cfo:
        Carrier frequency offset as a fraction of subcarrier spacing
        (same normalized units SchmidlCoxCFO estimates/corrects), or
        None to disable. Requires `cfo_fft_size`.
    cfo_fft_size:
        OFDM FFT size used to convert `cfo` into a phase-per-sample
        rotation: phase[n] = exp(j*(2*pi*cfo*n/cfo_fft_size + cfo_phase0)).
    cfo_phase0:
        Initial (constant) phase offset in radians, applied alongside cfo.
    seed:
        Seed for the AWGN generator (noise is generated on host via numpy
        regardless of backend, then moved to the active xp -- fine for a
        test/simulation utility, not a hot-path production concern).
    """

    def __init__(
        self,
        *,
        snr_db: Optional[float] = None,
        multipath_taps: Optional[Any] = None,
        cfo: Optional[float] = None,
        cfo_fft_size: Optional[int] = None,
        cfo_phase0: float = 0.0,
        seed: Optional[int] = None,
        backend=None,
    ) -> None:
        super().__init__(backend=backend)
        if cfo is not None and cfo_fft_size is None:
            raise ValueError("cfo_fft_size is required when cfo is set")

        self.snr_db = snr_db
        self.multipath_taps = (
            None if multipath_taps is None else self.xp.asarray(multipath_taps, dtype="complex64")
        )
        self.cfo = cfo
        self.cfo_fft_size = cfo_fft_size
        self.cfo_phase0 = cfo_phase0
        self._rng = np.random.default_rng(seed)
        self.batch_shape_doc = "(n_batch, n_samples) complex tx in -> (n_batch, n_samples) complex rx out"

    @staticmethod
    def random_multipath_taps(n_taps: int, seed: Optional[int] = None) -> Any:
        """Unit-total-energy complex Gaussian taps -- a quick stand-in
        Rayleigh-ish multipath channel, matching what the OFDM examples
        generated inline before this class existed. Normalized explicitly
        against its own realized energy (not just in expectation) so
        "unit energy" holds exactly for every draw, not just on average."""
        rng = np.random.default_rng(seed)
        taps = rng.standard_normal(n_taps) + 1j * rng.standard_normal(n_taps)
        taps = taps / np.sqrt(np.sum(np.abs(taps) ** 2))
        return taps.astype("complex64")

    def process(self, tx_iq: Any, **kwargs: Any) -> Any:
        xp = self.xp
        tx_iq = xp.asarray(tx_iq)
        if tx_iq.ndim == 1:
            tx_iq = tx_iq[None, :]
        n_batch, n_samples = tx_iq.shape

        rx = tx_iq
        if self.multipath_taps is not None:
            n_taps = self.multipath_taps.shape[-1]
            convolved = xp.empty((n_batch, n_samples + n_taps - 1), dtype="complex64")
            for b in range(n_batch):
                convolved[b] = xp.convolve(rx[b], self.multipath_taps)
            rx = convolved[:, :n_samples]  # truncate back to input length -- see module docstring

        if self.snr_db is not None:
            sig_power = xp.mean(xp.abs(rx) ** 2, axis=-1, keepdims=True)
            noise_power = sig_power / (10 ** (self.snr_db / 10))
            noise = (
                self._rng.standard_normal((n_batch, n_samples))
                + 1j * self._rng.standard_normal((n_batch, n_samples))
            )
            rx = rx + xp.asarray(noise, dtype="complex64") * xp.sqrt(noise_power / 2)

        if self.cfo is not None:
            n = xp.arange(n_samples)
            phase = xp.exp(1j * (2 * xp.pi * self.cfo * n / self.cfo_fft_size + self.cfo_phase0))
            rx = rx * phase[None, :]

        return rx.astype("complex64")
