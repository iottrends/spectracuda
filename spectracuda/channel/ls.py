"""LSChannelEstimator: pilot-based least-squares channel estimation with
linear interpolation across subcarriers.

No liquid-dsp precedent -- liquid-dsp's OFDM sync does its own fixed
pilot-based estimate internally rather than exposing it as a named,
swappable strategy (see docs/liquid-dsp-api-inventory.md). This is the
standard pilot-based LS estimate (H_ls = rx_pilot / tx_pilot at pilot
bins) plus linear frequency-domain interpolation across all subcarriers,
following standard OFDM receiver references (e.g. 802.11/3GPP-style),
designed from reference rather than ported.

Batch-shape contract: process(rx_pilots) takes (n_batch, n_pilot)
complex received pilot symbols -> (n_batch, fft_size) complex channel
estimate across every subcarrier (including null/pilot bins, which
callers should mask out before use -- only the data-subcarrier subset is
meaningful for equalization).
"""
from __future__ import annotations

from typing import Any

from ..block import Block
from ..registry import register


@register("channel_estimator", "ls")
class LSChannelEstimator(Block):
    """Parameters
    ----------
    pilot_indices:
        Subcarrier indices of the pilot tones (e.g. ResourceGrid.pilot_indices).
    fft_size:
        Total number of subcarriers.
    tx_pilots:
        Known transmitted pilot values at `pilot_indices`, shape (n_pilot,)
        (broadcast across the batch) or (n_batch, n_pilot).
    """

    def __init__(self, pilot_indices, fft_size: int, tx_pilots, *, backend=None, **kwargs) -> None:
        """**kwargs is a deliberate sink, not an oversight: `Ofdm.__init__`
        resolves `channel_estimator=` through one shared default_kwargs
        dict that also has to satisfy MMSEChannelEstimator's extra
        cp_len/noise_var params (see channel/mmse.py's docstring); this
        class only needs pilot_indices/fft_size/tx_pilots and silently
        ignores the rest, the same registry pattern used for cfo=."""
        super().__init__(backend=backend)
        self.fft_size = fft_size
        self.pilot_indices = self.xp.asarray(pilot_indices)
        self.tx_pilots = self.xp.asarray(tx_pilots)
        if self.pilot_indices.shape[0] < 2:
            raise ValueError("need at least 2 pilot subcarriers to interpolate")
        self.batch_shape_doc = (
            f"(n_batch, {self.pilot_indices.shape[0]}) complex rx pilots "
            f"in -> (n_batch, {fft_size}) complex channel estimate out"
        )

    def process(self, rx_pilots: Any, **kwargs: Any) -> Any:
        xp = self.xp
        rx_pilots = xp.asarray(rx_pilots)
        h_pilot = rx_pilots / self.tx_pilots  # (n_batch, n_pilot)

        all_bins = xp.arange(self.fft_size)
        n_batch = h_pilot.shape[0]
        h_full = xp.empty((n_batch, self.fft_size), dtype="complex64")
        # Per-batch-item interpolation: xp.interp is 1-D only. n_batch is
        # the outer loop here, not the hot inner path -- the FFT/equalize
        # stages carry the actual batch parallelism.
        for b in range(n_batch):
            real = xp.interp(all_bins, self.pilot_indices, xp.real(h_pilot[b]))
            imag = xp.interp(all_bins, self.pilot_indices, xp.imag(h_pilot[b]))
            h_full[b] = real + 1j * imag
        return h_full
