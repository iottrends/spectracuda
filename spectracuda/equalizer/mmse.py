"""MMSEEqualizer: MMSE per-subcarrier frequency-domain equalization.

No liquid-dsp precedent -- same gap noted for ZFEqualizer (see
docs/liquid-dsp-api-inventory.md); designed from the standard MMSE
closed-form weight w = conj(H) / (|H|^2 + noise_var).

Batch-shape contract: process(rx_data, channel_est) takes matching
(n_batch, n_data) complex received data symbols and channel estimate at
the same subcarrier indices -> (n_batch, n_data) complex equalized
symbols.
"""
from __future__ import annotations

from typing import Any

from ..block import Block
from ..registry import register


@register("equalizer", "mmse")
class MMSEEqualizer(Block):
    """Parameters
    ----------
    noise_var:
        Estimated noise variance (per subcarrier). Larger values pull the
        equalizer weight toward zero for weak-channel subcarriers instead
        of amplifying noise the way ZF does.
    """

    def __init__(self, noise_var: float = 1e-3, *, backend=None) -> None:
        super().__init__(backend=backend)
        self.noise_var = noise_var
        self.batch_shape_doc = (
            "(n_batch, n_data) complex rx + (n_batch, n_data) complex "
            "channel est in -> (n_batch, n_data) complex equalized out"
        )

    def process(self, rx_data: Any, channel_est: Any = None, **kwargs: Any) -> Any:
        if channel_est is None:
            raise ValueError("MMSEEqualizer.process requires channel_est=")
        xp = self.xp
        rx_data = xp.asarray(rx_data)
        channel_est = xp.asarray(channel_est)
        weight = xp.conj(channel_est) / (xp.abs(channel_est) ** 2 + self.noise_var)
        return rx_data * weight
