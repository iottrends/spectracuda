"""ZFEqualizer: zero-forcing per-subcarrier frequency-domain equalization.

No liquid-dsp precedent -- liquid-dsp's eqlms/eqrls are sample-adaptive
single-carrier equalizers, not per-subcarrier frequency-domain ZF (see
docs/liquid-dsp-api-inventory.md); designed from standard reference, not
ported.

Batch-shape contract: process(rx_data, channel_est) takes matching
(n_batch, n_data) complex received data symbols and channel estimate at
the same subcarrier indices -> (n_batch, n_data) complex equalized
symbols.
"""
from __future__ import annotations

from typing import Any

from ..block import Block
from ..registry import register


@register("equalizer", "zf")
class ZFEqualizer(Block):
    def __init__(self, *, backend=None) -> None:
        super().__init__(backend=backend)
        self.batch_shape_doc = (
            "(n_batch, n_data) complex rx + (n_batch, n_data) complex "
            "channel est in -> (n_batch, n_data) complex equalized out"
        )

    def process(self, rx_data: Any, channel_est: Any = None, **kwargs: Any) -> Any:
        if channel_est is None:
            raise ValueError("ZFEqualizer.process requires channel_est=")
        xp = self.xp
        return xp.asarray(rx_data) / xp.asarray(channel_est)
