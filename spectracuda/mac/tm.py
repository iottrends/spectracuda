"""TmEntity: Transparent Mode -- raw passthrough, no header, no
segmentation, no retransmission.

Matches real TM's actual constraint (not a diluted version of it): real
3GPP TM is used only for content that's pre-sized to fit exactly in one
PHY transport block (e.g. broadcast system information) -- there is no
out-of-band mechanism inventable here that would keep this "transparent"
while also handling oversized SDUs, so this class simply requires the SDU
to already fit and raises otherwise, rather than quietly doing something
TM was never meant to do.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class TmEntity:
    """Parameters
    ----------
    max_segment_bits:
        The PHY frame capacity (in bits) this entity's SDUs must fit
        within -- same quantity `MacLink` passes to Um/AmEntity, kept
        here only to validate `transmit()`, never used to segment
        (TM never segments).
    """

    def __init__(self, max_segment_bits: int, **kwargs: Any) -> None:
        """**kwargs is a deliberate sink, not an oversight: Mac(mode=...)
        (see mac.py) and MacLink (see session.py) want to construct any
        of TmEntity/UmEntity/AmEntity from one shared kwargs dict
        (window_size=, max_retries=, ...) without special-casing TM, the
        same reasoning already documented on SchmidlCoxCFO/
        LSChannelEstimator's own **kwargs sinks."""
        if max_segment_bits < 1:
            raise ValueError("max_segment_bits must be >= 1")
        self.max_segment_bits = max_segment_bits

    def transmit(self, sdu_bits: Any):
        sdu_bits = np.asarray(sdu_bits, dtype="uint8")
        if sdu_bits.shape[-1] % 8 != 0:
            raise ValueError(
                f"SDU is {sdu_bits.shape[-1]} bits, not a multiple of 8 -- "
                f"see Segmenter.segment()'s docstring (spectracuda/mac/"
                f"reassembly.py) for why MAC PDUs must be byte-aligned; "
                f"pad the raw SDU to a byte boundary yourself for now"
            )
        if sdu_bits.shape[-1] > self.max_segment_bits:
            raise ValueError(
                f"SDU is {sdu_bits.shape[-1]} bits, exceeding TM's "
                f"max_segment_bits={self.max_segment_bits} -- TM never "
                f"segments (see class docstring); use mode='um' or 'am' "
                f"for SDUs that don't fit in one PHY frame"
            )
        return [sdu_bits]

    def receive(self, pdu_bits: Any):
        """Identity: a TM PDU IS the SDU, no header to strip."""
        return np.asarray(pdu_bits, dtype="uint8")
