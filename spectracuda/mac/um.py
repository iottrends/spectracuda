"""UmEntity: Unacknowledged Mode -- segmentation + sequence-numbered
reassembly, best-effort delivery, no retransmission.

Composes Segmenter (tx-side splitting) and ReassemblyBuffer (rx-side
reassembly) around the pdu.py header codec -- a lost segment means the
SDU it belonged to is simply never completed (see ReassemblyBuffer's own
docstring for the bounded-window give-up behavior this implies); UmEntity
does not itself detect or report loss, it just delivers what completes.
"""
from __future__ import annotations

from typing import Any, List

import numpy as np

from .pdu import HEADER_LEN_BITS, SN_MODULUS, TYPE_DATA, decode_header, encode_header, sn_add
from .reassembly import ReassemblyBuffer, Segmenter


class UmEntity:
    """Parameters
    ----------
    max_segment_bits:
        Max PDU payload size, in bits -- the PHY frame capacity `MacLink`
        (see session.py) is built against.
    window_size:
        ReassemblyBuffer's pending-segment window (see its own docstring).
    """

    def __init__(self, max_segment_bits: int, *, window_size: int = 32, **kwargs: Any) -> None:
        """**kwargs is a deliberate sink (e.g. absorbs max_retries= when
        constructed from a shared kwargs dict alongside AmEntity) -- see
        TmEntity's constructor docstring for the full rationale."""
        self._segmenter = Segmenter(max_segment_bits)
        self._reassembly = ReassemblyBuffer(window_size=window_size)
        self._next_tx_sn = 0

    def transmit(self, sdu_bits: Any) -> List[np.ndarray]:
        sdu_bits = np.asarray(sdu_bits, dtype="uint8")
        pdus = []
        for si, so, segment_bits in self._segmenter.segment(sdu_bits):
            header = encode_header(pdu_type=TYPE_DATA, si=si, sn=self._next_tx_sn, so=so)
            pdus.append(np.concatenate([header, segment_bits]))
            self._next_tx_sn = sn_add(self._next_tx_sn, 1)
        return pdus

    def receive(self, pdu_bits: Any) -> List[np.ndarray]:
        pdu_bits = np.asarray(pdu_bits, dtype="uint8")
        header = decode_header(pdu_bits[:HEADER_LEN_BITS])
        segment_bits = pdu_bits[HEADER_LEN_BITS:]
        return self._reassembly.ingest(header["sn"], header["si"], header["so"], segment_bits)

    @property
    def expected_sn(self) -> int:
        """The next SN this entity's ReassemblyBuffer is waiting on (its
        cumulative reassembly point) -- 0 if nothing has arrived yet.
        Exposed publicly for AmEntity's status-report base_sn (see
        am.py's build_status()), rather than reaching into
        ReassemblyBuffer's private state from outside this class."""
        return self._reassembly._expected_sn

    @staticmethod
    def sn_space() -> int:
        return SN_MODULUS
