"""AmEntity: Acknowledged Mode -- everything UmEntity has (segmentation,
SN, reassembly), plus retransmission: a transmit-side buffer of
sent-but-unacknowledged PDUs, STATUS-PDU-driven ACK/NACK reporting, and a
bounded retry policy.

ARQ, not HARQ (see docs/mac.md's Scope boundary): a NACKed PDU is
retransmitted as a fresh copy and independently re-decoded -- there is no
soft-combining of a PDU with its earlier failed copy. Soft-combining would
need raw LLRs/IQ kept across retransmissions, which conflicts with this
codebase's existing hard-decision-bits-everywhere FEC interface (see
fec/ldpc.py's own documented limitation) -- scoping to plain ARQ avoids
reopening that gap here.

Symmetric API: both ends of a link use their OWN AmEntity instance. One
side's `transmit()`/`receive_status()` (its role as a sender) pairs with
the other side's `receive_data()`/`build_status()` (its role as a
receiver) for that direction of traffic -- a full-duplex link needs one
AmEntity per direction per endpoint, wired together by MacLink (see
session.py), not a single shared instance.

Status-report design (this project's own, not a 3GPP wire format -- see
pdu.py's module docstring for the same caveat on the header itself): a
STATUS pdu's header SN field is repurposed as `base_sn`, the receiver's
current cumulative reassembly point (UmEntity.expected_sn -- "I have
fully processed everything before this SN"); its payload is a fixed
`window_size`-bit bitmap, bit i = 1 if SN (base_sn+i) has been physically
received (regardless of whether its SDU has fully reassembled yet), 0 if
missing. This mirrors real RLC AM STATUS PDUs' actual ACK_SN + NACK-list
concept, just with fixed small window instead of real spec bit-packing.
"""
from __future__ import annotations

from typing import Any, List, Set

import numpy as np

from .pdu import (
    HEADER_LEN_BITS,
    SN_MODULUS,
    TYPE_DATA,
    TYPE_STATUS,
    decode_header,
    encode_header,
    sn_add,
    sn_precedes,
)
from .um import UmEntity


class AmEntity:
    """Parameters
    ----------
    max_segment_bits:
        Max PDU payload size, in bits (same meaning as UmEntity's).
    window_size:
        Both ReassemblyBuffer's pending-segment window AND the STATUS
        PDU's bitmap width -- kept as one parameter since they serve the
        same "how far ahead of the cumulative point do we track" role.
    max_retries:
        Per-SN retransmission cap. After this many NACKed rounds for the
        same SN, that PDU is given up on -- removed from the
        retransmission buffer and recorded in `self.failed_sns` (the
        caller decides what "permanent delivery failure" means for their
        use case; AmEntity itself never retries forever).
    """

    def __init__(self, max_segment_bits: int, *, window_size: int = 32, max_retries: int = 4) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self._um = UmEntity(max_segment_bits, window_size=window_size)
        self.window_size = window_size
        self.max_retries = max_retries
        # sn -> [pdu_bits, retry_count] -- sent, not yet cumulatively acked.
        self._retx_buffer: dict = {}
        # every SN whose segment has physically arrived on this side (as
        # a receiver), regardless of SDU reassembly completion -- what
        # build_status()'s bitmap reports.
        self._received_sns: Set[int] = set()
        self.failed_sns: Set[int] = set()

    # -- transmit side (this entity's own outgoing data) -----------------

    def transmit(self, sdu_bits: Any) -> List[np.ndarray]:
        pdus = self._um.transmit(sdu_bits)
        for pdu_bits in pdus:
            header = decode_header(pdu_bits[:HEADER_LEN_BITS])
            self._retx_buffer[header["sn"]] = [pdu_bits, 0]
        return pdus

    def receive_status(self, status_pdu_bits: Any) -> List[np.ndarray]:
        """Process a STATUS pdu received from the peer (reporting on
        traffic THIS entity sent via transmit()). Returns the PDUs that
        need retransmitting this round -- SNs that exceeded max_retries
        are dropped from the buffer and added to self.failed_sns instead
        of being returned."""
        status_pdu_bits = np.asarray(status_pdu_bits, dtype="uint8")
        header = decode_header(status_pdu_bits[:HEADER_LEN_BITS])
        if header["pdu_type"] != TYPE_STATUS:
            raise ValueError(f"expected a STATUS pdu (pdu_type={TYPE_STATUS}), got {header['pdu_type']}")
        base_sn = header["sn"]
        bitmap = status_pdu_bits[HEADER_LEN_BITS : HEADER_LEN_BITS + self.window_size]

        to_retransmit = []
        for sn in list(self._retx_buffer.keys()):
            if sn_precedes(sn, base_sn):
                # Cumulatively acked (base_sn = "the first SN NOT yet
                # fully processed" -- everything strictly before it has
                # been received; base_sn ITSELF has not, by definition,
                # and must fall through to the bitmap check below, not
                # be treated as acked here -- a real bug this line used
                # to have, caught by a single-dropped-segment test).
                del self._retx_buffer[sn]
                continue
            offset = (sn - base_sn) % SN_MODULUS  # forward distance from base_sn
            if offset >= self.window_size:
                continue  # not yet reported on this round -- leave buffered
            if bitmap[offset]:
                del self._retx_buffer[sn]  # received, just not yet cumulative
                continue

            entry = self._retx_buffer[sn]
            entry[1] += 1
            if entry[1] > self.max_retries:
                del self._retx_buffer[sn]
                self.failed_sns.add(sn)
            else:
                to_retransmit.append(entry[0])
        return to_retransmit

    @property
    def pending_pdus(self) -> List[np.ndarray]:
        """Every PDU still buffered awaiting acknowledgment -- exposed so
        a caller (MacLink, see session.py) has a well-defined fallback
        when a STATUS pdu itself is lost (no status this round = assume
        nothing new was acked, retry the whole outstanding buffer rather
        than stalling), without reaching into _retx_buffer directly."""
        return [entry[0] for entry in self._retx_buffer.values()]

    # -- receive side (this entity's own incoming data) -------------------

    def receive_data(self, pdu_bits: Any) -> List[np.ndarray]:
        pdu_bits = np.asarray(pdu_bits, dtype="uint8")
        header = decode_header(pdu_bits[:HEADER_LEN_BITS])
        if header["pdu_type"] != TYPE_DATA:
            raise ValueError(f"expected a DATA pdu (pdu_type={TYPE_DATA}), got {header['pdu_type']}")
        self._received_sns.add(header["sn"])
        return self._um.receive(pdu_bits)

    def build_status(self) -> np.ndarray:
        """Report on traffic received via receive_data(), for the peer's
        receive_status() to act on."""
        base_sn = self._um.expected_sn
        bitmap = np.zeros(self.window_size, dtype="uint8")
        for offset in range(self.window_size):
            sn = sn_add(base_sn, offset)
            if sn in self._received_sns:
                bitmap[offset] = 1
        header = encode_header(pdu_type=TYPE_STATUS, si=0, sn=base_sn, so=0)
        return np.concatenate([header, bitmap])
