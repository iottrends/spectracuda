"""Segmenter / ReassemblyBuffer: splitting an SDU too large for one PHY
frame into ordered PDU-sized segments, and reassembling them back into
SDUs on receive. Shared by UmEntity and AmEntity (TmEntity does neither --
see tm.py).

Plain classes, not Block subclasses (like framing/stats.py's compute_evm/
compute_rssi_db, not like Packetizer/HeaderCodec): this is small, host-
side, PDU-metadata bookkeeping -- there's no batched array math and no
meaningful backend/self.xp here, the same "framing bookkeeping is tiny
metadata work, not bulk DSP" reasoning already used throughout
spectracuda/framing/.

Segmenter is PHY-agnostic on purpose (mirrors Packetizer's own PHY-
agnostic design): it only knows `max_segment_bits`, a plain integer the
caller (MacLink, see session.py) supplies -- it has no idea what an Ofdm
frame even is.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .pdu import SI_FIRST, SI_FULL, SI_LAST, SI_MIDDLE, sn_add


class Segmenter:
    """Splits one SDU into (si, so, segment_bits) pieces, each at most
    `max_segment_bits` long. A single piece (si=SI_FULL, so=0) is
    returned when the whole SDU already fits."""

    def __init__(self, max_segment_bits: int) -> None:
        if max_segment_bits < 8 or max_segment_bits % 8 != 0:
            raise ValueError(
                f"max_segment_bits must be a positive multiple of 8, got "
                f"{max_segment_bits} -- see segment()'s docstring for why"
            )
        self.max_segment_bits = max_segment_bits

    def segment(self, sdu_bits: np.ndarray) -> List[Tuple[int, int, np.ndarray]]:
        """Every segment this produces -- including the final, possibly-
        shorter one -- is guaranteed a multiple of 8 bits IF `sdu_bits`
        itself is (checked below, not assumed): max_segment_bits is
        already a multiple of 8 (enforced in __init__), and the
        remainder segment's length (sdu_len % max_segment_bits) is then
        automatically a multiple of 8 too, since both operands are.
        This matters downstream, not just for its own sake: MacLink (see
        session.py) always requires the underlying Ofdm to have crc
        enabled, and Packetizer's CRC stage requires byte-aligned input
        (spectracuda/framing/packetizer.py) -- a non-byte-aligned PDU
        (header + segment) would fail there. Requiring byte-aligned SDUs
        here, once, is simpler and more honest than silently zero-padding
        a segment and having to track/strip that padding back off on
        reassembly."""
        sdu_bits = np.asarray(sdu_bits, dtype="uint8")
        n = sdu_bits.shape[-1]
        if n == 0:
            raise ValueError("cannot segment an empty SDU")
        if n % 8 != 0:
            raise ValueError(
                f"SDU is {n} bits, not a multiple of 8 -- MAC PDUs must be "
                f"byte-aligned (matching Packetizer's own CRC byte-"
                f"alignment requirement, see segment()'s docstring); pad "
                f"the raw SDU to a byte boundary yourself for now"
            )

        if n <= self.max_segment_bits:
            return [(SI_FULL, 0, sdu_bits)]

        pieces: List[Tuple[int, int, np.ndarray]] = []
        offset = 0
        while offset < n:
            end = min(offset + self.max_segment_bits, n)
            is_first = offset == 0
            is_last = end == n
            si = SI_FIRST if is_first else (SI_LAST if is_last else SI_MIDDLE)
            pieces.append((si, offset, sdu_bits[offset:end]))
            offset = end
        return pieces


class ReassemblyBuffer:
    """Reassembles segments (delivered out of order, by SN) back into
    complete, in-order SDUs.

    Scope, explicitly bounded rather than fully general: tracks ONE
    in-progress multi-segment SDU at a time (the segments of one SDU
    occupy a contiguous run of SNs, by construction -- see Segmenter/the
    entities that assign SN), plus a bounded window of segments that
    arrived early (SN ahead of the one currently expected). This is a
    real, stated simplification (see docs/mac.md) -- it does not support
    two SDUs' segment runs being reassembled concurrently/interleaved.
    For UM (no retransmission), a persistent gap at the expected SN is a
    real possibility (the missing segment is simply never coming) -- once
    the pending window fills up without the expected SN arriving, the
    stalled in-progress SDU is abandoned (dropped, not silently
    corrupted) and reassembly resyncs to the earliest pending SN. For AM,
    retransmission means a gap is expected to eventually fill, but the
    same bounded-window/give-up behavior still applies here as a safety
    net -- AM's own retry-exhaustion (see am.py) is the real backstop.
    """

    def __init__(self, window_size: int = 32, initial_expected_sn: int = 0) -> None:
        """`initial_expected_sn`: the TRUE starting SN of the stream this
        buffer will reassemble -- NOT inferred from whichever segment
        happens to physically arrive first (a real bug this class used
        to have: if the first-ARRIVING segment wasn't the first-SENT one,
        e.g. reordered delivery, seeding expected_sn from arrival order
        desyncs reassembly permanently -- caught directly by a genuine
        out-of-order-delivery test, not a hypothetical). Defaults to 0,
        matching every UmEntity/AmEntity's own SN counter starting point
        (see um.py) -- both ends of a link are expected to agree on
        this, the same way they already implicitly agree on SN 0 being
        where transmission starts."""
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.window_size = window_size
        self._expected_sn: int = initial_expected_sn
        self._pending: Dict[int, Tuple[int, int, np.ndarray]] = {}  # sn -> (si, so, bits)
        self._in_progress: List[np.ndarray] = []  # segment bits accumulated so far, in order

    def ingest(self, sn: int, si: int, so: int, segment_bits: np.ndarray) -> List[np.ndarray]:
        """Feed one arrived (sn, si, so, segment_bits) tuple in. Returns a
        list of completed SDU bit arrays (usually 0 or 1, structured as a
        list for generality) -- draining as many contiguous, complete
        SDUs as the newly-arrived segment unblocks."""
        segment_bits = np.asarray(segment_bits, dtype="uint8")
        self._pending[sn] = (si, so, segment_bits)

        if len(self._pending) > self.window_size and self._expected_sn not in self._pending:
            # The expected SN has been missing for too long -- give up on
            # whatever SDU was in progress and resync to the earliest
            # pending SN (see class docstring).
            self._in_progress = []
            self._expected_sn = min(self._pending)

        completed: List[np.ndarray] = []
        while self._expected_sn in self._pending:
            si_p, _so_p, bits_p = self._pending.pop(self._expected_sn)
            self._in_progress.append(bits_p)
            self._expected_sn = sn_add(self._expected_sn, 1)
            if si_p in (SI_FULL, SI_LAST):
                completed.append(np.concatenate(self._in_progress))
                self._in_progress = []
        return completed
