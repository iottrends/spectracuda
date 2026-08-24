"""compute_max_segment_bits(): shared PHY-capacity math, used by both
Mac (when constructed with ofdm_kwargs=, owning its own Ofdm) and
MacLink (which builds a Mac from an externally-supplied Ofdm). Lives in
its own module specifically to avoid a circular import -- mac.py needs
it now too, and session.py already does `from .mac import Mac`, so this
helper can't live in either of those two modules without one importing
the other in both directions.
"""
from __future__ import annotations

from typing import Any

from .pdu import HEADER_LEN_BITS


def compute_max_segment_bits(ofdm: Any, has_mac_header: bool) -> int:
    """Largest raw PDU size (bits, multiple of 8 -- Packetizer's CRC
    stage requires byte alignment) whose Packetizer-encoded length still
    fits within Ofdm.MAX_PAYLOAD_SYMBOLS worth of OFDM symbols, minus
    room for the MAC header itself (UM/AM only -- TM has none).

    Binary-searches against ofdm.packetizer.encoded_length() -- the
    SAME capacity accounting Ofdm's own generate_frame()/rx_process()
    already use (see docs/todo.md #1.10) -- rather than re-deriving FEC
    rate/CRC-overhead math independently, which would risk silently
    drifting out of sync with Ofdm's own real behavior."""
    limit = ofdm.MAX_PAYLOAD_SYMBOLS * ofdm.bits_per_ofdm_symbol
    lo, hi, best = 8, limit, 8
    while lo <= hi:
        mid = ((lo + hi) // 2 // 8) * 8
        if mid < 8:
            break
        try:
            fits = ofdm.packetizer.encoded_length(mid) <= limit
        except ValueError:
            fits = False
        if fits:
            best = mid
            lo = mid + 8
        else:
            hi = mid - 8
    return max(8, best - HEADER_LEN_BITS) if has_mac_header else best
