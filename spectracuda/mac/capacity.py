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
    drifting out of sync with Ofdm's own real behavior.

    History, not a hypothetical -- two real bugs found and fixed here,
    in sequence:

    (1) A block-oriented fec0 scheme ("rs_m8", any "ldpc_*" -- see
    fec/fec.py's `k_bits`) used to require its encoded_length() INPUT to
    be an exact multiple of `k_bits`, raising ValueError otherwise -- not
    "too big", a completely different failure ("wrong shape"). The plain
    byte-aligned binary search below couldn't tell those apart (both just
    looked like a caught exception), and arbitrary bisected guesses
    almost never land exactly on a multiple of a scheme like rs_m8's
    k_bits=1784, so the search got essentially no real signal and
    collapsed to the floor (8 bits) for EVERY rs_m8/LDPC config run
    through Mac(ofdm_kwargs=...) -- found while setting up a real
    fft=256/16-QAM/rs_m8+conv_v27 config, not by inspection. First fixed
    by searching whole BLOCKS instead of arbitrary sizes for any
    block-oriented fec0.

    (2) That whole-block workaround only papered over the real problem:
    forcing every message through an exact 1784-bit multiple meant even
    the bind handshake (104 bits) or an ordinary short message had no
    valid size at all. The actual fix was in the FEC layer itself --
    rs_m8 now supports "shortened" codewords (any byte-aligned length up
    to one full block, see fec/reed_solomon.py's encode()/decode()
    docstrings) -- so rs_m8's encoded_length() no longer raises for a
    non-multiple length, and the search below no longer needs the
    whole-block workaround AT ALL for rs_m8 -- back to the plain search,
    now genuinely correct again. `accepts_partial_block` (set on `FEC`,
    see fec.py) is what actually distinguishes rs_m8 (True) from LDPC
    (False, still exact-multiple only, a documented separate gap) here."""
    limit = ofdm.MAX_PAYLOAD_SYMBOLS * ofdm.bits_per_ofdm_symbol
    packetizer = ofdm.packetizer
    fec0 = packetizer.fec_codec
    needs_whole_block_search = (
        fec0 is not None
        and getattr(fec0, "k_bits", None) is not None
        and not getattr(fec0, "accepts_partial_block", False)
    )

    if needs_whole_block_search:
        block_bits = fec0.k_bits
        # Whole-block search: try n=1,2,3,... blocks of PRE-FEC (i.e.
        # post-CRC) bits. crc_overhead is already a multiple of 8 (CRC
        # key lengths are whole bytes), but n * block_bits landing on a
        # byte boundary once crc_overhead is subtracted back out is NOT
        # guaranteed (e.g. ldpc_648_r12's k_bits=324 is not itself a
        # multiple of 8 -- needs n=2 before `raw` is byte-aligned) --
        # skip any n that doesn't clear that bar rather than treating it
        # as "too big" the way the old code accidentally did for every n.
        #
        # NOT handled here (documented gap, not silently ignored): fec1
        # ALSO being block-oriented, with a block size incompatible with
        # fec0's, could in principle need a larger n than a naive 1-block
        # step finds -- the try/except below still protects against ever
        # RETURNING a wrong-shape size in that case, it just isn't
        # guaranteed to find the true optimum. Not the situation this fix
        # was written for (every case found so far pairs a block-oriented
        # fec0 with a bit-level fec1 like conv_v27, which has no
        # `k_bits` at all).
        crc_overhead = packetizer.crc_key_length_bytes * 8
        best = 8
        n = 1
        while True:
            pre_fec_bits = n * block_bits
            raw = pre_fec_bits - crc_overhead
            n += 1
            if raw < 8 or raw % 8 != 0:
                continue  # wrong shape for THIS n -- try the next block count, not "too big"
            try:
                fits = packetizer.encoded_length(raw) <= limit
            except ValueError:
                fits = False
            if not fits:
                break
            best = raw
        return max(8, best - HEADER_LEN_BITS) if has_mac_header else best

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
