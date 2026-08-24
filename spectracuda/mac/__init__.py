"""MAC layer: TM (Transparent), UM (Unacknowledged), AM (Acknowledged)
delivery modes -- a combined MAC+RLC layer (named after 3GPP RLC's three
modes; this project does not separate RLC/MAC into distinct sublayers,
see docs/mac.md) sitting above the OFDM PHY chain (spectracuda.pipeline.
Ofdm) and its framing layer (spectracuda.framing).

Simplified/custom behavior, not 3GPP-spec-exact PDU wire formats (see
pdu.py's module docstring) -- point-to-point only, no multi-UE scheduling
concept (this project is a single Ofdm tx/rx link). AM uses plain ARQ, not
HARQ soft-combining (see am.py's module docstring for why).

    Mac(mode="tm"|"um"|"am", max_segment_bits, ...)  # one entry point,
                                                       # string mode,
                                                       # matches FEC(scheme)/
                                                       # Modem(scheme)
    MacLink(ofdm, mode, channel=None)                 # wires a Mac to a
                                                       # real Ofdm (+
                                                       # optional sim.Channel)
                                                       # for genuine
                                                       # end-to-end delivery,
                                                       # including AM's
                                                       # retransmission rounds

MacLink also owns the binding handshake (bind.py) -- link.bind() must
succeed before link.send() will run -- and link-quality reporting
(quality.py) -- link.exchange_link_quality() reports aggregated
RSSI/EVM/delivery-ratio stats to the peer, over the same real PHY.
"""
from .am import AmEntity
from .bind import evaluate_bind_request
from .mac import Mac
from .quality import LinkQualityTracker
from .session import MacLink
from .tm import TmEntity
from .um import UmEntity

__all__ = [
    "Mac", "MacLink", "TmEntity", "UmEntity", "AmEntity",
    "evaluate_bind_request", "LinkQualityTracker",
]
