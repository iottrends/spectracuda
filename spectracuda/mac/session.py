"""MacLink: wires a Mac (TM/UM/AM) to a real Ofdm PHY (+ optionally a
spectracuda.sim.Channel impairment) to actually carry SDUs over the air,
including AM's real retransmission rounds.

Explicitly a demonstration/integration harness, NOT "the real chain" the
way Ofdm/Packetizer are -- same role spectracuda/sim/channel.py already
plays (present for testing/demoing a real end-to-end scenario, not
assumed to be how every deployment would use these pieces). The Mac/
TmEntity/UmEntity/AmEntity objects themselves stay pure, PHY-agnostic
logic (AmEntity.receive_status() takes a decoded status PDU and returns
PDUs to retransmit -- it never touches Ofdm or a channel itself), so
they're independently reusable/testable, matching the Packetizer/
HeaderCodec precedent from the framing work (docs/todo.md #1.1).

One Ofdm instance drives BOTH directions (matches Ofdm's own existing
"one object, both tx and rx" design) -- a DATA pdu travels tx_mac ->
Ofdm -> rx_mac in the forward direction; AM's STATUS pdu travels
rx_mac -> Ofdm -> tx_mac in reverse, through the exact same Ofdm/channel
pipeline (this is a point-to-point link, not two independently-configured
radios, so reusing one Ofdm for both directions is the honest model, not
a shortcut).

Real dependency, stated explicitly rather than assumed obvious:
distinguishing "this PDU arrived corrupted" from "this PDU arrived
correctly" needs Ofdm's own crc= enabled -- frame_found=False catches
total loss (see docs/todo.md #1.1's "frame not found" work), but a frame
that syncs/decodes/FEC-decodes to WRONG bits with no CRC configured looks
identical to a successful delivery. MacLink requires crc != "none" on the
Ofdm it's given for exactly this reason.

Binding and link-quality reporting (see bind.py/quality.py): `send()`
refuses to run before `bind()` succeeds -- a real behavioral gate, not
just a formality -- and every `_phy_round()` (any pdu type, any
success/failure) feeds `self.quality`, so `exchange_link_quality()`
reports on the WHOLE session's PHY-frame history, not just DATA traffic.
`bind()`'s accept/reject decision is delegated to
`bind.evaluate_bind_request()`, a pure function tested independently
with genuinely mismatched configs (see tests/test_mac_bind.py) -- here,
since `MacLink` builds `tx_mac`/`rx_mac` from one shared config (see
class docstring above), the request always matches what this same
object's own `tx_mac.max_segment_bits` supports, so `bind()` always
succeeds in THIS harness; it proves the message-exchange mechanism over
a real Ofdm, not mismatch-rejection (that's the standalone tests' job).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .bind import (
    decode_bind_request,
    decode_bind_response,
    encode_bind_request,
    encode_bind_response,
    evaluate_bind_request,
)
from .capacity import compute_max_segment_bits as _compute_max_segment_bits
from .mac import Mac
from .quality import LinkQualityTracker, decode_quality_report, encode_quality_report


class MacLink:
    """Parameters
    ----------
    ofdm:
        A configured `spectracuda.pipeline.ofdm.Ofdm` instance, used for
        both directions. Must have `crc != "none"` (see module docstring).
    mode:
        "tm", "um", or "am".
    channel:
        Optional `spectracuda.sim.Channel` instance, applied to every
        frame this link sends (both directions) -- None means an ideal
        (lossless) channel.
    max_rounds:
        AM only: upper bound on retransmission rounds per send() call
        (defaults to max_retries + 2 -- enough rounds for every buffered
        PDU to exhaust its retry budget, plus slack for the final status
        round). Ignored for tm/um (always exactly one PHY round per PDU
        -- neither mode retransmits).
    window_size, max_retries:
        Forwarded to Mac(mode, ...) -- see TmEntity/UmEntity/AmEntity's
        own constructors for what each means per mode (TM ignores both
        via its **kwargs sink).
    """

    def __init__(
        self,
        ofdm: Any,
        mode: str,
        *,
        channel: Optional[Any] = None,
        max_rounds: Optional[int] = None,
        window_size: int = 32,
        max_retries: int = 4,
    ) -> None:
        if ofdm.crc == "none":
            raise ValueError(
                "MacLink requires ofdm.crc != 'none' -- without a CRC, a "
                "corrupted-but-decoded frame is indistinguishable from a "
                "correctly-delivered one (see module docstring)"
            )
        self.ofdm = ofdm
        self.mode = mode
        self.channel = channel
        self.max_rounds = max_rounds if max_rounds is not None else max_retries + 2

        max_segment_bits = _compute_max_segment_bits(ofdm, has_mac_header=mode != "tm")
        self.max_segment_bits = max_segment_bits
        self._window_size = window_size
        self._max_retries = max_retries
        kwargs = {"window_size": window_size, "max_retries": max_retries}
        self.tx_mac = Mac(mode, max_segment_bits, **kwargs)
        self.rx_mac = Mac(mode, max_segment_bits, **kwargs)

        self.bound = False
        self.quality = LinkQualityTracker()

    def _phy_round(self, pdu_bits: np.ndarray) -> Optional[np.ndarray]:
        """One PDU, one over-the-air round trip. Returns the decoded PDU
        bits, or None if the PDU didn't arrive usably -- total loss
        (frame_found=False), a CRC failure, or an uncorrectable-FEC/
        header-decode exception (ValueError/NotImplementedError -- the
        same failure modes docs/todo.md #1.1/#1.2/LDPC's fail-loud
        convention deliberately raise rather than hide) are all treated
        identically here: this PDU is gone this round. Every attempt --
        success or failure, any pdu type -- feeds self.quality (see
        module docstring); the exception path is the one case with no
        rssi_db/evm to observe at all (rx_process never returned), so
        it's skipped there, not fabricated."""
        tx_iq = self.ofdm.generate_frame(np.asarray(pdu_bits, dtype="uint8")[None, :])
        if self.channel is not None:
            tx_iq = self.channel.process(tx_iq)
        try:
            result = self.ofdm.rx_process(tx_iq)
        except (ValueError, NotImplementedError):
            return None

        crc_valid = result["crc_valid"]
        delivered = bool(result["frame_found"]) and (crc_valid is None or bool(self._to_host(crc_valid)[0]))
        rssi_db = result["rssi_db"]
        evm = result["evm"]
        self.quality.observe(
            rssi_db=float(self._to_host(rssi_db)[0]),
            evm=None if evm is None else float(self._to_host(evm)[0]),
            delivered=delivered,
        )
        if not delivered:
            return None
        return self._to_host(result["bits"])[0].astype("uint8")

    def _to_host(self, arr: Any) -> np.ndarray:
        """arr may genuinely be a cupy.ndarray (whenever self.ofdm.backend
        == "cupy") -- plain np.asarray() raises on that (CuPy disallows
        implicit conversion), same real bug and fix as
        Mac._rx_one_frame() in mac.py -- caught on actual CUDA hardware
        (Colab Tesla T4, 2026-08-25)."""
        if self.ofdm.backend == "cupy":
            import cupy

            return cupy.asnumpy(arr)
        return np.asarray(arr)

    def bind(self) -> bool:
        """The binding handshake (docs/mac.md): send a BIND_REQUEST
        (this side's mode/max_segment_bits/window_size/max_retries),
        have the peer evaluate it against ITS OWN capacity via
        bind.evaluate_bind_request(), and adopt the result. Sets
        self.bound accordingly and returns it. send() refuses to run
        before this succeeds -- see module/class docstrings for why
        this always succeeds in THIS harness (one shared Ofdm/config)
        and where the real mismatch-rejection is actually proven."""
        request_pdu = encode_bind_request(self.mode, self.max_segment_bits, self._window_size, self._max_retries)
        arrived = self._phy_round(request_pdu)
        if arrived is None:
            self.bound = False
            return False

        request = decode_bind_request(arrived)
        # Acceptor-side evaluation, against the peer's OWN capacity --
        # here, the same self.max_segment_bits (one shared Ofdm/config,
        # see class docstring); a genuinely different acceptor capacity
        # is what tests/test_mac_bind.py exercises directly.
        decision = evaluate_bind_request(request, local_max_segment_bits=self.max_segment_bits)
        response_pdu = encode_bind_response(decision)
        response_arrived = self._phy_round(response_pdu)
        if response_arrived is None:
            self.bound = False
            return False

        response = decode_bind_response(response_arrived)
        self.bound = response["accepted"]
        return self.bound

    def exchange_link_quality(self) -> Dict[str, Any]:
        """Report this side's accumulated LinkQualityTracker stats to the
        peer, over one PHY round. Returns the decoded report dict (see
        quality.decode_quality_report()) -- the peer's own view of what
        was sent, i.e. this method's return value reflects what actually
        survived the trip, not necessarily byte-identical to
        self.quality.report_dict() if this very round itself experiences
        loss (in which case it raises, rather than fabricating a report
        -- see below)."""
        report_pdu = encode_quality_report(self.quality.report_dict())
        arrived = self._phy_round(report_pdu)
        if arrived is None:
            raise ValueError(
                "link-quality report PDU did not arrive usably this round -- "
                "no report to return (see _phy_round()'s fail modes)"
            )
        return decode_quality_report(arrived)

    def send(self, sdu_bits: Any) -> List[np.ndarray]:
        """Transmit one SDU through this link. Returns whatever complete
        SDU(s) the peer ends up delivering (0 or more -- TM/UM deliver at
        most one, matching one send() = one SDU; AM may deliver 0 if
        retries were exhausted before completion, see AmEntity.failed_sns
        to distinguish "still pending" from "gave up"). Requires bind()
        to have succeeded first -- the actual behavioral point of
        binding (see module docstring), not just a formality."""
        if not self.bound:
            raise ValueError("MacLink.send() requires a successful bind() first -- call link.bind()")
        sdu_bits = np.asarray(sdu_bits, dtype="uint8")
        delivered: List[np.ndarray] = []

        if self.mode == "am":
            pending = self.tx_mac.transmit(sdu_bits)
            for _ in range(self.max_rounds):
                if not pending:
                    break
                for pdu in pending:
                    arrived = self._phy_round(pdu)
                    if arrived is not None:
                        delivered.extend(self.rx_mac.receive_data(arrived))
                status = self.rx_mac.build_status()
                status_arrived = self._phy_round(status)
                pending = (
                    self.tx_mac.receive_status(status_arrived)
                    if status_arrived is not None
                    else self.tx_mac.pending_pdus  # status itself lost -> retry the whole buffer
                )
            return delivered

        # tm / um: single-shot, one PHY round per PDU, no retry (neither
        # mode retransmits -- that's the whole behavioral distinction).
        for pdu in self.tx_mac.transmit(sdu_bits):
            arrived = self._phy_round(pdu)
            if arrived is None:
                continue
            if self.mode == "tm":
                delivered.append(self.rx_mac.receive(arrived))
            else:
                delivered.extend(self.rx_mac.receive(arrived))
        return delivered
