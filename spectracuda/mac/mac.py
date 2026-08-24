"""Mac: Mac(mode="tm"|"um"|"am", max_segment_bits=None, ofdm_kwargs=None,
**kwargs) -- one entry class, string mode, matching this codebase's
established FEC(scheme)/Modem(scheme) convention (private _MODES dict,
NOT registry.py's register()/resolve() -- same reasoning fec.py already
documents: this is "one canonical mode per name," not a place where
passing a custom instance instead of a string is the expected usage
pattern).

Unlike FEC/Modem, TM/UM/AM genuinely don't share one method surface: TM
and UM both expose transmit()/receive(), but AM's ARQ role needs FOUR
methods (transmit/receive_data/receive_status/build_status -- see am.py's
docstring for why this isn't collapsible to two without losing the
sender-role/receiver-role distinction a real duplex link needs). Rather
than forcing an artificial common interface across all three, Mac
delegates via __getattr__ to whichever entity `mode` selected -- so
`Mac(mode="am", ...).build_status()` works, `Mac(mode="um",
...).build_status()` correctly doesn't exist (AttributeError, not a
silent no-op), and callers who already know their mode can just use the
entity's own real API through this one construction entry point.

Two construction modes (settled by direct design discussion, not a
speculative both-ways API):

  Mac(mode, max_segment_bits=N)              # PHY-agnostic (unchanged
                                                # from before ofdm_kwargs
                                                # existed) -- no Ofdm at
                                                # all, caller picks
                                                # max_segment_bits by
                                                # hand. This is what every
                                                # existing test_mac_am.py/
                                                # test_mac_um.py/
                                                # test_mac_tm.py test uses,
                                                # unchanged.

  Mac(mode, ofdm_kwargs={...})                # owns a real Ofdm, built
                                                # HERE from ofdm_kwargs.
                                                # max_segment_bits is
                                                # DERIVED by inspecting
                                                # that just-built Ofdm's
                                                # actual capacity
                                                # (capacity.compute_max_
                                                # segment_bits) -- NOT
                                                # accepted as a separate
                                                # argument in this mode,
                                                # specifically to prevent
                                                # a caller-supplied number
                                                # silently disagreeing
                                                # with what the Ofdm can
                                                # actually carry (a real
                                                # design tension flagged
                                                # and resolved during
                                                # design discussion, not
                                                # discovered as a bug
                                                # later).

Two genuinely independent HW units in this model are just two separate
`Mac(mode=..., ofdm_kwargs=...)` calls -- each builds its OWN Ofdm, never
shares one with any other Mac. See send()/receive() below for the
resulting IQ-level API this enables, and docs/mac.md for the full
design discussion (including why MacLink/session.py is a DIFFERENT,
still-supported path: one shared Ofdm across two roles, useful for what
it proves, but not this class's job).
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from .am import AmEntity
from .bind import (
    decode_bind_request,
    decode_bind_response,
    encode_bind_request,
    encode_bind_response,
    evaluate_bind_request,
)
from .capacity import compute_max_segment_bits
from .quality import LinkQualityTracker, decode_quality_report, encode_quality_report
from .tm import TmEntity
from .um import UmEntity

_MODES = {
    "tm": TmEntity,
    "um": UmEntity,
    "am": AmEntity,
}


class Mac:
    def __init__(
        self,
        mode: str,
        max_segment_bits: Optional[int] = None,
        *,
        ofdm_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"Unknown MAC mode {mode!r}; expected one of {sorted(_MODES)}")
        self.mode = mode

        if ofdm_kwargs is not None:
            # Local import -- avoids Mac (usable with zero PHY
            # involvement, see class docstring) importing the whole
            # Ofdm/pipeline machinery just to support the OTHER
            # construction mode. Ofdm is built HERE, owned by this Mac
            # alone, never shared with any other Mac.
            from spectracuda.pipeline import Ofdm

            self.ofdm = Ofdm(**ofdm_kwargs)
            if self.ofdm.crc == "none":
                raise ValueError(
                    "Mac(ofdm_kwargs=...) requires crc != 'none' in ofdm_kwargs -- "
                    "without a CRC, a corrupted-but-decoded frame is indistinguishable "
                    "from a correctly-delivered one (same requirement MacLink has, see "
                    "session.py's module docstring)"
                )
            derived = compute_max_segment_bits(self.ofdm, has_mac_header=(mode != "tm"))
            if max_segment_bits is not None:
                raise ValueError(
                    f"max_segment_bits must not be passed alongside ofdm_kwargs -- it's "
                    f"DERIVED from the Ofdm you're asking to build (here: {derived} bits), "
                    f"never independently specified, to avoid the two silently disagreeing"
                )
            max_segment_bits = derived
        else:
            self.ofdm = None
            if max_segment_bits is None:
                raise ValueError("max_segment_bits is required when ofdm_kwargs is not given")

        self.max_segment_bits = max_segment_bits
        self._impl = _MODES[mode](max_segment_bits, **kwargs)
        self.quality = LinkQualityTracker()
        self.bound = False
        # Captured uniformly regardless of mode -- bind.encode_bind_request()
        # carries all 4 fields in its wire format unconditionally (see
        # bind.py), the same way MacLink.bind() already did.
        self._window_size = kwargs.get("window_size", 32)
        self._max_retries = kwargs.get("max_retries", 4)

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes Mac itself doesn't define -- see
        # class docstring for why this is a deliberate mode-specific
        # passthrough, not a bug-prone catch-all.
        return getattr(self._impl, name)

    # -- IQ-level API, only usable when this Mac owns a real Ofdm --------
    # TM/UM/AM all get send_iq()/receive_iq() -- for AM this is only the
    # DATA-forward half (self._impl.transmit()/receive_data()); the
    # STATUS/retransmission round trip needs a return path (this Mac's
    # receiving side sending a STATUS pdu back over a DIFFERENT Ofdm than
    # the one DATA arrived on) which is inherently a second Mac/Ofdm pair,
    # not something a single object's send_iq()/receive_iq() could do
    # regardless -- orchestrated manually via build_status()/
    # receive_status() (PDU-level, __getattr__) instead, see
    # send_iq()'s docstring and docs/mac.md's "4-Mac/4-Ofdm" writeup.
    #
    # Named send_iq()/receive_iq(), NOT send()/receive() -- TmEntity/
    # UmEntity/AmEntity already have their own transmit()/receive() at
    # the PDU level, reached via __getattr__ (see above). A same-named
    # receive() defined directly on Mac would SHADOW that delegation
    # (Python only falls through to __getattr__ when normal lookup
    # fails) -- a real bug caught immediately by the existing MacLink
    # test suite: MacLink builds its Mac(s) WITHOUT ofdm_kwargs (it owns
    # the Ofdm itself) and calls .receive() expecting the PDU-level
    # passthrough; a same-named IQ-level receive() intercepted that call
    # and raised "requires ofdm_kwargs" instead. Distinct names avoid
    # the collision entirely rather than trying to make one method
    # smart enough to detect which layer its argument belongs to.

    def _require_ofdm(self, method_name: str) -> None:
        if self.ofdm is None:
            raise ValueError(
                f"{method_name}() requires this Mac to have been constructed with "
                f"ofdm_kwargs=... (it was constructed PHY-agnostic -- see class docstring)"
            )

    def send_iq(self, sdu_bits: Any) -> List[np.ndarray]:
        """SDU -> list of IQ arrays, one per PDU this SDU segments into
        (self._impl.transmit() -- for AM, this is AmEntity.transmit(),
        which ALSO buffers each PDU for retransmission, exactly as it
        does at the PDU level; send_iq() itself is just the DATA-forward
        half of AM's flow -- the STATUS/retransmission round trip is
        deliberately NOT built into send_iq()/receive_iq(), since it
        inherently needs a second Mac/Ofdm pair for the reverse
        direction (see docs/mac.md's "4-Mac/4-Ofdm" writeup) -- callers
        orchestrate that manually with the PDU-level build_status()/
        receive_status() (reached via __getattr__, same as always) plus
        a second Mac's own send_iq()-equivalent machinery, not a single
        method call here.

        Requires bind() to have succeeded first (self.bound) -- a real
        behavioral gate, not a formality, same precedent as MacLink.send()
        (see session.py) -- raises otherwise. receive_iq() is NOT gated:
        mirrors that same precedent's asymmetry -- if the sender isn't
        bound, nothing gets transmitted for a receiver to receive in the
        first place, so gating the receive side too would be redundant."""
        self._require_ofdm("send_iq")
        if not self.bound:
            raise ValueError("Mac.send_iq() requires a successful bind() first -- see build_bind_request()")
        pdus = self._impl.transmit(sdu_bits)
        return [self.ofdm.generate_frame(np.asarray(pdu, dtype="uint8")[None, :]) for pdu in pdus]

    def _rx_one_frame(self, iq_array: Any) -> Optional[np.ndarray]:
        """One arrived IQ frame -> the decoded raw bits (post header/FEC/
        CRC, still PDU-level -- not yet handed to self._impl.receive()),
        or None if it didn't arrive usably (frame_found=False, CRC
        failure, or an uncorrectable-FEC/header-decode exception -- the
        SAME failure modes MacLink._phy_round() treats as "this PDU is
        gone this round", reused here, not reinvented). Every attempt --
        success or failure -- feeds self.quality (see quality.py).

        Shared by receive_iq() and the bind/quality IQ handlers below --
        all of them need this exact rx_process/quality/failure-handling
        logic, not four separate copies of it."""
        iq_array = np.asarray(iq_array)
        try:
            result = self.ofdm.rx_process(iq_array)
        except (ValueError, NotImplementedError):
            return None
        crc_valid = result["crc_valid"]
        delivered = bool(result["frame_found"]) and (crc_valid is None or bool(np.asarray(crc_valid)[0]))
        rssi_db = result["rssi_db"]
        evm = result["evm"]
        self.quality.observe(
            rssi_db=float(np.asarray(rssi_db)[0]),
            evm=None if evm is None else float(np.asarray(evm)[0]),
            delivered=delivered,
        )
        if not delivered:
            return None
        return np.asarray(result["bits"])[0].astype("uint8")

    def receive_iq(self, iq_array: Any) -> Any:
        """One arrived DATA IQ frame -> whatever self._impl.receive()
        (UM) / self._impl.receive() (TM, identity) / self._impl.
        receive_data() (AM -- a DIFFERENT method name than UM/TM's
        receive(), see am.py: AM's receiver role is receive_data(), not
        receive()) returns for that PDU, or an empty result if the frame
        didn't arrive usably -- see _rx_one_frame()'s docstring for the
        failure modes and quality bookkeeping this delegates to.

        AM callers still need to separately call build_status()/
        receive_status() (PDU-level, via __getattr__) and drive those
        through a second Mac/Ofdm pair for the reverse direction --
        receive_iq() only ever does the DATA-forward half (see
        send_iq()'s docstring)."""
        self._require_ofdm("receive_iq")
        delivered_bits = self._rx_one_frame(iq_array)
        if delivered_bits is None:
            return [] if self.mode in ("um", "am") else None  # tm.receive() is one array, not a list
        if self.mode == "am":
            return self._impl.receive_data(delivered_bits)
        return self._impl.receive(delivered_bits)

    # -- binding + link-quality reporting, IQ-level ----------------------
    # Mode-agnostic (unlike send_iq()/receive_iq()) -- binding and
    # quality-reporting are control-plane PDU exchanges independent of
    # AM's data-flow limitation, so none of these four guard on mode.
    #
    # Migrated here from MacLink (see docs/mac.md) specifically because
    # THIS is what makes a genuine cross-object handshake possible for
    # the first time: MacLink.bind() always evaluates a request against
    # its OWN capacity (one shared Ofdm across both roles), so it can
    # only ever self-consistently succeed. Two real Mac(ofdm_kwargs=...)
    # objects have genuinely independent self.max_segment_bits values,
    # so handle_bind_request_iq() below can actually reject a
    # mismatched request now -- see tests/test_mac_two_units_simple.py
    # for the standing proof (something MacLink's own test suite
    # structurally cannot produce).

    def build_bind_request(self) -> np.ndarray:
        """Encode a BIND_REQUEST for this Mac's own mode/max_segment_bits/
        window_size/max_retries, as IQ. The caller delivers this to a
        PEER Mac's handle_bind_request_iq() -- fully manual wiring, no
        orchestrator class (see docs/mac.md)."""
        self._require_ofdm("build_bind_request")
        request_pdu = encode_bind_request(self.mode, self.max_segment_bits, self._window_size, self._max_retries)
        return self.ofdm.generate_frame(request_pdu[None, :])

    def handle_bind_request_iq(self, iq_array: Any) -> Optional[np.ndarray]:
        """Decode an arrived BIND_REQUEST, evaluate it against THIS Mac's
        OWN capacity (bind.evaluate_bind_request() -- a genuine,
        independent decision, see class docstring above), set self.bound
        from the outcome, and return the BIND_RESPONSE as IQ for the
        caller to deliver back to the requester. None if the request
        itself didn't arrive usably (self.bound is left unchanged in
        that case -- nothing was actually evaluated)."""
        self._require_ofdm("handle_bind_request_iq")
        arrived = self._rx_one_frame(iq_array)
        if arrived is None:
            return None
        request = decode_bind_request(arrived)
        decision = evaluate_bind_request(request, local_max_segment_bits=self.max_segment_bits)
        self.bound = decision["accepted"]
        response_pdu = encode_bind_response(decision)
        return self.ofdm.generate_frame(response_pdu[None, :])

    def handle_bind_response_iq(self, iq_array: Any) -> bool:
        """Decode an arrived BIND_RESPONSE, set self.bound accordingly,
        and return it. False (self.bound also set False) if the
        response itself didn't arrive usably."""
        self._require_ofdm("handle_bind_response_iq")
        arrived = self._rx_one_frame(iq_array)
        if arrived is None:
            self.bound = False
            return False
        response = decode_bind_response(arrived)
        self.bound = response["accepted"]
        return self.bound

    def build_quality_report(self) -> np.ndarray:
        """This Mac's own accumulated LinkQualityTracker stats, as IQ."""
        self._require_ofdm("build_quality_report")
        report_pdu = encode_quality_report(self.quality.report_dict())
        return self.ofdm.generate_frame(report_pdu[None, :])

    def handle_quality_report_iq(self, iq_array: Any) -> Any:
        """Decode an arrived LINK_QUALITY report, return the decoded
        dict (see quality.decode_quality_report()) -- this Mac's own
        view of what the peer reported. Raises ValueError if the report
        itself didn't arrive usably -- no report to return, matching
        MacLink.exchange_link_quality()'s existing convention."""
        self._require_ofdm("handle_quality_report_iq")
        arrived = self._rx_one_frame(iq_array)
        if arrived is None:
            raise ValueError(
                "link-quality report PDU did not arrive usably -- no report to return "
                "(see _rx_one_frame()'s fail modes)"
            )
        return decode_quality_report(arrived)
