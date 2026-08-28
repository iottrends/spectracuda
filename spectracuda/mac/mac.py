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

from typing import Any, Dict, List, Optional

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
from ..framing import compute_rssi_db
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
            # Kept verbatim (not just self.ofdm) so receive_iq_batch()
            # can later build independent Ofdm REPLICAS from the exact
            # same config -- see that method's own docstring for why a
            # pool of separate instances, not concurrent calls into this
            # one self.ofdm, is what real multi-core decode requires.
            self._ofdm_kwargs = dict(ofdm_kwargs)
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
            self._ofdm_kwargs = None
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

    def _rx_process_only(self, ofdm: Any, iq_array: Any) -> Optional[Dict[str, Any]]:
        """The PURE-decode half of what used to be _rx_one_frame() in one
        piece: just `ofdm.rx_process(iq_array)` (sync/CFO/OFDM-demod/
        channel-est/equalizer/FEC decode -- the expensive part), or None
        on the same ValueError/NotImplementedError this always treated
        as "didn't arrive usably." Touches ONLY the given `ofdm` instance
        -- no self.quality, no self._impl -- specifically so
        receive_iq_batch() below can run many of these concurrently, one
        thread per independent Ofdm replica, without two threads ever
        touching the same mutable state.

        Why a SEPARATE ofdm argument, not always self.ofdm: the native
        Viterbi/RS decoders underneath (fec/_native.py) each own ONE
        persistent C struct, reset at the START of every decode call and
        reused across calls -- correct for sequential reuse, but two
        threads calling decode() through the SAME instance concurrently
        would race on that struct's internal buffers (history_buffer,
        error_buffer), corrupting both results. A pool of independent
        Ofdm replicas (see _ofdm_replica_pool()), one per concurrent
        worker, sidesteps this entirely -- proven safe first as an
        isolated ThreadPoolExecutor experiment against bare
        ConvolutionalCode instances before wiring it in here.

        NotImplementedError still means "nothing usable, nothing to
        say" (None, as before) -- but ValueError specifically covers an
        uncorrectable-FEC payload decode failure (see ofdm.py's
        rx_process()/Packetizer.decode()), which happens AFTER sync and
        header decode already succeeded: a real, informative frame
        attempt, not silence. Found via examples/drone_tui/
        adaptive_mcs.py's own real-status-exchange test: at a marginal
        SNR, MOST corrupted qam64 frames failed this way rather than via
        crc_valid=False, and the old bare `return None` here made every
        one of them invisible to self.quality entirely (not merely
        "not delivered" -- literally never counted as an attempt) --
        directly contradicting _rx_one_frame()'s own documented
        "every attempt -- success or failure -- feeds self.quality"
        contract, and starving any decision (adaptive MCS included)
        that reads delivered_ratio of the very evidence it needs most.
        ofdm.rx_process() itself never gets to return its own dict in
        this case (the exception unwinds past its rssi_db/evm
        computation), so rssi_db is recomputed here the same way it
        does -- compute_rssi_db() over the whole raw iq_array,
        unconditionally, before any header/payload decode is attempted
        (see ofdm.py's rx_process()) -- evm has no such fallback (it's
        inherently a post-equalization, payload-content quantity) and is
        left None, same as any other frame_found=False result already
        leaves it. frame_found=False here (not True with crc_valid=
        False) is the honest characterization: this method only knows
        the payload was unrecoverable, not that the frame itself was
        genuinely present versus e.g. a borderline sync false-trigger."""
        try:
            return ofdm.rx_process(iq_array)
        except NotImplementedError:
            return None
        except ValueError:
            xp = ofdm.xp
            rx_iq = xp.asarray(iq_array)
            if rx_iq.ndim == 1:  # match rx_process()'s own top-of-function reshape (ofdm.py) --
                rx_iq = rx_iq[None, :]  # compute_rssi_db() needs a batch axis, not a bare (N,) array
            return {
                "frame_found": False,
                "crc_valid": None,
                "rssi_db": compute_rssi_db(xp, rx_iq),
                "evm": None,
                "bits": None,
            }

    def _apply_rx_result(self, result: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
        """The STATEFUL half of what used to be _rx_one_frame(): quality
        bookkeeping (self.quality.observe()) plus the delivered/CRC
        check, given an already-computed _rx_process_only() result (or
        None). Deliberately kept single-threaded-only -- self.quality is
        one shared LinkQualityTracker, so every call site (receive_iq()
        directly, receive_iq_batch() after its parallel decode phase)
        must call this in plain sequential order, never concurrently."""
        if result is None:
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
        logic, not four separate copies of it. Just
        _apply_rx_result(_rx_process_only(...)) against self.ofdm --
        split into those two pieces (see their own docstrings) so
        receive_iq_batch() can reuse each half on its own terms, not
        because this method itself needed to change.

        A real bug lived here until caught on actual CUDA hardware (a
        Colab Tesla T4, 2026-08-25): this used to run every value below
        through plain `np.asarray(...)`, which CuPy deliberately raises
        TypeError on ("implicit conversion to a NumPy array is not
        allowed") rather than silently doing a slow, unrequested
        device->host copy -- exactly what a numpy-only dev machine can
        never catch, since numpy's own asarray() is always a no-op on an
        already-numpy array. `self.ofdm.rx_process()`'s inputs/outputs
        live on `self.ofdm`'s own backend (genuinely cupy.ndarray when
        backend="cupy"), so iq_array is now passed straight through
        un-coerced (rx_process() does its own xp.asarray() internally,
        correctly handling either backend), and every value actually
        read out of `result` goes through self._to_host() first."""
        return self._apply_rx_result(self._rx_process_only(self.ofdm, iq_array))

    def _to_host(self, arr: Any) -> np.ndarray:
        """arr may genuinely be a cupy.ndarray (whenever self.ofdm.backend
        == "cupy") -- cupy.asnumpy() is required there since cupy raises
        on implicit conversion via plain np.asarray() (see
        _rx_one_frame's docstring for the real bug this fixes)."""
        if self.ofdm.backend == "cupy":
            import cupy

            return cupy.asnumpy(arr)
        return np.asarray(arr)

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
        return self._deliver(self._rx_one_frame(iq_array))

    def _deliver(self, delivered_bits: Optional[np.ndarray]) -> Any:
        """Shared by receive_iq() and receive_iq_batch(): mode-dependent
        hand-off of already-decoded PDU bits (or None) to self._impl --
        pulled out so both call sites apply the EXACT same per-mode
        logic, not two copies that could drift."""
        if delivered_bits is None:
            return [] if self.mode in ("um", "am") else None  # tm.receive() is one array, not a list
        if self.mode == "am":
            return self._impl.receive_data(delivered_bits)
        return self._impl.receive(delivered_bits)

    def _ofdm_replica_pool(self, n: int) -> List[Any]:
        """Lazily builds (and caches) a pool of >= n independent Ofdm
        instances, each constructed fresh from the exact same
        ofdm_kwargs self.ofdm itself was built from -- see
        receive_iq_batch()'s docstring for why real concurrent decode
        needs genuinely separate instances, not n threads sharing this
        Mac's own self.ofdm. Grows (never shrinks) the cached pool if a
        later call asks for more replicas than exist yet; a smaller
        later request just uses a prefix of the same cached list, so
        the replicas already spent effort caching their own compiled
        native FEC codecs stay warm across calls instead of being
        rebuilt from scratch every time."""
        if not hasattr(self, "_ofdm_replicas"):
            self._ofdm_replicas: List[Any] = []
        from spectracuda.pipeline import Ofdm

        while len(self._ofdm_replicas) < n:
            self._ofdm_replicas.append(Ofdm(**self._ofdm_kwargs))
        return self._ofdm_replicas[:n]

    def receive_iq_batch(self, iq_arrays: List[Any], n_workers: int = 2) -> List[Any]:
        """Like calling receive_iq() once per arrived IQ frame in
        iq_arrays, in the SAME order, with the SAME per-frame return
        values -- except the expensive part (sync/CFO/OFDM-demod/
        channel-est/equalizer/FEC decode, i.e. Ofdm.rx_process()) runs
        across up to n_workers frames AT ONCE, on real separate CPU
        cores, not n_workers=1 sequential calls.

        Why this needs its own method rather than just calling
        receive_iq() from n threads: the native Viterbi/RS decoders
        (fec/_native.py) each own ONE persistent C struct that gets
        reset, not recreated, at the start of every decode call --
        correct for sequential reuse of self.ofdm, but two threads
        decoding through that SAME struct concurrently would race on
        its internal history_buffer/error_buffer state and silently
        corrupt both results. This method instead round-robins frames
        across a POOL of independent Ofdm replicas (_ofdm_replica_pool,
        one per worker thread) built from the identical config
        self.ofdm itself uses -- proven necessary and sufficient first
        as an isolated ThreadPoolExecutor experiment against bare
        ConvolutionalCode instances, before wiring it in here.

        The stateful half of what a normal receive_iq() call does
        (self.quality.observe(), self._impl.receive()/receive_data())
        is deliberately run AFTERWARDS, single-threaded, in the
        original frame order -- self.quality and self._impl (the
        UM ReassemblyBuffer's sequence-number window, in particular)
        are shared mutable state this Mac owns ONCE, so touching them
        from multiple threads at once -- even just to append -- isn't
        something this method takes on faith is safe. Only the pure
        per-frame decode step (each frame's own independent replica)
        actually runs in parallel; everything that reads or writes
        self's own state runs exactly as it always did."""
        self._require_ofdm("receive_iq_batch")
        if not iq_arrays:
            return []
        n_workers = max(1, min(n_workers, len(iq_arrays)))
        if n_workers == 1:
            results = [self._rx_process_only(self.ofdm, iq) for iq in iq_arrays]
        else:
            import concurrent.futures

            replicas = self._ofdm_replica_pool(n_workers)
            results: List[Optional[Dict[str, Any]]] = [None] * len(iq_arrays)

            def _worker(worker_idx: int) -> None:
                # Each worker owns exactly one replica for its whole
                # life here -- frames are assigned round-robin, so no
                # two concurrently-running tasks ever touch the same
                # Ofdm instance (see this method's own docstring).
                ofdm = replicas[worker_idx]
                for i in range(worker_idx, len(iq_arrays), n_workers):
                    results[i] = self._rx_process_only(ofdm, iq_arrays[i])

            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
                list(ex.map(_worker, range(n_workers)))

        return [self._deliver(self._apply_rx_result(result)) for result in results]

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

    def set_tx_scheme(
        self, modem: Optional[str] = None, fec: Optional[str] = None, fec1: Optional[str] = None
    ) -> int:
        """Adaptive-MCS entry point: change this Mac's own OUTGOING
        modem/fec/fec1 choice (via Ofdm.reconfigure_tx_scheme() -- see
        its docstring for why that's an in-place reconfigure, not a
        rebuild-the-Ofdm operation), then propagate the resulting
        max_segment_bits change down into this mode's segmentation, so
        the NEXT send_iq() call segments against the new PHY capacity
        instead of a stale one.

        Caller's responsibility, not enforced here: serialize this
        against any concurrent send_iq() the same way examples/drone_tui/
        already serializes generate_frame() calls on one Ofdm (one
        shared push_lock) -- reconfigure_tx_scheme() mutates self.ofdm's
        modem/packetizer while a concurrent send_iq() could be mid-
        generate_frame() against the very same attributes. There is no
        mid-SDU hazard beyond that: send_iq() segments and returns an
        entire SDU's PDUs in one synchronous call (see its own
        docstring), so switching between two send_iq() calls is always
        a clean segmentation-generation boundary, never a partial one.

        Returns the new max_segment_bits."""
        self._require_ofdm("set_tx_scheme")
        self.ofdm.reconfigure_tx_scheme(modem=modem, fec=fec, fec1=fec1)
        self.max_segment_bits = compute_max_segment_bits(self.ofdm, has_mac_header=(self.mode != "tm"))
        # Segmentation bound lives on self._impl (TmEntity/UmEntity keep
        # their own copy passed at construction, see their docstrings;
        # AmEntity wraps a UmEntity internally, see am.py) -- self.
        # max_segment_bits above is this Mac's own record of the current
        # value (read by build_bind_request() etc.), not itself consulted
        # by any of these three entities' segmentation, so both must be
        # kept in step here rather than relying on one to imply the other.
        if self.mode == "tm":
            self._impl.max_segment_bits = self.max_segment_bits
        elif self.mode == "um":
            self._impl._segmenter.max_segment_bits = self.max_segment_bits
        else:  # am
            self._impl._um._segmenter.max_segment_bits = self.max_segment_bits
        return self.max_segment_bits

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
