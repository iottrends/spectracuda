"""Ofdm: the full OFDM tx+rx chain as ONE class, entirely
constructor-configured -- subcarrier/pilot/data/guard split, cyclic
prefix, modem, fec, sync, cfo, channel estimator, equalizer, backend.

This replaces the earlier OfdmRx/OfdmTx split sketched in
docs/architecture.md. Reasoning: tx and rx share almost every one of
these parameters (subcarrier allocation, CP length, preamble/training
content), and liquid-dsp's own OFDM example
(reference/liquid-dsp/examples/ofdmflexframesync_example.c) constructs
`ofdmflexframegen`/`ofdmflexframesync` by hand with the same
M/cp_len/taper_len/subcarrier values passed to both -- that's a
duplication C has to live with, not a design worth reproducing in
Python. One object owning both directions also removes a real class of
bug: tx and rx silently drifting out of sync on preamble/training-
sequence content, which had to be hand-matched across separate variables
in examples/ofdm_256_schmidl_cox_demo.py (the manual version this
replaces -- see that module's docstring for the two real bugs found
building it: the Schmidl-Cox R(d) boundary blowup, and the preamble-CP
plateau).

Usage is modeled on liquid-dsp's create-once-then-a-few-calls idiom, not
manual per-symbol slot-offset arithmetic:

    ofdm = Ofdm(fft_size=256, n_pilot=6, n_data=200, cp_len=32,
                modem="qpsk", fec="conv_v27",  # or "rs_m8", or "none"
                sync="schmidl_cox", cfo="schmidl_cox",
                channel_estimator="ls", equalizer="mmse")

    tx_iq = ofdm.generate_frame(payload_bits)
    result = ofdm.rx_process(rx_iq)   # -> {"bits": ..., "header": {...}, ...}

Frame structure this class builds/expects (each symbol slot is
cp_len + fft_size samples, except the preamble -- see
examples/ofdm_256_schmidl_cox_demo.py for why it has no CP). This maps
directly onto liquid-dsp's own OFDM frame structure (S0/S1/header/
payload, see reference/liquid-dsp/src/multichannel/src/ofdmframe.common.c
and src/framing/src/ofdmflexframegen.c) -- our preamble is their S0, our
training symbol is their S1:

    [ preamble (no CP) ] [ training symbol(s) ] [ header symbol(s) ] [ payload symbol(s) ]

Header: matches liquid-dsp's actual field layout (112 bits total, see
`reference/liquid-dsp/include/liquid.internal.h`'s OFDMFLEXFRAME_H_*
constants and `ofdmflexframegen_encode_header`), NOT a trimmed-down
version of it -- an earlier version of this class carried only a 16-bit
length field on a symbol shared with payload data; that was wrong for a
different reason than size (see below), and was replaced by this:

    byte 0:      protocol/version                 8 bits
    bytes 1-2:   payload length, in BITS           16 bits (liquid-dsp
                 counts bytes; ours counts bits, matching this project's
                 bit-oriented Modem interface throughout rather than
                 liquid-dsp's byte-oriented packetizer)
    byte 3:      mod_scheme (payload's modulation)  8 bits
    byte 4:      crc_type(3b) + fec0(5b)            8 bits
    byte 5:      fec1                                8 bits
    bytes 6-13:  user-defined data (8 bytes)        64 bits
                                                    --------
                                                    112 bits

Real correctness reason this exists (not just matching liquid-dsp for
its own sake): `Ofdm` is one Python object used for both tx and rx in
this codebase's tests/examples, which made it tempting to assume the
receiver already "knows" mod_scheme/fec because it's the same object's
constructor arguments. That assumption is wrong for the actual use case
this library is for -- a real receiver is a *separate device* that never
saw the transmitter's `Ofdm(...)` call; it only has the bits that arrived
over the air. So `rx_process()` decodes mod_scheme from the header and
dynamically builds the matching `Modem` for the payload -- it does NOT
assume `self.modem` (which is only "what this object uses when *it*
transmits") matches what's actually in the signal. `fec=` at
construction is still checked at construction time (only "none" is
implemented, Phase 3 gap), and the header's fec0/fec1/crc fields are
decoded and enforced too -- a decoded header claiming to need FEC/CRC we
don't have raises NotImplementedError, the same way the constructor does.

Header symbol(s), not a shared symbol: given 112 real bits, mixing them
onto part of a payload symbol (the noted-above earlier approach) doesn't
make sense the way it did for a 16-bit field -- liquid-dsp's own header
(224 bits after Golay FEC, ~5 symbols at their typical config) is
substantial enough that dedicating symbols to it is the right call, not
wasteful the way it would be for a token-sized field. `num_symbols_header
= ceil(HEADER_LEN_BITS / n_data)`, matching liquid-dsp's own formula
exactly. The header's 112 real bits are spread evenly across the full
flat capacity of all its symbol(s) combined (frequency diversity, same
reasoning as before) and scrambled with a fixed mask; any leftover
capacity (when HEADER_LEN_BITS isn't an exact multiple of n_data) is
filled with fixed random content -- not real data, not deterministic
padding -- matching liquid-dsp's own fix
(`modemcf_gen_rand_sym`/`scramble_data()` in `ofdmflexframegen_gen_header`
/`_encode_header`) for the actual bug found during development: nearly-
constant header content constructively interferes into a massive
time-domain PAPR spike (measured: 181x peak/average power vs ~5x for
ordinary payload content), which was corrupting the header specifically
under real multipath while payload (naturally varied content) decoded
fine under the identical channel -- confirmed by direct side-by-side
comparison, not assumed.

BPSK for the header regardless of what `modem=` is set to for the
payload -- matches liquid-dsp's own rationale
(`OFDMFLEXFRAME_H_MOD = LIQUID_MODEM_BPSK`): the header must be
decodable before the receiver knows anything else about the frame,
including what scheme the payload uses, so it needs the most robust
modulation available, independent of the payload's own choice.

FEC (`fec=`, `fec1=`): "none", "conv_v27" (rate-1/2 K=7 Viterbi),
"rs_m8" (RS(255,223)), or any "ldpc_<n>_r<rate>" variant -- see
spectracuda.fec.FEC. `fec=` is the INNER stage (liquid-dsp's fec0),
`fec1=` an optional OUTER stage (liquid-dsp's fec1, default "none" --
concatenated two-stage coding is a real but fairly specialized
technique; see docs/todo.md #1.2 and spectracuda.framing.Packetizer's
module docstring for the verified inner-then-outer encode / outer-
then-inner decode ordering, taken directly from liquid-dsp's own
`packetizer_create`/`packetizer_encode`/`packetizer_decode`, not
assumed from the "inner/outer" naming alone). `generate_frame()`
FEC-encodes the raw (CRC-appended) payload bits before modulating them;
the header's payload_len_bits records the RAW (pre-CRC, pre-FEC) bit
count (matching liquid-dsp's own payload_dec_len convention), and
`rx_process()` resolves the actual fec0/fec1 schemes from the decoded
header (not from self.fec_codec -- same reasoning as mod_scheme above)
to compute how many encoded bits/OFDM symbols to expect, then decodes
back to the original raw bits via `spectracuda.framing.Packetizer`.

CRC (`crc=`): "none", "checksum", "crc8", "crc16", "crc24", or "crc32" --
see spectracuda.fec.CRC (self-contained, byte-exact port of liquid-dsp's
crc.c, unlike FEC's from-scratch reimplementations). Applied to the RAW
payload bits BEFORE FEC-encoding on transmit, and checked AFTER FEC-
decoding on receive -- the same order as liquid-dsp's own
packetizer_encode/decode. `rx_process()` resolves the actual crc0 scheme
from the decoded header (same reasoning as mod_scheme/fec0) and returns
a `crc_valid` bool array in its result dict (None when crc="none");
mismatches are never raised as exceptions, matching liquid-dsp's
crc_validate_message -- the caller decides what to do with a bad
result. This is the only way to catch a silently-wrong Viterbi decode:
conv_v27 has no other failure signal (it always returns *a* path,
right or wrong), unlike RS which can at least detect its own overflow.

Framing (header codec + CRC/FEC composition): the bit-level header
encode/decode (`HeaderCodec`) and the CRC + (up to two-stage) FEC
composition (`Packetizer`) both live in `spectracuda.framing` now, not
inline in this class -- see docs/todo.md #1.1 for the original gap
("you can't reuse 'decode a framed packet' logic outside Ofdm itself")
and #1.2 for the fec1/two-stage-FEC wiring. This object owns one
`self.header_codec`/`self.packetizer` for its OWN tx-side choice of
crc/fec/fec1, and builds a throwaway `Packetizer` per `rx_process()`
call from whatever the DECODED header says (same "resolve from the
wire, not from self" principle as mod_scheme/fec0/fec1/crc above).
`_encode_header_bits`/`_decode_header_bits` remain as thin delegating
wrapper methods for backward compatibility.

`rx_process()`'s "frame not found" handling (`sync_threshold=`,
`DEFAULT_SYNC_THRESHOLD`): every `sync=` strategy returns a best-
candidate `start_index`/`metric` even for pure noise with no preamble in
it at all (it's a "best window found" search, not a detector with a
built-in null hypothesis) -- previously this meant garbage sync results
marched all the way through CFO correction and header demod, either
getting caught by an unrelated sanity check (a "decoded mod_scheme isn't
known" ValueError) or, worse, not caught at all. `rx_process()` now
gates on item 0's sync metric against `self.sync_threshold` (default
`DEFAULT_SYNC_THRESHOLD=0.3`, empirically calibrated -- see its own
docstring) and returns a clean, fully-defined `frame_found=False` result
(every other field `None`) before attempting anything else. See
`rx_process()`'s own docstring for the complete, stable result-dict
schema (including `evm`/`rssi_db`, liquid-dsp's
`ofdmflexframesync_get_framedatastats()`-equivalent readouts that were
simply missing before -- see `spectracuda.framing.stats`).

Known gaps, called out rather than hidden:
  - Per-payload-symbol pilots (n_pilot) are carried in every payload
    grid but not yet used for anything (future phase/drift *tracking*
    across a longer burst); the channel estimate itself comes entirely
    from the training symbol(s).
  - "Frame not found" is now well-defined; "header found but garbage"
    (e.g. a corrupted-but-plausible-looking header) still surfaces as a
    raised exception (ValueError/NotImplementedError) rather than a
    `header_valid=False`-style result -- only the sync stage has a
    defined null outcome today.

iq_dtype ("float16" | "float32", default "float32"): simulates a
finite-resolution ADC/DAC front end by quantizing IQ samples to the
requested resolution at the tx-output and rx-input boundaries
(generate_frame()'s return value, rx_process()'s input). This is
deliberately NOT "run the DSP math in float16" -- no FFT library this
project can rely on (not numpy's pocketfft, not CuPy's standard FFT API)
actually transforms half-precision complex data, and numpy/cupy have no
native complex32 dtype at all (only complex64 = 2xfloat32, complex128 =
2xfloat64). Genuine float16 compute would need a hand-built complex16
wrapper type with its own add/multiply/conj/abs and an FFT that upcasts
internally -- a much larger, invasive change touching nearly every
block's internals, and still wouldn't give a real "16-bit FFT". Boundary
quantization is what's actually achievable and answers the real question
this is usually asked for: how much does N-bit ADC/DAC resolution hurt
BER, with all DSP compute still at complex64 (the precision this whole
project is built around -- see docs/architecture.md on why float64/
double precision is avoided in the compute path: Jetson-class GPUs have
drastically lower FP64 throughput than FP32).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from ..backend import BackendName, default_backend
from ..block import Block
from ..framing import HeaderCodec, Packetizer, compute_evm, compute_rssi_db
from ..framing.header import CRC_SCHEME_CODES as _CRC_SCHEME_CODES
from ..framing.header import FEC_SCHEME_CODES as _FEC_SCHEME_CODES
from ..modem import Modem
from ..ofdm import OfdmDemodulator, OfdmModulator, ResourceGrid
from ..registry import resolve

# Importing these registers "schmidl_cox" (and any future strategies)
# into the sync/cfo/equalizer registries as a side effect.
from ..sync import SchmidlCoxSync  # noqa: F401
from ..cfo import SchmidlCoxCFO  # noqa: F401
from ..equalizer import MMSEEqualizer, ZFEqualizer  # noqa: F401

# Header field-code tables (mod_scheme/crc0/fec0 <-> wire code) now live
# in framing/header.py -- they're header-wire-format concerns, not OFDM
# ones (see docs/todo.md #1.1). Only the two tables this module's own
# constructor-time validation needs are imported directly above;
# HeaderCodec owns encoding/decoding the bits themselves.


class Ofdm(Block):
    #: Full liquid-dsp-matching header size in bits (14 bytes) -- see
    #: class docstring for the field layout.
    #: Single source of truth is HeaderCodec now (spectracuda/framing/
    #: header.py) -- these class attrs are kept as aliases so existing
    #: call sites (Ofdm.HEADER_LEN_BITS/.PROTOCOL_VERSION) keep working.
    HEADER_LEN_BITS = HeaderCodec.HEADER_LEN_BITS
    PROTOCOL_VERSION = HeaderCodec.PROTOCOL_VERSION
    #: Default sync-detection threshold (see rx_process()'s "frame not
    #: found" handling) -- empirically calibrated, not a rigorous
    #: detection-theoretic bound: pure-noise buffers (no preamble at
    #: all) scored metric<=0.21 for SchmidlCoxSync and <=0.13 for
    #: ZadoffChuSync across fft_size in {64, 256} (30 trials each),
    #: while genuine embedded signals scored >=0.74/>=0.89 respectively
    #: even at a modest 10 dB SNR. 0.3 sits comfortably in the gap for
    #: both strategies at fft sizes in that tested range; tune via
    #: sync_threshold= for other configurations rather than assuming
    #: this generalizes to every fft_size/strategy combination untested.
    DEFAULT_SYNC_THRESHOLD = 0.3
    #: Hard cap on payload symbols per frame -- not a wire-format limit
    #: (the header's 16-bit payload_len_bits field could represent much
    #: longer frames), a physical one: beyond roughly this many OFDM
    #: symbols, the RF channel has likely changed enough (coherence time)
    #: that the single channel estimate taken from the training symbol(s)
    #: at the start of the frame no longer applies, and/or sync has
    #: drifted -- there's no per-symbol tracking to compensate (see class
    #: docstring, "per-payload-symbol pilots ... not yet used"). Enforced
    #: on both generate_frame() (refuse to build an over-long frame) and
    #: rx_process() (refuse to trust an over-long *decoded* count, which
    #: doubles as a defensive check against a corrupted header -- this is
    #: exactly the failure mode found during development: a bad header
    #: decode returning a huge garbage symbol count that then crashed
    #: deep in slot extraction instead of failing clearly up front).
    MAX_PAYLOAD_SYMBOLS = 128

    #: rx_streaming()'s SEEKING-state search window cap, in multiples of
    #: fft_size -- bounds how much accumulated history gets re-run through
    #: self.sync.process()'s matched-filter correlation on each call while
    #: still searching for a preamble. Without a cap, a long silent/noisy
    #: stream with no frame in it would make the search buffer (and the
    #: correlation cost) grow without bound. Only applies while SEEKING --
    #: never trims once a frame is actually being decoded (WAITING_HEADER/
    #: WAITING_PAYLOAD), where every accumulated sample is needed.
    STREAM_SEARCH_WINDOW_SYMBOLS = 8

    def __init__(
        self,
        fft_size: int,
        n_pilot: int,
        n_data: int,
        cp_len: int,
        *,
        dc_null: bool = True,
        modem: str = "qpsk",
        fec: str = "none",
        fec1: str = "none",
        crc: str = "none",
        interleaver: str = "none",
        interleaver_kwargs: Optional[Dict[str, Any]] = None,
        sync: Any = "schmidl_cox",
        cfo: Any = "schmidl_cox",
        channel_estimator: Any = "ls",
        equalizer: Any = "mmse",
        n_training_symbols: int = 1,
        pilot_values: Optional[Any] = None,
        preamble_seed: int = 123,
        training_seed: int = 999,
        backend: Optional[BackendName] = None,
        iq_dtype: str = "float32",
        sync_threshold: Optional[float] = None,
        strict_fec_check: bool = False,
    ) -> None:
        if fec != "none" and fec not in _FEC_SCHEME_CODES:
            raise ValueError(
                f"fec={fec!r}; expected one of {sorted(_FEC_SCHEME_CODES)} "
                f"(LDPC/Polar aren't implemented -- see class docstring, "
                f"matches liquid-dsp not having them either)"
            )
        if fec1 != "none" and fec1 not in _FEC_SCHEME_CODES:
            raise ValueError(
                f"fec1={fec1!r}; expected one of {sorted(_FEC_SCHEME_CODES)} "
                f"(fec1 is the OUTER FEC stage -- see docs/todo.md #1.2 and "
                f"spectracuda.framing.Packetizer's module docstring)"
            )
        if crc not in _CRC_SCHEME_CODES:
            raise ValueError(f"crc={crc!r}; expected one of {sorted(_CRC_SCHEME_CODES)}")
        if n_training_symbols < 1:
            raise ValueError("n_training_symbols must be >= 1")
        if iq_dtype not in ("float16", "float32"):
            raise ValueError(
                f"iq_dtype={iq_dtype!r}; expected 'float16' or 'float32' "
                f"(see class docstring for why float64 isn't offered here "
                f"and why float16 quantizes at the boundaries rather than "
                f"running the DSP math itself at half precision)"
            )
        self.iq_dtype = iq_dtype
        self.sync_threshold = self.DEFAULT_SYNC_THRESHOLD if sync_threshold is None else sync_threshold

        resolved_backend = backend or default_backend()
        super().__init__(backend=resolved_backend)
        xp = self.xp

        self.fft_size = fft_size
        self.cp_len = cp_len
        self.slot_len = cp_len + fft_size
        self.fec = fec
        self.fec1 = fec1
        self.crc = crc
        # Off by default -- preserves this class's existing, deliberately-
        # tested "receiver is a separate device that never saw the
        # transmitter's Ofdm(...) call, decode fec0/fec1 from the header
        # alone" behavior (see tests/test_ofdm_class.py's
        # test_rx_process_resolves_*_fec0_from_header_not_self_fec_codec
        # and the LDPC/two-stage end-to-end tests). Opt into True for a
        # receiver that should instead reject any decoded fec0/fec1 not
        # equal to this object's own self.fec/self.fec1 -- see
        # _decode_header_from_sync()'s own comment for why: it stops a
        # false sync detection (SEEKING triggering on noise, no real
        # frame at all) from constructing a fresh, potentially-expensive
        # codec (LDPC's GF(2) matrix inversion above all -- measured as a
        # real multi-hundred-ms-to-multi-second stall on a Raspberry Pi
        # 5, see debug/pluto_rx_standalone_test.py) for a scheme this
        # receiver was never going to legitimately see. True is the right
        # choice for a receiver whose peer is known in advance to always
        # use this exact fec/fec1 (e.g. this project's drone link, whose
        # own adaptive-MCS controller -- examples/drone_tui/
        # adaptive_mcs.py -- never varies fec/fec1 at all); leave False
        # for a receiver that must stay generic across arbitrarily-
        # configured senders.
        self.strict_fec_check = strict_fec_check
        # interleaver choice is a deliberate EXCEPTION to the "resolve
        # from the wire" rule below -- it's never signaled in the
        # header (liquid-dsp doesn't signal its own interleaver's
        # parameters over the air either), so rx_process() rebuilds its
        # throwaway Packetizer using THIS object's own interleaver/
        # interleaver_kwargs, not anything decoded from the frame -- see
        # framing/packetizer.py's module docstring for the full
        # reasoning. Both ends of a link must already agree on this
        # out-of-band, the same way preamble_seed/training_seed do.
        self.interleaver = interleaver
        self.interleaver_kwargs = interleaver_kwargs
        # This object's OWN choice of crc/fec0/fec1 when *it* transmits --
        # the receive path resolves the actual scheme from the decoded
        # header instead (same reasoning as mod_scheme; see class
        # docstring), via a freshly-built Packetizer(header_fields["crc"],
        # header_fields["fec0"], header_fields["fec1"], ...) rather than
        # assuming this one matches (see rx_process()). Packetizer
        # (spectracuda/framing/packetizer.py) is the extracted,
        # independently-reusable CRC+FEC composition logic that used to
        # live inline here -- see docs/todo.md #1.1/#1.2. fec_codec/
        # crc_codec are kept as aliases into it so existing call sites
        # (ofdm.fec_codec.k_bits, etc.) still work unchanged.
        self.packetizer = Packetizer(
            crc=crc, fec=fec, fec1=fec1, interleaver=interleaver,
            interleaver_kwargs=interleaver_kwargs, backend=resolved_backend,
        )
        self.fec_codec = self.packetizer.fec_codec
        self.crc_codec = self.packetizer.crc_codec
        # Single-entry memo for the RX-side "throwaway" payload_modem/
        # payload_packetizer built in _decode_header_from_sync() from the
        # DECODED header's mod_scheme/crc/fec0/fec1 -- see that method's
        # own comment for why those must be resolved from the wire, not
        # assumed from self.packetizer. In practice a link doesn't
        # renegotiate scheme every frame, so rebuilding those objects
        # (fresh native Viterbi/RS trellis+GF-table construction included)
        # on every single received frame was pure waste on the hot path --
        # measured ~0.15ms/frame. Keyed on the resolved (mod_scheme, crc,
        # fec0, fec1) tuple (interleaver/interleaver_kwargs are always
        # THIS object's own fixed values here, never resolved from the
        # header -- see below -- so they don't need to be part of the key);
        # a scheme change on the wire still rebuilds correctly, just no
        # longer needlessly on the common no-change path.
        self._rx_payload_codec_cache = None  # (key_tuple, payload_modem, payload_packetizer) or None
        self.n_training_symbols = n_training_symbols
        self.batch_shape_doc = (
            "generate_frame(bits): (n_batch, k * n_data * bits_per_symbol) "
            "bits -> (n_batch, n_samples) complex iq. "
            "rx_process(iq): (n_batch, n_samples) complex iq -> dict with a "
            "stable key set -- see rx_process()'s own docstring for the full "
            "schema (frame_found, start_index, sync_metric, rssi_db, "
            "cfo_estimate, channel_estimate, header, n_payload_symbols, "
            "bits, crc_valid, evm)."
        )

        self.grid = ResourceGrid(fft_size=fft_size, n_data=n_data, n_pilot=n_pilot, dc_null=dc_null)
        self.modem = Modem(modem, backend=resolved_backend)
        # Always BPSK, independent of `modem=` -- see class docstring
        # (matches liquid-dsp's OFDMFLEXFRAME_H_MOD = LIQUID_MODEM_BPSK).
        self.header_modem = Modem("bpsk", backend=resolved_backend)
        #: Payload bit capacity of one full OFDM symbol at this object's
        #: own `modem=` scheme (used when *this* object transmits).
        self.bits_per_ofdm_symbol = self.grid.n_data * self.modem.bits_per_symbol
        # Fixed (not secret, not needed by the receiver -- it's simply
        # never demodulated past the last real bit, see rx_process())
        # filler for automatic partial-last-payload-symbol padding (see
        # docs/todo.md #1.10 and generate_frame()) -- same "fixed random,
        # not constant" rationale as the header's own filler (see class
        # docstring's PAPR-bug paragraph): the padding tail of a payload
        # symbol shouldn't be predictable/constant content either, even
        # though it's only ever a fraction of one symbol. Sized to the
        # max padding that could ever be needed (one full symbol's worth
        # minus 1 bit).
        self._payload_filler_bits = np.random.default_rng(7777).integers(
            0, 2, size=self.bits_per_ofdm_symbol
        ).astype("uint8")

        # Header dedicated symbol(s), matching liquid-dsp's
        # num_symbols_header = ceil(header_sym_len / M_data) exactly.
        self.num_symbols_header = math.ceil(self.HEADER_LEN_BITS / self.grid.n_data)
        total_header_slots = self.num_symbols_header * self.grid.n_data

        # Spread the 112 real header bits evenly across the full flat
        # capacity of all header symbol(s) combined (frequency
        # diversity), rather than packing them densely into the first
        # slots -- see class docstring for the PAPR bug this (plus
        # scrambling) fixes.
        self._header_positions_flat = np.unique(
            np.linspace(0, total_header_slots - 1, self.HEADER_LEN_BITS).round().astype(int)
        )
        if len(self._header_positions_flat) != self.HEADER_LEN_BITS:
            raise ValueError(
                f"n_data={n_data} produced only {len(self._header_positions_flat)} "
                f"distinct spread positions for the header's {self.HEADER_LEN_BITS} "
                f"bits across {self.num_symbols_header} symbol(s) (rounding "
                f"collision) -- use a larger n_data"
            )
        # Precomputed once here (both operands are fixed for this object's
        # lifetime) instead of recomputing via setdiff1d/arange on every
        # single _build_header_symbols() call -- that was a real per-TX-frame
        # cost (measured ~0.27ms/frame) for a result that never changes.
        self._header_filler_positions = np.setdiff1d(
            np.arange(total_header_slots), self._header_positions_flat
        )
        self.header_codec = HeaderCodec(scramble_seed=42)
        # Alias for existing call sites (tests included) that reach into
        # this directly -- the real mask now lives on HeaderCodec.
        self._header_scramble_mask = self.header_codec._scramble_mask
        # Fixed (not secret, not needed by the receiver -- it's discarded
        # on decode) filler for the header's leftover capacity, so it
        # isn't constant/predictable content either.
        n_filler = total_header_slots - self.HEADER_LEN_BITS
        self._header_filler_bits = np.random.default_rng(2024).integers(
            0, 2, size=n_filler
        ).astype("uint8")

        self.mod = OfdmModulator(fft_size, cp_len, backend=resolved_backend)
        self.demod = OfdmDemodulator(fft_size, cp_len, backend=resolved_backend)

        self.sync = resolve("sync", sync, fft_size=fft_size, backend=resolved_backend)
        self.equalizer = resolve("equalizer", equalizer, backend=resolved_backend)

        if not hasattr(self.sync, "generate_preamble"):
            raise TypeError(
                f"sync={sync!r} has no generate_preamble(); Ofdm currently "
                f"requires a sync strategy that can generate its own preamble"
            )
        self.preamble_seed = preamble_seed
        self._preamble_time = self.sync.generate_preamble(seed=preamble_seed)

        self.pilot_values = (
            xp.ones((self.grid.n_pilot,), dtype="complex64")
            if pilot_values is None
            else xp.asarray(pilot_values, dtype="complex64")
        )

        # cfo resolved here (not alongside sync/equalizer above) because
        # PilotBasedCFO needs pilot_indices/tx_pilots/n_repeats, which
        # aren't available until self.pilot_values exists -- SchmidlCoxCFO
        # only needs fft_size and ignores the rest (see its constructor's
        # **kwargs sink), so this one shared default_kwargs dict still
        # works for either strategy, same registry pattern as
        # channel_estimator below.
        self.cfo = resolve(
            "cfo",
            cfo,
            fft_size=fft_size,
            cp_len=cp_len,
            pilot_indices=self.grid.pilot_indices,
            tx_pilots=self.pilot_values,
            n_repeats=n_training_symbols,
            backend=resolved_backend,
        )

        # Deterministic, known training-symbol content -- shared
        # automatically between tx and rx because both live on this one
        # object. The SAME training symbol is repeated n_training_symbols
        # times (like 802.11's Long Training Field, sent twice) so the
        # receiver can average across repetitions to reduce noise; see
        # rx_process().
        self.training_seed = training_seed
        rng = np.random.default_rng(training_seed)
        alphabet = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype="complex64") / np.sqrt(2)
        train_data_values = xp.asarray(alphabet[rng.integers(0, 4, size=self.grid.n_data)])
        self._train_grid_freq = self.grid.scatter(
            xp, self.pilot_values[None, :], train_data_values[None, :]
        )[0]
        self._train_known_indices = xp.asarray(
            np.sort(np.concatenate([self.grid.pilot_indices, self.grid.data_indices]))
        )
        train_known_values = self._train_grid_freq[self._train_known_indices]

        from ..channel import LSChannelEstimator, MMSEChannelEstimator  # noqa: F401 (registers "ls"/"mmse")

        self.channel_estimator = resolve(
            "channel_estimator",
            channel_estimator,
            pilot_indices=self._train_known_indices,
            fft_size=fft_size,
            tx_pilots=train_known_values,
            cp_len=cp_len,
            backend=resolved_backend,
        )

    def reconfigure_tx_scheme(
        self, modem: Optional[str] = None, fec: Optional[str] = None, fec1: Optional[str] = None
    ) -> int:
        """Change this object's OWN transmit-side modem/fec/fec1 choice
        (what `generate_frame()` uses next) WITHOUT rebuilding the rest
        of this Ofdm -- for adaptive MCS. Each argument left as None
        keeps that piece unchanged; at least one should be given or this
        is a no-op.

        Deliberately NOT "throw this Ofdm away, build a new one with
        different kwargs" -- two real reasons, not just avoided
        boilerplate:

        1. This same object also owns the RX-side streaming state
           (`self._stream_buffer`/`self._stream_header`, see
           reset_stream()/rx_streaming()) for the OTHER direction of a
           full-duplex Mac -- discarding and recreating the object would
           lose whatever partial frame is sitting in that buffer, purely
           because THIS side's own outgoing scheme changed (the peer's
           transmit scheme is completely independent). Every field
           touched below is scheme-derived only; grid/header_codec/sync/
           mod/demod/equalizer/preamble/_stream_buffer/
           _rx_payload_codec_cache are untouched.
        2. Rebuilding `Packetizer`/`FEC` from scratch reconstructs native
           Viterbi trellis / RS GF-table structs -- real, non-free setup
           this class already goes out of its way to avoid paying
           needlessly (see `_rx_payload_codec_cache`'s own docstring
           above, "~0.15ms/frame" for exactly this construction cost).
           Changing only what actually changed avoids paying it on every
           MCS transition for parameters that didn't move.

        A modem-only change needs no corresponding RX-side action:
        rx_process()/rx_streaming() resolve mod_scheme from the DECODED
        header, never from self, so an unmodified peer keeps decoding
        fine, regardless of strict_fec_check (see __init__). A fec/fec1
        change is DIFFERENT for a receiver constructed with
        strict_fec_check=True: _decode_header_from_sync() rejects
        (raises ValueError) any decoded fec0/fec1 that doesn't equal
        that receiver's own self.fec/self.fec1 -- see __init__'s own
        comment for why (constructing a codec, LDPC's GF(2) matrix
        inversion above all, for whatever a false sync detection
        decodes out of pure noise is a real, measured multi-hundred-ms-
        to-multi-second stall on real hardware). A strict_fec_check=True
        receiver whose peer DOES negotiate fec/fec1 changes over the air
        (e.g. via LINK_QUALITY-driven scheme selection) must have
        reconfigure_tx_scheme() called on BOTH ends' Ofdm -- the
        sender's to change what it transmits, the receiver's to accept
        what it now expects to decode -- not just the sender's. This
        project's own adaptive-MCS controller, examples/drone_tui/
        adaptive_mcs.py, sidesteps the whole question by only ever
        varying modem, never fec/fec1.
        A strict_fec_check=False receiver (the default) is unaffected by
        any of this -- it keeps resolving fec0/fec1 from the header
        alone, same as always.

        Returns the (possibly unchanged) `self.bits_per_ofdm_symbol` --
        callers with their own segmentation math derived from it (e.g.
        `Mac.max_segment_bits`, via `mac/capacity.py`) need to know
        whether/how it moved."""
        if fec is not None and fec != "none" and fec not in _FEC_SCHEME_CODES:
            raise ValueError(f"fec={fec!r}; expected one of {sorted(_FEC_SCHEME_CODES)}")
        if fec1 is not None and fec1 != "none" and fec1 not in _FEC_SCHEME_CODES:
            raise ValueError(f"fec1={fec1!r}; expected one of {sorted(_FEC_SCHEME_CODES)}")

        if modem is not None and modem != self.modem.scheme:
            self.modem = Modem(modem, backend=self.backend)
            self.bits_per_ofdm_symbol = self.grid.n_data * self.modem.bits_per_symbol
            # Same seed as __init__ (7777) -- deterministic/reproducible
            # content, not security-sensitive (see __init__'s own
            # comment), just resized to the new bits_per_ofdm_symbol.
            self._payload_filler_bits = np.random.default_rng(7777).integers(
                0, 2, size=self.bits_per_ofdm_symbol
            ).astype("uint8")

        new_fec = self.fec if fec is None else fec
        new_fec1 = self.fec1 if fec1 is None else fec1
        if new_fec != self.fec or new_fec1 != self.fec1:
            self.packetizer = Packetizer(
                crc=self.crc, fec=new_fec, fec1=new_fec1, interleaver=self.interleaver,
                interleaver_kwargs=self.interleaver_kwargs, backend=self.backend,
            )
            self.fec = new_fec
            self.fec1 = new_fec1
            self.fec_codec = self.packetizer.fec_codec
            self.crc_codec = self.packetizer.crc_codec

        return self.bits_per_ofdm_symbol

    # -- internal helpers ---------------------------------------------------

    def _to_host(self, arr: Any) -> np.ndarray:
        """Plain-numpy view of an array, regardless of backend -- header
        byte/bit packing is tiny metadata work, not bulk DSP, so it
        always runs in plain numpy rather than threading xp through it."""
        if self.backend == "cupy":
            import cupy

            return cupy.asnumpy(arr)
        return np.asarray(arr)

    def _bits_to_bytes(self, bits: Any) -> np.ndarray:
        """(n_batch, n_bits) -> (n_batch, n_bits//8) uint8, MSB-first
        (np.packbits' own default bitorder). Host-side, tiny metadata/
        framing work, same rationale as _to_host -- n_bits must already
        be a multiple of 8 (checked by the caller with a clear error;
        this is CRC's byte-oriented boundary, not a general Ofdm one)."""
        host_bits = self._to_host(bits).astype("uint8")
        return np.packbits(host_bits, axis=-1)

    def _bytes_to_bits(self, byte_arr: np.ndarray) -> Any:
        """Inverse of _bits_to_bytes, returned on self.xp (the caller's
        compute backend) rather than staying host-numpy."""
        bits = np.unpackbits(byte_arr, axis=-1)
        return self.xp.asarray(bits)

    def _quantize(self, iq: Any) -> Any:
        """Simulate a finite-resolution ADC/DAC: round real/imag through
        `self.iq_dtype` and back to float32, then re-pack as complex64
        (the DSP compute dtype throughout this project -- see class
        docstring for why this is boundary quantization, not float16
        compute). A no-op when iq_dtype="float32" (the default)."""
        if self.iq_dtype == "float32":
            return iq
        xp = self.xp
        real_q = xp.real(iq).astype("float16").astype("float32")
        imag_q = xp.imag(iq).astype("float16").astype("float32")
        return (real_q + 1j * imag_q).astype("complex64")

    def _extract_slot(self, rx: Any, item_starts: Any, slot_len: int) -> Any:
        """Gather a (n_batch, slot_len) block starting at a different
        offset per batch item. A Python loop is unavoidable here (offsets
        genuinely differ per item, from `sync`'s per-item detection) --
        the same pattern already used in LSChannelEstimator/SchmidlCoxCFO
        for the same reason. All actual DSP work stays one vectorized
        call across the gathered batch."""
        xp = self.xp
        n_batch = rx.shape[0]
        out = xp.empty((n_batch, slot_len), dtype=rx.dtype)
        for b in range(n_batch):
            s = int(item_starts[b])
            out[b] = rx[b, s : s + slot_len]
        return out

    def _encode_header_bits(
        self,
        payload_len_bits: int,
        mod_scheme: str,
        fec0: str,
        user_data: Optional[bytes],
        crc0: str = "none",
        fec1: str = "none",
    ) -> np.ndarray:
        """Thin wrapper delegating to self.header_codec.encode_bits() --
        the real logic (and the field-code tables it uses) now lives in
        spectracuda/framing/header.py (see docs/todo.md #1.1/#1.2). Kept
        as a method here (rather than removed) so existing call sites
        keep working unchanged."""
        return self.header_codec.encode_bits(payload_len_bits, mod_scheme, fec0, user_data, crc0, fec1)

    def _decode_header_bits(self, bits: np.ndarray) -> Dict[str, Any]:
        """Thin wrapper delegating to self.header_codec.decode_bits() --
        see _encode_header_bits's docstring."""
        return self.header_codec.decode_bits(bits)

    def _build_header_symbols(
        self,
        payload_len_bits: int,
        mod_scheme: str,
        fec0: str,
        user_data: Optional[bytes],
        n_batch: int,
        crc0: str = "none",
        fec1: str = "none",
    ) -> Any:
        """(n_batch, num_symbols_header * slot_len) time-domain samples
        for the dedicated header symbol(s). Same content for every batch
        item (one frame call = one frame type/length)."""
        xp = self.xp
        content_bits = self._encode_header_bits(payload_len_bits, mod_scheme, fec0, user_data, crc0, fec1)

        total_slots = self.num_symbols_header * self.grid.n_data
        flat_bits = np.empty(total_slots, dtype="uint8")
        flat_bits[self._header_positions_flat] = content_bits
        flat_bits[self._header_filler_positions] = self._header_filler_bits

        symbol_chunks = flat_bits.reshape(self.num_symbols_header, self.grid.n_data)
        pilots_batch = xp.tile(self.pilot_values, (n_batch, 1))

        time_chunks = []
        for i in range(self.num_symbols_header):
            syms = self.header_modem.modulate(xp.asarray(symbol_chunks[i])[None, :])[0]  # (n_data,)
            syms_batch = xp.tile(syms, (n_batch, 1))
            freq = self.grid.scatter(xp, pilots_batch, syms_batch)
            time_chunks.append(self.mod.process(freq))
        return xp.concatenate(time_chunks, axis=-1)

    def _decode_header_symbols(self, header_bits_chunks: Any) -> Dict[str, Any]:
        """header_bits_chunks: (n_batch, num_symbols_header * n_data) BPSK-
        demodulated bits (already concatenated across header symbols).
        Uses item 0 as the shared header content (see class docstring:
        one frame call = one frame type/length across the whole batch)."""
        flat_bits = self._to_host(header_bits_chunks)[0]
        content_bits = flat_bits[self._header_positions_flat]
        return self._decode_header_bits(content_bits)

    # -- public API -----------------------------------------------------------

    def generate_frame(self, payload_bits: Any, user_data: Optional[bytes] = None) -> Any:
        xp = self.xp
        payload_bits = xp.asarray(payload_bits)
        if payload_bits.ndim == 1:
            payload_bits = payload_bits[None, :]
        n_batch = payload_bits.shape[0]
        raw_bit_count = payload_bits.shape[-1]

        # CRC-append then FEC-encode, delegated to self.packetizer
        # (spectracuda/framing/packetizer.py -- extracted out of this
        # method, see docs/todo.md #1.1). Matches liquid-dsp's own
        # packetizer_encode order exactly. The header's payload_len_bits
        # still records the RAW count (excluding the CRC key), matching
        # liquid-dsp's own payload_dec_len convention -- the receiver
        # derives crc_length itself from the decoded crc0 scheme, the
        # same way it derives the encoded bit count from decoded fec0
        # (see rx_process()).
        modulated_bits = self.packetizer.encode(payload_bits)
        encoded_bit_count = modulated_bits.shape[-1]

        # Automatic partial-last-symbol padding (docs/todo.md #1.10): the
        # receiver needs no new wire field for this -- it already derives
        # encoded_bit_count itself from the header's (unaffected,
        # RAW/pre-CRC/pre-FEC) payload_len_bits plus the decoded crc/fec0/
        # fec1 schemes (see rx_process()), so it knows exactly how many
        # of the demodulated bits in the last symbol are real vs filler,
        # without spectracuda needing to say so on the wire.
        n_payload_symbols = math.ceil(encoded_bit_count / self.bits_per_ofdm_symbol)
        if n_payload_symbols > self.MAX_PAYLOAD_SYMBOLS:
            raise ValueError(
                f"payload needs {n_payload_symbols} OFDM symbols, exceeding "
                f"MAX_PAYLOAD_SYMBOLS={self.MAX_PAYLOAD_SYMBOLS} (a coherence-"
                f"time limit, not a wire-format one -- see class docstring). "
                f"Split into multiple frames."
            )
        padding_bits = n_payload_symbols * self.bits_per_ofdm_symbol - encoded_bit_count
        if padding_bits > 0:
            filler = xp.tile(xp.asarray(self._payload_filler_bits[:padding_bits]), (n_batch, 1))
            modulated_bits = xp.concatenate([modulated_bits, filler], axis=-1)

        pilots_batch = xp.tile(self.pilot_values, (n_batch, 1))
        preamble_batch = xp.tile(self._preamble_time, (n_batch, 1))

        train_time_one = self.mod.process(self._train_grid_freq[None, :])[0]
        train_batch = xp.tile(train_time_one, (n_batch, self.n_training_symbols))

        header_batch = self._build_header_symbols(
            raw_bit_count, self.modem.scheme, self.fec, user_data, n_batch, self.crc, self.fec1
        )

        # Batched across payload symbols in ONE call each, not a Python loop
        # per symbol: modem.modulate()/grid.scatter()/mod.process() all
        # already accept an arbitrary leading batch dimension (their own
        # batch-shape contracts, docs/architecture.md), so folding
        # (n_batch, n_payload_symbols) into one combined axis and calling
        # each once lets numpy's own vectorized C loop (ifft's axis=-1 in
        # particular) do the repetition instead of paying Python-level
        # call overhead ~n_payload_symbols times per frame -- a real,
        # measured bottleneck (examples/benchmark_x86_stages_v2.py profiling
        # found this loop as the single largest remaining tx cost once FEC
        # encode was fixed). Reshape order is batch-major/symbol-minor
        # throughout (grouped_bits' own axis order), so this produces the
        # exact same per-batch-item symbol sequence the old per-symbol loop
        # did, just without the loop.
        grouped_bits = modulated_bits.reshape(n_batch, n_payload_symbols, self.bits_per_ofdm_symbol)
        combined_bits = grouped_bits.reshape(n_batch * n_payload_symbols, self.bits_per_ofdm_symbol)
        pilots_combined = xp.tile(self.pilot_values, (n_batch * n_payload_symbols, 1))
        data_symbols_all = self.modem.modulate(combined_bits)
        freq_all = self.grid.scatter(xp, pilots_combined, data_symbols_all)
        payload_time_all = self.mod.process(freq_all)  # (n_batch*n_payload_symbols, fft_size+cp_len)
        symbol_len = self.fft_size + self.cp_len
        payload_time = payload_time_all.reshape(n_batch, n_payload_symbols * symbol_len)

        frame = xp.concatenate([preamble_batch, train_batch, header_batch, payload_time], axis=-1)
        return self._quantize(frame)  # simulate the DAC's resolution -- see class docstring

    def rx_process(self, rx_iq: Any, n_payload_symbols: Optional[int] = None, **kwargs: Any) -> Dict[str, Any]:
        """n_payload_symbols: normally left as None so it's derived from
        the header's decoded payload_len_bits (see class docstring); pass
        an explicit value to override (e.g. for debugging a suspected
        header decode error).

        Return value: a dict with a stable, fully-defined key set (see
        docs/todo.md #1.1 -- this used to be "whatever fields happened to
        be convenient to add during development"). Every key below is
        ALWAYS present; most are `None` when `frame_found` is False:

            frame_found      bool -- see sync_threshold=/DEFAULT_SYNC_THRESHOLD.
                             False means item 0's sync metric never cleared
                             the bar; nothing past sync ran at all.
            start_index      (n_batch,) int -- best-effort even if frame_found
                             is False (whatever sync.process() found).
            sync_metric      (n_batch,) float -- the raw sync detection score
                             (see sync.process()'s own docstring for its range).
            rssi_db          (n_batch,) float -- relative received power in dB,
                             computed unconditionally (see framing/stats.py).
            cfo_estimate     (n_batch,) float or None.
            channel_estimate array or None.
            header           dict or None (see HeaderCodec.decode_bits()).
            n_payload_symbols int or None.
            bits             array or None -- the decoded raw payload bits.
            crc_valid        (n_batch,) bool array, or None if the decoded
                             header's crc is "none" or frame_found is False.
            evm              (n_batch,) float or None -- normalized RMS EVM
                             of the payload's equalized symbols against the
                             receiver's own hard-decision (see
                             framing/stats.py's compute_evm)."""
        xp = self.xp
        rx_iq = xp.asarray(rx_iq)
        if rx_iq.ndim == 1:
            rx_iq = rx_iq[None, :]
        rx_iq = self._quantize(rx_iq)  # simulate the ADC's resolution -- see class docstring

        # RSSI is computed regardless of whether a frame is actually
        # found below -- it describes received energy, not frame content
        # (see framing/stats.py's module docstring for the "relative,
        # not calibrated dBm" caveat).
        rssi_db = compute_rssi_db(xp, rx_iq)

        sync_result = self.sync.process(rx_iq)
        start_index = sync_result["start_index"]
        sync_metric = sync_result["metric"]

        # "Frame not found" -- the real gap this used to have (see
        # docs/todo.md #1.1): sync.process() always returns SOME
        # start_index/metric, even for pure noise with no preamble in it
        # at all, because it's a "best candidate window" search, not a
        # detector with a built-in null hypothesis. Gated on item 0's
        # metric only, matching this class's existing "one call = one
        # frame" convention (_decode_header_symbols already only reads
        # item 0's content for the same reason) -- not a per-item
        # decision. Below threshold, every field a real frame's decode
        # would need is None rather than best-effort garbage; nothing
        # past this point (CFO correction, channel/header/payload decode)
        # is attempted at all.
        item0_metric = float(np.asarray(self._to_host(sync_metric))[0])
        if item0_metric < self.sync_threshold:
            return {
                "frame_found": False,
                "start_index": start_index,
                "sync_metric": sync_metric,
                "rssi_db": rssi_db,
                "cfo_estimate": None,
                "channel_estimate": None,
                "header": None,
                "n_payload_symbols": None,
                "bits": None,
                "crc_valid": None,
                "evm": None,
            }

        h = self._decode_header_from_sync(rx_iq, start_index, n_payload_symbols)
        p = self._decode_payload_from_header(
            h["rx_corrected"], h["pos"], h["h_hat_data"], h["payload_modem"],
            h["payload_packetizer"], h["encoded_bit_count"], h["n_payload_symbols"],
        )

        return {
            "frame_found": True,
            "start_index": start_index,
            "sync_metric": sync_metric,
            "rssi_db": rssi_db,
            "cfo_estimate": h["cfo_estimate"],
            "channel_estimate": h["h_hat_data"],
            "n_payload_symbols": h["n_payload_symbols"],
            "header": h["header_fields"],
            "bits": p["bits"],
            "crc_valid": p["crc_valid"],
            "evm": p["evm"],
        }

    def _decode_header_from_sync(self, rx_iq: Any, start_index: Any, n_payload_symbols: Optional[int] = None) -> Dict[str, Any]:
        """Everything from CFO correction through header decode and
        payload-length computation -- extracted out of rx_process() (not
        duplicated) so rx_streaming() (see below) can reuse this EXACT
        logic once it has accumulated enough samples for the header,
        before it has any payload samples at all. rx_process() itself is
        refactored to call this -- zero change to its own signature or
        observable behavior, verified by the full existing test suite
        passing unchanged.

        Returns a dict: cfo_estimate, rx_corrected, h_hat_data,
        header_fields, payload_modem, payload_packetizer,
        encoded_bit_count, n_payload_symbols (resolved: either the
        decoded value or the caller's override), pos (sample offset of
        the payload's own start, immediately after the header)."""
        xp = self.xp
        cfo_estimate = self.cfo.process(rx_iq, start_index=start_index)
        rx_corrected = self.cfo.correct(rx_iq, cfo_estimate)

        pos = start_index + self.fft_size  # preamble has no CP -- see class docstring

        h_hat_data_sum = None
        for _ in range(self.n_training_symbols):
            train_slot = self._extract_slot(rx_corrected, pos, self.slot_len)
            pos = pos + self.slot_len
            train_rx_grid = self.demod.process(train_slot)
            train_rx_known = train_rx_grid[:, self._train_known_indices]
            h_hat_full = self.channel_estimator.process(train_rx_known)
            h_hat_data = h_hat_full[:, self.grid.data_indices]
            h_hat_data_sum = h_hat_data if h_hat_data_sum is None else h_hat_data_sum + h_hat_data
        h_hat_data = h_hat_data_sum / self.n_training_symbols

        header_bits_chunks = []
        for _ in range(self.num_symbols_header):
            header_slot = self._extract_slot(rx_corrected, pos, self.slot_len)
            pos = pos + self.slot_len
            header_rx_grid = self.demod.process(header_slot)
            header_rx_data = self.grid.extract_data(xp, header_rx_grid)
            header_equalized = self.equalizer.process(header_rx_data, channel_est=h_hat_data)
            header_bits_chunks.append(self.header_modem.demodulate(header_equalized))
        header_fields = self._decode_header_symbols(xp.concatenate(header_bits_chunks, axis=-1))

        # strict_fec_check (see __init__'s own comment): reject any
        # decoded fec0/fec1 this receiver wasn't itself configured for,
        # BEFORE touching the codec cache/construction below -- cheap
        # (two attribute comparisons), and it's what stops a false sync
        # detection (SEEKING triggering on plain noise, no peer
        # transmitting at all) from decoding garbage header bits into a
        # random-but-VALID fec0 code -- FEC_SCHEME_CODES has 17 entries,
        # 12 of them LDPC variants, so most such garbage draws land on
        # one -- and paying for a full LDPCCode construction (GF(2)
        # matrix inversion + edge-index tables, up to ~1.6s measured on
        # a Pi 5 for the largest 1944-bit variant) to decode a frame
        # that was never really there. Real-hardware root cause:
        # debug/pluto_rx_standalone_test.py showed random multi-hundred-
        # ms-to-multi-second rx_streaming() stalls with no TX active at
        # all; profiled to LDPCCode.__init__ via _rx_payload_codec_cache
        # thrashing on every such false positive (single-entry cache,
        # keyed on the noise-derived fec0/fec1, essentially never hits).
        #
        # Off by default: the class's existing, deliberately-tested
        # "receiver never needs to have seen the transmitter's own
        # Ofdm(...) call" generality (tests/test_ofdm_class.py's
        # dynamic-fec0-resolution tests) is preserved unless a caller
        # opts in.
        if self.strict_fec_check and (
            header_fields["fec0"] != self.fec or header_fields["fec1"] != self.fec1
        ):
            raise ValueError(
                f"decoded header fec0={header_fields['fec0']!r}/fec1={header_fields['fec1']!r} "
                f"doesn't match this receiver's configured fec={self.fec!r}/fec1={self.fec1!r} "
                f"-- rejecting before constructing a codec for it (strict_fec_check=True; "
                f"likely a false sync detection decoding noise, not a real frame from a "
                f"differently-configured peer)"
            )

        # Dynamically resolve the payload's modem AND crc composition
        # from the DECODED header -- not from self.modem/self.packetizer.
        # A real receiver is a separate device that never saw this
        # object's constructor call (see class docstring). The
        # Packetizer built here is throwaway/rx-only, mirroring
        # self.packetizer's tx-side role but for whatever crc/fec0/fec1
        # the wire actually says (fec0/fec1 already validated to match
        # self.fec/self.fec1 above, so this is never a surprising
        # scheme in practice) -- EXCEPT interleaver, which is
        # deliberately NOT resolved from the header (it's never signaled
        # there -- see self.interleaver's own comment above and
        # framing/packetizer.py's module docstring): this receiver must
        # already be configured with the matching interleaver out-of-
        # band, so THIS object's own self.interleaver/self.interleaver_kwargs
        # are used here.
        codec_key = (header_fields["mod_scheme"], header_fields["crc"], header_fields["fec0"], header_fields["fec1"])
        cached = self._rx_payload_codec_cache
        if cached is not None and cached[0] == codec_key:
            payload_modem, payload_packetizer = cached[1], cached[2]
        else:
            payload_modem = Modem(header_fields["mod_scheme"], backend=self.backend)
            payload_packetizer = Packetizer(
                crc=header_fields["crc"],
                fec=header_fields["fec0"],
                fec1=header_fields["fec1"],
                interleaver=self.interleaver,
                interleaver_kwargs=self.interleaver_kwargs,
                backend=self.backend,
            )
            self._rx_payload_codec_cache = (codec_key, payload_modem, payload_packetizer)
        bits_per_symbol_payload = self.grid.n_data * payload_modem.bits_per_symbol

        raw_payload_len_bits = header_fields["payload_len_bits"]
        try:
            encoded_bit_count = payload_packetizer.encoded_length(raw_payload_len_bits)
        except ValueError as exc:
            raise ValueError(
                f"decoded payload_len_bits={raw_payload_len_bits} is not "
                f"compatible with decoded crc={header_fields['crc']!r}/"
                f"fec0={header_fields['fec0']!r}/fec1={header_fields['fec1']!r} "
                f"-- likely header corruption ({exc})"
            ) from exc
        # Ceiling, not exact division: generate_frame() automatically
        # pads a partial last payload symbol with filler bits (see its
        # own docstring/docs/todo.md #1.10) -- the receiver doesn't need
        # a new wire field to know how many of the demodulated bits in
        # that last symbol are real vs filler, because encoded_bit_count
        # itself (derived above from the header's unaffected raw
        # payload_len_bits) already says exactly where the real data
        # ends; anything demodulated past it is discarded below.
        decoded_n_payload_symbols = math.ceil(encoded_bit_count / bits_per_symbol_payload)
        if decoded_n_payload_symbols > self.MAX_PAYLOAD_SYMBOLS:
            # Defensive check, not just a coherence-time rule here: this is
            # exactly the failure mode found during development -- a bad
            # header decode returning a huge garbage symbol count that
            # then crashed deep inside slot extraction instead of failing
            # clearly right here.
            raise ValueError(
                f"decoded header claims {decoded_n_payload_symbols} payload "
                f"symbols, exceeding MAX_PAYLOAD_SYMBOLS={self.MAX_PAYLOAD_SYMBOLS} "
                f"-- likely header corruption, not a legitimately long frame"
            )
        if n_payload_symbols is None:
            n_payload_symbols = decoded_n_payload_symbols
        elif n_payload_symbols > self.MAX_PAYLOAD_SYMBOLS:
            raise ValueError(
                f"n_payload_symbols={n_payload_symbols} (explicit override) "
                f"exceeds MAX_PAYLOAD_SYMBOLS={self.MAX_PAYLOAD_SYMBOLS}"
            )

        return {
            "cfo_estimate": cfo_estimate,
            "rx_corrected": rx_corrected,
            "h_hat_data": h_hat_data,
            "header_fields": header_fields,
            "payload_modem": payload_modem,
            "payload_packetizer": payload_packetizer,
            "encoded_bit_count": encoded_bit_count,
            "n_payload_symbols": n_payload_symbols,
            "pos": pos,
        }

    def _decode_payload_from_header(
        self, rx_corrected: Any, pos: Any, h_hat_data: Any, payload_modem: Any,
        payload_packetizer: Any, encoded_bit_count: int, n_payload_symbols: int,
    ) -> Dict[str, Any]:
        """Payload extraction through FEC-decode/CRC-check/EVM -- the
        other half of rx_process(), extracted for the same reuse reason
        as _decode_header_from_sync() above. Returns bits, crc_valid, evm."""
        xp = self.xp
        n_batch = rx_corrected.shape[0]

        # Batched across payload symbols in ONE call each, not a Python loop
        # per symbol -- the tx-side mirror of this fix (generate_frame()'s
        # own module comment has the full rationale: modem/grid/demod/
        # equalizer all already accept an arbitrary leading batch dim, so
        # folding (n_batch, n_payload_symbols) into one combined axis lets
        # numpy's own vectorized C loop do the repetition instead of paying
        # Python-level call overhead ~n_payload_symbols times per frame).
        #
        # Slot gathering: pos[b] is a genuinely per-item sync offset, but
        # the SPACING between consecutive symbols (self.slot_len) is the
        # same for every item -- so every (item, symbol) slot start is
        # pos[b] + i*self.slot_len, computable by broadcasting (no data
        # dependency across symbols), then gathered in one fancy-indexing
        # call rather than n_payload_symbols separate _extract_slot() calls.
        symbol_starts = pos[:, None] + xp.arange(n_payload_symbols)[None, :] * self.slot_len  # (n_batch, n_payload_symbols)
        sample_idx = symbol_starts[:, :, None] + xp.arange(self.slot_len)[None, None, :]  # (n_batch, n_payload_symbols, slot_len)
        # Explicit bounds check, matching _extract_slot()'s OLD per-symbol
        # basic-slice behavior: a plain `rx[b, s:s+slot_len]` on a
        # corrupted/truncated buffer silently clips to a shorter array,
        # which then failed downstream as a ValueError (broadcasting that
        # short slice into a fixed-size row) -- caught by
        # mac/session.py's `except (ValueError, NotImplementedError)`.
        # Advanced (fancy) indexing has no such silent-clip behavior -- it
        # raises IndexError, which that except clause does NOT catch -- so
        # this checks explicitly and raises the SAME exception type the
        # rest of this pipeline already uses for "corrupted/implausible
        # input", rather than letting a new, uncaught exception type leak
        # out of what used to be a handled failure mode (real regression
        # caught by tests/test_mac_session.py::
        # test_am_recovers_from_channel_loss_that_defeats_um, not assumed).
        if sample_idx.size and int(sample_idx.max()) >= rx_corrected.shape[1]:
            raise ValueError(
                f"payload extraction needs samples up to index {int(sample_idx.max())} "
                f"but rx_corrected only has {rx_corrected.shape[1]} -- likely a "
                f"corrupted/truncated frame (insufficient samples for "
                f"{n_payload_symbols} payload symbols)"
            )
        batch_idx = xp.arange(n_batch)[:, None, None]
        all_slots = rx_corrected[batch_idx, sample_idx]  # (n_batch, n_payload_symbols, slot_len)

        combined_slots = all_slots.reshape(n_batch * n_payload_symbols, self.slot_len)
        combined_rx_grid = self.demod.process(combined_slots)
        combined_rx_data = self.grid.extract_data(xp, combined_rx_grid)
        h_hat_combined = xp.repeat(h_hat_data, n_payload_symbols, axis=0)  # batch-major/symbol-minor, matches combined_slots' own row order
        equalized_combined = self.equalizer.process(combined_rx_data, channel_est=h_hat_combined)
        demod_bits_combined = payload_modem.demodulate(equalized_combined)  # (n_batch*n_payload_symbols, bits_per_symbol_payload) -- UNTRUNCATED

        n_data = equalized_combined.shape[-1]
        bits_per_symbol_payload = demod_bits_combined.shape[-1]
        equalized_flat = equalized_combined.reshape(n_batch, n_payload_symbols * n_data)
        encoded_bits = demod_bits_combined.reshape(n_batch, n_payload_symbols * bits_per_symbol_payload)

        # Discard any automatic partial-last-symbol padding (see
        # generate_frame()'s docstring/docs/todo.md #1.10) -- the last
        # symbol may carry filler bits beyond encoded_bit_count, which
        # would otherwise be fed into FEC-decode as if they were real
        # codeword bits. (EVM below deliberately uses the UNTRUNCATED
        # demod_bits_combined/equalized_combined, matching the original
        # per-symbol-chunk behavior -- the last symbol's filler bits are
        # still real, meaningful EVM data, just not real FEC codeword bits.)
        encoded_bits = encoded_bits[:, :encoded_bit_count]

        # FEC-decode then CRC-strip+check, delegated to payload_packetizer
        # (may raise ValueError if a codeword has more errors than the
        # decoded fec0 scheme can correct -- see FEC.decode()/the
        # underlying scheme's docstring). crc_valid is None when
        # crc="none" (nothing to check), else a per-batch-item bool array
        # so the caller decides what to do (retry, drop, log) -- never
        # raised as an exception, matching liquid's crc_validate_message.
        decode_result = payload_packetizer.decode(encoded_bits)
        raw_bits = decode_result["bits"]
        crc_valid = decode_result["crc_valid"]

        # EVM: standard normalized RMS EVM against the receiver's OWN
        # hard-decision re-modulated symbols (no ground truth needed --
        # see framing/stats.py's module docstring).
        ideal_combined = payload_modem.modulate(demod_bits_combined)
        ideal_flat = ideal_combined.reshape(n_batch, n_payload_symbols * n_data)
        evm = compute_evm(xp, equalized_flat, ideal_flat)

        return {"bits": raw_bits, "crc_valid": crc_valid, "evm": evm}

    # -- streaming receiver ----------------------------------------------
    # rx_process() (above) assumes the caller already has one complete,
    # correctly-bounded frame buffer -- fine for simulation, not how real
    # IQ arrives (an arbitrary-chunked, arbitrarily-aligned continuous
    # stream, e.g. 64/128/256/1024 samples at a time from an SDR, with no
    # relationship to frame/symbol boundaries -- docs/todo.md #2.5).
    # rx_streaming() is the real streaming receiver; it does NOT replace
    # or change rx_process()'s behavior (confirmed by the full existing
    # test suite passing unchanged after the _decode_header_from_sync()/
    # _decode_payload_from_header() extraction above).
    #
    # Design checked against liquid-dsp's OWN real implementation before
    # writing this (reference/liquid-dsp/src/multichannel/src/
    # ofdmframesync.c, ofdmframesync_execute()) rather than inventing a
    # state machine from assumption: liquid-dsp processes one sample at a
    # time through states SEEKPLCP -> PLCPSHORT0/1 -> PLCPLONG ->
    # RXSYMBOLS, where every length past "sync found" is already known/
    # fixed from the object's own config, so it never has to guess how
    # many more samples it needs at any stage. This reuses that exact
    # structural insight (SEEKING -> WAITING_HEADER -> WAITING_PAYLOAD),
    # expressed through this project's EXISTING batch-vectorized
    # sync/cfo/channel_estimator/equalizer/demod primitives -- re-run on
    # accumulated slices once enough samples exist for the current stage
    # -- rather than a literal per-sample NCO/timer port. Deliberate,
    # stated simplification, not an oversight.
    #
    # Single-stream only (n_batch=1) for this first version -- multi-
    # stream batched streaming (N independent streams, each possibly in a
    # different state at any moment) is a real, separate extension, not
    # attempted here. Matches liquid-dsp's own precedent too: one
    # ofdmframesync instance handles one stream; you'd instantiate N for
    # N streams.

    def reset_stream(self) -> None:
        """(Re)initialize rx_streaming()'s state. Called automatically by
        rx_streaming() on first use; call it explicitly to abandon
        whatever partial frame is in flight and start over (e.g. after a
        long enough gap that resuming doesn't make sense)."""
        xp = self.xp
        self._stream_state = "SEEKING"
        self._stream_buffer = xp.zeros((1, 0), dtype="complex64")
        self._stream_frame_start = None
        self._stream_header: Optional[Dict[str, Any]] = None

    def rx_streaming(self, chunk: Any) -> Optional[Dict[str, Any]]:
        """Feed one arbitrary-sized chunk of IQ samples (any length, any
        alignment relative to frame/symbol boundaries -- see the class-
        level comment above). Returns a rx_process()-shaped result dict
        (same key set, `frame_found` always True when this method
        actually returns a dict -- there is no False case here the way
        rx_process() has one, since this method simply returns None
        instead of ever reporting a negative result) the instant a
        complete frame finishes decoding during this call, else None
        (still searching/accumulating -- call again with the next chunk).

        A failed header or payload decode (ValueError -- header
        corruption, a false-positive sync detection, or an uncorrectable
        FEC codeword) does NOT raise and does NOT kill the stream, unlike
        rx_process()'s own behavior -- a streaming receiver has to keep
        running indefinitely across many frames, so one bad frame is
        discarded (this call returns None) and the state machine resumes
        searching for the next one, exactly as a real receiver must."""
        if not hasattr(self, "_stream_state"):
            self.reset_stream()
        xp = self.xp
        chunk = xp.asarray(chunk, dtype="complex64")
        if chunk.ndim == 1:
            chunk = chunk[None, :]
        if chunk.shape[0] != 1:
            raise ValueError(
                f"rx_streaming() is single-stream only (n_batch=1) for this first "
                f"version -- got a chunk with {chunk.shape[0]} batch items"
            )
        self._stream_buffer = xp.concatenate([self._stream_buffer, chunk], axis=-1)

        if self._stream_state == "SEEKING":
            cap = self.STREAM_SEARCH_WINDOW_SYMBOLS * self.fft_size
            if self._stream_buffer.shape[-1] > cap:
                self._stream_buffer = self._stream_buffer[:, -cap:]

            if self._stream_buffer.shape[-1] < self.fft_size:
                return None  # not even one preamble-length's worth yet

            sync_result = self.sync.process(self._stream_buffer)
            metric = float(np.asarray(self._to_host(sync_result["metric"]))[0])
            if metric >= self.sync_threshold:
                self._stream_frame_start = int(np.asarray(self._to_host(sync_result["start_index"]))[0])
                self._stream_state = "WAITING_HEADER"
            else:
                return None

        if self._stream_state == "WAITING_HEADER":
            header_end = (
                self._stream_frame_start + self.fft_size
                + self.n_training_symbols * self.slot_len
                + self.num_symbols_header * self.slot_len
            )
            if self._stream_buffer.shape[-1] < header_end:
                return None  # keep accumulating

            start_index_arr = xp.asarray([self._stream_frame_start])
            try:
                self._stream_header = self._decode_header_from_sync(self._stream_buffer, start_index_arr)
            except (ValueError, NotImplementedError):
                # False positive or corrupted header -- discard past the
                # detected start (advance by 1 sample so the same false
                # peak isn't immediately re-detected) and resume searching.
                self._stream_buffer = self._stream_buffer[:, self._stream_frame_start + 1 :]
                self._stream_state = "SEEKING"
                self._stream_frame_start = None
                return None
            self._stream_state = "WAITING_PAYLOAD"

        if self._stream_state == "WAITING_PAYLOAD":
            h = self._stream_header
            # h["pos"] is an xp array (start_index was passed in as one,
            # see WAITING_HEADER above) -- pull it to a plain Python int
            # here, since it's used below for buffer-length comparisons
            # and slicing, neither of which accept an array index.
            pos_scalar = int(np.asarray(self._to_host(h["pos"]))[0])
            frame_end = pos_scalar + h["n_payload_symbols"] * self.slot_len
            if self._stream_buffer.shape[-1] < frame_end:
                return None  # keep accumulating

            # Real correctness point, not obvious at a glance: h["rx_corrected"]
            # was CFO-corrected over the buffer as it stood during the HEADER
            # stage -- it does NOT include the payload samples that have
            # arrived since (the buffer has grown). Re-apply CFO correction
            # (self.cfo.correct is just a phase multiply, cheap) to the FULL,
            # now-complete buffer using the SAME already-estimated
            # cfo_estimate, rather than reusing the stale, too-short array.
            rx_corrected_full = self.cfo.correct(self._stream_buffer, h["cfo_estimate"])

            try:
                p = self._decode_payload_from_header(
                    rx_corrected_full, h["pos"], h["h_hat_data"], h["payload_modem"],
                    h["payload_packetizer"], h["encoded_bit_count"], h["n_payload_symbols"],
                )
            except (ValueError, NotImplementedError):
                self._stream_buffer = self._stream_buffer[:, frame_end:]
                self._stream_state = "SEEKING"
                self._stream_frame_start = None
                self._stream_header = None
                return None

            result = {
                "frame_found": True,
                "start_index": xp.asarray([self._stream_frame_start]),
                "sync_metric": None,  # not retained across the multi-call accumulation
                "rssi_db": compute_rssi_db(xp, self._stream_buffer[:, : self.fft_size]),
                "cfo_estimate": h["cfo_estimate"],
                "channel_estimate": h["h_hat_data"],
                "n_payload_symbols": h["n_payload_symbols"],
                "header": h["header_fields"],
                "bits": p["bits"],
                "crc_valid": p["crc_valid"],
                "evm": p["evm"],
            }

            self._stream_buffer = self._stream_buffer[:, frame_end:]
            self._stream_state = "SEEKING"
            self._stream_frame_start = None
            self._stream_header = None
            return result

        return None

    def process(self, batch: Any, **kwargs: Any) -> Dict[str, Any]:
        """Alias for rx_process() -- required to satisfy Block's abstract
        process() contract. Call rx_process() directly; that's the name
        this class actually documents and expects to be used."""
        return self.rx_process(batch, **kwargs)
