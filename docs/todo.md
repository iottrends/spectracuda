# TODO: gaps to close

Source of truth for each claim below: [`docs/liquid-dsp-api-inventory.md`](liquid-dsp-api-inventory.md)
(what liquid-dsp actually exposes) and [`docs/architecture.md`](architecture.md)
(spectracuda's own phased build plan). This document turns those into a
concrete, checkable list. Items are ordered by priority within each
section, not by file layout.

A reminder from the architecture doc that governs how to read this list:
spectracuda is **not** trying to port liquid-dsp 1:1. `sync=`/`cfo=`/
`channel_estimator=`/`equalizer=` as swappable strategies have no
liquid-dsp precedent at all — liquid-dsp bakes one fixed algorithm for
each of those into `ofdmflexframesync`'s internals and only exposes
read-outs (`get_cfo()`, `get_rssi()`). So "parity" for those sections
means "finish the strategies spectracuda itself designed," not "match
liquid-dsp's menu" — there is no menu to match. `modem` and `fec` are the
opposite case: liquid-dsp *does* expose a clean scheme-enum there, so
parity for those two sections means literally closing the scheme-count
gap.

---

## Part 1 — OFDM pipeline gaps

Everything here sits directly on the `sync → cfo → channel_estimator →
equalizer → modem → fec → framing` chain that `Ofdm`
(`spectracuda/pipeline/ofdm.py`) wires together.

### 1.1 Framing layer — done (deframer/packetizer, stats schema, frame-not-found)
`spectracuda/framing/` is no longer a 5-line stub: `header.py`
(`HeaderCodec`), `packetizer.py` (`Packetizer`), and `stats.py`
(`compute_evm`/`compute_rssi_db`) now hold the logic that used to live
as ~150 lines inline inside `Ofdm.rx_process()`/`generate_frame()`,
reusable independent of any OFDM machinery (mirroring liquid-dsp's own
separation: `packetizer` knows nothing about `ofdmflexframegen`/
`ofdmflexframesync`). See `tests/test_framing_header.py`/
`tests/test_framing_packetizer.py` for standalone proof of that reuse.

- [x] Deframer/packetizer: `Packetizer` (CRC-append-then-FEC-encode /
      FEC-decode-then-CRC-check, exactly liquid-dsp's `packetizer_encode`/
      `packetizer_decode` order) and `HeaderCodec` (112-bit header <->
      field-dict, previously `Ofdm._encode_header_bits`/
      `_decode_header_bits`'s inline bodies) are both standalone,
      Ofdm-independent classes now. `Ofdm` owns one instance of each for
      its own tx-side crc/fec choice, and builds a throwaway `Packetizer`
      per `rx_process()` call from the DECODED header (same "resolve
      from the wire" principle as mod_scheme/fec0/crc). Old private call
      sites (`ofdm._encode_header_bits(...)`, `ofdm.fec_codec`,
      `ofdm.crc_codec`, etc.) kept working via thin delegating
      wrappers/aliases — no test churn from the extraction.
- [x] CRC-check-on-receive — unchanged behavior, now implemented via
      `Packetizer.decode()`'s `crc_valid` (never raises, matching
      liquid-dsp's `crc_validate_message`).
- [x] Packet stats readout — `rx_process()` now returns a dict with a
      **stable, fully-enumerated key set** (see its own docstring):
      `frame_found`, `start_index`, `sync_metric`, `rssi_db`,
      `cfo_estimate`, `channel_estimate`, `header`, `n_payload_symbols`,
      `bits`, `crc_valid`, `evm`. EVM and RSSI (the two liquid-dsp
      `ofdmflexframesync_get_framedatastats()` readouts that were
      previously entirely missing) are computed in
      `spectracuda/framing/stats.py`: EVM against the receiver's own
      hard-decision re-modulated symbols (standard practical EVM
      methodology, no transmitted-ground-truth needed); RSSI as a
      relative (NOT calibrated/absolute dBm — honestly documented, this
      codebase's IQ has no real antenna/ADC gain reference) received-
      power readout, computed unconditionally, even when no frame is
      found.
- [x] "Frame not found" — the sharpest of the three original gaps, now
      fixed: every `sync=` strategy always returns a best-candidate
      `start_index`/`metric` (a "best window found" search, not a
      detector with a built-in null hypothesis), which used to march
      genuine noise all the way through CFO/header decode, either
      getting caught by an unrelated sanity check or not caught at all.
      `rx_process()` now gates on item 0's sync metric against a new
      `sync_threshold=` constructor param (default
      `Ofdm.DEFAULT_SYNC_THRESHOLD=0.3`, empirically calibrated against
      both `SchmidlCoxSync` and `ZadoffChuSync` at fft_size in {64, 256}
      — see the constant's own docstring for the exact noise-floor-vs-
      signal numbers) and returns a clean `frame_found=False` result
      (every other field `None`) before attempting anything else.
      Remaining, explicitly-documented gap: a header that decodes to a
      *plausible-looking but wrong* value still raises an exception
      (ValueError/NotImplementedError) rather than a `header_valid=False`
      result — only the sync stage has a defined null outcome so far.

### 1.2 Second FEC stage (`fec1`) — done
Two-stage (concatenated) FEC is wired in end to end: `Ofdm(..., fec=...,
fec1=...)`, threaded through `HeaderCodec.encode_bits`/`decode_bits`
(the `fec1` header field no longer forces `NotImplementedError` — any
real FEC scheme is a genuinely accepted value) and
`spectracuda.framing.Packetizer` (`fec`=fec0=inner, `fec1`=outer).

- [x] Two-stage encode/decode pipeline: `Packetizer.encode()` runs
      fec0 (inner) then fec1 (outer); `Packetizer.decode()` runs the
      reverse (fec1/outer first, fec0/inner last) — `Ofdm.generate_frame`/
      `rx_process` delegate to it, same as single-stage FEC already did.
      A decode failure at either stage raises `ValueError` naming WHICH
      stage failed (`"fec1 (outer, ...) decode failed"` / `"fec0 (inner,
      ...) decode failed"`), not a bare, ambiguous "FEC decode failed".
- [x] Encode order decided + documented: fec0=INNER, fec1=OUTER,
      inner-then-outer on encode / outer-then-inner on decode — taken
      directly from liquid-dsp's own `packetizer_create()`/
      `packetizer_encode()`/`packetizer_decode()`
      (`reference/liquid-dsp/src/fec/src/packetizer.c`), verified by
      reading `plan[0].fs = fec0`/`plan[1].fs = fec1` and the encode
      loop running `plan[0]` then `plan[1]` while decode runs
      `plan[plan_len-1]` down to `plan[0]` — not assumed from the
      "inner/outer" naming alone. Documented in
      `spectracuda/framing/packetizer.py`'s module docstring (the
      authoritative copy) and referenced from `pipeline/ofdm.py`'s class
      docstring. `fec1` defaults to `"none"` (purely additive; existing
      single-stage `fec=` behavior is unchanged).
- Real coding-gain proof, not just plumbing: `tests/test_ofdm_class.py::
  test_two_stage_fec_corrects_errors_that_would_defeat_inner_alone`
  confirms (self-verifying, not just documented) that the exact same
  bit-flip pattern defeats `conv_v27` alone before proving the full
  `conv_v27`(inner)+`ldpc_648_r12`(outer) pipeline corrects it inside a
  real `Ofdm` frame.
- Interleaving between fec0/fec1 — **done**, see §1.12: the specific
  real-world benefit of concatenating Viterbi with RS (mopping up
  Viterbi's bursty decode-error residue) needs `rs_m8` as `fec0` and
  `conv_v27` as `fec1` (Viterbi decoded FIRST, facing the channel
  directly; RS decoded LAST) — the OPPOSITE assignment from the
  `conv_v27`(fec0)/`rs_m8`(fec1) example used just above in this
  section, which is still a correct, working `Packetizer` composition
  (fec0/fec1 plumbing is scheme-agnostic), just not the physically
  motivated pairing for that specific real-world benefit. See §1.12 for
  the full correction and why it was found only while building the
  demonstration test, not caught earlier.

### 1.3 `sync` — both planned strategies done
- [x] `SchmidlCoxSync` — implemented (`spectracuda/sync/schmidl_cox.py`)
- [x] `ZadoffChuSync` — implemented (`spectracuda/sync/zadoff_chu.py`):
      FFT-based matched-filter correlation against a standard Zadoff-Chu
      (CAZAC) sequence template, the batch-whole-buffer reimagining of
      `qdetector_create_linear`'s core idea (see its module docstring
      for exactly what carries over vs what doesn't from qdetector.c's
      streaming state machine). Known, documented simplification: no
      integer-bin CFO-offset search across candidates (qdetector.c has
      one) — pairs with `PilotBasedCFO`, not `SchmidlCoxCFO` (see §1.4).

### 1.4 `cfo` — both planned strategies done
- [x] `SchmidlCoxCFO` — implemented (`spectracuda/cfo/schmidl_cox.py`).
      Pairs specifically with `SchmidlCoxSync` — depends on the
      preamble's repeated-halves shape (confirmed during development:
      pairing it with `ZadoffChuSync` produces a garbage CFO estimate
      that corrupts the whole frame).
- [x] `PilotBasedCFO` — implemented (`spectracuda/cfo/pilot_based.py`):
      phase-slope estimate from repeated known OFDM (training) symbols
      at pilot subcarriers, no preamble-shape dependency — the correct
      pairing for `ZadoffChuSync` (also works with `SchmidlCoxSync`).
      Known, documented trade-off: noticeably noisier than SchmidlCoxCFO
      at typical small pilot counts (e.g. n_pilot=6) and ordinary OFDM
      test SNRs (20-25 dB) — needs more pilots and/or higher SNR for
      comparably clean decode (see its module docstring for the
      empirical characterization).

### 1.5 `channel_estimator` — both planned strategies done
- [x] `LSChannelEstimator` — implemented (`spectracuda/channel/ls.py`),
      pilot-based LS + linear interpolation across subcarriers
- [x] `MMSEChannelEstimator` — implemented (`spectracuda/channel/mmse.py`):
      Wiener/MMSE estimate from a closed-form assumed-uniform-PDP
      frequency correlation model (max_delay defaults to cp_len), fully
      vectorized as one precomputed weight matrix + a single batched
      matmul (no per-batch-item loop, unlike LS's `xp.interp` loop).
      Rigorously beats naive LS+interpolation NMSE when correctly
      configured (matched noise_var, pilot count >= max_delay — see
      tests/test_channel_mmse.py); known, documented limitation: an
      irreducible NMSE floor when underdetermined (fewer pilots than
      max_delay), and the generic default noise_var/max_delay won't
      always beat LS against every real (e.g. sparser-than-assumed)
      channel realization. Distinct from `MMSEEqualizer` (still done,
      unrelated pipeline stage) — don't conflate the two.

### 1.6 `equalizer` — both planned strategies done
- [x] `ZFEqualizer` — implemented
- [x] `MMSEEqualizer` — implemented
- `MMSEChannelEstimator` has now landed (§1.5); pairing
  `equalizer="mmse"` with `channel_estimator="mmse"` specifically
  (rather than each only ever being tested against the other's LS/ZF
  counterpart) is still an open combination worth a dedicated test, not
  yet done.

### 1.7 `modem` — scheme coverage gap (real parity target)
Implemented: BPSK, QPSK, 16-QAM, 64-QAM, 256-QAM (5 of liquid-dsp's 53
`modulation_scheme` enum values). Missing, in rough order of how likely
they are to matter for an OFDM PHY:

- [ ] 8-PSK / higher-order PSK variants (`LIQUID_MODEM_PSK8`…`PSK256`)
- [ ] APSK4–256 (used in DVB-S2-style links)
- [ ] π/4-DQPSK
- [ ] DPSK, ASK, ARB* (arbitrary constellations via `modem_create_arbitrary`)
- Out of scope for the OFDM PHY specifically (not really a gap for this
  project): GMSK, CPFSK, FSK, analog FM/AM (`freqmod`/`ampmodem`) — these
  are liquid-dsp modem-module members but not OFDM subcarrier modulations

### 1.8 `fec` — scheme coverage gap (real parity target) + a deliberate non-parity addition
Implemented, matching liquid-dsp parity: `conv_v27` (Viterbi), `rs_m8`
(Reed-Solomon RS(255,223)) — 2 of liquid-dsp's 28 `fec_scheme` enum
values. Missing, still a real parity gap:

- [ ] Punctured convolutional-rate variants: `V27P23`…`P78`, `V29P23`…`P78`
- [ ] Higher-constraint-length convolutional: `V29`, `V39`, `V615`
- [ ] Block/Hamming family: `HAMMING74`, `HAMMING84`, `HAMMING128`
- [ ] `GOLAY2412`
- [ ] SEC-DED family: `SECDED2216`, `SECDED3932`, `SECDED7264`
- [ ] No-op/repeat codes: `NONE`(already trivially supported), `REP3`, `REP5`
- Confirmed non-gap: **no Polar** — liquid-dsp doesn't have it either,
  so its absence here isn't a parity gap, just a shared limitation.

- [x] **LDPC — implemented as a deliberate scope expansion BEYOND
  liquid-dsp parity** (see `docs/ldpc.md` for the original implementation
  plan): the full 12-variant IEEE 802.11n QC-LDPC family (`ldpc_648_r12`
  … `ldpc_1944_r56` — 4 rates x 3 codeword lengths), `spectracuda/fec/
  ldpc.py` + `ldpc_tables.py`. liquid-dsp has no LDPC at all, so this was
  previously listed as a "confirmed non-gap" above -- added anyway,
  the same "no liquid-dsp precedent -> design from a standard reference
  instead of deferring" reasoning as `LSChannelEstimator`/`ZFEqualizer`/
  `MMSEEqualizer`. Base matrices sourced from `simgunz/802.11n-ldpc`
  (a citable MATLAB reference implementation of the published IEEE
  802.11n tables), transcribed programmatically (not by hand) and
  verified two independent ways before trusting them: the MATLAB
  `circshift` shift-direction convention checked directly against
  `numpy.roll`, and every one of the 12 variants' parity submatrix
  checked full-rank over GF(2) (the property the standard's own
  construction guarantees and the systematic encoder relies on). This is
  also the first FEC codec in this codebase whose iterative decode core
  (normalized min-sum belief propagation) is genuinely GPU-batched
  across `(n_batch, ...)` via gather-only `self.xp` operations (no
  scatter, no custom kernels) — Reed-Solomon's Berlekamp-Massey, by
  contrast, is a real per-codeword Python loop. Known, explicitly
  documented limitation: the whole `FEC` interface is strictly hard-
  decision bits in/out (no LLR pathway exists anywhere in this
  codebase's `Modem`/`FEC` today), so this decoder assumes a binary
  symmetric channel with a caller-supplied crossover probability `p`
  rather than consuming true soft channel information — functionally
  correct, but leaves real coding gain on the table versus a future
  soft-input pathway (see `ldpc.py`'s module docstring).

- [x] **`rs_m8` now supports "shortened" codewords — a real, load-bearing
  bug found and fixed, not a preemptive feature.** Discovered while
  wiring up a real `Mac(ofdm_kwargs=dict(fft_size=256, modem="qam16",
  fec="rs_m8", fec1="conv_v27", ...))` config (a two-process drone-link
  demo): `rs_m8` used to accept *only* an exact 223-byte message — not
  even its own bind handshake (104 bits) could be sent, since nothing
  short of a full block was ever valid. Padding every short message up
  to 223 bytes was considered and explicitly rejected (17x waste on a
  real link for a 13-byte message). Fixed with the actual standard
  technique instead — shortened Reed-Solomon (used in e.g. CCSDS): treat
  a message shorter than 223 bytes as if `(223 - real_k)` leading zero
  bytes were really there, compute the real 32 parity bytes against
  them (needs zero changes to the core GF(256) math — `ReedSolomonCode`
  was already systematic), but never transmit those implicit zeros —
  only `real_k + 32` bytes cross the air, scaling with the real message,
  not the block size. `spectracuda/fec/reed_solomon.py`'s `encode()`/
  `decode()` now accept any `1 <= real_k <= 223` (`decode()` recovers
  `real_k` directly from the codeword's own length, no new parameter
  needed); `spectracuda/fec/fec.py`'s wrapper now handles a message as
  N full blocks + at most one shortened leftover block (the same
  mechanism serves both the tiny-bind-request case and the leftover-
  segment-of-a-big-message case); a second, smaller real bug caught
  *while building this fix* (not by inspection): a byte-**mis**aligned
  leftover length (e.g. 100 bits) used to silently succeed through
  `np.packbits`'s implicit zero-padding and come back wrong on decode,
  not loud — now raises `ValueError` explicitly, same "fail loud, don't
  guess" convention as the rest of this codebase's FEC code.
  `spectracuda/mac/capacity.py`'s `compute_max_segment_bits()` had
  needed a stopgap "search whole blocks only" workaround for exactly
  this gap (an intermediate real fix in its own right, since arbitrary
  bisected guesses almost never land on an exact block multiple and the
  search was collapsing to an 8-bit floor) — simplified back to its
  original plain search now that `rs_m8` genuinely accepts any
  byte-aligned length; `FEC.accepts_partial_block` (new attribute) is
  what lets `capacity.py` tell `rs_m8` (now `True`) apart from LDPC
  (still `False` — see below).
  **Explicitly out of scope, a separate documented gap, not silently
  left behind**: every `ldpc_*` variant hits the identical "exact
  multiple only" wall through `Mac(ofdm_kwargs=...)` — confirmed
  directly, same symptom, same root cause — but LDPC's parity-check-
  matrix structure isn't the same simple systematic-zero-pad case RS is,
  so "shortening" it isn't the same fix. Still broken for `Mac`+LDPC
  today. Regression tests: `tests/test_fec_reed_solomon.py`,
  `tests/test_fec.py`, and `tests/test_mac_block_oriented_fec.py` (the
  full real config — bind handshake, a short message, and a
  multi-segment message, all through `Mac`+`rs_m8`+`conv_v27` — none of
  which could complete at all before this fix).

### 1.9 Layer 1 infra — implemented, lower priority to revisit
`OfdmModulator`/`OfdmDemodulator` (FFT/IFFT + CP) and `ResourceGrid`
(pilot/data/null mapping) are done and match liquid-dsp's
`ofdmframe_init_default_sctype`-style subcarrier allocation model. No
action items here beyond what's already tracked (the `complex64`
precision-discipline notes in `docs/architecture.md`).

### 1.10 `generate_frame()` partial-last-symbol padding — done
Found during manual verification (a quick smoke test with `crc="crc16"`,
`fec="conv_v27"`, at an arbitrary raw bit count): `generate_frame()` used
to require the CRC+FEC-encoded payload bit count to divide evenly into
`n_data * bits_per_symbol`, raising `ValueError` otherwise.

- [x] Automatic padding of a partial last payload OFDM symbol, no new
      wire field needed: `generate_frame()` now pads the encoded bit
      stream up to a whole number of OFDM symbols with fixed (not
      secret, not constant-content -- same "avoid a PAPR-bug-style
      artifact" rationale as the header's own filler) random filler
      bits, and `rx_process()` strips it back off using ONLY the
      header's existing `payload_len_bits` (RAW, pre-CRC/pre-FEC count)
      combined with the decoded crc/fec0/fec1 schemes — exactly the
      value it already needed for other bookkeeping
      (`Packetizer.encoded_length()`), confirmed to need no new field
      and no conflict with its existing meaning. `MAX_PAYLOAD_SYMBOLS`
      is still enforced against the padded (ceiling-divided) symbol
      count, so padding can't be used to sneak an over-long frame past
      the cap. See `tests/test_ofdm_class.py`'s
      `test_generate_frame_automatically_pads_a_partial_last_symbol`
      and neighboring tests (arbitrary bit counts, combined with crc+fec,
      spanning multiple symbols, batched with different per-item
      content, and the still-enforced MAX_PAYLOAD_SYMBOLS cap).

### 1.11 Long-frame decode reliability degrades at small `fft_size` — open, not root-caused
Found while building the MAC layer (`docs/mac.md`), which needed
multi-symbol payloads (segmentation across many OFDM symbols) and so
exercised a scale of frame this project's own prior tests mostly hadn't.
At `fft_size=64`/`n_pilot=6` (`sync=`/`cfo="schmidl_cox"`, AWGN-only
channel): a SHORT payload (~1 OFDM symbol) decodes reliably (20/20 clean
trials at `snr_db=25`), but a LONG payload (~19 symbols) has a real,
non-trivial per-attempt failure rate even at the same generous SNR
(17/20 at `snr_db=25`, reproducibly re-confirmed, not a one-off). The
same long payload at `fft_size=256`/`n_pilot=8` (this project's other
already-validated config) does NOT show this degradation as severely at
comparable SNR — pointing at frame LENGTH (number of OFDM symbols in one
frame), not just small `fft_size` alone, as the relevant variable.

**Isolated directly** (not just inferred): a SYMBOL-COUNT-MATCHED
comparison at `fft_size=128`/`n_pilot=7` — same ~19-symbol frame length
as the `fft=64` test above, achieved with a proportionally longer SDU —
scored **15/20** at `snr_db=25`, i.e. *worse* than `fft=64`'s 17/20
despite the larger FFT. This rules out "small `fft_size`" as the
explanation on its own and confirms frame length (number of OFDM symbols
per frame) is the real correlated variable — `fft_size=128`'s own
short-payload (~1 symbol) trials are a clean 20/20, matching both other
configs, so it's specifically the many-symbols case that degrades,
consistently across all three `fft_size` values tested.

- [ ] Root-cause the degradation. Leading, UNCONFIRMED hypothesis:
      `SchmidlCoxCFO`'s one-shot CFO estimate/correction (a single
      constant phase-ramp correction applied once, see its own
      docstring) leaves a small residual error that's negligible over a
      short frame but grows large enough over many more samples (a long
      multi-symbol frame) to meaningfully degrade EVM by the end of the
      frame — consistent with the observed pattern (fails more at longer
      frames, not just smaller `fft_size`), but NOT verified directly
      (e.g. by checking whether EVM measurably increases from the first
      to the last payload symbol within a single long frame — the direct
      test that would confirm or rule this out hasn't been run).
- [ ] If confirmed, the fix is almost certainly NOT re-estimating CFO
      per-symbol (no closed-loop carrier tracking object exists yet --
      see §2.1, `docs/todo.md`'s own note on this exact gap) — this
      finding is itself a concrete argument for prioritizing that item.
- Not currently blocking anything: `docs/mac.md`'s `MacLink` tests work
  around it by testing both `fft_size=64` and `fft_size=256` explicitly,
  with separately-calibrated channel parameters per config, rather than
  assuming one config's behavior transfers to the other.
- `tests/test_ofdm_combination_matrix.py` (fft_size 64/128/256 x modem
  qpsk/16qam/64qam x rate 1/2/2/3/3/4 x viterbi/ldpc x sync/cfo pairing,
  49 tests) broadens coverage in the SHORT-frame regime this item
  already found reliable — it does NOT probe the long-frame degradation
  itself (every case is a single FEC block, a handful of OFDM symbols);
  see that file's own module docstring for the full combination listing
  and why it's scoped to Ofdm directly rather than through `MacLink`.

### 1.12 Interleaver between fec0/fec1 — done, four algorithms, selectable via Ofdm's own constructor
`spectracuda/interleaver/` (`base.py`, `block.py`, `permutation.py`,
`convolutional.py`, `liquid.py`) implements four interleaver
strategies, registered under `spectracuda.registry` (the same
string-or-instance pattern as `sync=`/`cfo=`/`channel_estimator=`/
`equalizer=`, NOT the private-dict FEC/CRC/Modem pattern — each
interleaver has real tunable parameters worth exposing). Wired into
`Packetizer` (`interleaver=`, `interleaver_kwargs=`, sitting between
fec0's encoded output and fec1's input) and threaded straight through
as `Ofdm(..., interleaver=..., interleaver_kwargs=...)`.

**Real correctness finding from testing this, not assumed working from
the design alone: interleaving must move whole BYTES, not individual
BITS, to protect a byte-oriented outer code (`rs_m8`) — bit-granularity
interleaving can make things WORSE, not better.** Confirmed directly:
a ~50-bit contiguous Viterbi decode-error burst that already fit inside
Reed-Solomon's `t=16`-byte-per-codeword budget without any interleaving
(it landed as ~7 concentrated byte errors in one codeword, zero in the
other) got FRAGMENTED by a bit-level permutation into 32 and 20 byte
errors across the two codewords — WORSE than no interleaving at all,
because scattering individual bits (rather than whole bytes) turns a
handful of concentrated byte errors into many more, smaller ones, each
still counted as one full byte error by RS. Byte-granularity
interleaving of the exact same scenario left only 2 and 4 byte errors
per codeword. Fixed via a `unit_bits` parameter (default 1, set to 8 --
or a multiple -- to move whole bytes as indivisible blocks) on
`BlockInterleaver`/`PermutationInterleaver`/`ConvolutionalInterleaver`;
`LiquidInterleaver` is exempt (its mixed byte/sub-byte granularity per
pass is intrinsic to faithfully reproducing liquid-dsp's own algorithm,
not expressible as one uniform unit size). This also matches liquid-dsp's
own design choice: `interleaver.c`'s pass 1 swaps whole bytes, not
individual bits (see below) — not a coincidence once the mechanism is
understood, the same reason DVB/CCSDS interleavers are byte-oriented.
Pinned as a permanent regression test:
`tests/test_interleaver.py::test_byte_granularity_genuinely_helps_rs_where_bit_granularity_hurts_it`.

**Correction to this document's own earlier text (§1.2), found while
building the demonstration test above, not caught before**: the
"classic RS-cleans-up-Viterbi's-bursty-residual" architecture needs
Viterbi decoded FIRST (facing the channel directly) and RS decoded
LAST (mopping up Viterbi's residual). liquid-dsp's `packetizer_decode`
runs `fec1` first, `fec0` last (verified directly against
`packetizer.c` — see §1.2) — so for that architecture, **`conv_v27`
must be `fec1` and `rs_m8` must be `fec0`**, the OPPOSITE of the
`fec="conv_v27", fec1="rs_m8"` example this document (and a session
explanation) originally used. `fec0`/`fec1` composition itself was
never wrong (§1.2's plumbing/ordering is correct and liquid-verified);
only the earlier *illustrative example* of which physical scheme plays
which role had the assignment backwards.

The four algorithms:
  - **`"block"`** (`BlockInterleaver`, RECOMMENDED default) — the
    textbook M x N matrix interleaver, pure reshape+transpose,
    trivially vectorizes across the batch on `self.xp`.
  - **`"permutation"`** (`PermutationInterleaver`) — one fixed
    pseudo-random shuffle table (same fixed-seed-randomization
    technique as the header scramble mask / payload filler bits).
  - **`"convolutional"`** (`ConvolutionalInterleaver`) — a finite-block
    adaptation of the classic Forney/Ramsey-type interleaver CCSDS/
    DVB-S specify (explicit, documented deviation from the true
    unbounded-streaming construct — see its own module docstring).
  - **`"liquid"`** (`LiquidInterleaver`) — a verified-correct port of
    liquid-dsp's actual `interleaver.c` algorithm, confirmed (by
    reading the source directly) to be neither of the two textbook
    designs above: grid dimensions `M = 1+floor(sqrt(n_bytes))`, 4
    passes by default (pass 1 swaps whole bytes via a stateful indexed
    walk across the M x N grid; passes 2-4 repeat the same walk with a
    per-pass N offset, restricted to specific BITS within the paired
    bytes via a fixed mask `0x0f`/`0x55`/`0x33`), decode running the
    same passes in exact reverse order. Verified two independent ways
    before being trusted: `decode(encode(x))==x` on real byte data
    using the literal ported swap-loop, AND the derived bit-level
    permutation array applied as a plain gather producing byte-for-byte
    identical output to that same literal swap-loop — both checked
    across sizes including 255 (RS(255,223)'s own codeword size). A
    real bug was caught this way during development: an early
    refactor tried to reconstruct the algorithm's internal `n = n_bytes
    / 3` starting counter from `n2 = n_bytes / 2` alone
    (`(2*n2)//3`), which silently diverges from the true `n_bytes // 3`
    whenever n_bytes is an ODD MULTIPLE OF 3 — including 255 itself,
    which would have been a silent, serious correctness bug for the
    single most relevant real-world size. Fixed by passing n_bytes
    through directly instead of re-deriving it.

Deliberate, explicit exception to the "resolve from the wire, not from
self" rule every other header-carried field follows (mod_scheme/crc/
fec0/fec1): interleaver choice is NOT signaled in the header at all —
liquid-dsp doesn't signal its own interleaver's parameters over the air
either, both ends of a real link already have to agree on it
out-of-band. `Ofdm.rx_process()` builds its throwaway `Packetizer`
using the RECEIVING object's own `self.interleaver`/
`self.interleaver_kwargs`, the same way `preamble_seed`/`training_seed`
already work, not anything decoded from the frame — a receiver
misconfigured with the wrong interleaver fails to decode, same as a
genuinely mismatched preamble (verified directly: swapping in a
different interleaver type/params on the rx side reliably breaks
decode, proving this is actually wired in, not a cosmetic parameter).

`interleaver != "none"` requires `fec1 != "none"` at `Packetizer`
construction (raises `ValueError` otherwise) — interleaving between
fec0/fec1 only protects fec0's residual before fec1 sees it; with no
fec1 there's nothing downstream for it to protect.

Verification: `tests/test_interleaver.py` (89 tests: bijection +
round-trip + genuine-permutation + burst-spreading-property for all
three unit-based classes, the liquid port cross-checked against an
independently-typed reference implementation across 11 sizes including
the 255-byte regression case, unit_bits behavior, and the byte-vs-bit
granularity RS demonstration above) plus `tests/test_framing_packetizer.py`
(Packetizer-level interleaver wiring, all 4 types) plus manual `Ofdm`-level
verification (all 4 types round-trip end to end; a receiver configured
with the WRONG interleaver reliably fails, proving the setting is live).
Full suite: 505 passed, 1 pre-existing unrelated skip.

---

## Part 2 — Everything else (non-OFDM-pipeline liquid-dsp surface)

These are liquid-dsp capabilities with **no spectracuda module at all**
yet. Lower priority than Part 1 by design — the project's stated target
is the OFDM PHY, not general SDR infrastructure — but listed here so the
gap is visible rather than silently assumed away.

### 2.1 Timing / carrier tracking infrastructure
- [ ] `symsync` equivalent — polyphase-filterbank symbol timing recovery
      (liquid-dsp: `symsync_{rrrf,crcf}`). spectracuda currently assumes
      batch-aligned symbols with no fractional-timing-offset tracking.
- [ ] Standalone NCO/PLL carrier-phase tracking (liquid-dsp: `nco_crcf`
      + `_pll_*`). Today CFO is a one-shot batch estimate/correct
      (`SchmidlCoxCFO`), not a closed-loop tracking object.
- [ ] `symtrack`-equivalent (combined AGC + timing + carrier tracking) —
      not applicable to the OFDM path specifically, but note it's the
      liquid-dsp answer for single-carrier/linear-modem receivers if
      that ever becomes in scope

### 2.2 AGC
- [ ] No equivalent to `agc_{rrrf,crcf}` (gain control with lock/bandwidth
      params) anywhere in spectracuda today

### 2.3 General filter infrastructure
liquid-dsp's `filter` module (`firfilt`, `firdecim`/`firinterp`,
`firpfb`/`firpfbch`(r) channelizer, `firfarrow`, `iirfilt`, `fftfilt`,
`resamp`/`rresamp`/`msresamp`, `firdespm`, `fdelay`) has no spectracuda
counterpart. `docs/architecture.md` notes the intended path is "build on
CuPy/`cupyx.scipy.signal`" rather than hand-port these — that build
hasn't started.

- [ ] Pulse shaping / matched filtering (root-raised-cosine via `firfilt`
      equivalent) — relevant once non-OFDM or oversampled-front-end
      scenarios are in scope
- [ ] Arbitrary/rational resampling (`resamp`/`msresamp` equivalent) —
      relevant for real-ADC-rate front ends that don't sample at exactly
      `fft_size`-aligned rate

### 2.4 Non-OFDM framing PHYs — out of scope by design
`framegen64`/`framesync64`, `flexframegen`/`flexframesync`,
`bpacketgen`/`bpacketsync`, `dsssframegen`/`dsssframesync`,
`gmskframegen`/`gmskframesync`, `fskframegen`/`fskframesync`,
`qpilotgen`/`qpilotsync`. Not a gap against the project's stated goal
(OFDM PHY only) — listed for completeness, not as an action item.

### 2.5 Streaming API shim — done, `Ofdm.rx_streaming()`
liquid-dsp's entire concurrency model is `execute(samples, n)` + callback
on arbitrary-sized chunks; `rx_process(whole_buffer)` remains batch-only
by deliberate GPU-throughput design (`docs/architecture.md`, "Batch-shape
contract") and is UNCHANGED — `rx_streaming()` is additive.

- [x] `Ofdm.rx_streaming(chunk) -> Optional[dict]` — feed one arbitrary-
      sized, arbitrarily-aligned chunk of IQ (any length, no relationship
      to frame/symbol boundaries required); returns a `rx_process()`-
      shaped result the instant a complete frame finishes decoding, else
      `None` (still accumulating/searching). Design checked directly
      against liquid-dsp's own `ofdmframesync_execute()`
      (`reference/liquid-dsp/src/multichannel/src/ofdmframesync.c`)
      before writing this — its real state machine (`SEEKPLCP` →
      `PLCPSHORT0/1` → `PLCPLONG` → `RXSYMBOLS`) confirmed the key
      insight this reuses: every length past "sync found" is already
      known/fixed from the object's own config, so the receiver never
      has to guess how many more samples it needs at any stage. Expressed
      through this project's EXISTING batch-vectorized sync/cfo/
      channel_estimator/equalizer/demod primitives (re-run on accumulated
      slices once enough samples exist for the current stage) rather than
      a literal per-sample NCO/timer port — a deliberate, stated
      simplification. Single-stream only (`n_batch=1`) for this first
      version; multi-stream batched streaming is a separate, unattempted
      extension.
  - Required one behavior-preserving internal refactor of `rx_process()`
    (verified via the full pre-existing suite passing unchanged
    afterward): its header-decode and payload-decode logic extracted
    into `_decode_header_from_sync()`/`_decode_payload_from_header()`,
    which `rx_process()` itself now calls — avoids duplicating ~90 lines
    of delicate math that `rx_streaming()` needs reused at two different
    points in time, rather than risking two copies silently drifting
    apart.
  - Real correctness bug caught during implementation, not left for a
    user to find: the CFO-corrected buffer computed at the header stage
    is stale by the time payload samples arrive (it was corrected over a
    shorter buffer that didn't yet include them) — fixed by re-applying
    `self.cfo.correct()` to the full, now-complete buffer with the SAME
    already-estimated `cfo_estimate` once the payload is ready, rather
    than reusing the stale array. Caught by comparing streaming output
    against `rx_process()`'s reference EVM directly (matched to 1e-6,
    not just "close enough"), not assumed correct.
  - A search-window cap (`Ofdm.STREAM_SEARCH_WINDOW_SYMBOLS`, default 8×
    `fft_size`) bounds accumulation while searching, so a long silent/
    noisy stream with no frame in it doesn't grow the buffer (or the
    matched-filter correlation cost) without bound — never trims once a
    frame is actually being decoded.
  - Failure handling deliberately diverges from `rx_process()`: a failed
    header or payload decode (corrupted header, false-positive
    detection, uncorrectable FEC) does NOT raise here — it's discarded
    (`None` returned) and the state machine resumes searching, since a
    streaming receiver must survive one bad frame and keep running
    indefinitely, unlike a single one-shot `rx_process()` call.
  - `tests/test_ofdm_streaming.py` (12 tests): bit-identical match
    against `rx_process()` across 5 chunk sizes including a deliberately
    non-aligned one (37), multiple frames back-to-back in one continuous
    stream, pure-noise streams (never a false completion, buffer stays
    bounded), and a corrupted-frame-then-recovers case (one bad frame
    doesn't prevent finding the next real one).
  - **Real gap found and closed, not assumed covered**: the above 12
    tests all used ONE fixed `Ofdm` config throughout (`fft_size=256`/
    `sync="schmidl_cox"`/`cfo="schmidl_cox"`/etc.) — zero variation
    across sync strategy, CFO strategy, `fft_size`, `cp_len`,
    `channel_estimator`, or `equalizer` had actually been exercised
    through `rx_streaming()` specifically, even though
    `tests/test_ofdm_combination_matrix.py` already covered this exact
    matrix for `rx_process()`. Closed by
    `tests/test_ofdm_streaming_combination_matrix.py` (49 tests,
    mirroring that file's own 36+9+1+3 structure) — the full fft_size x
    modem x FEC/rate grid, all 3 valid sync/cfo pairings, the known-
    incompatible pairing (confirmed it never completes via streaming
    rather than raising, a genuinely different code path from
    `rx_process()`'s own negative-case test), and the real multipath/
    AWGN/CFO subset, all fed through `rx_streaming()` with a
    deliberately non-aligned chunk size (97) throughout. All 49 pass.

### 2.6 Correctness-validation debt
- [x] `CRC` — verified byte-exact against liquid-dsp's own
      `crc_autotest.c` vectors
- [ ] Viterbi / Reed-Solomon have no liquid-dsp reference to diff
      against (liquid-dsp wraps external `libfec` for both, no internal
      fallback implementation) — worth cross-checking against a
      standard external reference (e.g. a known-good Python RS/Viterbi
      implementation or GNU Radio) instead, since "match liquid-dsp
      bit-for-bit" isn't achievable here even in principle
- [ ] No benchmark suite comparing spectracuda (CuPy/GPU) throughput
      against liquid-dsp's SIMD-optimized C path — worth having before
      making any performance claims, since none of the current tests
      (`tests/test_*.py`) measure this

### 2.7 Documentation housekeeping — done
- [x] `spectracuda/__init__.py`'s module docstring and `README.md` both
      used to say "Layer 2/3 blocks... not implemented yet" and reference
      `OfdmRx`/`OfdmTx` (removed long ago, replaced by the unified `Ofdm`
      class). `README.md` rewritten with a verified-runnable quick-start
      example, a swappable-strategies table, the two-stage-FEC+
      interleaver `fec0`/`fec1` ordering explanation (see docs/todo.md
      #1.12's correction), a MAC-layer example, and a docs map; found via
      a direct user question ("is there a readme...") rather than
      proactively, worth noting since it means this had been stale for a
      while without being caught by testing (docs aren't covered by the
      test suite by nature).

### 2.8 Tutorial book — Part I done (PHY chain), Part II outlined
- [x] **"The OFDM Field Guide"** — a PySDR-style tutorial, published as an
      interactive HTML artifact (linked from `README.md`'s new "The book"
      section), spectracuda-focused rather than ground-up (assumes basic
      DSP/SDR literacy, points to PySDR itself for that). Part I (the
      Ofdm object, synchronization, CFO, channel estimation/equalization,
      the modem) is fully written, each chapter grounded in a real,
      generated figure — `docs/book_figures/generate_figures.py` imports
      spectracuda's actual classes (`Modem`, `SchmidlCoxSync`, `Ofdm`,
      `Channel`) and runs them; nothing is a hand-drawn illustration. Two
      real things worth recording, found while building the figures (not
      library bugs, but real, measured facts a reader benefits from
      knowing):
      - The real `SchmidlCoxCFO` estimator has a narrower reliable range
        than an illustrative offset does — swept empirically at 25 dB SNR,
        this implementation decodes correctly through ~1.0 subcarrier of
        offset and fails past ~1.5. The book's CFO chapter states this
        measured range explicitly rather than picking a flattering number.
      - A long payload through a real multipath channel can shift the
        detected frame start a few samples later than the noiseless case
        (delay spread), and `generate_frame()`'s output has zero built-in
        trailing margin beyond exactly what a nominal-start decode needs —
        a late-detected start then runs the last payload symbol past the
        end of the buffer. Worked around in the figure script by padding
        trailing silence (as any real captured buffer would have); the
        underlying question of how much margin a streaming capture should
        keep by default is still open, adjacent to #1.11 above, not fixed
        here.
      Part II (the MAC layer: TM/UM/AM, two independent radios, binding,
      bidirectional AM) is outlined in the book's nav with "coming soon"
      tags and real pointers to `docs/mac.md`/`examples/`, not written yet
      — disclosed honestly rather than left blank or faked.
      `pyproject.toml` gained a `docs` extra (`matplotlib`, dev-only,
      never imported by `spectracuda` itself) to make figure regeneration
      reproducible.
- [x] **Live hosted docs site (spectracuda.readthedocs.io) + the book now
      exists in two synchronized forms.** `docs/conf.py`/`.readthedocs.yaml`
      set up a Sphinx+MyST build (parses `docs/*.md` unchanged, plus a new
      `docs/book/*.md` directory holding the same chapters as the
      standalone HTML artifact, MyST syntax instead of hand-authored HTML).
      `docs/book_figures/book_template.html`/`build_book.py` (the
      artifact's own source) were themselves missing from the repo at
      first — built straight into an ephemeral scratchpad during the
      original session — moved into `docs/book_figures/` once flagged, so
      the whole book is now reproducible from source like everything else
      here, not recoverable only by reading the published artifact back.
- [x] **Chapter 06 — FEC, LDPC & Interleaving, written (both forms).**
      `conv_v27`/`rs_m8` vs. liquid-dsp parity, the 12-variant LDPC family
      as deliberate non-parity scope, the "fail loud" uncorrectable-
      codeword convention, and the `fec0`/`fec1` two-stage ordering
      requirement (`docs/todo.md` §1.12's own correction, cited directly
      rather than re-derived). Centered on one real, measured figure — a
      genuine frame-error-rate-vs-SNR sweep (`Ofdm(fec=...)` +
      `Channel(snr_db=...)` + `rx_process()`, 40 real trials per point,
      `conv_v27`/`rs_m8`/`ldpc_648_r12`/`none`) — not a textbook reference
      curve. Two real, measured findings the chapter reports honestly:
      `conv_v27` and `ldpc_648_r12` both show a genuine ~4 dB coding gain
      over uncoded; `rs_m8` ALONE shows almost no gain at all against pure
      AWGN, which is real RS behavior (a burst/symbol-oriented code has
      little to work with against scattered independent bit errors, not a
      bug) and the actual, measured reason a real system pairs it with an
      inner code rather than using it alone — exactly the fec0/fec1
      section's point, now backed by a real number instead of only
      asserted. Getting there required finding each scheme's own valid
      payload-size constraint by hand (`rs_m8`: exact multiples of 1784
      bits; `ldpc_648_r12`: multiples of 324, forced to 632 bits of
      payload once crc16's 16-bit/byte-alignment requirement is also
      satisfied) and reusing the same late-detected-start trailing-margin
      workaround the channel-equalization figure already needed. Verified
      to actually build: a real `sphinx-build -b html docs ... -E -a` run
      (forced, not relying on a possibly-stale cached environment) came
      back clean, zero warnings, with the new chapter's table/figure/
      admonitions all present in the rendered output.

---

## Suggested next-up order

Given the above, the highest-leverage next steps for reaching a genuinely
end-to-end OFDM rx (not just individually-tested stages) are:

1. ~~`framing/`~~ — **done** (§1.1): `HeaderCodec`/`Packetizer` extracted
   as reusable objects, `rx_process()` now has a stable, fully-enumerated
   result schema (including EVM/RSSI, previously missing entirely), and
   a calibrated `sync_threshold=` turns "the best correlation we found
   was still weak" into a clean `frame_found=False` result instead of
   marching on to decode a header that isn't there. Remaining, smaller
   gap noted in §1.1: a *plausible-but-wrong* decoded header still
   raises rather than returning `header_valid=False`.
2. ~~`ZadoffChuSync`~~ — **done** (§1.3), along with its correct `cfo=`
   pairing `PilotBasedCFO` (§1.4) and `MMSEChannelEstimator` (§1.5) — all
   three sibling strategies from the original §1.3-1.5 gaps now exist;
   `sync=`/`cfo=`/`channel_estimator=` are each genuinely swappable now,
   not single-option
3. ~~`fec1` second stage~~ — **done** (§1.2): two-stage (inner/outer)
   FEC wired end to end through `HeaderCodec`/`Packetizer`/`Ofdm`,
   ordering decided and documented (fec0=inner, fec1=outer, verified
   against liquid-dsp's own `packetizer.c`), real coding-gain proof
   inside the full pipeline.
4. Remaining breadth: `equalizer="mmse"` paired specifically with
   `channel_estimator="mmse"` (§1.6, untested combination), remaining
   modem/FEC scheme coverage (§1.7/§1.8)
5. ~~`generate_frame()`'s no-partial-symbol-padding gap~~ — **done**
   (§1.10): automatic filler padding, no new wire field, existing
   `MAX_PAYLOAD_SYMBOLS` cap still enforced against the padded count.
6. ~~MAC layer (TM/UM/AM)~~ — **done** (Part 3, new ground beyond the
   original OFDM-pipeline scope): see `docs/mac.md`.

---

## Part 3 — MAC layer (TM / UM / AM)

New ground, not part of the original liquid-dsp-parity scope (liquid-dsp
has no MAC/RLC concept at all — this section exists purely because it was
explicitly requested, same footing as LDPC's "added anyway" framing).

- [x] **MAC layer implemented** — `spectracuda/mac/` (`Mac`, `MacLink`,
  `TmEntity`/`UmEntity`/`AmEntity`), a combined MAC+RLC layer offering
  3GPP-RLC-named TM/UM/AM delivery modes with a simplified/custom (not
  spec-exact) PDU format, wired end-to-end into a real `Ofdm` PHY chain
  via `MacLink`. See `docs/mac.md` for the full design, two real bugs
  found and fixed during implementation (a reassembly out-of-order-
  desync bug and a STATUS-PDU cumulative-ack boundary bug — both caught
  by dedicated tests, not just described), and the reproducible
  channel-loss proof that AM recovers where UM/TM genuinely fail.
  Scope, explicitly bounded: point-to-point only (no multi-UE
  scheduling), ARQ not HARQ (no soft-combining), simplified PDU wire
  format (not 3GPP TS 36.322/38.322-exact).
- [x] **Binding handshake + link-quality reporting implemented** —
  `spectracuda/mac/bind.py` (`evaluate_bind_request()`, a pure
  accept/reject function tested with genuinely mismatched configs, not
  just self-consistent plumbing) and `spectracuda/mac/quality.py`
  (`LinkQualityTracker` aggregating `Ofdm.rx_process()`'s existing
  `rssi_db`/`evm` into RSSI/EVM/delivery-ratio trend stats — the latter
  an explicit BER-trend proxy, not true BER, honestly documented as
  such). `MacLink.bind()`/`.exchange_link_quality()` wire both into the
  real PHY; `send()` now refuses to run before `bind()` succeeds — a
  real behavioral gate, not a formality. Required widening `pdu.py`'s
  `TYPE` field from 1 to 3 bits (purely additive: DATA=0/STATUS=1 keep
  their values). One more real bug found and fixed: gating `send()` on
  `bind()` silently broke every calibrated lossy-channel test seed,
  because `bind()`'s own PHY rounds consume some of the shared
  `Channel`'s RNG stream before `send()` runs — fixed by binding on a
  clean channel first (also the more realistic model), not by
  re-sweeping new constants. See `docs/mac.md` for the full writeup.
- Not done / out of scope for this pass: multi-SDU concurrent/
  interleaved reassembly (`ReassemblyBuffer` tracks one in-progress SDU
  at a time — a stated limitation, see `docs/mac.md`), HARQ soft-
  combining, real 3GPP PDU wire-format fidelity.
- [x] **`MacLink`'s "full two-independently-configured-endpoint" gap,
  closed** — `Mac(mode=..., ofdm_kwargs=...)` now optionally builds and
  owns its OWN `Ofdm` (derived `max_segment_bits`, never a separately-
  specified number that could disagree with it), so two real HW units
  are just two independent `Mac(...)` calls with zero shared object
  identity (`hw1.ofdm is not hw2.ofdm`, confirmed, not assumed).
  `Mac.send_iq()`/`receive_iq()` (TM/UM/AM — AM's DATA-forward half,
  see below for its full STATUS round trip) carry data over real IQ,
  fully manually wired by
  the caller (no orchestrator class, by explicit request — "it is fully
  manual it is responsibility of developer"). Binding/link-quality
  reporting were THEN migrated from `MacLink` onto `Mac` itself
  (`build_bind_request()`/`handle_bind_request_iq()`/
  `handle_bind_response_iq()`/`build_quality_report()`/
  `handle_quality_report_iq()`) — this is what finally makes the
  mismatch-rejection proof genuine: `MacLink.bind()` could only ever
  self-consistently succeed (one shared `Ofdm` across both roles), but
  two real `Mac` objects have real, independent capacity, so
  `test_real_cross_object_bind_genuinely_rejects_a_mismatched_capacity`
  is the first time this rejection has been proven through the actual
  wire mechanism rather than only as a standalone pure-function call.
  `MacLink` itself is untouched (still 21/21 passing) — a deliberate
  scope boundary, not an oversight (its `tx_mac`/`rx_mac` are
  PHY-agnostic by design and structurally can't use the new IQ-transport
  methods). Two real bugs found building this, both fixed: `bind.py`'s
  `max_segment_bits` wire field was a `u16` (max 65535) and silently
  worked in every `MacLink`-only test until a genuine two-object test at
  `fft_size=256`/`modem="qam64"` derived a real value of 165840 — widened
  to a `u32`; and gating `send_iq()` on `bind()` made an existing
  mismatched-PHY test's original scenario unreachable (the handshake
  itself now fails first on a broken link, before data would ever be
  sent) — rewritten to demonstrate the failure at the bind stage instead
  of papering over it. See `docs/mac.md`'s "Two-node redesign, step 1"
  and "Binding + link-quality reporting, migrated onto `Mac`" sections
  for the full writeup. Full suite: 583 passed, 1 pre-existing unrelated
  skip.
- [x] **The full bidirectional 4-`Mac`/4-`Ofdm` picture, and AM over
  `send_iq()`/`receive_iq()`, both done.** The `mode == "am"` guards on
  `send_iq()`/`receive_iq()` are removed — AM's DATA-forward half now
  works exactly like UM's over the wire. The STATUS/retransmission round
  trip needed a genuine second Mac/Ofdm pair for the reverse direction,
  which a single object's send/receive pair structurally can't provide —
  demonstrated in two new example scripts (`examples/mac_bidirectional_am_
  batch_demo.py` via `Ofdm.rx_process()`, `mac_bidirectional_am_streaming_
  demo.py` via `Ofdm.rx_streaming()`), each building the full 4-object
  model (`HW1.tx_mac`→`Ofdm_A`, `HW2.rx_mac`→`Ofdm_A'`,
  `HW2.tx_mac`→`Ofdm_B`, `HW1.rx_mac`→`Ofdm_B'`), binding both direction
  pairs independently, then proving a real dropped-PDU → STATUS → targeted
  retransmission → recovery round trip through an actual `Channel`, plus a
  clean no-retransmit-needed reverse direction. Converted to
  `tests/test_mac_bidirectional_am.py` (9 tests). Two real bugs found
  along the way: the stale `test_am_mode_iq_methods_not_supported_yet`
  (asserted the now-removed guard — replaced with a test proving AM's
  DATA-forward half actually works) and all three PRE-EXISTING example
  scripts (`mac_two_units_demo.py`, `mac_half_duplex_demo.py`,
  `mac_streaming_demo.py`) turning out to be broken ever since the
  `bound`-gate was added to `send_iq()` — none of them had been updated
  to call `bind()` first, and nothing in the test suite caught it because
  they're standalone scripts nothing imports; caught only by actually
  running them, not by inspection. See `docs/mac.md`'s "AM's
  `send_iq()`/`receive_iq()` opened up, plus the 4-Mac/4-Ofdm
  bidirectional picture" section for the full writeup, including a
  real constraint hit while building the demo (the PDU header's 16-bit
  `so` field caps a segmented SDU at 65535 bits, and the small-PHY config
  needed for real 3-way segmentation ran into the still-open §1.11
  long-frame-reliability gap at the SNRs used elsewhere — worked around
  with a higher demo SNR, not a fix for §1.11 itself). Full suite: 592
  passed, 1 pre-existing unrelated skip.
