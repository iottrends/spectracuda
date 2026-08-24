# liquid-dsp public API inventory

Source: `reference/liquid-dsp` (jgaeddert/liquid-dsp, cloned shallow), extracted
from `include/liquid.h` (~11k lines, 20 `// MODULE :` sections). This is the
ground truth for what liquid-dsp actually exposes to users today — used to
ground spectracuda's top-level API instead of designing from assumption.

## The 20 modules

`agc`, `audio`, `buffer`, `channel`, `dotprod`, `equalization`, `fec`, `fft`,
`filter`, `framing`, `math`, `matrix`, `modem`, `multichannel`, `nco`,
`optimization`, `quantization`, `random`, `sequence`, `utility`, `vector`.

Every object follows the same C idiom: `<name>_create(...)` → `<name>_execute`
/ `_step` / `_encode` / `_decode` → `<name>_destroy`. Many are type-generic via
X-macros (e.g. `firfilt_rrrf`/`firfilt_crcf`/`firfilt_cccf` = real/complex
input-output-coefficient variants of one `firfilt` design) — that macro
pattern *is* liquid-dsp's version of "one block, several interchangeable
instantiations," conceptually close to what a Python registry pattern gives us
for free.

## Relevant to an OFDM receive chain

### Synchronization / detection
- `qdetector_{cccf}` — generalized correlator-based preamble/frame detector:
  `_create`, `_create_linear` (arbitrary sequence, i.e. **this is liquid-dsp's
  Zadoff-Chu-style / linear-sequence sync**), `_create_gmsk`, `_create_cpfsk`.
- `qdsync` — the newer post-detection synchronizer that continues tracking
  after `qdetector` fires (`_create_linear`, `_create_cpfsk`).
- `detector_cccf` — older/simpler single-sequence correlation detector.
- `bsync` — binary/msequence sync (`_create`, `_create_msequence`).
- `presync` — pre-demodulation sync helper.
- `symsync_{rrrf,crcf}` — **timing recovery** (polyphase-filterbank symbol
  timing synchronizer), `_create`, `_create_rnyquist`, `_create_kaiser`.
- `symtrack_{rrrf,cccf}` — combined AGC + timing + carrier tracking in one
  object (`_create`, `_create_default`) — liquid-dsp's answer to "give me a
  synced constellation from a raw stream" for the non-OFDM (linear modem)
  path.
- **No standalone "Schmidl-Cox" object.** S&C-style self-correlation sync
  isn't a named liquid-dsp block — it's what `ofdmframesync`'s internal S0/S1
  preamble logic already implements (see below), not something exposed for
  reuse outside OFDM framing.

### Frequency/phase
- `nco_crcf` / `nco` — numerically-controlled oscillator + **carrier
  tracking PLL** (`_create(LIQUID_NCO)` vs `_create(LIQUID_VCO)`, `_pll_*`
  methods) — this is liquid-dsp's CFO/phase-tracking primitive; there's no
  separate "cfo_estimator" object — CFO estimation is folded into
  `ofdmframesync`/`symtrack`/`qdsync` internally and only *exposed* as a
  read-out (`ofdmframesync_get_cfo`), not as an independent swappable
  strategy.

### OFDM (the core of the target use case)
- `ofdmframegen` / `ofdmframesync` — **low-level, single-symbol** API:
  `write_S0a/S0b/S1/writesymbol/writetail` on the gen side, `execute()` +
  callback on the sync side. Subcarrier allocation is a plain
  null/pilot/data byte array (`ofdmframe_init_default_sctype`,
  `_init_sctype_range`), not a config object.
- `ofdmflexframegen` / `ofdmflexframesync` — **high-level, packet-oriented**
  wrapper around the above: `assemble(header, payload)` → `write(buf)` on
  the tx side; `execute(samples, n)` streaming into a **user callback** on
  the rx side (`framesync_callback`), with `get_rssi()`, `get_cfo()`,
  `get_framedatastats()` as the only externally queryable "channel state."
- **Important finding:** liquid-dsp's OFDM sync is a single monolithic,
  *stateful, streaming* object — packet detection, CFO estimation/tracking,
  timing, per-subcarrier equalization, and header/payload demod+FEC-decode
  all happen *inside* `ofdmflexframesync_execute()` and are not separately
  swappable. There is no `channel_estimator="ls"` / `equalizer="mmse"` knob
  in liquid-dsp — LS/MMSE channel estimation and ZF/MMSE equalization aren't
  named, user-selectable strategy objects anywhere in the library; the
  built-in OFDM sync just does its own fixed pilot-based LS-style estimate
  internally. **This is a real gap between the original proposal's declarative
  API sketch and liquid-dsp's actual API shape** — the
  `channel_estimator=`/`equalizer=` swappability spectracuda wants is a
  deliberate *improvement* over liquid-dsp, not something being mirrored from
  it, and should be flagged as such rather than assumed to have a liquid-dsp
  precedent.

### Equalization (as its own module — separate from OFDM)
- `eqlms_{rrrf,cccf}` — adaptive LMS equalizer: `_create`, `_create_rnyquist`,
  `_create_lowpass`.
- `eqrls_{rrrf,cccf}` — adaptive RLS equalizer: `_create`.
- These are generic **single-carrier, sample-adaptive** equalizers (train on
  a known sequence or decision-direct), not OFDM per-subcarrier
  frequency-domain LS/MMSE/ZF equalizers. Another gap vs. the proposal's
  framing — LS/MMSE/ZF as named strategies simply don't exist in liquid-dsp;
  they'd need to be designed from textbook/standards references (e.g.
  802.11/3GPP reference receivers), not ported.

### AGC
- `agc_{rrrf,crcf}` — `_create(void)`, then `_execute`/`_lock`/set
  bandwidth/gain limits. Simple, single object type, no strategy variants.

### Modem (linear digital modulation)
- One `modem` object type, parameterized entirely by the
  `modulation_scheme` enum (53 schemes): PSK2–256, DPSK2–256, ASK2–256,
  **QAM4–256** (covers the target list — BPSK/QPSK/16/64/256-QAM are all
  named enum members, not separate objects), APSK4–256, plus BPSK/QPSK/OOK
  aliases, several "optimal"/arbitrary constellations (ARB*), V.29, π/4-DQPSK.
  `_create(scheme)` / `_create_arbitrary(table)` / `modulate` / `demodulate`
  (+ soft-decision LLR variant). This maps very cleanly to a single
  `modem`/`mapper` block in spectracuda parameterized by a scheme string,
  exactly like the proposal envisioned.
- Also in this module (not needed for OFDM PHY v1): `gmskmod/dem`,
  `cpfskmod/dem`, `fskmod/dem`, `freqmod/dem` (analog FM), `ampmodem` (AM).

### FEC (confirms the v1 scope decision already made)
- One `fec` object type, parameterized by `fec_scheme` enum
  (`LIQUID_FEC_NUM_SCHEMES = 28`):
  - No-op / repeat codes: NONE, REP3, REP5
  - Block/Hamming family: HAMMING74/84/128, GOLAY2412, SECDED2216/3932/7264
  - **Convolutional (Viterbi-decoded)**: CONV_V27, V29, V39, V615, plus
    punctured-rate variants V27P23…P78, V29P23…P78 (this *is* liquid-dsp's
    "Viterbi" — there's no separate object named `viterbi`, it's just
    `fec_scheme = LIQUID_FEC_CONV_*` decoded via `fec_decode`/`fec_decode_soft`)
  - **Reed-Solomon: exactly one variant**, `LIQUID_FEC_RS_M8` (GF(2^8),
    n=255, k=223) — not a general parameterized RS, a single fixed code.
  - **No LDPC, no Polar codes anywhere in liquid-dsp** — there's no
    liquid-dsp reference implementation to mirror for either, so both
    would have to be designed from scratch/other references, not
    ported. (Polar is still deferred for spectracuda too. LDPC was
    later added anyway as a deliberate scope expansion BEYOND
    liquid-dsp parity — see `docs/ldpc.md`, `docs/todo.md` §1.8, and
    `spectracuda/fec/ldpc.py` — precisely because no liquid-dsp
    reference existed to defer to in the first place.)
- `packetizer` — chains CRC + interleaver + two-stage FEC (inner/outer) into
  one encode/decode object — this is liquid-dsp's framing-level FEC
  composition, worth mirroring conceptually for spectracuda's `fec=` handling
  of header vs. payload coding.
- `interleaver`, standalone CRC (`crc_*`, referenced from framing) round out
  the module.

### Framing (non-OFDM, for completeness)
- `framegen64`/`framesync64` — fixed 64-byte payload frame (simplest case).
- `flexframegen`/`flexframesync` — variable-length payload, same
  assemble/write + execute/callback idiom as `ofdmflexframe*`.
- `bpacketgen`/`bpacketsync`, `dsssframegen`/`sync` (DSSS), `gmskframegen`/
  `sync`, `fskframegen`/`sync` — alternate PHYs, not relevant to the OFDM
  target.
- `qpilotgen`/`qpilotsync` — pilot-sequence generation/detection helper used
  internally by the newer frame types.

### Filter (used internally by the above, and independently useful)
`firfilt`, `firdecim`, `firinterp`, `firhilb`/`iirhilb`, `firpfb`
(polyphase filterbank), `firpfbch`/`firpfbchr` (channelizer), `firfarrow`
(fractional-delay), `iirfilt`/`iirfiltsos`, `fftfilt`, `resamp`/`rresamp`
(arbitrary + rational-rate resamplers), `msresamp` (multi-stage resampler),
`firdespm` (Parks-McClellan filter design), `ordfilt` (order-statistic/median
filter), `fdelay` (fractional delay line). All generic DSP infrastructure —
maps to "build on CuPy/`cupyx.scipy.signal`" rather than anything
SDR-specific to reimplement by hand.

### Everything else (present but out of scope for the OFDM receiver v1)
`audio` (codecs), `buffer`/`cbuffer`/`wdelay`/`window` (ring buffers),
`channel`/`tvmpch` (channel *simulation*, i.e. for generating test
impairments — actually directly useful for spectracuda's own test/integration
harness, not the receiver itself), `dotprod` (SIMD-optimized vector
dot-product, an implementation-level primitive CuPy/cuBLAS already covers),
`math`/`matrix`/`smatrix`/`vector` (numeric utilities, sparse matrix),
`multichannel` (multi-signal channelizer via `firpfbch`), `optimization`
(gradient descent / genetic-algorithm parameter search — not PHY-relevant),
`quantization` (compander), `random` (distributions), `sequence`
(m-sequence/complementary-code generators, used by `bsync`/`qdetector` for
known sync sequences), `utility` (misc helpers).

## Key takeaways for mapping to spectracuda's top-level API

1. **liquid-dsp's real granularity is coarser than the original proposal
   assumed.** For OFDM specifically, liquid-dsp gives you one strategy per
   concern (one preamble detector shape via `qdetector_create_linear`, one
   built-in CFO/timing tracker, one internal channel-estimate-and-equalize
   step) — not a menu of interchangeable `"ls"` vs `"mmse"` style options.
   spectracuda's swappable-block registry (`sync=`, `cfo_estimator=`,
   `channel_estimator=`, `equalizer=`) is a genuine value-add over liquid-dsp,
   not a mirror of it, and should be framed to the user/README that way.
2. **Where liquid-dsp *does* give a clean enum-of-strategies model — `modem`
   (53 schemes) and `fec` (28 schemes) — spectracuda should copy that pattern
   almost directly**: one block type, a scheme-name string parameter, rather
   than a class-per-scheme registry. This is the part of "liquid-dsp-like"
   that transfers most literally.
3. **The streaming `execute(samples, n) + callback` idiom is liquid-dsp's
   entire concurrency/statefulness model** — every sync/frame/equalizer
   object is designed to be fed arbitrary-sized chunks continuously and
   maintain internal state across calls. spectracuda's batch-first
   `rx.process(iq_samples)` (whole buffer in, packets out) is a deliberate
   *departure* from this, driven by the GPU-batching requirement established
   earlier — worth being explicit that spectracuda trades liquid-dsp's
   arbitrary-chunk streaming flexibility for GPU throughput, and that a
   thin streaming-shim wrapper (accumulate-then-batch) may be worth offering
   later for users who want liquid-dsp-style incremental feeding. **Update:
   this shim now exists** — `Ofdm.rx_streaming(chunk)`, additive alongside
   `rx_process()`, modeled directly on `ofdmframesync_execute()`'s own state
   machine. See `docs/architecture.md`'s batch-shape-contract section and
   `docs/mac.md`'s "How MAC data actually crosses the air" for the design
   and how `spectracuda.mac` drives it.
4. **FEC v1 scope (Viterbi + Reed-Solomon) maps exactly onto liquid-dsp's own
   `LIQUID_FEC_CONV_*` and `LIQUID_FEC_RS_M8`** — good confirmation this
   choice is "parity with liquid-dsp," not a reduced subset of it.
5. Genuinely reusable-as-reference, not just descriptive: `qdetector`'s
   linear-sequence correlation detector is the concrete algorithm to mirror
   for spectracuda's `sync="zc"`; `ofdmframesync`'s S0/S1 preamble handling is
   the concrete algorithm to mirror for `sync="schmidl_cox"` (since, as noted,
   liquid-dsp doesn't expose it as an independent named block — it has to be
   extracted from `ofdmframesync`'s internal logic, which the cloned source
   under `reference/liquid-dsp/src/framing/src/ofdmframesync.c` has in full).
