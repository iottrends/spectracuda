# spectracuda architecture

## Framing

spectracuda is not a CUDA port of liquid-dsp. The [API inventory](liquid-dsp-api-inventory.md)
shows liquid-dsp never exposed `channel_estimator="ls"` / `equalizer="mmse"` /
`cfo_estimator="schmidl_cox"` as interchangeable components — those are
algorithms buried inside monolithic, stateful objects (`ofdmflexframesync`),
not swappable strategies. spectracuda's job is:

```
liquid-dsp  →  extract algorithms  →  modernize into swappable abstractions  →  GPU implementation
```

The two places liquid-dsp *does* already expose a clean "one object type, many
named schemes" model — `modem` (53 schemes) and `fec` (28 schemes) — are
copied almost verbatim. Everywhere else (sync, CFO, channel estimation,
equalization), the algorithm is a reference, the API is new.

## Layered model

**Layer 1 — fixed infrastructure** (configured by params, not swappable —
there's no competing "algorithm" to choose between):
- FFT/IFFT + cyclic-prefix add/remove
- Resource-grid / pilot-subcarrier mapping
- Framing / deframing, CRC, packet stats

**Layer 2 — swappable strategies** (real algorithmic alternatives exist, per
the liquid-dsp inventory and standard OFDM receiver design):
- `sync`: `ZadoffChuSync`, `SchmidlCoxSync` (reference: `qdetector_create_linear`
  for ZC; `ofdmframesync.c`'s S0/S1 preamble logic for Schmidl-Cox — liquid-dsp
  doesn't expose the latter as a reusable block, so it's extracted from source)
- `cfo`: `SchmidlCoxCFO`, `PilotBasedCFO` — **kept as its own strategy class**,
  independent of which `sync` is used (decided over the alternative of folding
  CFO into the sync object's output — liquid-dsp itself only exposes CFO as a
  readout off the frame sync object, but we're deliberately decoupling it here
  so any sync can pair with any CFO method)
- `channel_estimator`: `LSChannelEstimator`, (later) `MMSEChannelEstimator` —
  no liquid-dsp precedent; designed from standard pilot-based LS/MMSE
  reference derivations (e.g. 802.11/3GPP-style), same gap noted in the
  inventory
- `equalizer`: `ZFEqualizer`, `MMSEEqualizer` — liquid-dsp's `eqlms`/`eqrls`
  are sample-adaptive single-carrier equalizers, not per-subcarrier
  frequency-domain ZF/MMSE; again designed from reference, not ported
- `modem`: `Modem("qpsk" | "qam16" | "qam64" | ...)` — one class, scheme-name
  string, mirrors `modem_create(scheme)` directly
- `fec`: `FEC("conv_v27" | "rs_m8" | "ldpc_648_r12" | ...)` — one class,
  scheme-name string, mirrors `fec_create(scheme)` directly. `conv_v27`/
  `rs_m8` match liquid-dsp's own `LIQUID_FEC_CONV_V27`/`LIQUID_FEC_RS_M8`
  exactly (Polar still doesn't exist in liquid-dsp either, so its
  absence here is still parity, not a gap). LDPC, however, is now
  implemented as a deliberate scope expansion BEYOND liquid-dsp parity
  — the full 12-variant IEEE 802.11n QC-LDPC family — following the same
  "no liquid-dsp precedent → design from a standard reference instead of
  deferring" reasoning already used for `LSChannelEstimator`/
  `ZFEqualizer`/`MMSEEqualizer` above. It's also the first FEC codec in
  this codebase whose iterative decode core (normalized min-sum belief
  propagation) is genuinely GPU-batched across `(n_batch, ...)` via
  gather-only `self.xp` operations, rather than Reed-Solomon's per-
  codeword Python loop — see `spectracuda/fec/ldpc.py`'s module
  docstring.

**Layer 3 — tx+rx composition**: **one** class, `Ofdm`, wires Layer 1 +
Layer 2 into the full chain — both directions, one constructor. This
revises the original `OfdmRx`/`OfdmTx` split below: tx and rx share
almost every parameter (subcarrier allocation, CP length, preamble/
training content), and liquid-dsp's own idiom of constructing
`ofdmflexframegen`/`ofdmflexframesync` by hand with matching
M/cp_len/taper_len/subcarrier values (see
`reference/liquid-dsp/examples/ofdmflexframesync_example.c`) is a
duplication C has to live with, not a pattern worth reproducing in
Python — one object owning both directions also removes a real class of
bug (tx/rx drifting out of sync on preamble/training content, which had
to be hand-matched across separate variables before `Ofdm` existed; see
`spectracuda/pipeline/ofdm.py`).

## The string-or-instance rule (applies to every Layer 2 param)

Every swappable-strategy constructor argument accepts **either** a scheme
string (resolved via a small registry to a default-configured instance) **or**
a pre-built instance (for custom params or subclassing) — the sklearn/Keras
pattern. This is what keeps the original pitch's `sync="zc"` ergonomics alive
while still allowing `sync=ZadoffChuSync(seq_root=25, seq_len=63)`:

```python
ofdm = Ofdm(
    fft_size=256, n_pilot=6, n_data=200, cp_len=32,
    sync="schmidl_cox",                       # string -> SchmidlCoxSync() with defaults
    cfo=SchmidlCoxCFO(fft_size=256),          # or a configured instance
    channel_estimator="ls",
    equalizer="mmse",
    modem="qpsk",
    fec="none",                                # only "none" works today -- Phase 3
    backend="cupy",                            # "numpy" | "cupy"
)
tx_iq = ofdm.generate_frame(payload_bits)      # batch in, iq out
result = ofdm.rx_process(rx_iq)                 # batch in, {"bits": ..., "cfo_estimate": ..., ...} out
```

## Precision: complex64 everywhere, float64 nowhere in the compute path

Every block computes in `complex64`/`float32`, never `complex128`/`float64` —
Jetson-class GPUs have drastically lower FP64 throughput than FP32, so any
code path that silently upcasts is a real, hidden performance bug on the
actual target hardware, not just a style nit.

This was violated in two places found during development, both fixed:
- `numpy.fft.fft`/`ifft` has **no single-precision code path at all** —
  it always computes internally in double precision and returns
  `complex128` regardless of input dtype (numpy's own documented
  behavior). `OfdmModulator`/`OfdmDemodulator` (`ofdm/fft.py`) now
  explicitly `.astype("complex64")` after every fft/ifft call. This also
  closed a cross-backend correctness gap: cuFFT (via CuPy) *does*
  preserve `complex64`, so numpy and cupy backends were silently
  disagreeing on output precision for the identical config.
- `Modem`'s `_pam_level` built its PAM-level array in `float64` before
  downcasting the final symbol to `complex64` — paying the double-
  precision cost on every single symbol modulated/demodulated for
  nothing. Now builds directly in `float32`.

`Ofdm`'s `iq_dtype` param (`"float16"` | `"float32"`, default `"float32"`)
simulates a finite-resolution ADC/DAC by quantizing IQ samples to the
requested resolution at the tx-output/rx-input boundaries only — not by
running the DSP math itself at half precision. That's a deliberate,
necessary compromise: no FFT library this project can rely on (not numpy,
not CuPy's standard API) actually transforms half-precision complex data,
and numpy/cupy have no native `complex32` dtype at all. A genuine
float16-compute mode would need a hand-built complex16 wrapper type with
its own arithmetic and an FFT that upcasts internally — a much larger,
invasive change touching nearly every block, for an FFT that still
wouldn't run at 16 bits. Boundary quantization answers the real question
this is usually asked for (how much does N-bit ADC/DAC resolution hurt
BER) while keeping the compute path exactly the complex64 this whole
project is built around.

Layer 1 blocks are never string/registry-driven — just plain constructor
params (`fft_size=`, `cp_len=`), since there's nothing to choose between.

## Backend abstraction

Every block holds `self.xp` (`numpy` or `cupy`, selected once at `Ofdm`
construction and threaded down to every stage). This is not a convenience —
it's what makes the whole correctness test suite runnable in CI with no GPU
present at all (Jetson hardware won't be in a normal CI runner), matching the
"NumPy fallback mode" requirement from the Phase 0 feasibility plan.

Control loops that are inherently sequential, small-signal, per-symbol
(residual carrier-phase tracking, AGC, fine timing feedback) stay CPU-resident
regardless of the selected backend — GPU throughput doesn't help a tight
feedback loop, and this boundary should stay explicit rather than be
"optimized" onto GPU later without re-justifying it.

## Batch-shape contract

Every Layer 1/2 block's `process()`/`__call__` operates on a batch dimension
as the primary shape (`(n_batch, fft_size)` complex array in/out, or
equivalent) — never a single scalar sample/symbol. This is the hard
requirement established during the feasibility discussion: GPU acceleration
on Jetson Orin Nano only pays off once operations are batched across symbols/
packets/antenna streams; kernel-launch and Python/CuPy dispatch overhead
dominates at batch=1. Each block's docstring must state its exact batch-shape
contract.

This is a deliberate departure from liquid-dsp's `execute(samples, n)` +
callback streaming idiom, which assumes arbitrary-chunk incremental feeding.
`Ofdm.rx_process()` — the core receive API this batch-shape contract governs
— stays batch: one call, one already-bounded frame's worth of IQ.

A separate, additive streaming receiver now exists alongside it —
`Ofdm.rx_streaming(chunk)` — for exactly the case liquid-dsp's idiom was
built for: a real receive chain that gets arbitrary, unaligned chunks of a
continuous sample stream rather than one cleanly-bounded frame at a time.
It's modeled directly on liquid-dsp's own `ofdmframesync_execute()` state
machine (`SEEKPLCP → PLCPSHORT0/1 → PLCPLONG → RXSYMBOLS`) rather than
invented from scratch, and reuses the exact same header/payload-decode
logic `rx_process()` calls internally (see `Ofdm._decode_header_from_sync()`/
`_decode_payload_from_header()`) — not a parallel, drifting implementation.
It does not replace `rx_process()` or the batch-shape contract above (every
Layer 1/2 block underneath is still batch-shaped) — it's the one place in
the API surface that accumulates single-stream state across calls, by
design, because that's what a real streaming receive chain requires. See
`docs/mac.md`'s "How MAC data actually crosses the air" section and
`docs/todo.md` #2.5 for the full design and how `spectracuda.mac` drives it.

## Directory layout (unchanged from the earlier design pass — already
organized by pipeline stage, not by liquid-dsp's 20 C modules)

```
spectracuda/
├── backend.py          # numpy/cupy selection, self.xp threading
├── registry.py         # string -> default-instance resolution
├── block.py            # Block base class, batch-shape contract
├── sync/                # ZadoffChuSync, SchmidlCoxSync
├── cfo/                  # SchmidlCoxCFO, PilotBasedCFO
├── ofdm/                  # FFT/IFFT+CP (Layer 1), resource grid, pilot extract
├── channel/                # LSChannelEstimator, (later) MMSE
├── equalizer/                # ZFEqualizer, MMSEEqualizer
├── modem/                     # Modem(scheme)
├── fec/                        # FEC(scheme), CRC
├── framing/                     # deframer, packet stats (Layer 1)
├── pipeline/
│   └── ofdm.py                    # Ofdm -- the whole tx+rx chain, one class
├── mac/                             # NEW LAYER, sits above Ofdm/framing --
│   ├── mac.py                         # Mac(mode="tm"|"um"|"am", ...) -- either
│   │                                     # PHY-agnostic, or owns its OWN Ofdm
│   │                                     # (ofdm_kwargs=...) for genuinely
│   │                                     # independent multi-object topologies
│   ├── tm.py / um.py / am.py            # combined MAC+RLC (docs/mac.md):
│   │                                     # pure, PHY-agnostic entity logic
│   ├── pdu.py / reassembly.py             # simplified/custom PDU format,
│   │                                        # segmentation/reassembly
│   ├── bind.py / quality.py / capacity.py   # BIND handshake, link-quality
│   │                                          # reporting, derived capacity
│   └── session.py                              # MacLink -- wires Mac to a
│                                                  # real, SHARED Ofdm (+
│                                                  # optional sim.Channel)
└── sim/                             # NOT part of the real chain -- test/
    └── channel.py                   # simulation only. Channel: impairment
                                      # simulator (awgn/multipath/cfo),
                                      # mirrors liquid-dsp's channel_cccf
```

`mac/` is new ground beyond liquid-dsp parity -- liquid-dsp has no MAC/RLC
concept at all (single-link OFDM framing only). See `docs/mac.md` for the
full design; `MacLink` (session.py) plays the same "demonstration/
integration harness, not the real chain" role `sim/channel.py` already
does -- the `Mac`/`TmEntity`/`UmEntity`/`AmEntity` objects themselves stay
pure, PHY-agnostic, reusable logic.

## v1 object list

```
Ofdm                              # replaces the earlier OfdmRx/OfdmTx split
Modem, FEC
ZadoffChuSync, SchmidlCoxSync
SchmidlCoxCFO, PilotBasedCFO
LSChannelEstimator
ZFEqualizer, MMSEEqualizer
```

Note: this is the target **API surface**; build order still follows the
earlier phased plan's dependency sequence (FFT/channel-est/equalizer/modem
first, validated against synthetic perfectly-timed symbols with no sync
needed yet; sync/CFO second; FEC third) — the object list above is what
Phase 1-3 collectively produce, not a suggestion to build top-down in this
order.
