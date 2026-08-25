# spectracuda

**Docs: [spectracuda.readthedocs.io](https://spectracuda.readthedocs.io/en/latest/)**

GPU-accelerated, liquid-dsp-inspired SDR PHY (+ MAC) framework for NVIDIA
Jetson (Orin Nano first, NX/AGX Orin and desktop CUDA GPUs as additional
targets).

Not a CUDA port of liquid-dsp. liquid-dsp never exposed swappable
`channel_estimator=`/`equalizer=`/`cfo=` strategies — those are algorithms
buried inside monolithic C objects, not interchangeable components.
spectracuda extracts the algorithms liquid-dsp *does* have a reference
implementation for, ports the ones that are self-contained (CRC, the
Zadoff-Chu-adjacent `qdetector` idea, the `interleaver` algorithm),
re-derives the ones that only exist wrapped around an external C library
(Viterbi, Reed-Solomon), and designs the rest from standard references
where liquid-dsp has no precedent at all (LS/MMSE channel estimation,
ZF/MMSE equalization, LDPC, the MAC layer) — everything batch-first,
GPU-first, with a NumPy fallback so it runs anywhere.

## Install

```sh
pip install -e ".[dev]"
pytest
```

No GPU is required for development or the test suite — everything runs on
`backend="numpy"` by default when no working CUDA runtime is detected.
Install the `cuda` extra (`pip install -e ".[cuda]"`) on a CUDA-capable
machine (e.g. the Jetson) to exercise `backend="cupy"`. Every block accepts
`backend="numpy"|"cupy"` explicitly if you want to pin it.

## Quick start

One `Ofdm` object owns the whole tx+rx chain — build it once, then call
`generate_frame()`/`rx_process()`:

```python
import numpy as np
from spectracuda.pipeline import Ofdm

ofdm = Ofdm(
    fft_size=256, n_pilot=8, n_data=200, cp_len=32,
    modem="qpsk", fec="conv_v27", crc="crc32",
    sync="schmidl_cox", cfo="schmidl_cox",
    channel_estimator="ls", equalizer="mmse",
)

payload_bits = np.random.default_rng(0).integers(0, 2, size=(1, 64)).astype("uint8")
tx_iq = ofdm.generate_frame(payload_bits)      # -> (1, n_samples) complex64 IQ
result = ofdm.rx_process(tx_iq)                # -> dict, see schema below

result["frame_found"]   # True — a real frame was detected
result["bits"]          # decoded payload bits (None if frame_found is False)
result["crc_valid"]     # per-item bool array, or None if crc="none"
result["evm"], result["rssi_db"], result["cfo_estimate"], result["header"]
```

`rx_process()`'s return value is a **stable, fully-enumerated dict** —
every key above is always present; most are `None` when `frame_found` is
`False` (e.g. you fed it noise with no real frame in it) rather than the
call raising or returning a partial/ad hoc structure. See `Ofdm.rx_process`'s
own docstring for the complete schema.

A receiver never needs to be told the transmitter's `modem=`/`fec=`/`crc=`
choice — `Ofdm` carries all of that in a liquid-dsp-style 112-bit frame
header and resolves it dynamically from the decoded frame on receive, the
same way a real, separate receiving device would have to.

### Swappable strategies

Every algorithmic stage is swappable via a scheme-name string (or pass an
already-configured instance for custom tuning):

| Constructor arg | Options | liquid-dsp precedent? |
|---|---|---|
| `modem=` | `"bpsk"`, `"qpsk"`, `"qam16"`, `"qam64"`, `"qam256"` | yes — `modem_create(scheme)` |
| `fec=` (`fec0`, inner) | `"none"`, `"conv_v27"`, `"rs_m8"`, `"ldpc_648_r12"`…`"ldpc_1944_r56"` (12 rate/length variants) | conv/RS yes; LDPC is a deliberate non-liquid-dsp addition |
| `crc=` | `"none"`, `"checksum"`, `"crc8"`, `"crc16"`, `"crc24"`, `"crc32"` | yes — byte-exact port of `crc.c` |
| `sync=` | `"schmidl_cox"`, `"zc"` (Zadoff-Chu) | no menu to match — designed here |
| `cfo=` | `"schmidl_cox"`, `"pilot_based"` | no menu to match — designed here |
| `channel_estimator=` | `"ls"`, `"mmse"` | no menu to match — designed here |
| `equalizer=` | `"zf"`, `"mmse"` | no menu to match — designed here |

**Not every `sync=`/`cfo=` pairing is valid**: `SchmidlCoxCFO` depends on
the preamble's repeated-halves structure and only pairs with
`sync="schmidl_cox"`. `PilotBasedCFO` has no such dependency and works
with either `sync=` value. `sync="zc"` + `cfo="schmidl_cox"` reliably
fails — use `cfo="pilot_based"` with it instead.

### Two-stage (concatenated) FEC + interleaving

A second, outer FEC stage (`fec1=`) and an interleaver between the two
stages are both supported, mirroring liquid-dsp's own `packetizer`
inner/outer composition:

```python
ofdm = Ofdm(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32, modem="qpsk",
    fec="rs_m8", fec1="conv_v27",                       # fec0=inner, fec1=outer
    interleaver="block", interleaver_kwargs={"unit_bits": 8},
)
```

**Getting `fec`/`fec1` the right way round matters.** `fec0` (`fec=`) is
applied first on encode and decoded *last*; `fec1` is applied second
(closest to the channel) and decoded *first*. For the classic "Viterbi
faces the channel, Reed-Solomon mops up its bursty decode-error residue"
benefit, Viterbi must be `fec1` and RS must be `fec0` — i.e.
`fec="rs_m8", fec1="conv_v27"`, not the reverse. `interleaver_kwargs={"unit_bits": 8}`
also matters, not just style: interleaving at individual-bit granularity
can make a byte-oriented outer code (RS) *worse off*, not better — it
fragments a few concentrated byte errors into many more of them.
`interleaver=` picks from `"block"` (recommended default), `"permutation"`,
`"convolutional"`, or `"liquid"` (a verified port of liquid-dsp's own
`interleaver.c` algorithm). See `spectracuda/interleaver/`'s docstrings
and `docs/todo.md` §1.12 for the full derivation — this is the one place
in the library where a wrong assumption was made and caught during
development, worth reading if you're deciding how to configure it.

Interleaver choice is the one setting **not** carried in the frame header
(liquid-dsp doesn't signal it over the air either) — both ends of a link
must already agree on `interleaver=`/`interleaver_kwargs=` out of band.

### MAC layer (segmentation, sequencing, retransmission)

`spectracuda.mac` sits above `Ofdm`, named after 3GPP RLC's TM/UM/AM modes
(not a 3GPP-accurate implementation — the real TM/UM/AM *behavior* is what
carries over: raw passthrough / numbered best-effort / numbered-with-ARQ).
Two ways to use it:

**`MacLink`** — a demonstration/integration harness (same role as
`spectracuda.sim.Channel`) wiring a `Mac(mode=...)` entity to one shared
`Ofdm` object playing both the tx and rx role:

```python
from spectracuda.mac import MacLink

link = MacLink(ofdm, mode="am")   # requires ofdm.crc != "none"
link.bind()                       # handshake; send() refuses to run before this
delivered = link.send(sdu_bits)   # segments, transmits, retransmits (AM), reassembles
link.exchange_link_quality()      # RSSI/EVM/delivery-ratio report round trip
```

**`Mac(mode=, ofdm_kwargs=)`** — each `Mac` owns its own, genuinely
independent `Ofdm` (`hw1.ofdm is not hw2.ofdm`, always). Two real
endpoints are just two separate `Mac(...)` calls with **no shared object
identity at all** — the only thing that crosses between them is an IQ
array, exactly like two real radios:

```python
from spectracuda.mac import Mac

phy = dict(fft_size=256, n_pilot=8, n_data=216, cp_len=32, modem="qpsk",
           crc="crc16", sync="schmidl_cox", cfo="schmidl_cox")
hw1 = Mac(mode="um", ofdm_kwargs=phy)   # builds & owns its OWN Ofdm
hw2 = Mac(mode="um", ofdm_kwargs=phy)   # a second, independent Ofdm

req_iq = hw1.build_bind_request()               # real 3-message handshake,
resp_iq = hw2.handle_bind_request_iq(req_iq)     # hw2 evaluates against ITS
assert hw1.handle_bind_response_iq(resp_iq)      # OWN capacity, can reject

for iq in hw1.send_iq(sdu_bits):                # segment, encode, IQ out
    delivered = hw2.receive_iq(iq)              # decode, reassemble, in
```

A full **bidirectional AM** link (retransmission needs a STATUS pdu
travelling the opposite direction from the DATA it reports on) needs
**four** `Mac`/`Ofdm` objects, two per endpoint, one per direction — see
`docs/mac.md`'s "How MAC data actually crosses the air" section for the
full model and diagram, and
[`examples/mac_bidirectional_am_batch_demo.py`](examples/mac_bidirectional_am_batch_demo.py)
/
[`examples/mac_bidirectional_am_streaming_demo.py`](examples/mac_bidirectional_am_streaming_demo.py)
for the complete, runnable version — including a dropped PDU, real
STATUS-driven recovery, and a full hex/header printout of every pdu that
crosses each link.

**Batch vs. streaming receive**: `Ofdm.rx_process(iq)` decodes one
already-bounded frame's worth of IQ in a single call — the natural shape
for simulation, and what every `Mac` method above uses internally.
`Ofdm.rx_streaming(chunk)` is the real-receiver-shaped alternative: feed
it arbitrary, unaligned chunks of a continuous sample stream (any size,
any alignment) and it finds/tracks/assembles the frame itself via an
internal state machine modeled on liquid-dsp's own
`ofdmframesync_execute()`, returning `None` while still searching and a
decoded result the instant a frame completes. `Mac`'s own methods are
`rx_process()`-only; see `examples/mac_streaming_demo.py` and
`examples/mac_bidirectional_am_streaming_demo.py` for how to drive
`rx_streaming()` underneath the same MAC logic by hand.

The `TmEntity`/`UmEntity`/`AmEntity` classes underlying all of the above
are pure, PHY-agnostic logic, independently usable/testable without any
`Ofdm` involved. See `docs/mac.md` for the full design and the real bugs
found building it.

### Channel simulation

```python
from spectracuda.sim import Channel

channel = Channel(snr_db=20.0, multipath_taps=Channel.random_multipath_taps(3, seed=0),
                   cfo=0.05, cfo_fft_size=256, seed=0)
rx_iq = channel.process(tx_iq)
```

## The book

**The OFDM Field Guide** — a PySDR-style tour of this codebase's real OFDM
PHY chain, assuming you already know what an IQ sample and an FFT are
(PySDR itself covers that ground). Every figure is generated by actually
running spectracuda's own classes, never a hand-drawn stand-in. Two
published forms of the same Part I content:

- **[spectracuda.readthedocs.io](https://spectracuda.readthedocs.io/en/latest/)**
  — the canonical, always-current site (built by Read the Docs from
  `docs/` on every push to `main`; see `docs/conf.py`/`.readthedocs.yaml`),
  also hosting the full reference doc set (`architecture`/`mac`/`ldpc`/
  `todo`) alongside the book chapters.
- [**The OFDM Field Guide (standalone artifact)**](https://claude.ai/code/artifact/3a82c56d-fd78-4bd2-9ed9-2aa8944036a2)
  — the original single self-contained HTML build, figures base64-embedded
  inline, no server required. Fully reproducible from
  `docs/book_figures/` (`pip install -e ".[docs]"` first, for matplotlib —
  dev-only, never a runtime dependency):

```sh
python docs/book_figures/generate_figures.py   # regenerate the *.png figures
python docs/book_figures/build_book.py         # rebuild book.html (base64-embeds them)
```

`book_template.html` is the editable source; `book.html` is the
self-contained build. Part I (the PHY chain: sync, CFO, channel
estimation, modem) is written in both forms; Part II (the MAC layer) is
outlined in `docs/book/roadmap.md` but not yet written — the book says so
honestly rather than pretending otherwise.

## Docs map

- [`docs/architecture.md`](docs/architecture.md) — the design: layered
  API model (Layer 1 fixed infra / Layer 2 swappable strategies / Layer 3
  `Ofdm` composition), the string-or-instance rule, backend abstraction,
  batch-shape contract.
- [`docs/liquid-dsp-api-inventory.md`](docs/liquid-dsp-api-inventory.md) —
  what liquid-dsp actually exposes, used as the algorithm reference
  throughout.
- [`docs/mac.md`](docs/mac.md) — the MAC layer (TM/UM/AM), as actually
  built, including real bugs found and fixed during implementation; see
  its "How MAC data actually crosses the air" section for the
  batch-vs-streaming, two-node/four-node bidirectional explanation.
- [`examples/`](examples/) — runnable scripts, from a single tx-only/
  rx-only pair through full bidirectional AM over 4 independent
  `Mac`/`Ofdm` objects, both batch (`rx_process()`) and streaming
  (`rx_streaming()`) receive.
- [`docs/ldpc.md`](docs/ldpc.md) — the LDPC implementation plan and
  final state (12-variant IEEE 802.11n QC-LDPC family).
- [`docs/todo.md`](docs/todo.md) — the concrete, checkable gap list
  against liquid-dsp and against this project's own stated scope; the
  living source of truth for what's done vs. open, including the reasoning
  behind real bugs/corrections found along the way (not just a task list).

## Status

The core OFDM PHY chain (sync/CFO/channel estimation/equalization/modem/
FEC/CRC/framing), the LDPC and two-stage-FEC-with-interleaving FEC
extensions, and the MAC layer are all implemented and tested — see
`docs/todo.md` for the precise, itemized state of every piece (what's
done, what's a deliberate scope boundary, and what's still genuinely
open, e.g. a not-yet-root-caused long-frame reliability question at small
`fft_size`, §1.11).

`backend="cupy"` has now actually run on real CUDA hardware (a Colab
Tesla T4, 2026-08-25) — not just designed against the API, as it was
before this point. A direct `Ofdm(backend="cupy")` `generate_frame()` →
`rx_process()` round trip produced genuine `cupy.ndarray` IQ and a
correct CRC, and the full suite went from 600 passed/1 skipped (skip
being the one cupy-parametrized test, `tests/conftest.py`'s `backend`
fixture, with no CUDA runtime to run against) to **601 passed, 0
skipped** on that GPU. Jetson-specific behavior (actual throughput, the
CPU-resident stages noted in `docs/architecture.md`) is still
unvalidated — a T4 on Colab confirms CUDA correctness, not Jetson
performance.

## Development

```sh
pytest                          # full suite
pytest tests/test_ofdm_class.py -v   # one area
pytest -k "ldpc or interleaver" -v   # by keyword
```

Every test defaults to `backend="numpy"` so the suite runs identically
with or without a GPU present.
