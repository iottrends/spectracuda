# MAC layer: TM / UM / AM modes

Status: **implemented.** See `spectracuda/mac/` (`pdu.py`, `reassembly.py`,
`tm.py`, `um.py`, `am.py`, `mac.py`, `bind.py`, `quality.py`, `capacity.py`,
`session.py`) and `tests/test_mac_{pdu,tm,um,am,bind,quality,session,
two_units_simple,half_duplex,bidirectional_am}.py` (97 tests) for the
final state. Two ways to use it: `MacLink` (`session.py`), a demonstration
harness wiring one shared `Ofdm` to both a `tx_mac`/`rx_mac` role, and
`Mac(mode=, ofdm_kwargs=)`, which owns its own independent `Ofdm` — the
model that makes genuine multi-object (two-node, four-node) topologies
possible. See "How MAC data actually crosses the air" below for the
reader-facing explanation of how that works, batch and streaming; the
rest of this doc is a chronological design/bugs-found log.

## Context

A combined MAC+RLC layer (named after 3GPP RLC's TM/UM/AM modes; this
project does not separate RLC/MAC into distinct sublayers — an explicit
choice, not spec-accurate 3GPP layering), sitting above the OFDM PHY chain
(`spectracuda.pipeline.Ofdm`) and its framing layer (`spectracuda.framing`).
Simplified/custom PDU wire format, not derived from a 3GPP spec table — but
the actual TM/UM/AM *behavior* (raw passthrough / numbered best-effort /
numbered-with-retransmission) is the genuine distinction real RLC draws.

## Scope boundary

- **Point-to-point only** — this project is a single `Ofdm` tx/rx link, no
  multi-UE scheduling concept, so "MAC" here means segmentation/reassembly
  + sequence numbering + (AM only) retransmission, not resource allocation.
- **ARQ, not HARQ** — AM retransmits a fresh, independently-decoded copy of
  a NACKed PDU; no soft-combining across retransmissions (would need raw
  LLRs/IQ kept across rounds, conflicting with this codebase's existing
  hard-decision-bits-everywhere FEC interface — see `fec/ldpc.py`).

## How MAC data actually crosses the air (batch and streaming)

This section is the reader-facing explanation — everything else in this
doc is a chronological log of what was built and why. Start here if you
want the mental model rather than the history.

### Two independent `Mac` objects, not one shared object playing both roles

`Mac(mode="um"|"am", ofdm_kwargs={...})` builds and owns its own `Ofdm`.
Two real endpoints are just two separate `Mac(...)` calls — `hw1.ofdm is
not hw2.ofdm`, always, by construction. Nothing is shared: no config
object, no class instance, no in-process shortcut. The **only** thing
that crosses between them is an IQ array, handed from one object's
`generate_frame()`/`send_iq()` output to the other's `rx_process()`/
`receive_iq()` input — exactly what a real antenna-to-antenna link would
carry, simulated as a plain function call (optionally through
`spectracuda.sim.Channel` for realistic noise/multipath/CFO in between).
This is deliberate — see the design discussion this doc's later sections
narrate: "we dont share any config between the transmitter and receiver
... everything on the receiver has to be decoded on its own."

```
hw1_mac.send_iq(sdu_bits)  ->  [iq_frame, ...]
                                     │
                              (channel.process(), optional)
                                     ▼
hw2_mac.receive_iq(iq_frame)  ->  decoded SDU bits
```

`send_iq()`/`receive_iq()` are thin: `send_iq()` is
`self._impl.transmit(sdu) -> [self.ofdm.generate_frame(pdu) for pdu in
pdus]`; `receive_iq()` is `self.ofdm.rx_process(iq) -> CRC/frame_found
check -> self._impl.receive(bits)` (or `receive_data()` for AM — see
below). Everything above the `Ofdm` call is pure PHY-agnostic MAC logic
(`TmEntity`/`UmEntity`/`AmEntity`); everything at/below it is the real
OFDM chain. Two `Mac`s never touch each other's internals — only the IQ.

### Binding: a real handshake, not a formality

Before any `send_iq()` call is allowed, both sides must complete a
3-message exchange, itself carried as real IQ over the same `Ofdm`:

```python
req_iq  = mac_a.build_bind_request()          # encodes mode/capacity/window/retries
resp_iq = mac_b.handle_bind_request_iq(req_iq) # mac_b's OWN accept/reject decision
accepted = mac_a.handle_bind_response_iq(resp_iq)
```

`handle_bind_request_iq()` evaluates the request against `mac_b`'s own,
independently-derived `max_segment_bits` (`bind.evaluate_bind_request()`)
— it can genuinely reject (not silently clamp) a request that asks for
more than it can carry. `send_iq()` raises `ValueError` if called before
`bound` is `True`.

### Why AM needs a SECOND `Mac`/`Ofdm` pair per direction

TM and UM are one-directional: `send_iq()` on one side, `receive_iq()` on
the other, done. AM adds retransmission, which means the receiving side
has to send something *back* — a STATUS pdu reporting which PDUs arrived.
That STATUS pdu is real traffic that needs its own PHY frame, physically
travelling the opposite direction. A single `Mac`/`Ofdm` object's
`send_iq()`/`receive_iq()` pair can't carry traffic in both directions at
once (one `Ofdm` per object, one direction of `generate_frame()` calls
per link in this design) — so a full bidirectional AM link needs **four**
`Mac`/`Ofdm` objects, two per endpoint, one per direction:

```
                       Ofdm_A  (forward PHY: hw1 -> hw2)
        HW1.tx_mac ───────DATA──────────▶ HW2.rx_mac
             ▲                                │
             │                                │ build_status()
             │        Ofdm_B (reverse PHY)     ▼
        HW1.rx_mac ◀──────STATUS────────── HW2.tx_mac
             │
             │ receive_status()  <- a DIFFERENT Mac object than the one
             ▼                       that physically decoded the STATUS!
        HW1.tx_mac ──────retransmit───▶ HW2.rx_mac
```

The wrinkle worth internalizing: the STATUS pdu is **decoded** by
`hw1_rx_mac` (it owns `Ofdm_B'`, the receive side matching `Ofdm_B`), but
its *content* is only meaningful to `hw1_tx_mac` (the object that
actually holds the retransmission buffer for the DATA it sent). So the
example code explicitly does:

```python
status_bits = <decode iq arriving on hw1_rx_mac.ofdm>   # hw1_rx_mac decodes it
retransmits = hw1_tx_mac.receive_status(status_bits)     # hw1_tx_mac acts on it
```

— one Mac object physically receiving, a *different* Mac object at the
same endpoint logically consuming what it received. This only shows up
once you have independently-owned `Ofdm`s; with one shared `Ofdm` (the
`MacLink` model, see below) the question never arises because there's
only one receive path to begin with.

`Mac.receive_iq()` can't be used for the STATUS leg at all — for AM it
always assumes an arriving frame is DATA (`self._impl.receive_data()`)
and raises on anything else. So this whole routing is written out
manually in the example code (no orchestrator class does it for you, by
explicit design choice) via a small local helper mirroring
`Mac._rx_one_frame()`'s logic without assuming the pdu_type.

See `examples/mac_bidirectional_am_batch_demo.py` /
`mac_bidirectional_am_streaming_demo.py` for the complete, runnable
version of this diagram — including a deliberately dropped PDU, the real
STATUS-driven retransmission that recovers it, and a full hex/header
printout of every BIND_REQUEST, BIND_RESPONSE, DATA, and STATUS pdu that
crosses each link (run either script directly to see it).

### Batch (`rx_process()`) vs. streaming (`rx_streaming()`)

Both example scripts build the *identical* 4-object model and run the
*identical* MAC logic (`transmit()`/`receive_data()`/`build_status()`/
`receive_status()`) — the only thing that differs is how each receiving
`Ofdm` turns arriving IQ samples into decoded bits:

- **`rx_process(iq)`** — one call, the whole frame's worth of IQ handed
  over at once, already correctly sliced (the caller knows exactly where
  one `generate_frame()` output ends, because it's the return value of a
  single Python call). This is the natural shape for simulation/testing,
  and it's what `send_iq()`/`receive_iq()`/`build_bind_request()`/etc.
  use internally — but it's not how a real antenna ever hands samples to
  a receiver.
- **`rx_streaming(chunk)`** — a real receive chain never gets "one frame,
  cleanly bounded" — it gets a continuous stream of samples in arbitrary,
  unaligned pieces (64, 128, 1024 samples, whatever the radio/USB/network
  layer happens to deliver next) and has to find the frame itself. Feed
  it one chunk at a time; it returns `None` while still searching/
  accumulating, and returns the same `rx_process()`-shaped result dict
  the instant a complete frame finishes decoding *during that call* —
  possibly mid-chunk. Internally it's a real state machine (`SEEKING →
  WAITING_HEADER → WAITING_PAYLOAD`), modeled directly on liquid-dsp's
  own `ofdmframesync_execute()` state machine (`SEEKPLCP → PLCPSHORT0/1 →
  PLCPLONG → RXSYMBOLS`, see `reference/liquid-dsp/src/multichannel/src/
  ofdmframesync.c`) rather than invented from scratch. A bad/false-
  positive frame doesn't kill the stream — it's discarded and the state
  machine resumes searching, since a real streaming receiver has to keep
  running indefinitely across many frames, unlike `rx_process()`, which
  raises on a bad decode.

Because `Mac` itself has no streaming-aware IQ-level method (`send_iq()`/
`receive_iq()` are `rx_process()`-only — see docs/todo.md #2.5), the
streaming example writes its own small `_stream_receive_iq()`/
`_stream_rx_raw_pdu()` helpers that feed `ofdm.rx_streaming()` chunk by
chunk and then hand the decoded bits to the exact same
`receive_data()`/`receive_status()` calls the batch version uses — MAC
logic is completely unaware of which receive mechanism produced the
bits it's given. Every *receiving* `Ofdm` needs its own independent
streaming state (`ofdm.reset_stream()`, called once per object, not once
per frame) — the same way each `Mac`'s `UmEntity`/`AmEntity` already
keeps its own independent SN/reassembly state.

## Design, as built

### `spectracuda/mac/pdu.py` — the 32-bit UM/AM header
`TYPE`(1) + `SI`(2) + `SN`(10, modulo-1024) + `SO`(16, bit offset — not
byte, matching this project's bit-oriented convention throughout) + 3
reserved bits. `sn_add`/`sn_precedes` do real modulo-1024 arithmetic
(wraparound-safe), not a naive `<` comparison. TM PDUs have no header at
all — a TM PDU *is* the SDU.

### `spectracuda/mac/reassembly.py` — `Segmenter` / `ReassemblyBuffer`
`Segmenter.segment()` requires the SDU's bit length to be a multiple of 8
(see "Bugs found" below for why) and guarantees every produced segment,
including the final possibly-shorter one, is also byte-aligned as a
result. `ReassemblyBuffer` tracks one in-progress multi-segment SDU at a
time plus a bounded pending window; a persistent gap past the window
abandons the stalled SDU and resyncs — a stated scope limitation (no
concurrent interleaved multi-SDU reassembly), not a bug.

### `spectracuda/mac/{tm,um,am}.py` — the three entities
- `TmEntity`: raw passthrough; requires the SDU to already fit in one PHY
  frame (matches real TM's actual constraint — no invented out-of-band
  length mechanism that would stop being "transparent").
- `UmEntity`: `Segmenter` + SN header + `ReassemblyBuffer`, no
  retransmission — a lost segment means its SDU is never completed.
- `AmEntity`: composes a `UmEntity` internally for the DATA-PDU path, adds
  a retransmission buffer (`transmit`/`receive_status`, the sender role)
  and STATUS-PDU reporting (`receive_data`/`build_status`, the receiver
  role) — see its own module docstring for why AM's real API is four
  methods, not two: a full-duplex link needs one `AmEntity` per direction
  per endpoint. Status reports are a `base_sn` (the receiver's cumulative
  reassembly point) + a fixed-width received/missing bitmap over a
  window — this project's own design, not a 3GPP STATUS PDU format, but
  the same ACK_SN + NACK-list concept. Bounded `max_retries`: exceeding it
  drops the PDU from the retransmission buffer into `failed_sns` rather
  than retrying forever.

### `spectracuda/mac/mac.py` — `Mac(mode=, max_segment_bits, **kwargs)`
One entry point matching `FEC(scheme)`/`Modem(scheme)`'s convention, but
TM/UM/AM genuinely don't share one method surface (AM needs four methods,
TM/UM need two) — `Mac` delegates via `__getattr__` rather than forcing an
artificial common interface, so `Mac(mode="um").build_status()` correctly
raises `AttributeError` instead of silently no-op'ing.

### `spectracuda/mac/session.py` — `MacLink`
Explicitly a demonstration/integration harness (same role as
`spectracuda/sim/channel.py`), not "the real chain" the way `Ofdm`/
`Packetizer` are — the entities themselves stay pure PHY-agnostic logic.
`MacLink` requires `ofdm.crc != "none"` (a corrupted-but-decoded frame is
indistinguishable from a correct one without CRC — see module docstring).
`_compute_max_segment_bits()` binary-searches against
`ofdm.packetizer.encoded_length()` — the SAME capacity accounting
`Ofdm.generate_frame()`/`rx_process()` already use (docs/todo.md #1.10) —
rather than re-deriving FEC-rate/CRC-overhead math independently. AM's
`send()` drives real rounds: DATA PDUs forward, a STATUS PDU in reverse
each round, `AmEntity.receive_status()` deciding retransmissions; if the
STATUS pdu itself is lost, `MacLink` falls back to `AmEntity.pending_pdus`
(retry the whole outstanding buffer) rather than stalling.

### `spectracuda/mac/pdu.py` — TYPE widened from 1 to 3 bits
Added later (see "Binding and link-quality reporting" below): `TYPE_DATA=0`/
`TYPE_STATUS=1` keep their original values (purely additive, not a
renumbering), `TYPE_BIND_REQUEST=2`/`TYPE_BIND_RESPONSE=3`/
`TYPE_LINK_QUALITY=4` added. Header stays 32 bits total
(`TYPE(3)+SI(2)+SN(10)+SO(16)+RESERVED(1)`) — the reserved bits absorbed
the widening, no other field moved.

### Binding — `spectracuda/mac/bind.py`, `MacLink.bind()`
A lightweight BIND_REQUEST/BIND_RESPONSE handshake — this project's own
design, no 3GPP precedent claimed (same footing as the rest of `mac/`).
Real behavioral point: `MacLink.send()` now refuses to run before
`bind()` succeeds (`ValueError`), not just a formality. The actual
accept/reject decision, `bind.evaluate_bind_request()`, is a **pure
function** independent of `Ofdm`/`MacLink` — accepts iff the requested
mode is known and `max_segment_bits` doesn't exceed the acceptor's own
capacity, **rejecting** (not silently clamping) an over-large request —
tested directly with two independently-chosen configs in
`tests/test_mac_bind.py`, proving genuine mismatch-rejection, not just
self-consistent plumbing. `MacLink.bind()` wires this into a real PHY
round trip; since `MacLink` builds `tx_mac`/`rx_mac` from one shared
config (see class docstring), it always succeeds in this harness — it
proves the message-exchange mechanism, not mismatch-rejection (that's
what the standalone tests prove).

### Link-quality reporting — `spectracuda/mac/quality.py`, `MacLink.exchange_link_quality()`
`LinkQualityTracker` aggregates `Ofdm.rx_process()`'s existing per-frame
`rssi_db`/`evm` (docs/todo.md #1.1) across every `_phy_round()` — any pdu
type, success or failure — into running means plus a `delivered_ratio`.
Honesty note, same spirit as RSSI already being "relative, not
calibrated dBm": `delivered_ratio` is a BER-*trend* proxy, not a true
bit-error-rate — there's no ground-truth bits to compute real BER
against. `exchange_link_quality()` sends the current report to the peer
over one PHY round (push model, same shape as AM's STATUS PDU) and
returns what the peer actually decoded — which can differ from the raw
local stats if that very round experiences loss, in which case it raises
rather than fabricating a report.

## Bugs found during implementation (verified fixed, not just described)

1. **Reassembly desync on out-of-order first arrival.**
   `ReassemblyBuffer` used to seed `expected_sn` from whichever segment
   physically arrived *first*, not the stream's true starting SN. A
   genuine out-of-order-delivery test (segments fed in shuffled order)
   caught this immediately: the buffer treated an early-arriving LAST
   segment as if it were the whole SDU. Fixed by requiring
   `initial_expected_sn` (default 0, matching every entity's own SN
   counter start) at construction instead of inferring it from arrival
   order. Covered by
   `test_mac_um.py::test_multi_segment_reassembly_survives_out_of_order_delivery`.
2. **STATUS PDU cumulative-ack boundary bug.** `AmEntity.receive_status()`
   originally treated `sn == base_sn` as cumulatively acknowledged, but
   `base_sn` means "the first SN *not yet* fully processed" — the exact
   opposite. A dropped-segment-at-the-window-edge test caught this: the
   missing segment was silently treated as acked and never retransmitted.
   Fixed by only treating `sn_precedes(sn, base_sn)` (strictly before) as
   acked, letting `sn == base_sn` fall through to the real bitmap check.
   Covered by
   `test_mac_am.py::test_recovers_a_single_dropped_segment_via_retransmission`.
3. **Byte-alignment requirement, discovered not assumed.** `Packetizer`'s
   CRC stage requires byte-aligned input; a segment length that isn't a
   multiple of 8 (the final/remainder segment, generically) would fail
   there. Fixed at the source: `Segmenter` now requires byte-aligned SDUs
   and guarantees byte-aligned segments as a consequence, rather than
   padding-and-tracking after the fact.
4. **`fft_size`-vs-frame-length reliability finding — now explicitly
   tested at THREE configs, not avoided.** An early draft of the
   `MacLink` integration tests used `fft_size=64` and saw unreliable
   decode on the suite's long (~19 OFDM-symbol) 800-bit test SDU, even at
   20-30dB SNR — initially mistaken for a MAC-layer bug. Root-caused by
   direct comparison: at `fft_size=64`/`n_pilot=6`, a SHORT payload (~1
   symbol) decodes reliably (confirmed: 20/20 clean trials at
   `snr_db=25`), but a LONG payload spanning many symbols has a real,
   non-trivial per-attempt failure rate even at generous SNR (confirmed:
   17/20 at `snr_db=25`, reproducibly re-confirmed). Testing a THIRD
   config, `fft_size=128`/`n_pilot=7`, isolated the real variable: a
   frame-length-MATCHED comparison (same ~19 symbols, achieved via a
   proportionally longer SDU) scored **15/20** at `fft=128` — *worse*
   than `fft=64`'s 17/20 despite the larger FFT, while `fft=128`'s own
   short-payload trials are a clean 20/20 like every other config. This
   rules out "small `fft_size`" as the explanation and confirms frame
   LENGTH (OFDM symbols per frame), not `fft_size`, is what actually
   correlates with the degradation — root cause still not pinned down
   (residual-CFO/timing-drift accumulation over more OFDM symbols is the
   leading hypothesis, unconfirmed). **All three `fft_size` values
   (64/128/256) are now explicitly, permanently tested** in
   `test_mac_session.py` (`test_clean_channel_round_trip` is
   parametrized across all three; each has its own separately-calibrated
   AM-recovers-from-loss proof, since each config's per-attempt
   reliability differs enough that one config's `(snr_db, seed)` pairing
   doesn't transfer to another — each was found by its own direct sweep,
   not reused blindly). The underlying reliability-vs-frame-length
   question itself remains an open PHY item, tracked in `docs/todo.md`
   §1.11, not silently worked around.
5. **`bind()` silently shifted the calibrated lossy-channel test seeds.**
   The moment `send()` was gated on `bind()`, every one of the three
   `test_am_recovers_from_channel_loss_that_defeats_um*` tests broke —
   `bind()`'s own PHY rounds consume some of the shared `Channel`
   object's seeded RNG stream *before* `send()` ever runs, so the noise
   realization `send()` actually sees at a given `seed=` is no longer
   the one the `(snr_db, seed)` pair was originally calibrated against.
   Caught immediately by the full regression run (not a silent drift —
   a direct, reproducible test failure), not fixed by re-sweeping new
   constants: fixed at the design level instead, by binding on a clean
   (`channel=None`) round first and only attaching the lossy `Channel`
   afterward, right before `send()` — which is also the more realistic
   model (a real link binds before conditions degrade, not while already
   in a lossy state) and keeps every existing calibrated `(snr_db,
   seed)` pair exactly valid. See `test_mac_session.py`'s
   `_bind_clean_then_attach_channel()` helper.

## Two-node redesign, step 1: `Mac` can own its own `Ofdm`

`MacLink` shares ONE `Ofdm` object across both its `tx_mac`/`rx_mac` roles
— fine for what it proves (the message-exchange mechanism), but it can
never demonstrate genuinely independent decode: nothing stops the two
roles from implicitly depending on shared object state just because
they're the same Python object. Settled by direct design discussion (not
re-derived here): the fix is for `Mac` itself to optionally own its own
`Ofdm`, so two real "HW units" are just two separate `Mac(mode=...,
ofdm_kwargs=...)` calls with zero shared object identity.

- `Mac(mode, ofdm_kwargs={...})` — builds `self.ofdm = Ofdm(**ofdm_kwargs)`
  internally; `max_segment_bits` is **derived** from that just-built
  `Ofdm`'s real capacity (`capacity.compute_max_segment_bits()`, relocated
  out of `session.py` to avoid a circular import), never accepted as a
  separate argument — passing both raises `ValueError`, specifically to
  prevent the two silently disagreeing (a design tension flagged and
  resolved during discussion, not found as a bug afterward).
- `Mac(mode, max_segment_bits=N)` (no `ofdm_kwargs`) — unchanged,
  PHY-agnostic, exactly as before this step.
- New IQ-level methods `send_iq()`/`receive_iq()` (TM/UM only — AM needs
  a return status path, deferred to the bidirectional two-unit step),
  reusing the same frame_found/crc_valid/exception handling
  `MacLink._phy_round()` already established, not reinvented.

**Real bug found and fixed while wiring this in**: the first version
named the new receive method `receive()` — which *shadowed* the existing
`__getattr__`-delegated `receive()` that `TmEntity`/`UmEntity` already
expose at the PDU level (the one `MacLink` depends on for its
PHY-agnostic `Mac` instances). The moment it was added, 7 of
`MacLink`'s own tests failed immediately — `self.rx_mac.receive(arrived)`
was being intercepted by the new IQ-level method and raising "requires
ofdm_kwargs" instead of reaching `TmEntity`/`UmEntity`'s real PDU-level
`receive()`. Fixed by naming the new methods `send_iq()`/`receive_iq()`
instead — distinct names avoid the collision entirely rather than trying
to make one method smart enough to detect which layer its argument
belongs to. Covered by
`test_mac_two_units_simple.py::test_existing_pdu_level_api_unaffected_by_ofdm_kwargs_addition`,
a permanent regression guard.

Verified: `tests/test_mac_two_units_simple.py` (10 tests) — two
genuinely independent `Ofdm` objects (`is not`, not just equal config),
HW1-tx→HW2-rx round trip both on a clean and a real lossy
`spectracuda.sim.Channel`, and a genuine physical-mismatch case
(different `sync=`) that actually fails to decode rather than silently
working. `MacLink`/`session.py` untouched — still 21/21 passing. Full
suite: 515 passed, 1 pre-existing unrelated skip (505 prior + 10 new).

Explicitly NOT this step (agreed, deliberate sequencing, not scope
creep): the full bidirectional two-unit picture (4 `Mac`s, 4 `Ofdm`s,
paired by direction, AM's status round trip working over `send_iq()`/
`receive_iq()`), and migrating `MacLink`'s `bind()`/
`exchange_link_quality()` behavior onto `Mac` itself.

## Binding + link-quality reporting, migrated onto `Mac`

The item flagged above as deliberately deferred. `MacLink.bind()` can
only ever self-consistently succeed — it evaluates a request against its
OWN capacity, since `tx_mac`/`rx_mac` are both built from one shared
`Ofdm`. Two real `Mac(mode=..., ofdm_kwargs=...)` objects have genuinely
independent `self.max_segment_bits`, so migrating this logic onto `Mac`
itself is what finally makes a REAL cross-object handshake possible —
not just moving code around.

Four new mode-agnostic methods (binding/quality-reporting don't depend
on AM's `send_iq()`/`receive_iq()` data-flow limitation, so none of them
guard on `mode`): `build_bind_request()`, `handle_bind_request_iq()`
(the acceptor's OWN `evaluate_bind_request()` call — a genuine decision
now), `handle_bind_response_iq()`, `build_quality_report()`,
`handle_quality_report_iq()`. A real handshake is three calls, fully
manual (no orchestrator class, per the established convention):

```python
req_iq = mac_a.build_bind_request()
resp_iq = mac_b.handle_bind_request_iq(req_iq)   # mac_b's OWN evaluation
accepted = mac_a.handle_bind_response_iq(resp_iq)
```

`send_iq()` gained the same `self.bound` gate `MacLink.send()` already
had — a real behavioral requirement, not decorative.

**`MacLink` is untouched, deliberately, not by oversight**: its
`tx_mac`/`rx_mac` are constructed WITHOUT `ofdm_kwargs` (PHY-agnostic by
design; `MacLink` itself owns the one shared `Ofdm`), so they structurally
cannot call the new IQ-transport methods (those require `self.ofdm` on
the SAME object). `MacLink`'s own bind/quality logic stays a separate
implementation — confirmed still 21/21 passing, unchanged.

**Two real bugs found while wiring this in, not hypothetical:**

1. **`bind.py`'s wire format couldn't carry a real `max_segment_bits`.**
   `max_segment_bits` was encoded as a `u16` (max 65535) in both
   `BIND_REQUEST` and `BIND_RESPONSE`. This silently worked in every test
   ever written against `MacLink`, because `MacLink.bind()` only ever
   evaluates its own, coincidentally-small-scale-tested capacity against
   itself. The first genuine two-object test at `fft_size=256`/
   `modem="qam64"` derived a real `max_segment_bits` of 165840 —
   comfortably over 65535 — and `encode_bind_request()` raised
   immediately. Fixed by widening the field to a `u32` (4 bytes) in both
   PDU types, adjusting the now-larger payload sizes and every
   downstream byte offset accordingly (request: 5→7 bytes; response:
   7→9 bytes). This is exactly the kind of bug that a self-consistent
   handshake can never surface — it took a genuinely independent
   acceptor capacity to expose it.
2. **The mismatched-`sync=` test could no longer reach its own scenario.**
   `test_mac_two_units_simple.py`'s existing mismatched-sync test used to
   `send_iq()` anyway and observe the receive side fail to decode. With
   `send_iq()` now gated on `bind()`, that path is unreachable for a
   permanently-mismatched pair — the bind handshake itself travels over
   the same broken PHY and never completes, so `send_iq()` raises
   *before* ever reaching the old failure point. Rewritten
   (`test_mismatched_sync_prevents_even_binding`) to demonstrate the
   failure at the bind stage instead — arguably the more honest outcome
   (a real radio that can't complete a handshake doesn't get to transmit
   either), not a regression papered over.

**The actual payoff, proven directly**:
`test_real_cross_object_bind_genuinely_rejects_a_mismatched_capacity` —
two real `Mac`s (`modem="qam64"` vs `modem="bpsk"`, a genuine, not staged,
capacity asymmetry) exchange a real `BIND_REQUEST`/`BIND_RESPONSE` pair
over real (if trivial) IQ, and the smaller side's OWN
`evaluate_bind_request()` genuinely rejects it. `MacLink`'s test suite
structurally cannot produce this — this is the first time it's been
proven through the actual wire mechanism rather than only as a standalone
pure-function call.

Verified: `tests/test_mac_bind.py` (11, unaffected by the u16→u32 fix —
the changed field lives past every other byte those tests inspect) +
`tests/test_mac_two_units_simple.py`/`test_mac_half_duplex.py` (now
binding for real before every `send_iq()` call) + `test_mac_session.py`
(21/21, `MacLink` confirmed unchanged) — 49 tests across those four
files, all passing. Full suite: 583 passed, 1 pre-existing unrelated skip.

## AM's `send_iq()`/`receive_iq()` opened up, plus the 4-Mac/4-Ofdm
## bidirectional picture

The `mode == "am"` guards on `Mac.send_iq()`/`receive_iq()` (added when
those methods were first built, before a genuine second Mac/Ofdm pair
for the reverse direction existed to make AM's STATUS round trip
possible) are now removed. AM's DATA-forward half behaves exactly like
UM's over the wire — `send_iq()` calls `AmEntity.transmit()` (which also
buffers each PDU for retransmission, same as at the PDU level);
`receive_iq()` calls `AmEntity.receive_data()` instead of `receive()`
(the name AM's receiver role actually uses, see `am.py`) and returns
`[]`, not `None`, on a failed decode (`AmEntity.receive_data()` is
list-returning, same shape as UM's `receive()`). Neither method grew a
STATUS/retransmission round trip of its own — that inherently needs a
*second* Mac/Ofdm pair for the reverse direction, which a single
object's send/receive pair structurally cannot provide regardless of how
smart it's made.

**Bug found**: removing the guard immediately broke
`test_am_mode_iq_methods_not_supported_yet`, which had hard-coded the
old "AM isn't supported" error message. Not papered over — replaced with
`test_am_mode_data_forward_half_works_over_send_iq_receive_iq`, which
proves AM's DATA-forward half actually works end-to-end (bind, segment,
send, decode, matches the original SDU) rather than continuing to assert
a limitation that no longer exists.

**Second bug found, larger**: all three existing example scripts
(`mac_two_units_demo.py`, `mac_half_duplex_demo.py`,
`mac_streaming_demo.py`) turned out to be broken by the *previous*
change (the `self.bound` gate added to `send_iq()`, see above) — none of
them had ever been updated to call `bind()` first, and none of this was
caught by the test suite because these are standalone example scripts,
not imported by any test. Running each one directly (`python
examples/mac_*_demo.py`) is what surfaced it — the same "run it, don't
just read it" verification discipline applied to example code, not only
to library code. Fixed by adding the same `_bind()` helper used in the
test files to each script.

**The actual new work**: two new example scripts implementing the full
4-Mac/4-Ofdm bidirectional AM picture —

```
HW1.tx_mac  -- owns Ofdm_A   (forward-direction PHY: hw1 -> hw2)
HW2.rx_mac  -- owns Ofdm_A'  (SAME config as Ofdm_A, separate object)
HW2.tx_mac  -- owns Ofdm_B   (reverse-direction PHY: hw2 -> hw1)
HW1.rx_mac  -- owns Ofdm_B'  (SAME config as Ofdm_B, separate object)
```

`examples/mac_bidirectional_am_batch_demo.py` decodes via
`Ofdm.rx_process()`; `examples/mac_bidirectional_am_streaming_demo.py`
decodes via `Ofdm.rx_streaming()` — otherwise identical scenarios, both
converted to permanent tests in `tests/test_mac_bidirectional_am.py`
(9 tests, parametrized across seeds plus a batch-vs-streaming agreement
check).

The wrinkle AM adds on top of the earlier two-unit/half-duplex UM demos:
a STATUS pdu reporting on traffic received in one direction has to
physically travel in the *other* direction, and is decoded by a
*different* Mac object than the one whose retransmission buffer it's
actually about — e.g. `hw2_rx_mac` (received hw1's DATA over `Ofdm_A`)
builds the STATUS; it travels hw2→hw1 over `Ofdm_B` (`hw2_tx_mac`'s
transmit chain, the same one hw2 uses for its own reverse-direction
DATA); `hw1_rx_mac` (owns `Ofdm_B'`) physically decodes it; `hw1_tx_mac`
(a *different* Mac object at HW1, the one actually holding the
retransmission buffer) is the one that calls `receive_status()` on the
decoded bits. This is genuinely fully-manual wiring — no orchestrator
class, per the established design stance — and `Mac.receive_iq()` can't
be used for the STATUS leg at all, since it assumes DATA `pdu_type` for
AM mode and would raise on a STATUS pdu; both example scripts write out
a small local helper (`_rx_raw_pdu()` / `_stream_rx_raw_pdu()`) that
does the same `rx_process()`/`rx_streaming()` + CRC + quality-tracking
logic `Mac._rx_one_frame()` uses internally, but without assuming the
pdu_type in advance.

**Third thing found, not a bug but a real constraint worth recording**:
proving genuine multi-PDU segmentation for this demo required care on
two fronts. First, the PDU header's `so` (segment offset) field is 16
bits (`SO_BITS`, `pdu.py`) — a segmented SDU's total size is capped at
65535 bits regardless of how large the underlying PHY's per-PDU capacity
is, so a large-capacity PHY (e.g. `fft_size=256`) can only ever be
coaxed into 2 segments for a demo, not 3+. Second, a genuinely small PHY
(`fft_size=64`, `n_data=16`) that segments a moderate SDU into 3 PDUs
ran into the still-open "long-frame reliability" gap (`docs/todo.md`
§1.11 — EVM measurably degrades with frame length, root cause not yet
identified) at the 25–30 dB SNR used elsewhere in these demos: the
longest (first) segment's CRC intermittently failed depending on the
channel noise realization. Both example scripts use `snr_db=40.0`
specifically to stay clear of that threshold — not a fix for #1.11
(still open, still worth root-causing separately) but a deliberate,
documented choice so this demo exercises the AM retransmission logic
against a real encode/channel/decode round trip, rather than against a
channel too marginal for the small-PHY config regardless of AM.

Verified: both example scripts run standalone (`python examples/mac_*.py`)
and produce `correct=True` for both directions across 6 different seeds
each; `tests/test_mac_bidirectional_am.py` (9/9); full suite 592 passed,
1 pre-existing unrelated skip (up from 583 — the 9 new tests).

**Follow-up: made the bind handshake and every PDU actually visible.**
Both scripts originally called `Mac.build_bind_request()`/
`handle_bind_request_iq()`/`send_iq()`/`receive_iq()` directly — correct,
but those methods only ever hand back/accept opaque IQ arrays, so the
BIND_REQUEST/BIND_RESPONSE bytes and every DATA/STATUS PDU's actual
content were invisible in the printed output. Rewritten to inline those
methods' exact same logic (mirroring `mac.py`'s own `send_iq()`/
`handle_bind_request_iq()` bodies 1:1 — verified by re-running across 6
seeds and confirming identical `forward_pdus`/`forward_retransmits`/
`correct` results before and after) so each stage can print the PDU as
hex plus its decoded header (`pdu_type`/`si`/`sn`/`so`) before it becomes
IQ and right after it's decoded back out of IQ — including the
BIND_RESPONSE's `accepted`/`reason` decision, not just its raw bytes.
Sample output (batch demo, forward direction):

```
[bind fwd (hw1->hw2, Ofdm_A)] BIND_REQUEST  sent     BIND_REQUEST si=0 sn=0 so=0 hex=4000000002000007d02004
[bind fwd (hw1->hw2, Ofdm_A)] BIND_RESPONSE received BIND_RESPONSE si=0 sn=0 so=0 hex=60000000010002000007d02004  -> accepted=True reason='none'
[fwd] DATA PDU #0 sent     DATA si=1 sn=0 so=0 hex=08000000e07ff66e5007677ebc36ce93...(254 bytes total)
[fwd] DATA PDU #1 DROPPED  (never reaches hw2_rx_mac at all)
[fwd status] STATUS sent      STATUS si=0 sn=1 so=0 hex=2002000040000000  (hw2_rx_mac -- built)
[fwd status] routed hw2_rx_mac --(Ofdm_B)--> hw1_rx_mac --> hw1_tx_mac.receive_status(): 1 PDU(s) to retransmit
```

Long DATA payloads are truncated to 16 bytes of hex with an explicit
`...(N bytes total)` marker (the decoded header, always shown in full,
is what actually identifies the frame); BIND/STATUS pdus, always small,
print in full. `tests/test_mac_bidirectional_am.py` unaffected — the
returned result dicts are identical, only stdout changed.

## Verification, as actually run

1. `pytest tests/test_mac_pdu.py tests/test_mac_tm.py tests/test_mac_um.py
   tests/test_mac_am.py tests/test_mac_bind.py tests/test_mac_quality.py
   -v` — 50 standalone tests, zero `Ofdm` involvement.
2. `pytest tests/test_mac_session.py -v` — 21 tests, at THREE fft_size
   configs (256/128/64, see "Bugs found" #4), including three
   separately-calibrated core proofs: `test_am_recovers_from_channel_loss_that_defeats_um`
   (`fft=256`, `snr_db=8.0`/`seed=0`),
   `test_am_recovers_from_channel_loss_that_defeats_um_at_fft128`
   (`fft=128`, `snr_db=12.0`/`seed=0`), and
   `test_am_recovers_from_channel_loss_that_defeats_um_at_fft64` (`fft=64`,
   `snr_db=20.0`/`seed=1`) — each a specific, reproducible condition
   (empirically found by a direct seed/SNR sweep, not a hand-picked
   outlier) where UM genuinely fails to deliver while AM, under the
   *identical* channel realization, succeeds via retransmission — through
   a real `Ofdm` PHY chain and a real `spectracuda.sim.Channel`, not a
   mocked/idealized link. Also covers `bind()`-then-`send()`,
   `send()`-before-`bind()` raising, and `exchange_link_quality()`
   reporting sane numbers after a real multi-round lossy session. Re-run
   3x consecutively to confirm determinism (all seeded, zero flakiness
   observed).
3. `pytest` (full suite): **367 passed, 1 pre-existing unrelated skip**
   (296 prior + 71 new, zero regressions).
