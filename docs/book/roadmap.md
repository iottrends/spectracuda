# Coming next: framing, streaming, and the MAC layer

These chapters aren't written yet — rather than leave a blank space,
here's what each will cover and where to read the real material right
now. Every topic below already has finished, tested code and reference
documentation behind it; what's missing is narrative prose in this style,
not functionality.

## Part I, continued

**07 — Framing & the Header.** The 112-bit liquid-dsp-style frame header,
the deframer, and how `rx_process()` tells a genuinely-absent frame apart
from a corrupted one.
*See: {doc}`../todo` §1.1.*

**08 — Streaming Receive.** `Ofdm.rx_streaming(chunk)` — a real chunked
receiver state machine modeled directly on liquid-dsp's own
`ofdmframesync_execute()`, additive alongside the batch `rx_process()`
every Part I chapter so far has used.
*See: `examples/mac_streaming_demo.py`, {doc}`../todo` §2.5.*

## Part II — the MAC layer

liquid-dsp has no MAC/RLC concept at all — this is new ground, named after
3GPP RLC's TM/UM/AM modes for the behavior they describe (raw passthrough
/ numbered best-effort / numbered-with-retransmission), not as a
spec-accurate implementation.

**09 — TM / UM / AM.** The 32-bit PDU header, segmentation and
reassembly, and the three entities' genuinely different method surfaces —
why AM needs four methods where TM/UM need two.
*See: {doc}`../mac`.*

**10 — Two Independent Radios.** Already implemented, not just planned:
`Mac(mode=, ofdm_kwargs=)` lets each `Mac` build and own its own real
`Ofdm` internally, so two genuinely independent endpoints are two
separate objects with zero shared identity — IQ is the only thing that
crosses between them. `MacLink`/`session.py` remains a separate,
still-supported path (one shared `Ofdm` across both roles) for what
*that* proves; it isn't this design's job.
*See: `examples/mac_two_units_demo.py`, {doc}`../mac`.*

**11 — Binding & Link Quality.** The real BIND_REQUEST/BIND_RESPONSE
handshake, and why a genuinely independent second `Mac` was needed before
mismatch-rejection could be proven through the actual wire mechanism, not
just as a pure function.
*See: {doc}`../mac`, "Binding + link-quality reporting".*

**12 — Bidirectional AM.** The full 4-Mac/4-Ofdm picture: why AM's
STATUS/retransmission round trip needs a second Mac/Ofdm pair per
direction, the STATUS-routes-the-other-way wrinkle, and the same scenario
over both batch and streaming receive.
*See: `examples/mac_bidirectional_am_batch_demo.py`,
`examples/mac_bidirectional_am_streaming_demo.py`.*

---

Nothing in this book substitutes for {doc}`../architecture`,
{doc}`../mac`, or {doc}`../todo` — those are the actual source of truth,
including every real bug found and fixed along the way. This book is the
guided tour; those documents are the map.
