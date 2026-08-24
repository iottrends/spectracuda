# Chapter 04 — Channel Estimation & Equalization

Multipath doesn't distort a signal uniformly — each subcarrier sees a
different complex gain, because each corresponds to a different frequency
and the channel's frequency response varies across the band. A single
global correction can't fix that; equalization has to happen *per
subcarrier*, which means the receiver first needs to know what each
subcarrier's gain actually is.

That's what pilot subcarriers are for: known symbols, scattered across the
resource grid at fixed positions, that the receiver can compare against
what it actually received to back out the channel's effect at those
positions.

| Stage | Strategies | The tradeoff each represents |
|---|---|---|
| `channel_estimator=` | `"ls"`, `"mmse"` | LS solves the pilot equations directly (no assumptions, more noise-sensitive); MMSE folds in a noise-aware prior (biased toward smoother estimates, less noise-sensitive). |
| `equalizer=` | `"zf"`, `"mmse"` | Zero-forcing inverts the channel exactly (can blow up where the channel is weak); MMSE balances inversion against noise amplification. |

Neither pairing has a liquid-dsp precedent to match against — liquid-dsp's
own `eqlms`/`eqrls` are sample-adaptive single-carrier equalizers, not
per-subcarrier frequency-domain ZF/MMSE — so both are designed here from
standard OFDM receiver references rather than ported from existing
source.

Run against a real, randomly-generated 4-tap multipath channel, an
`Ofdm(channel_estimator="ls", equalizer="mmse")` receiver recovers a
channel-magnitude estimate that tracks the true frequency response
closely across every data subcarrier, and the equalized constellation
comes back tight:

```{figure} ../book_figures/channel_equalization.png
:alt: Left panel, a line plot comparing the true multipath channel frequency response magnitude against the LS channel estimate magnitude across 216 data subcarriers, the two curves nearly overlapping. Right panel, a tight four-cluster QPSK constellation after MMSE equalization, EVM 0.0856.

**Left:** `result["channel_estimate"]` (the real LS estimate a
`rx_process()` call returns) plotted against the true channel's FFT,
evaluated at the exact data-subcarrier bins `ofdm.grid.data_indices`
reports. **Right:** the real equalized payload symbols from that same
decode — same capture-and-verify technique as Chapter 03's
recovered-constellation figure.
```

```{note}
**A real edge case, found while building this figure.** A long payload
through a real multipath channel can shift the detected frame start a few
samples later than the noiseless case (delay spread), and
`generate_frame()`'s output carries no built-in trailing margin beyond
exactly what a nominal-start decode needs — so a late-detected start can
run the last payload symbol past the end of the buffer. The figure script
works around it by padding trailing silence, exactly as any real captured
buffer would have some idle samples after a frame anyway; the underlying
question of how much margin a real streaming capture should keep is still
open territory (see `docs/todo.md` #1.11's neighborhood) rather than
something quietly papered over here.
```
