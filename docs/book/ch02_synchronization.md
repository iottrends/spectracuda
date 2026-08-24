# Chapter 02 — Synchronization

Before anything else can happen, a receiver has to answer one question:
*where, in this stream of samples, does the frame actually start?*
Everything downstream — CFO correction, channel estimation, demodulation —
assumes that question is already answered. Get it wrong by even a handful
of samples and every subcarrier's phase is wrong too.

spectracuda offers two independent `sync=` strategies, chosen by string
like every other swappable stage:

| Strategy | Idea | liquid-dsp precedent |
|---|---|---|
| `"zc"` | Cross-correlate against a known Zadoff-Chu sequence — you know exactly what you're looking for. | `qdetector_create_linear` |
| `"schmidl_cox"` | Self-correlate a preamble made of two identical halves — you don't need to know the sequence, only that it repeats. | extracted from `ofdmframesync.c`'s S0/S1 handling — not exposed as its own reusable block |

## How Schmidl-Cox actually decides

The 1997 Schmidl & Cox algorithm builds a preamble whose first and second
halves are identical in the time domain (achieved by putting energy only
on even subcarriers before the IFFT). A receiver then slides a window
across the incoming stream and, at every candidate offset `d`, asks "how
well does the first half of this window correlate with the second half?"
The metric:

```text
P(d) = sum_{m=0}^{L-1} conj(r[d+m]) * r[d+m+L]
R(d) = 0.5 * ( sum |r[d+m]|^2 + sum |r[d+m+L]|^2 )
M(d) = |P(d)|^2 / R(d)^2
```

```{warning}
**A real deviation from the textbook formula.** The original 1997 paper
normalizes by second-half energy only. That denominator collapses toward
zero whenever a candidate window straddles the boundary between the
preamble and a low-energy region right after it — confirmed empirically
during development, where such a boundary produced a metric of ~77
against a true peak bounded at 1.0. This implementation uses the
symmetric energy shown above (both halves), which needs *both* windows to
be low-energy for the denominator to vanish, and is exact in the
noiseless case just like the original.
```

Run against a synthetic stream — 400 samples of noise, then a real
preamble, then more noise — the metric traces a sharp, unambiguous peak
exactly where the preamble begins:

```{figure} ../book_figures/sync_metric.png
:alt: Schmidl-Cox timing metric plotted against candidate sample offset, showing a sharp peak of 1.000 at the true preamble start, with low, noisy metric values everywhere else in the stream.

Real output of `SchmidlCoxSync.process()` on a 256-sample preamble buried
in noise. The dashed line is where the block actually reports
`start_index`; it lands exactly on the dotted true-start line. (The full
curve isn't something `process()` returns directly — it only returns the
peak — so this script mirrors the exact formula above to plot every
candidate, then asserts its own argmax matches the real block's
`start_index` before trusting the plot.)
```

```{note}
**A real, GPU-shaped design choice.** Candidate offsets aren't evaluated
with a Python loop sliding one sample at a time. `P(d)`/`R(d)` are
computed for *every* candidate at once via prefix-sum cumulative arrays —
the same "many candidates evaluated as one batched array op" reframing
this whole codebase applies wherever liquid-dsp's own C idiom would reach
for a sequential loop.
```

## Not every `sync`/`cfo` pairing is valid

`SchmidlCoxCFO` (Chapter 03) depends on the preamble's repeated-halves
structure and only pairs with `sync="schmidl_cox"`. `PilotBasedCFO` has no
such dependency and works with either `sync=` value. `sync="zc"` +
`cfo="schmidl_cox"` reliably fails — use `cfo="pilot_based"` with a
Zadoff-Chu preamble instead.
