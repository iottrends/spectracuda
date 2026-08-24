# Chapter 03 — Carrier Frequency Offset

A transmitter's local oscillator and a receiver's local oscillator are
never running at exactly the same frequency — even a few parts per
million of mismatch, at gigahertz carrier frequencies, becomes a real
offset of tens to thousands of hertz. Left uncorrected, that offset shows
up as a steadily rotating phasor multiplying every received sample.

```text
r[n] = s[n] * exp( j * 2*pi * cfo * n / fft_size )     # spectracuda/sim/channel.py
```

`cfo` here is expressed in units of subcarrier spacing — a `cfo` of 1.0
means the offset equals exactly one subcarrier's width. Applied to a
stream of QPSK symbols with no correction, that steady phase ramp smears
every symbol around a full circle:

```{figure} ../book_figures/cfo_rotation.png
:alt: Two constellation scatter plots side by side. The left plot, uncorrected, shows QPSK symbols smeared into a full circle by carrier frequency offset. The right plot, corrected, shows the same symbols collapsed back into four tight clusters at the QPSK constellation points.

Real `Modem("qpsk")` symbols, with the exact phase ramp above applied and
then exactly removed (a known offset — no estimator involved in this
panel). This is the effect a CFO estimator exists to undo automatically,
without knowing the offset in advance.
```

spectracuda offers two `cfo=` strategies:

| Strategy | How it estimates |
|---|---|
| `"schmidl_cox"` | Reuses the sync preamble's own repeated-halves phase difference — cheap, but only valid alongside `sync="schmidl_cox"`. |
| `"pilot_based"` | Tracks phase drift across pilot subcarriers symbol-to-symbol — independent of which `sync=` found the frame, works with either. |

Run through the real pipeline — `Ofdm(cfo="schmidl_cox")`, a `Channel`
with a known offset applied, a full `generate_frame()` → `rx_process()`
round trip — the estimator recovers the offset and the equalizer's output
collapses back to tight clusters:

```{figure} ../book_figures/cfo_recovered.png
:alt: A tight, four-cluster QPSK constellation plot showing the real decoded payload symbols after SchmidlCoxCFO correction, with estimated CFO of 0.898 against a true CFO of 0.900.

The actual equalized payload symbols from one real `rx_process()` call —
captured by temporarily wrapping the equalizer's own output during a real
decode, then double-checked by recomputing EVM from those captured
symbols and confirming it matches the value `rx_process()` itself
reported (0.1049 both ways) before trusting the plot.
```

```{warning}
**An honest, measured limit — not a spec number.** The illustration above
uses `cfo=3.5` because it's a directly-applied, exactly-known offset — no
estimator involved, so any magnitude makes the point. The *real*
`SchmidlCoxCFO` estimator has a narrower reliable range: sweeping true
offsets from 0.1 to 3.5 subcarriers at 25 dB SNR, this implementation
decodes correctly through roughly 1.0 subcarrier of offset and fails past
about 1.5 — which is why the second figure above uses 0.9, not 3.5. That
sweep lives in `generate_figures.py`'s own commit history, not asserted
from memory.
```
