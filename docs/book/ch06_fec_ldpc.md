# Chapter 06 — FEC, LDPC & Interleaving

A CRC only tells a receiver *that* a frame is wrong, never how to fix it.
Forward error correction is the layer that can actually recover from
channel noise — trading transmitted bits (redundancy) for the ability to
correct errors after the fact, without asking for a retransmission.

## The three schemes, in one table

| `fec=` value | What it is | liquid-dsp precedent |
|---|---|---|
| `"conv_v27"` | Rate-1/2, constraint-length-7 convolutional code, hard-decision Viterbi decode | yes — `LIQUID_FEC_CONV_V27`, exact parity |
| `"rs_m8"` | Reed-Solomon(255, 223) over GF(256) | yes — `LIQUID_FEC_RS_M8`, exact parity |
| `"ldpc_648_r12"` … `"ldpc_1944_r56"` | 12-variant IEEE 802.11n QC-LDPC family (3 block lengths × 4 rates), normalized min-sum belief-propagation decode | **no** — a deliberate scope expansion *beyond* liquid-dsp, which has no LDPC (or Polar) at all |

Convolutional and Reed-Solomon are both direct liquid-dsp parity —
matched exactly, not reduced subsets of what liquid-dsp offers. LDPC is
new ground: liquid-dsp simply doesn't have it, so this codebase designs
the full 12-variant family from the IEEE 802.11n standard reference
rather than leaving the gap unaddressed. It's also the first FEC codec
in this codebase whose iterative decode core is genuinely GPU-batched
across `(n_batch, ...)` via gather-only `self.xp` operations, rather than
Reed-Solomon's inherently sequential per-codeword Python loop — see
`spectracuda/fec/ldpc.py`'s module docstring.

```python
from spectracuda.pipeline import Ofdm

ofdm = Ofdm(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32, modem="qpsk",
    fec="ldpc_648_r12", crc="crc16",
    sync="schmidl_cox", cfo="schmidl_cox",
)
```

```{note}
**A "fail loud" convention, not a silent best-effort.** When a codeword
has more errors than the decoded scheme can correct, `Ofdm.rx_process()`
raises `ValueError` rather than returning a garbage best guess — the same
convention this codebase's LDPC/Reed-Solomon decoders already document
for their own capacity limits. A caller has to explicitly decide what
"uncorrectable" means for their use case (retry, drop, log), rather than
receiving silently-wrong bits with no signal anything went wrong.
```

## Does it actually help? A real, measured answer

Four schemes, one PHY config, one AWGN channel model, swept across SNR —
40 independent trials per point, each a genuine `generate_frame()` →
`Channel.process()` → `rx_process()` round trip, counted as a failure on
either a caught decode exception or a CRC miss:

```{figure} ../book_figures/fec_comparison.png
:alt: A frame-error-rate-versus-SNR waterfall plot comparing four FEC schemes -- none, conv_v27, rs_m8, and ldpc_648_r12 -- for QPSK over an AWGN channel. The none and rs_m8 curves fall together around 10-12 dB SNR; conv_v27 and ldpc_648_r12 fall together roughly 4 dB earlier, around 6-8 dB.

Real `Ofdm(fec=...)` + `Channel(snr_db=...)` + `rx_process()`, 40 trials
per (scheme, SNR) point, `crc16` deciding pass/fail. Not a textbook BER
curve reference image — every point is a measured fraction of 40 real
frame attempts.
```

Two real, measured things this curve shows, not asserted from theory:

**`conv_v27` and `ldpc_648_r12` both deliver a genuine ~4 dB coding gain**
over uncoded — reaching the same reliability roughly 4 dB of SNR earlier
than `"none"` does. That's the actual payoff FEC exists for, demonstrated
here rather than just claimed.

**`rs_m8` alone barely helps at all** against this channel — its curve
tracks `"none"` almost exactly. This isn't a bug; it's Reed-Solomon
behaving exactly as RS theory predicts. RS corrects *symbol* errors (each
GF(256) symbol is 8 bits), and it can correct up to `(n-k)/2 = 16` wrong
symbols per 255-symbol codeword regardless of how many bits are wrong
*within* those symbols. Scattered, independent random bit errors from
AWGN don't concentrate into a small number of symbols — they spread
thinly across many, and a codeword with, say, 40 different symbols each
carrying one wrong bit exceeds RS's 16-symbol correction budget even
though only 40 of 2040 bits are actually wrong. RS is a *burst*-error
code; against uniformly scattered bit errors it isn't the right tool
used alone — which is exactly why real systems pair it with an inner
code.

```{warning}
**Read this curve for what it actually measured, not more.** One PHY
config, one payload size per scheme (chosen to satisfy each scheme's own
block-size constraint, not rate-matched to carry identical net
information across schemes), pure AWGN with no multipath or burst
structure. This demonstrates real, qualitative coding gain and RS's real
weakness against scattered bit errors — it is not a rate-normalized
Shannon-bound comparison, and no burst/fading channel exists yet in
`spectracuda.sim.Channel` to demonstrate RS's actual strength against
bursty errors directly (see the two-stage section below for where that
strength is supposed to show up instead).
```

## Two-stage FEC: why `fec0`/`fec1` order genuinely matters

A second, outer FEC stage (`fec1=`) is supported, mirroring liquid-dsp's
own `packetizer` inner/outer composition:

```python
ofdm = Ofdm(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32, modem="qpsk",
    fec="rs_m8", fec1="conv_v27",                # fec0 = inner, fec1 = outer
    interleaver="block", interleaver_kwargs={"unit_bits": 8},
)
```

`fec0` (`fec=`) is applied **first** on encode and decoded **last**;
`fec1` is applied **second** — closest to the channel — and decoded
**first**. Given the previous section's real result (RS alone barely
helps against scattered bit errors, Viterbi gives a real ~4 dB gain),
the classic concatenated-coding benefit needs Viterbi to be the code that
actually faces the channel and Reed-Solomon to mop up whatever bursty
residue a Viterbi decode error leaves behind — which means **Viterbi
must be `fec1` and Reed-Solomon must be `fec0`**: `fec="rs_m8",
fec1="conv_v27"`, not the reverse. Getting this backwards doesn't error
out — it just quietly gives up the benefit the combination exists for.

`interleaver_kwargs={"unit_bits": 8}` matters for the same reason, not
just style: interleaving at individual-bit granularity can make a
byte-oriented outer code (RS) *worse* off, not better — it fragments a
few concentrated byte errors into many more scattered ones, working
directly against RS's own strength. `interleaver=` picks from `"block"`
(recommended default), `"permutation"`, `"convolutional"`, or `"liquid"`
(a verified port of liquid-dsp's own `interleaver.c` algorithm).

```{note}
**A wrong assumption, caught during development, not shipped silently.**
The `unit_bits=8` requirement above wasn't obvious from the start — see
`docs/todo.md` §1.12 for the full derivation of how a bit-granularity
interleaver was initially assumed fine, then found to actively hurt a
byte-oriented outer code once tested directly. Worth reading if you're
deciding how to configure `interleaver_kwargs=` for your own `fec0`/
`fec1` combination.
```
