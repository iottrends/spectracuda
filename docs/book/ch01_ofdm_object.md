# Chapter 01 — The `Ofdm` Object

Everywhere liquid-dsp splits transmit and receive into separate objects
(`ofdmflexframegen` / `ofdmflexframesync`), spectracuda collapses both
directions into one class: `Ofdm`. That's not a simplification for its own
sake — tx and rx share almost every parameter (subcarrier allocation, CP
length, preamble content, training symbols), and keeping them separate
means hand-matching those values across two objects every time you change
one. One object removes an entire class of bug: tx and rx silently
drifting out of sync on preamble or training content.

```python
from spectracuda.pipeline import Ofdm
import numpy as np

ofdm = Ofdm(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32,
    modem="qpsk", fec="conv_v27", crc="crc32",
    sync="schmidl_cox", cfo="schmidl_cox",
    channel_estimator="ls", equalizer="mmse",
)

bits = np.random.default_rng(0).integers(0, 2, size=(1, 64)).astype("uint8")
tx_iq = ofdm.generate_frame(bits)      # -> (1, n_samples) complex64
result = ofdm.rx_process(tx_iq)        # -> dict, stable key set

result["frame_found"]   # True
result["bits"]          # decoded payload bits
result["crc_valid"]     # per-item bool array
result["evm"], result["rssi_db"], result["cfo_estimate"], result["header"]
```

A receiver is never told what the transmitter chose for
`modem=`/`fec=`/`crc=` — `Ofdm` packs all of that into a 112-bit frame
header (a liquid-dsp-style convention) and resolves it dynamically from
the decoded frame, exactly the way a real, physically separate receiving
device has to. That's what makes the two-radio examples in Part II
meaningful rather than a simulation shortcut: each side's `Ofdm` is built
independently and never inspects the other's configuration.

## The tx→channel→rx chain, at a glance

```text
generate_frame() — tx, left to right:
payload bits → FEC + CRC → interleave → Modem → resource grid → IFFT + CP → preamble + training
```

```text
rx_process() — rx, left to right:
sync → CFO correct → FFT + CP strip → channel est. → equalize → demod → deinterleave → FEC decode + CRC
```

Every stage after "payload bits" and before "demod"/"CRC" is
**batch-shaped**: every block operates on `(n_batch, …)` arrays, never a
single scalar sample. That's a deliberate departure from liquid-dsp's own
`execute(samples, n)` streaming idiom — GPU throughput on a Jetson-class
part only pays off once work is batched, so spectracuda batches by design
and offers a separate, additive streaming receiver (`rx_streaming()`,
Chapter 08) for the arbitrary-chunk case liquid-dsp's idiom targets.

The whole compute path runs in `complex64`/`float32`, never `float64` —
Jetson-class GPUs have drastically lower double-precision throughput, so a
silent upcast (numpy's own `fft`/`ifft` always computes in double
precision internally, regardless of input dtype — a real bug found and
fixed during development) is a hidden performance cliff on the actual
target hardware, not a style nit.
