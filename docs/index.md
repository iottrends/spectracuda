# spectracuda

GPU-accelerated, liquid-dsp-inspired SDR PHY (+ MAC) framework for NVIDIA
Jetson (Orin Nano first, NX/AGX Orin and desktop CUDA GPUs as additional
targets).

Not a CUDA port of liquid-dsp. liquid-dsp never exposed swappable
`channel_estimator=`/`equalizer=`/`cfo=` strategies -- those are algorithms
buried inside monolithic C objects, not interchangeable components.
spectracuda extracts the algorithms liquid-dsp *does* have a reference
implementation for, ports the ones that are self-contained (CRC, the
Zadoff-Chu-adjacent `qdetector` idea, the `interleaver` algorithm),
re-derives the ones that only exist wrapped around an external C library
(Viterbi, Reed-Solomon), and designs the rest from standard references
where liquid-dsp has no precedent at all (LS/MMSE channel estimation,
ZF/MMSE equalization, LDPC, the MAC layer) -- everything batch-first,
GPU-first, with a NumPy fallback so it runs anywhere.

Source: [github.com/iottrends/spectracuda](https://github.com/iottrends/spectracuda)

## Quick start

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

See {doc}`book/ch01_ofdm_object` for the full walkthrough, or the
[README](https://github.com/iottrends/spectracuda#readme) for install
instructions and every swappable strategy.

```{toctree}
:maxdepth: 2
:caption: The OFDM Field Guide — Part I

book/ch01_ofdm_object
book/ch02_synchronization
book/ch03_cfo
book/ch04_channel_estimation
book/ch05_modem
book/roadmap
```

```{toctree}
:maxdepth: 1
:caption: Reference

architecture
mac
ldpc
liquid-dsp-api-inventory
todo
```
