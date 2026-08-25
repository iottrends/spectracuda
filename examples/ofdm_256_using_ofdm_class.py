"""The same 256-subcarrier / 6-pilot / 200-data / 1/8-CP / Schmidl-Cox
scenario as examples/ofdm_256_schmidl_cox_demo.py, but through the Ofdm
class (spectracuda/pipeline/ofdm.py) and the Channel impairment
simulator (spectracuda/sim/channel.py) instead of manual slot-offset
arithmetic and hand-rolled noise/multipath/CFO code. Compare the three
files: this is what building real classes around the validated logic
buys you.
"""
from __future__ import annotations

import numpy as np

from spectracuda.backend import default_backend
from spectracuda.pipeline import Ofdm
from spectracuda.sim import Channel

FFT_SIZE = 256
CP_LEN = FFT_SIZE // 8  # 32


def run(seed: int = 0, snr_db: float = 25.0, eps_true: float = 0.15, verbose: bool = True,
        backend: str = None):
    backend = backend or default_backend()  # "cupy" if a working CUDA runtime is present, else "numpy"
    ofdm = Ofdm(
        fft_size=FFT_SIZE, n_pilot=6, n_data=200, cp_len=CP_LEN,
        modem="qpsk", fec="none",
        sync="schmidl_cox", cfo="schmidl_cox",
        channel_estimator="ls", equalizer="mmse",
        n_training_symbols=2,
        backend=backend,
    )

    rng = np.random.default_rng(seed)
    tx_bits = rng.integers(0, 2, size=(1, ofdm.grid.n_data * ofdm.modem.bits_per_symbol)).astype("uint8")
    tx_iq = ofdm.generate_frame(tx_bits)[0]

    # unknown timing offset: pad with silence before/after the frame
    pad_before, pad_after = 40, 40
    padded = np.concatenate(
        [np.zeros(pad_before, dtype="complex64"), tx_iq, np.zeros(pad_after, dtype="complex64")]
    )

    taps = Channel.random_multipath_taps(3, seed=seed + 1000)
    channel = Channel(
        snr_db=snr_db, multipath_taps=taps, cfo=eps_true, cfo_fft_size=FFT_SIZE,
        seed=seed + 1000, backend=backend,
    )
    rx_iq = channel.process(padded)[0]

    # -- receiver: this is the entire receive chain -------------------------
    result = ofdm.rx_process(rx_iq)

    ber = float(np.mean(result["bits"] != tx_bits))
    if verbose:
        print(f"timing: detected={int(result['start_index'][0])} true={pad_before}")
        print(f"CFO:     estimated={result['cfo_estimate'][0]:.5f} true={eps_true:.5f}")
        print(f"BER:     {ber:.6f}")
    return {
        "true_start_index": pad_before,
        "detected_start_index": int(result["start_index"][0]),
        "true_cfo": eps_true,
        "estimated_cfo": float(result["cfo_estimate"][0]),
        "ber": ber,
    }


if __name__ == "__main__":
    run()
