"""Worked example: 256-subcarrier OFDM receiver using SchmidlCoxSync +
SchmidlCoxCFO, built entirely from currently-implemented spectracuda
blocks (see docs/architecture.md for the full design; OfdmRx doesn't
exist yet to wire this declaratively -- Phase 2/3 still owed).

Requested configuration:
    - 256 subcarriers (fft_size)
    - 6 pilots, 200 data, the remaining 50 nulled as guard bands
    - cyclic prefix = 1/8 of the symbol time (256 // 8 = 32 samples)
    - Schmidl-Cox timing sync + coarse CFO estimation/correction

Frame structure (each symbol slot is cp_len + fft_size samples,
uniformly, including the preamble):

    [ preamble (Schmidl-Cox, two identical halves) ]
    [ training symbol -- ALL 206 non-null subcarriers carry known values ]
    [ payload symbol  -- ResourceGrid layout: 6 pilot + 200 data ]

Why a training symbol at all: LS + linear interpolation from just 6
pilots across 200 data subcarriers would be badly underdetermined for a
frequency-selective channel (verified during development -- an
i.i.d.-random-per-bin test channel produced >20% NMSE with sparse
interpolation; see tests/test_channel_and_equalizer.py). Real systems
solve this with a fully-known training symbol for the initial channel
estimate (this example) and use sparse per-symbol pilots only for
lightweight phase/drift *tracking* across a longer burst -- that
tracking step is a Phase 2 gap, not implemented here (this example
reuses the training-symbol estimate for the one payload symbol that
follows it, which is accurate as long as the channel doesn't change
meaningfully within that short a window).

Impairments simulated: a random 3-tap multipath channel (time-domain
convolution, so the cyclic prefix does its actual job), AWGN, an unknown
timing offset (leading silence), and a known carrier frequency offset.

The preamble is transmitted WITHOUT its own cyclic prefix -- found during
development that giving it one creates a timing-metric plateau roughly
cp_len samples wide (the CP duplicates part of the repeated-halves
structure the correlator is looking for, so the detector locks onto
anywhere in that plateau, off by up to cp_len samples once a multipath
channel and noise are added). This is a well-documented weakness of the
classic Schmidl & Cox algorithm; real preambles (e.g. 802.11's short
training field) are likewise sent without a CP for exactly this reason.
Confirmed empirically: with the preamble's CP removed, timing error drops
to 1-2 samples (ordinary channel-delay-spread smearing, well inside what
the *data* symbols' CP_LEN=32 already absorbs) instead of up to ~30.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from spectracuda.cfo import SchmidlCoxCFO
from spectracuda.channel import LSChannelEstimator
from spectracuda.equalizer import MMSEEqualizer
from spectracuda.modem import Modem
from spectracuda.ofdm import OfdmDemodulator, OfdmModulator, ResourceGrid
from spectracuda.sync import SchmidlCoxSync

FFT_SIZE = 256
CP_LEN = FFT_SIZE // 8  # 32, i.e. 1/8 of the symbol time
N_DATA = 200
N_PILOT = 6  # -> n_null = 256 - 200 - 6 = 50 guard/DC subcarriers


def run(seed: int = 0, snr_db: float = 25.0, eps_true: float = 0.15, verbose: bool = True) -> Dict[str, Any]:
    grid = ResourceGrid(fft_size=FFT_SIZE, n_data=N_DATA, n_pilot=N_PILOT, dc_null=True)
    modem = Modem("qpsk", backend="numpy")
    mod = OfdmModulator(FFT_SIZE, CP_LEN, backend="numpy")
    demod = OfdmDemodulator(FFT_SIZE, CP_LEN, backend="numpy")
    sync = SchmidlCoxSync(FFT_SIZE, backend="numpy")
    cfo_block = SchmidlCoxCFO(FFT_SIZE, backend="numpy")

    # -- build the TX frame -------------------------------------------------
    # No CP on the preamble itself -- see module docstring.
    preamble_time = sync.generate_preamble(seed=123)  # fixed seed: known to "both sides"

    tx_pilots = np.ones((grid.n_pilot,), dtype="complex64")

    train_rng = np.random.default_rng(999)  # fixed seed: known training sequence
    train_bits = train_rng.integers(0, 2, size=2 * grid.n_data).astype("uint8")
    train_symbols = modem.modulate(train_bits.reshape(1, -1))[0]
    train_grid_freq = grid.scatter(np, tx_pilots[None, :], train_symbols[None, :])[0]
    train_time = mod.process(train_grid_freq[None, :])[0]

    payload_rng = np.random.default_rng(seed)  # the actual "unknown" payload
    tx_bits = payload_rng.integers(0, 2, size=2 * grid.n_data).astype("uint8")
    tx_data_symbols = modem.modulate(tx_bits.reshape(1, -1))[0]
    payload_grid = grid.scatter(np, tx_pilots[None, :], tx_data_symbols[None, :])[0]
    payload_time = mod.process(payload_grid[None, :])[0]

    frame = np.concatenate([preamble_time, train_time, payload_time]).astype("complex64")

    # -- channel impairments --------------------------------------------------
    rng = np.random.default_rng(seed + 1000)
    n_taps = 3
    taps = (rng.standard_normal(n_taps) + 1j * rng.standard_normal(n_taps)) / np.sqrt(2 * n_taps)

    pad_before, pad_after = 40, 40  # unknown timing offset + convolution edge margin
    padded = np.concatenate(
        [np.zeros(pad_before, dtype="complex64"), frame, np.zeros(pad_after, dtype="complex64")]
    )
    channeled = np.convolve(padded, taps)[: len(padded)].astype("complex64")

    sig_power = float(np.mean(np.abs(channeled) ** 2))
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = (rng.standard_normal(len(channeled)) + 1j * rng.standard_normal(len(channeled))) * np.sqrt(
        noise_power / 2
    )

    n = np.arange(len(channeled))
    cfo_ramp = np.exp(1j * 2 * np.pi * eps_true * n / FFT_SIZE)
    rx = ((channeled + noise) * cfo_ramp).astype("complex64")[None, :]

    # -- receiver ---------------------------------------------------------------
    sync_result = sync.process(rx)
    start_index = sync_result["start_index"]

    cfo_est = cfo_block.process(rx, start_index=start_index)
    rx_corrected = cfo_block.correct(rx, cfo_est)

    d = int(start_index[0])
    train_slot_start = d + FFT_SIZE
    train_slot = rx_corrected[:, train_slot_start : train_slot_start + CP_LEN + FFT_SIZE]
    train_rx_grid = demod.process(train_slot)

    payload_slot_start = train_slot_start + CP_LEN + FFT_SIZE
    payload_slot = rx_corrected[:, payload_slot_start : payload_slot_start + CP_LEN + FFT_SIZE]
    payload_rx_grid = demod.process(payload_slot)

    # channel estimate from the fully-known training symbol (all 206
    # non-null bins -- no sparse-pilot interpolation needed here)
    train_known_indices = np.sort(np.concatenate([grid.pilot_indices, grid.data_indices]))
    train_known_values = train_grid_freq[train_known_indices]
    train_rx_at_known = train_rx_grid[:, train_known_indices]

    est = LSChannelEstimator(train_known_indices, FFT_SIZE, train_known_values, backend="numpy")
    h_hat_data = est.process(train_rx_at_known)[:, grid.data_indices]

    payload_rx_data = grid.extract_data(np, payload_rx_grid)
    eq = MMSEEqualizer(noise_var=noise_power, backend="numpy")
    equalized = eq.process(payload_rx_data, channel_est=h_hat_data)

    rx_bits = modem.demodulate(equalized)
    ber = float(np.mean(rx_bits != tx_bits.reshape(1, -1)))

    result = {
        "true_start_index": pad_before,
        "detected_start_index": d,
        "true_cfo": eps_true,
        "estimated_cfo": float(cfo_est[0]),
        "ber": ber,
    }
    if verbose:
        print(f"grid: fft_size={FFT_SIZE} cp_len={CP_LEN} n_data={grid.n_data} "
              f"n_pilot={grid.n_pilot} n_null={len(grid.null_indices)}")
        print(f"timing: detected={result['detected_start_index']} true={result['true_start_index']}")
        print(f"CFO:     estimated={result['estimated_cfo']:.5f} true={result['true_cfo']:.5f}")
        print(f"BER:     {result['ber']:.6f}")
    return result


if __name__ == "__main__":
    run()
