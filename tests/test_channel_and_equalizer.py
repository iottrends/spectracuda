"""Frequency-domain-only tests for LSChannelEstimator + ZF/MMSEEqualizer.

Deliberately skips OfdmModulator/OfdmDemodulator here: for a cyclic-prefixed
OFDM system, a channel impulse response no longer than the CP reduces to a
per-subcarrier complex multiply in the frequency domain -- so modeling the
channel as the FFT of a short (n_taps <= cp_len) random time-domain impulse
response is a standard, physically-motivated way to test channel-estimation/
equalization algebra in isolation from the FFT/CP layer (covered separately
in test_ofdm_fft.py). A short-tap channel is frequency-smooth (correlated
across nearby subcarriers), which is what makes sparse-pilot LS + linear
interpolation meaningful in the first place -- an i.i.d.-random-per-bin
"channel" would defeat interpolation entirely and isn't physically realistic.
"""
import numpy as np
import pytest

from spectracuda.channel import LSChannelEstimator
from spectracuda.equalizer import MMSEEqualizer, ZFEqualizer
from spectracuda.ofdm import ResourceGrid


def _make_grid_and_channel(rng, fft_size=64, n_data=40, n_pilot=16, n_taps=3):
    grid = ResourceGrid(fft_size=fft_size, n_data=n_data, n_pilot=n_pilot, dc_null=True)
    taps = (rng.standard_normal(n_taps) + 1j * rng.standard_normal(n_taps)) / np.sqrt(2 * n_taps)
    h_time = np.zeros(fft_size, dtype="complex64")
    h_time[:n_taps] = taps
    h_true = np.fft.fft(h_time).astype("complex64")
    return grid, h_true


def _nmse(a, b):
    return float(np.mean(np.abs(a - b) ** 2) / np.mean(np.abs(b) ** 2))


def test_ls_estimate_matches_truth_no_noise():
    rng = np.random.default_rng(0)
    grid, h_true = _make_grid_and_channel(rng)
    n_batch = 4

    tx_pilots = np.ones(grid.n_pilot, dtype="complex64")
    rx_pilots = np.tile(tx_pilots * h_true[grid.pilot_indices], (n_batch, 1))

    est = LSChannelEstimator(grid.pilot_indices, grid.fft_size, tx_pilots, backend="numpy")
    h_hat = est.process(rx_pilots)

    assert h_hat.shape == (n_batch, grid.fft_size)
    # Pilot bins are recovered exactly (direct division, no interpolation
    # involved there); data bins are only approximate (linear interpolation
    # of a smooth-but-not-piecewise-linear frequency response) -- checked
    # via NMSE below, not exact equality.
    np.testing.assert_allclose(
        h_hat[:, grid.pilot_indices], np.tile(h_true[grid.pilot_indices], (n_batch, 1)), atol=1e-5
    )
    data_nmse = _nmse(h_hat[:, grid.data_indices], np.tile(h_true[grid.data_indices], (n_batch, 1)))
    assert data_nmse < 0.01


@pytest.mark.parametrize("Equalizer,kwargs", [(ZFEqualizer, {}), (MMSEEqualizer, {"noise_var": 1e-6})])
def test_equalizer_recovers_symbols_no_noise(Equalizer, kwargs):
    rng = np.random.default_rng(1)
    grid, h_true = _make_grid_and_channel(rng)
    n_batch = 3

    tx_pilots = np.ones(grid.n_pilot, dtype="complex64")
    tx_data = (rng.standard_normal((n_batch, grid.n_data)) + 1j * rng.standard_normal((n_batch, grid.n_data))).astype(
        "complex64"
    )
    tx_data /= np.sqrt(np.mean(np.abs(tx_data) ** 2))  # normalize power like Modem does

    rx_pilots = np.tile(tx_pilots * h_true[grid.pilot_indices], (n_batch, 1))
    rx_data = tx_data * h_true[grid.data_indices]

    est = LSChannelEstimator(grid.pilot_indices, grid.fft_size, tx_pilots, backend="numpy")
    h_hat_full = est.process(rx_pilots)
    h_hat_data = h_hat_full[:, grid.data_indices]

    eq = Equalizer(backend="numpy", **kwargs)
    recovered = eq.process(rx_data, channel_est=h_hat_data)

    # Elementwise recovery is only approximate (channel-estimate interpolation
    # error, occasionally amplified near spectral nulls) -- NMSE over the
    # whole batch is the meaningful, non-flaky metric here.
    assert _nmse(recovered, tx_data) < 0.02


def test_equalizer_requires_channel_est_kwarg():
    eq = ZFEqualizer(backend="numpy")
    with pytest.raises(ValueError):
        eq.process(np.zeros(4, dtype="complex64"))
