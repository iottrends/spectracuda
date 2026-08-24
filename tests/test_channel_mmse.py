import numpy as np
import pytest

from spectracuda.channel.mmse import MMSEChannelEstimator, _uniform_pdp_correlation


def test_uniform_pdp_correlation_matches_direct_sum():
    """Closed-form R_H[k] checked against a direct (slow) sum over l,
    for several (k, N, L) combinations, before being trusted."""
    N, L = 64, 16
    for k in [0, 1, 5, -3, 63, 64, -64, 100]:
        direct = sum(np.exp(-1j * 2 * np.pi * k * l / N) for l in range(L)) / L
        closed = _uniform_pdp_correlation(np.array([k]), N, L)[0]
        assert closed == pytest.approx(direct, abs=1e-9)


def test_dimensions_and_batch_shape():
    fft_size = 64
    pilot_indices = np.arange(0, 64, 8)  # 8 pilots
    tx_pilots = np.ones(len(pilot_indices), dtype="complex64")
    est = MMSEChannelEstimator(pilot_indices, fft_size, tx_pilots, cp_len=8, backend="numpy")
    rx = np.ones((3, len(pilot_indices)), dtype="complex64")
    h_full = est.process(rx)
    assert h_full.shape == (3, fft_size)


def test_reduces_nmse_versus_ls_interpolation_under_noise():
    """MMSE (with a correctly-matched max_delay assumption) must beat
    plain LS+linear-interpolation's NMSE against the true channel, at a
    realistic pilot count >= the assumed delay spread (well-determined
    regime) -- checked directly, not assumed."""
    fft_size = 64
    max_delay = 16
    rng = np.random.default_rng(0)
    taps = (rng.standard_normal(max_delay) + 1j * rng.standard_normal(max_delay)) / np.sqrt(2 * max_delay)
    h_true = np.fft.fft(taps, fft_size)

    pilot_indices = np.sort(rng.choice(fft_size, size=24, replace=False))
    tx_pilots = np.ones(len(pilot_indices), dtype="complex64")
    noise_var = 0.05
    noise = (rng.standard_normal(len(pilot_indices)) + 1j * rng.standard_normal(len(pilot_indices)))
    noise *= np.sqrt(noise_var / 2)
    rx_pilots = (h_true[pilot_indices] + noise).astype("complex64")[None, :]

    est = MMSEChannelEstimator(pilot_indices, fft_size, tx_pilots, cp_len=max_delay,
                                noise_var=noise_var, backend="numpy")
    h_mmse = np.asarray(est.process(rx_pilots))[0]
    nmse_mmse = np.mean(np.abs(h_mmse - h_true) ** 2) / np.mean(np.abs(h_true) ** 2)

    m = np.arange(fft_size)
    real_i = np.interp(m, pilot_indices, rx_pilots[0].real)
    imag_i = np.interp(m, pilot_indices, rx_pilots[0].imag)
    h_interp = real_i + 1j * imag_i
    nmse_interp = np.mean(np.abs(h_interp - h_true) ** 2) / np.mean(np.abs(h_true) ** 2)

    assert nmse_mmse < nmse_interp


def test_converges_to_near_perfect_when_well_determined_and_low_noise():
    """When n_pilot >= max_delay and noise_var is small, MMSE with the
    correctly-matched model must recover the true channel almost
    exactly (verified empirically during development: n_pilot=8 against
    max_delay=16 -- an underdetermined case -- left a large NMSE floor
    even at near-zero noise, while n_pilot=32 converged to <1e-9; this
    test pins the well-determined side of that behavior)."""
    fft_size = 64
    max_delay = 16
    rng = np.random.default_rng(1)
    taps = (rng.standard_normal(max_delay) + 1j * rng.standard_normal(max_delay)) / np.sqrt(2 * max_delay)
    h_true = np.fft.fft(taps, fft_size)
    pilot_indices = np.sort(rng.choice(fft_size, size=32, replace=False))
    tx_pilots = np.ones(len(pilot_indices), dtype="complex64")

    est = MMSEChannelEstimator(pilot_indices, fft_size, tx_pilots, cp_len=max_delay,
                                noise_var=1e-8, backend="numpy")
    rx_pilots = h_true[pilot_indices][None, :].astype("complex64")  # no noise
    h_mmse = np.asarray(est.process(rx_pilots))[0]
    nmse = np.mean(np.abs(h_mmse - h_true) ** 2) / np.mean(np.abs(h_true) ** 2)
    assert nmse < 1e-6


def test_underdetermined_case_has_bounded_but_nonzero_floor():
    """The flip side, documented honestly rather than hidden: with fewer
    pilots than the assumed max_delay, even zero actual noise leaves a
    real reconstruction-error floor (not a bug -- there are more assumed
    unknowns than observations)."""
    fft_size = 64
    max_delay = 16
    rng = np.random.default_rng(0)
    taps = (rng.standard_normal(max_delay) + 1j * rng.standard_normal(max_delay)) / np.sqrt(2 * max_delay)
    h_true = np.fft.fft(taps, fft_size)
    pilot_indices = np.sort(rng.choice(fft_size, size=8, replace=False))
    tx_pilots = np.ones(len(pilot_indices), dtype="complex64")

    est = MMSEChannelEstimator(pilot_indices, fft_size, tx_pilots, cp_len=max_delay,
                                noise_var=1e-8, backend="numpy")
    rx_pilots = h_true[pilot_indices][None, :].astype("complex64")
    h_mmse = np.asarray(est.process(rx_pilots))[0]
    nmse = np.mean(np.abs(h_mmse - h_true) ** 2) / np.mean(np.abs(h_true) ** 2)
    assert 0.1 < nmse < 2.0  # real, bounded floor -- not ~0, not blown up


def test_defaults_when_cp_len_not_given():
    fft_size = 64
    pilot_indices = np.arange(0, 64, 4)
    tx_pilots = np.ones(len(pilot_indices), dtype="complex64")
    est = MMSEChannelEstimator(pilot_indices, fft_size, tx_pilots, backend="numpy")
    assert est.max_delay == fft_size // 4


def test_requires_at_least_one_pilot():
    with pytest.raises(ValueError):
        MMSEChannelEstimator(np.array([], dtype=int), 64, np.array([], dtype="complex64"), backend="numpy")
