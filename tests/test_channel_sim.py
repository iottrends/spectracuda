import numpy as np
import pytest

from spectracuda.sim import Channel


def test_no_impairments_is_identity():
    channel = Channel(backend="numpy")
    x = (np.random.default_rng(0).standard_normal((2, 50)) + 1j * 0).astype("complex64")
    y = channel.process(x)
    np.testing.assert_allclose(y, x, atol=1e-6)


def test_awgn_adds_noise_at_requested_snr():
    channel = Channel(snr_db=20.0, seed=0, backend="numpy")
    x = np.ones((1, 10000), dtype="complex64")
    y = channel.process(x)
    noise = y - x
    measured_snr_db = 10 * np.log10(np.mean(np.abs(x) ** 2) / np.mean(np.abs(noise) ** 2))
    assert measured_snr_db == pytest.approx(20.0, abs=1.0)


def test_multipath_output_shape_matches_input():
    taps = Channel.random_multipath_taps(3, seed=1)
    channel = Channel(multipath_taps=taps, backend="numpy")
    x = np.ones((2, 20), dtype="complex64")
    y = channel.process(x)
    assert y.shape == x.shape


def test_random_multipath_taps_unit_energy():
    taps = Channel.random_multipath_taps(5, seed=2)
    assert taps.shape == (5,)
    np.testing.assert_allclose(np.sum(np.abs(taps) ** 2), 1.0, atol=1e-5)


def test_cfo_rotates_phase_as_expected():
    fft_size = 64
    eps = 0.25
    channel = Channel(cfo=eps, cfo_fft_size=fft_size, backend="numpy")
    x = np.ones((1, fft_size), dtype="complex64")
    y = channel.process(x)
    n = np.arange(fft_size)
    expected = np.exp(1j * 2 * np.pi * eps * n / fft_size)
    np.testing.assert_allclose(y[0], expected, atol=1e-6)


def test_cfo_without_fft_size_raises():
    with pytest.raises(ValueError):
        Channel(cfo=0.1, backend="numpy")


def test_1d_input_is_promoted_to_batch_of_one():
    channel = Channel(snr_db=30.0, seed=0, backend="numpy")
    x = np.ones(16, dtype="complex64")
    y = channel.process(x)
    assert y.shape == (1, 16)


def test_impairments_compose_awgn_multipath_cfo():
    taps = Channel.random_multipath_taps(3, seed=3)
    channel = Channel(snr_db=30.0, multipath_taps=taps, cfo=0.1, cfo_fft_size=32, seed=3, backend="numpy")
    x = np.ones((2, 32), dtype="complex64")
    y = channel.process(x)
    assert y.shape == x.shape
    assert not np.allclose(y, x)  # something actually happened
