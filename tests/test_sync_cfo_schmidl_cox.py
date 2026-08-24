import numpy as np
import pytest

from spectracuda.cfo import SchmidlCoxCFO
from spectracuda.sync import SchmidlCoxSync


def _embed(preamble, true_offset, tail_len, dtype="complex64"):
    zeros_before = np.zeros(true_offset, dtype=dtype)
    zeros_after = np.zeros(tail_len, dtype=dtype)
    return np.concatenate([zeros_before, preamble, zeros_after]).astype(dtype)


def test_odd_fft_size_rejected():
    with pytest.raises(ValueError):
        SchmidlCoxSync(63, backend="numpy")
    with pytest.raises(ValueError):
        SchmidlCoxCFO(63, backend="numpy")


def test_preamble_has_two_identical_halves():
    fft_size = 64
    sync = SchmidlCoxSync(fft_size, backend="numpy")
    preamble = sync.generate_preamble(seed=1)
    L = fft_size // 2
    np.testing.assert_allclose(preamble[:L], preamble[L:], atol=1e-6)


def test_exact_timing_and_metric_no_noise():
    fft_size = 64
    sync = SchmidlCoxSync(fft_size, backend="numpy")
    preamble = sync.generate_preamble(seed=1)
    true_offset = 37
    rx = _embed(preamble, true_offset, tail_len=50)[None, :]

    result = sync.process(rx)
    assert int(result["start_index"][0]) == true_offset
    assert result["metric"][0] == pytest.approx(1.0, abs=1e-4)


def test_boundary_regression_no_spurious_metric_blowup():
    """Regression test for a real bug found during development: the
    textbook second-half-only R(d) blows up (metric >> 1, e.g. ~77 was
    observed) for candidate windows straddling the preamble/silence
    boundary. The symmetric-energy R(d) fix must keep every metric value
    bounded at/below ~1 (up to floating-point slack)."""
    fft_size = 64
    sync = SchmidlCoxSync(fft_size, backend="numpy")
    preamble = sync.generate_preamble(seed=1)
    true_offset = 37
    rx_clean = _embed(preamble, true_offset, tail_len=50)

    rng = np.random.default_rng(0)
    sig_power = float(np.mean(np.abs(preamble) ** 2))
    noise_std = np.sqrt((sig_power / 10 ** 3) / 2)  # 30 dB SNR
    noise = (rng.standard_normal(rx_clean.shape[-1]) + 1j * rng.standard_normal(rx_clean.shape[-1])) * noise_std
    rx = (rx_clean + noise).astype("complex64")[None, :]

    result = sync.process(rx)
    assert int(result["start_index"][0]) == true_offset
    assert result["metric"][0] < 1.1  # bounded, not the ~77 regression


def test_cfo_estimate_matches_known_offset_no_noise():
    fft_size = 64
    sync = SchmidlCoxSync(fft_size, backend="numpy")
    cfo_block = SchmidlCoxCFO(fft_size, backend="numpy")
    preamble = sync.generate_preamble(seed=1)

    true_offset = 20
    eps_true = 0.3
    n = np.arange(fft_size)
    preamble_cfo = (preamble * np.exp(1j * 2 * np.pi * eps_true * n / fft_size)).astype("complex64")
    rx = _embed(preamble_cfo, true_offset, tail_len=30)[None, :]

    result = sync.process(rx)
    assert int(result["start_index"][0]) == true_offset

    est = cfo_block.process(rx, start_index=result["start_index"])
    assert est[0] == pytest.approx(eps_true, abs=1e-3)


def test_cfo_correct_undoes_the_offset():
    fft_size = 64
    sync = SchmidlCoxSync(fft_size, backend="numpy")
    cfo_block = SchmidlCoxCFO(fft_size, backend="numpy")
    preamble = sync.generate_preamble(seed=1)

    eps_true = -0.4
    n = np.arange(fft_size)
    preamble_cfo = (preamble * np.exp(1j * 2 * np.pi * eps_true * n / fft_size)).astype("complex64")
    rx = preamble_cfo[None, :]

    est = cfo_block.process(rx, start_index=np.array([0]))
    corrected = cfo_block.correct(rx, est)
    np.testing.assert_allclose(corrected[0], preamble, atol=1e-4)


def test_cfo_requires_start_index():
    cfo_block = SchmidlCoxCFO(64, backend="numpy")
    with pytest.raises(ValueError):
        cfo_block.process(np.zeros((1, 64), dtype="complex64"))


def test_too_short_signal_raises():
    sync = SchmidlCoxSync(64, backend="numpy")
    with pytest.raises(ValueError):
        sync.process(np.zeros((1, 30), dtype="complex64"))
