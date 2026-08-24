import numpy as np
import pytest

from spectracuda.sync import ZadoffChuSync


def _embed(preamble, true_offset, tail_len, dtype="complex64"):
    zeros_before = np.zeros(true_offset, dtype=dtype)
    zeros_after = np.zeros(tail_len, dtype=dtype)
    return np.concatenate([zeros_before, preamble, zeros_after]).astype(dtype)


def test_root_not_coprime_with_fft_size_raises():
    # gcd(2, 64) = 2 != 1
    with pytest.raises(ValueError):
        ZadoffChuSync(64, root=2, backend="numpy")


def test_preamble_is_unit_amplitude_and_right_length():
    fft_size = 63  # odd, exercises the N-odd formula branch
    sync = ZadoffChuSync(fft_size, backend="numpy")
    preamble = sync.generate_preamble()
    assert preamble.shape == (fft_size,)
    np.testing.assert_allclose(np.abs(preamble), 1.0, atol=1e-6)


def test_even_fft_size_also_works():
    fft_size = 64
    sync = ZadoffChuSync(fft_size, backend="numpy")
    preamble = sync.generate_preamble()
    assert preamble.shape == (fft_size,)
    np.testing.assert_allclose(np.abs(preamble), 1.0, atol=1e-6)


def test_sequence_has_ideal_cyclic_autocorrelation():
    """The defining CAZAC property: circular autocorrelation of a
    Zadoff-Chu sequence with a cyclic-shifted copy of itself is (near)
    zero at every nonzero shift, and equal to N at zero shift -- checked
    directly against the sequence generator, independent of the sync
    detection algorithm built on top of it."""
    fft_size = 63
    sync = ZadoffChuSync(fft_size, root=1, backend="numpy")
    x = np.asarray(sync.generate_preamble())
    for shift in [1, 5, 20, 40, 62]:
        shifted = np.roll(x, shift)
        autocorr = np.sum(np.conj(x) * shifted)
        assert abs(autocorr) < 1e-6 * fft_size, f"shift={shift}: |autocorr|={abs(autocorr)}"
    autocorr_zero = np.sum(np.conj(x) * x)
    assert autocorr_zero == pytest.approx(fft_size, abs=1e-4)


def test_matches_brute_force_reference_correlation():
    """Regression/correctness check for the FFT-based matched-filter
    indexing itself (see module docstring): compare against a direct,
    unoptimized double-loop correlation on a small synthetic signal --
    the same validation done ad hoc before trusting the FFT approach,
    now kept as a permanent test."""
    fft_size = 16
    sync = ZadoffChuSync(fft_size, backend="numpy")
    template = np.asarray(sync.generate_preamble())

    rng = np.random.default_rng(0)
    n_samples = 40
    rx = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)).astype("complex64")

    n_candidates = n_samples - fft_size + 1
    brute_force = np.array(
        [np.sum(np.conj(template) * rx[d : d + fft_size]) for d in range(n_candidates)]
    )

    result = sync.process(rx[None, :])
    # Reconstruct the raw (unnormalized) correlation the same way process() does,
    # to compare directly against the brute-force reference at every candidate --
    # not just at the argmax.
    kernel = np.conj(template[::-1])
    conv_len = n_samples + fft_size - 1
    n_fft = 1
    while n_fft < conv_len:
        n_fft *= 2
    full_conv = np.fft.ifft(np.fft.fft(rx, n=n_fft) * np.fft.fft(kernel, n=n_fft))
    y = full_conv[fft_size - 1 : fft_size - 1 + n_candidates]
    np.testing.assert_allclose(y, brute_force, atol=1e-3)
    assert result["start_index"][0] == np.argmax(np.abs(brute_force) ** 2)


def test_detects_known_offset_noiseless():
    fft_size = 64
    sync = ZadoffChuSync(fft_size, backend="numpy")
    preamble = sync.generate_preamble()
    true_offset = 37
    rx = _embed(preamble, true_offset, tail_len=50)[None, :]

    result = sync.process(rx)
    assert int(result["start_index"][0]) == true_offset
    assert result["metric"][0] == pytest.approx(1.0, abs=1e-4)


def test_metric_high_for_true_preamble_low_for_pure_noise():
    """Detection-separation property: a genuine embedded preamble scores
    near 1.0, while pure noise of comparable power never does -- the
    normalized matched filter must actually discriminate, not just
    return some bounded-but-meaningless number."""
    fft_size = 64
    sync = ZadoffChuSync(fft_size, backend="numpy")
    preamble = sync.generate_preamble()
    true_offset = 20
    rx_signal = _embed(preamble, true_offset, tail_len=40)[None, :]
    result_signal = sync.process(rx_signal)
    assert result_signal["metric"][0] > 0.9

    rng = np.random.default_rng(0)
    n_samples = rx_signal.shape[-1]
    noise_only = (
        (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) / np.sqrt(2)
    ).astype("complex64")[None, :]
    result_noise = sync.process(noise_only)
    assert result_noise["metric"][0] < 0.5


def test_batch_with_different_offsets_per_item():
    fft_size = 64
    sync = ZadoffChuSync(fft_size, backend="numpy")
    preamble = sync.generate_preamble()
    offsets = [10, 55, 30]
    rows = [_embed(preamble, off, tail_len=80 - off) for off in offsets]
    rx = np.stack(rows, axis=0)

    result = sync.process(rx)
    np.testing.assert_array_equal(result["start_index"], offsets)
    assert (result["metric"] > 0.9).all()


def test_survives_awgn():
    fft_size = 64
    sync = ZadoffChuSync(fft_size, backend="numpy")
    preamble = sync.generate_preamble()
    true_offset = 37
    rx_clean = _embed(preamble, true_offset, tail_len=50)

    rng = np.random.default_rng(0)
    sig_power = float(np.mean(np.abs(preamble) ** 2))
    noise_std = np.sqrt((sig_power / 10 ** 2) / 2)  # 20 dB SNR
    noise = (rng.standard_normal(rx_clean.shape[-1]) + 1j * rng.standard_normal(rx_clean.shape[-1])) * noise_std
    rx = (rx_clean + noise).astype("complex64")[None, :]

    result = sync.process(rx)
    assert int(result["start_index"][0]) == true_offset


def test_too_short_signal_raises():
    sync = ZadoffChuSync(64, backend="numpy")
    with pytest.raises(ValueError):
        sync.process(np.zeros((1, 30), dtype="complex64"))
