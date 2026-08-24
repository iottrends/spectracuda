import numpy as np
import pytest

from spectracuda.ofdm import OfdmDemodulator, OfdmModulator


def test_round_trip_identity_no_channel():
    rng = np.random.default_rng(0)
    fft_size, cp_len = 64, 16
    mod = OfdmModulator(fft_size, cp_len, backend="numpy")
    demod = OfdmDemodulator(fft_size, cp_len, backend="numpy")

    freq = (rng.standard_normal((5, fft_size)) + 1j * rng.standard_normal((5, fft_size))).astype(
        "complex64"
    )
    time_domain = mod.process(freq)
    assert time_domain.shape == (5, fft_size + cp_len)

    recovered = demod.process(time_domain)
    assert recovered.shape == freq.shape


def test_output_dtype_is_complex64_not_complex128():
    """Regression test: numpy.fft has no single-precision code path and
    always computes/returns complex128 internally, regardless of input
    dtype. Left unfixed, every OFDM symbol would silently double in
    precision (and cost) on the numpy backend, and numpy/cupy backends
    would disagree on output precision for the same config (cupy's FFT,
    via cuFFT, does preserve complex64)."""
    fft_size, cp_len = 32, 8
    mod = OfdmModulator(fft_size, cp_len, backend="numpy")
    demod = OfdmDemodulator(fft_size, cp_len, backend="numpy")

    freq = np.ones((2, fft_size), dtype="complex64")
    time_domain = mod.process(freq)
    assert time_domain.dtype == np.complex64

    recovered = demod.process(time_domain)
    assert recovered.dtype == np.complex64
    np.testing.assert_allclose(recovered, freq, atol=1e-4)


def test_cp_len_zero_is_allowed():
    mod = OfdmModulator(32, 0, backend="numpy")
    demod = OfdmDemodulator(32, 0, backend="numpy")
    freq = np.ones((2, 32), dtype="complex64")
    time_domain = mod.process(freq)
    assert time_domain.shape == (2, 32)
    np.testing.assert_allclose(demod.process(time_domain), freq, atol=1e-4)


def test_cp_len_out_of_range_raises():
    with pytest.raises(ValueError):
        OfdmModulator(32, 32, backend="numpy")
    with pytest.raises(ValueError):
        OfdmModulator(32, -1, backend="numpy")


def test_wrong_input_shape_raises():
    mod = OfdmModulator(32, 8, backend="numpy")
    with pytest.raises(ValueError):
        mod.process(np.zeros((2, 31), dtype="complex64"))

    demod = OfdmDemodulator(32, 8, backend="numpy")
    with pytest.raises(ValueError):
        demod.process(np.zeros((2, 39), dtype="complex64"))


def test_cyclic_prefix_matches_tail_of_time_domain_symbol():
    fft_size, cp_len = 16, 4
    mod = OfdmModulator(fft_size, cp_len, backend="numpy")
    freq = np.zeros((1, fft_size), dtype="complex64")
    freq[0, 3] = 1.0  # single active subcarrier -> nonzero cp to check
    out = mod.process(freq)
    np.testing.assert_allclose(out[0, :cp_len], out[0, -cp_len:], atol=1e-6)
