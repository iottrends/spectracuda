import numpy as np
import pytest

from spectracuda.modem import Modem

SCHEMES = ["bpsk", "qpsk", "qam16", "qam64", "qam256"]


@pytest.mark.parametrize("scheme", SCHEMES)
def test_round_trip_no_noise(scheme):
    rng = np.random.default_rng(0)
    modem = Modem(scheme, backend="numpy")
    n_symbols = 500
    bits = rng.integers(0, 2, size=(3, n_symbols * modem.bits_per_symbol)).astype(
        "uint8"
    )
    symbols = modem.modulate(bits)
    recovered = modem.demodulate(symbols)
    assert recovered.shape == bits.shape
    assert np.array_equal(recovered, bits)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_modulate_output_dtype_is_complex64(scheme):
    """Regression guard: _pam_level used to build its array in float64
    before the final downcast, paying double-precision compute for every
    symbol for nothing (see mapper.py's _pam_level docstring). This
    checks the final dtype contract, not the intermediate dtype -- the
    real fix is in mapper.py itself."""
    modem = Modem(scheme, backend="numpy")
    bits = np.zeros((1, 8 * modem.bits_per_symbol), dtype="uint8")
    assert modem.modulate(bits).dtype == np.complex64


@pytest.mark.parametrize("scheme", SCHEMES)
def test_average_power_is_normalized(scheme):
    rng = np.random.default_rng(1)
    modem = Modem(scheme, backend="numpy")
    bits = rng.integers(0, 2, size=(1, 20000 * modem.bits_per_symbol)).astype("uint8")
    symbols = modem.modulate(bits)
    avg_power = float(np.mean(np.abs(symbols) ** 2))
    assert avg_power == pytest.approx(1.0, abs=0.02)


def test_qpsk_known_mapping():
    modem = Modem("qpsk", backend="numpy")
    # bits (I, Q): 0->-1 level, 1->+1 level (gray-trivial for 1 bit/axis)
    bits = np.array([[0, 0, 0, 1, 1, 0, 1, 1]], dtype="uint8")
    symbols = modem.modulate(bits)
    norm = 1.0 / np.sqrt(2)
    expected = np.array([[-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j]]) * norm
    np.testing.assert_allclose(symbols, expected, atol=1e-6)


def test_unknown_scheme_raises():
    with pytest.raises(ValueError):
        Modem("qam1024", backend="numpy")


def test_bit_count_not_multiple_of_bits_per_symbol_raises():
    modem = Modem("qam16", backend="numpy")
    with pytest.raises(ValueError):
        modem.modulate(np.zeros((1, 5), dtype="uint8"))


def test_process_is_alias_for_modulate():
    modem = Modem("qpsk", backend="numpy")
    bits = np.array([[0, 1, 1, 0]], dtype="uint8")
    np.testing.assert_array_equal(modem.process(bits), modem.modulate(bits))
