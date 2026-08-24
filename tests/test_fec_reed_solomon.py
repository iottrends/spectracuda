import numpy as np
import pytest

from spectracuda.fec.reed_solomon import ReedSolomonCode


def _rs():
    return ReedSolomonCode(backend="numpy")


def test_dimensions():
    rs = _rs()
    assert (rs.n, rs.k, rs.nroots, rs.t) == (255, 223, 32, 16)


def test_clean_round_trip_batched():
    rs = _rs()
    rng = np.random.default_rng(0)
    msg = rng.integers(0, 256, size=(4, 223)).astype("uint8")
    codeword = rs.encode(msg)
    assert codeword.shape == (4, 255)
    decoded = rs.decode(codeword)
    np.testing.assert_array_equal(decoded, msg)


@pytest.mark.parametrize("n_errors", [1, 5, 10, 16])
def test_corrects_up_to_t_symbol_errors(n_errors):
    """t=16 is RS(255,223)'s theoretical maximum correctable symbol
    errors per codeword (nroots=32 -> t=nroots//2)."""
    rs = _rs()
    rng = np.random.default_rng(0)
    msg = rng.integers(0, 256, size=(1, 223)).astype("uint8")
    codeword = rs.encode(msg)[0]

    err_rng = np.random.default_rng(n_errors * 7)
    positions = err_rng.choice(255, size=n_errors, replace=False)
    noisy = codeword.copy()
    for p in positions:
        noisy[p] ^= int(err_rng.integers(1, 256))

    decoded = rs.decode(noisy[None, :])
    np.testing.assert_array_equal(decoded[0], msg[0])


def test_corrects_errors_at_boundary_positions():
    """Regression-style check for the MSB-first/LSB-first indexing bug
    found during development (Chien search's error-location convention
    didn't match this class's array indexing) -- confirmed at the first
    and last codeword positions specifically, since that's where such a
    mismatch would be most obviously wrong."""
    rs = _rs()
    rng = np.random.default_rng(0)
    msg = rng.integers(0, 256, size=(1, 223)).astype("uint8")
    codeword = rs.encode(msg)[0]
    noisy = codeword.copy()
    noisy[0] ^= 200
    noisy[254] ^= 55
    decoded = rs.decode(noisy[None, :])
    np.testing.assert_array_equal(decoded[0], msg[0])


def test_batch_with_different_error_counts_per_item():
    rs = _rs()
    rng = np.random.default_rng(0)
    msg = rng.integers(0, 256, size=(3, 223)).astype("uint8")
    codeword = rs.encode(msg)
    noisy = codeword.copy()
    for b, n_err in enumerate([0, 8, 16]):
        rng_b = np.random.default_rng(b + 1)
        positions = rng_b.choice(255, size=n_err, replace=False)
        for p in positions:
            noisy[b, p] ^= int(rng_b.integers(1, 256))
    decoded = rs.decode(noisy)
    np.testing.assert_array_equal(decoded, msg)


def test_over_capacity_errors_detected_not_silently_wrong():
    """17 errors exceeds t=16 -- must raise ValueError (detected via the
    Berlekamp-Massey-degree/Chien-search-root-count mismatch), not
    silently return incorrect data."""
    rs = _rs()
    rng = np.random.default_rng(0)
    msg = rng.integers(0, 256, size=(1, 223)).astype("uint8")
    codeword = rs.encode(msg)[0]
    err_rng = np.random.default_rng(99)
    positions = err_rng.choice(255, size=17, replace=False)
    noisy = codeword.copy()
    for p in positions:
        noisy[p] ^= int(err_rng.integers(1, 256))
    with pytest.raises(ValueError):
        rs.decode(noisy[None, :])


def test_wrong_message_length_raises():
    rs = _rs()
    with pytest.raises(ValueError):
        rs.encode(np.zeros((1, 100), dtype="uint8"))


def test_wrong_codeword_length_raises():
    rs = _rs()
    with pytest.raises(ValueError):
        rs.decode(np.zeros((1, 100), dtype="uint8"))


def test_process_is_alias_for_encode():
    rs = _rs()
    msg = np.zeros((1, 223), dtype="uint8")
    np.testing.assert_array_equal(rs.process(msg), rs.encode(msg))
