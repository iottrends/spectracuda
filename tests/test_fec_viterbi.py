import numpy as np
import pytest

from spectracuda.fec.viterbi import ConvolutionalCode


def test_clean_round_trip():
    cc = ConvolutionalCode(backend="numpy")
    rng = np.random.default_rng(0)
    k = 50
    bits = rng.integers(0, 2, size=(4, k)).astype("uint8")
    encoded = cc.encode(bits)
    assert encoded.shape == (4, 2 * (k + cc.tail_bits))
    decoded = cc.decode(encoded)
    np.testing.assert_array_equal(decoded, bits)


def test_corrects_a_few_bit_errors():
    cc = ConvolutionalCode(backend="numpy")
    rng = np.random.default_rng(0)
    k = 50
    bits = rng.integers(0, 2, size=(4, k)).astype("uint8")
    encoded = cc.encode(bits)
    noisy = encoded.copy()
    n_batch, n = noisy.shape
    for b in range(n_batch):
        positions = rng.choice(n, size=2, replace=False)  # well within dfree=10 correction capability
        noisy[b, positions] ^= 1
    decoded = cc.decode(noisy)
    np.testing.assert_array_equal(decoded, bits)


def test_k_equals_one_edge_case():
    cc = ConvolutionalCode(backend="numpy")
    bits = np.array([[1]], dtype="uint8")
    encoded = cc.encode(bits)
    decoded = cc.decode(encoded)
    np.testing.assert_array_equal(decoded, bits)


def test_stress_reduces_ber_substantially_at_moderate_noise():
    """Broad statistical check across many random codewords/error
    patterns: at ~3% raw bit-error rate (fairly harsh for a rate-1/2
    K=7, dfree=10 code), most trials should decode perfectly, and the
    residual BER after decoding should be far below the raw BER."""
    cc = ConvolutionalCode(backend="numpy")
    rng = np.random.default_rng(1)
    k, n_trials, n_batch = 100, 40, 8
    perfect = 0
    raw_errors, residual_errors = [], []
    for _ in range(n_trials):
        bits = rng.integers(0, 2, size=(n_batch, k)).astype("uint8")
        encoded = cc.encode(bits)
        flip_mask = rng.random(encoded.shape) < 0.03
        noisy = encoded ^ flip_mask.astype("uint8")
        decoded = cc.decode(noisy)
        if np.array_equal(decoded, bits):
            perfect += 1
        raw_errors.append(flip_mask.sum())
        residual_errors.append(np.sum(decoded != bits))

    assert perfect / n_trials > 0.8
    assert np.mean(residual_errors) < 0.05 * np.mean(raw_errors)


def test_process_is_alias_for_encode():
    cc = ConvolutionalCode(backend="numpy")
    bits = np.array([[0, 1, 1, 0]], dtype="uint8")
    np.testing.assert_array_equal(cc.process(bits), cc.encode(bits))
