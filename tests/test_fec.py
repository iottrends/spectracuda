import numpy as np
import pytest

from spectracuda.fec import FEC


def test_conv_v27_bit_level_round_trip():
    fec = FEC("conv_v27", backend="numpy")
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(3, 40)).astype("uint8")
    encoded = fec.encode(bits)
    noisy = encoded.copy()
    noisy[:, ::37] ^= 1  # sprinkle a few bit errors, well within correction capability
    decoded = fec.decode(noisy)
    np.testing.assert_array_equal(decoded, bits)


def test_rs_m8_presents_a_uniform_bit_level_interface():
    """ReedSolomonCode itself operates on GF(256) byte symbols, but FEC
    must present the same bits-in/bits-out contract regardless of
    scheme (matching Modem's convention) -- callers should never need
    to know rs_m8 is byte-oriented underneath."""
    fec = FEC("rs_m8", backend="numpy")
    assert fec.k_bits == 223 * 8
    assert fec.n_bits == 255 * 8

    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(2, fec.k_bits)).astype("uint8")
    encoded = fec.encode(bits)
    assert encoded.shape == (2, fec.n_bits)
    decoded = fec.decode(encoded)
    np.testing.assert_array_equal(decoded, bits)


def test_rs_m8_corrects_byte_level_errors_through_the_bit_interface():
    fec = FEC("rs_m8", backend="numpy")
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(2, fec.k_bits)).astype("uint8")
    encoded = fec.encode(bits)

    noisy = encoded.copy().reshape(2, 255, 8)
    for b in range(2):
        rng_b = np.random.default_rng(b + 5)
        byte_positions = rng_b.choice(255, size=16, replace=False)  # t=16 max
        for p in byte_positions:
            noisy[b, p] ^= rng_b.integers(0, 2, size=8).astype("uint8")
    noisy = noisy.reshape(2, fec.n_bits)

    decoded = fec.decode(noisy)
    np.testing.assert_array_equal(decoded, bits)


def test_unknown_scheme_raises():
    """"ldpc" bare (no size/rate suffix) used to be this test's example
    of an unimplemented-but-plausible-looking scheme name; now that LDPC
    is real (12 "ldpc_<n>_r<rate>" variants, see test_fec_ldpc.py), swap
    in a genuinely nonexistent name instead, so this test can't be
    accidentally "passing" only because it names a real scheme wrong."""
    with pytest.raises(ValueError):
        FEC("polar", backend="numpy")


def test_rs_m8_wrong_bit_count_raises():
    """100 used to be the "wrong length" example -- it no longer is: FEC
    now supports "shortened" rs_m8 (N full 1784-bit blocks + at most one
    leftover block of ANY byte-aligned size, see fec.py's encoded_length()
    docstring) -- the genuinely invalid case moved to a NON-byte-aligned
    leftover (100 % 8 != 0), which decode() could never unambiguously
    reverse (np.packbits would otherwise silently zero-pad it -- a real
    bug caught while building this, see the encode()/decode() comments)."""
    fec = FEC("rs_m8", backend="numpy")
    with pytest.raises(ValueError):
        fec.encode(np.zeros((1, 100), dtype="uint8"))  # 100 % 8 != 0
    # a genuinely byte-aligned, non-block-multiple size is fine now:
    fec.encode(np.zeros((1, 96), dtype="uint8"))  # no raise


def test_rs_m8_shortened_leftover_block_round_trips():
    """A real message that isn't an exact multiple of 1784 bits (the
    actual, common case -- e.g. a 13-byte bind request is 104 bits after
    CRC) now round-trips, encoded as ONLY the real bytes + parity, not
    padded up to a full 1784-bit block -- see fec.py's encoded_length()
    docstring for why."""
    fec = FEC("rs_m8", backend="numpy")
    bits = np.random.default_rng(0).integers(0, 2, size=(1, 104)).astype("uint8")
    encoded = fec.encode(bits)
    assert encoded.shape[-1] == 104 + 8 * 32  # real bits + parity, NOT padded to 2040
    np.testing.assert_array_equal(fec.decode(encoded), bits)


def test_rs_m8_multi_block_plus_leftover_round_trips():
    """N full blocks + one shortened leftover block together -- the
    general case a big, multi-segment message actually needs, not just
    the small-message case above."""
    fec = FEC("rs_m8", backend="numpy")
    n_bits = 1784 * 2 + 800  # 2 full blocks + an 800-bit leftover
    bits = np.random.default_rng(1).integers(0, 2, size=(1, n_bits)).astype("uint8")
    encoded = fec.encode(bits)
    assert encoded.shape[-1] == fec.encoded_length(n_bits)
    np.testing.assert_array_equal(fec.decode(encoded), bits)


def test_process_is_alias_for_encode():
    fec = FEC("conv_v27", backend="numpy")
    bits = np.array([[0, 1, 1, 0]], dtype="uint8")
    np.testing.assert_array_equal(fec.process(bits), fec.encode(bits))
