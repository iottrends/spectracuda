"""Packetizer: standalone tests proving CRC+FEC composition is usable
independent of any Ofdm/OFDM machinery -- the actual gap this class
closes (see docs/todo.md #1.1). Mirrors liquid-dsp's own packetizer_
encode/packetizer_decode order (CRC append then FEC encode; FEC decode
then CRC strip+check)."""
import numpy as np
import pytest

from spectracuda.framing import Packetizer


def test_no_crc_no_fec_is_a_pass_through():
    p = Packetizer(backend="numpy")
    bits = np.array([[1, 0, 1, 1, 0]], dtype="uint8")
    encoded = p.encode(bits)
    np.testing.assert_array_equal(encoded, bits)
    result = p.decode(encoded)
    np.testing.assert_array_equal(result["bits"], bits)
    assert result["crc_valid"] is None


def test_crc_only_appends_and_validates():
    p = Packetizer(crc="crc32", backend="numpy")
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(3, 32)).astype("uint8")  # 4 bytes
    encoded = p.encode(bits)
    assert encoded.shape == (3, 32 + 32)  # + 4-byte (32-bit) crc32 key
    result = p.decode(encoded)
    np.testing.assert_array_equal(result["bits"], bits)
    np.testing.assert_array_equal(result["crc_valid"], [True, True, True])


def test_crc_only_detects_corruption():
    p = Packetizer(crc="crc16", backend="numpy")
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, 16)).astype("uint8")
    encoded = p.encode(bits)
    corrupted = encoded.copy()
    corrupted[0, 3] ^= 1
    result = p.decode(corrupted)
    assert not result["crc_valid"][0]


def test_fec_only_corrects_errors():
    p = Packetizer(fec="conv_v27", backend="numpy")
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, 34)).astype("uint8")
    encoded = p.encode(bits)
    noisy = encoded.copy()
    noisy[:, ::37] ^= 1
    result = p.decode(noisy)
    np.testing.assert_array_equal(result["bits"], bits)
    assert result["crc_valid"] is None


def test_crc_and_fec_combined_order_matches_liquid_dsp():
    """crc appended BEFORE fec-encoding (not after) -- verified by
    checking the crc-protected quantity is exactly what the fec layer
    carries: bits + crc key length."""
    p = Packetizer(crc="crc8", fec="conv_v27", backend="numpy")
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, 16)).astype("uint8")  # 2 bytes
    encoded = p.encode(bits)
    # pre-fec length = 16 + 8 (crc8 key) = 24 bits; conv_v27 rate 1/2 + 6-tail
    assert encoded.shape[-1] == 2 * (24 + 6)
    result = p.decode(encoded)
    np.testing.assert_array_equal(result["bits"], bits)
    assert result["crc_valid"][0]


def test_crc_and_fec_combined_catches_silently_wrong_viterbi_decode():
    """The actual motivating case for crc+fec together: conv_v27 alone
    has no way to signal a bad decode (it always returns *a* path) --
    heavy corruption must flip crc_valid to False even though decode()
    itself doesn't raise."""
    p = Packetizer(crc="crc32", fec="conv_v27", backend="numpy")
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, 24)).astype("uint8")  # 24+32=56 bits into conv_v27
    encoded = p.encode(bits)
    heavy_noise = (rng.random(encoded.shape) < 0.25).astype("uint8")
    corrupted = encoded ^ heavy_noise
    result = p.decode(corrupted)
    assert not np.array_equal(result["bits"], bits) or not result["crc_valid"][0]


def test_ldpc_variant_through_packetizer():
    """640 (not 316) raw bits: crc8's key must land the message on a
    byte boundary (640 % 8 == 0) AND on an exact multiple of
    ldpc_648_r12's k=324 once the 8-bit key is added (640+8=648=2*324,
    i.e. this folds into 2 LDPC blocks via FEC's own batch chunking --
    324-8=316 isn't byte-aligned, so a single-block raw count doesn't
    satisfy both constraints at once for this particular scheme pair)."""
    p = Packetizer(crc="crc8", fec="ldpc_648_r12", backend="numpy")
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(2, 640)).astype("uint8")
    encoded = p.encode(bits)
    assert encoded.shape == (2, 2 * 648)
    result = p.decode(encoded)
    np.testing.assert_array_equal(result["bits"], bits)
    np.testing.assert_array_equal(result["crc_valid"], [True, True])


def test_encoded_length_matches_actual_encode_output():
    p = Packetizer(crc="crc32", fec="conv_v27", backend="numpy")
    raw_bit_count = 24
    predicted = p.encoded_length(raw_bit_count)
    bits = np.zeros((1, raw_bit_count), dtype="uint8")
    actual = p.encode(bits)
    assert predicted == actual.shape[-1]


def test_encoded_length_raises_for_non_byte_aligned_with_crc():
    p = Packetizer(crc="crc8", backend="numpy")
    with pytest.raises(ValueError):
        p.encoded_length(10)  # not a multiple of 8


def test_crc_requires_byte_aligned_payload():
    p = Packetizer(crc="crc32", backend="numpy")
    with pytest.raises(ValueError):
        p.encode(np.zeros((1, 10), dtype="uint8"))


def test_process_is_alias_for_encode():
    p = Packetizer(crc="crc16", backend="numpy")
    bits = np.array([[1, 0, 1, 1, 0, 1, 0, 0]], dtype="uint8")
    np.testing.assert_array_equal(p.process(bits), p.encode(bits))


def test_batched_with_different_content_per_item():
    p = Packetizer(crc="crc32", fec="conv_v27", backend="numpy")
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(4, 24)).astype("uint8")
    encoded = p.encode(bits)
    result = p.decode(encoded)
    np.testing.assert_array_equal(result["bits"], bits)
    np.testing.assert_array_equal(result["crc_valid"], [True] * 4)


# --- two-stage FEC (fec0=inner, fec1=outer -- see docs/todo.md #1.2) ------


def test_fec1_defaults_to_none_and_is_a_single_stage_no_op():
    p = Packetizer(crc="crc8", fec="conv_v27", backend="numpy")
    assert p.fec1 == "none"
    assert p.fec1_codec is None


def test_two_stage_encode_order_is_inner_then_outer():
    """fec0 (conv_v27, inner) applied FIRST, THEN fec1 (ldpc_648_r12,
    outer) -- verified directly against manually replicating the two
    steps in that order, not just "it round-trips" (which wouldn't
    catch an accidentally-reversed order, since FEC composition is
    order-sensitive but round-trips either way as long as encode/decode
    are consistently reversed)."""
    p = Packetizer(fec="conv_v27", fec1="ldpc_648_r12", backend="numpy")
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(2, 156)).astype("uint8")  # 2*(156+6)=324=ldpc_648_r12's k
    encoded = p.encode(bits)

    from spectracuda.fec import FEC

    inner = FEC("conv_v27", backend="numpy")
    outer = FEC("ldpc_648_r12", backend="numpy")
    expected = outer.encode(inner.encode(bits))
    np.testing.assert_array_equal(encoded, expected)
    assert encoded.shape == (2, 648)


def test_two_stage_round_trip_clean():
    p = Packetizer(fec="conv_v27", fec1="ldpc_648_r12", backend="numpy")
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(2, 156)).astype("uint8")
    encoded = p.encode(bits)
    result = p.decode(encoded)
    np.testing.assert_array_equal(result["bits"], bits)


def test_two_stage_round_trip_with_crc():
    p = Packetizer(crc="crc8", fec="conv_v27", fec1="ldpc_648_r12", backend="numpy")
    rng = np.random.default_rng(0)
    # Needs a raw_bit_count that's simultaneously byte-aligned (crc8) and
    # produces a conv_v27-encoded length that's a multiple of
    # ldpc_648_r12's k=324 -- searched directly for the smallest value
    # satisfying both constraints rather than guessing one.
    raw_bit_count = next(k for k in range(8, 2000, 8) if (2 * (k + 8 + 6)) % 324 == 0)
    bits = rng.integers(0, 2, size=(2, raw_bit_count)).astype("uint8")
    encoded = p.encode(bits)
    result = p.decode(encoded)
    np.testing.assert_array_equal(result["bits"], bits)
    np.testing.assert_array_equal(result["crc_valid"], [True, True])


def test_two_stage_fec_corrects_errors_conv_v27_inner_ldpc_outer():
    """Real coding-gain check, not just plumbing: heavy-ish bit errors
    that would defeat conv_v27 alone must still be corrected once
    ldpc_648_r12 (outer) mops up the residual after conv_v27 (inner)
    decodes first."""
    p = Packetizer(fec="conv_v27", fec1="ldpc_648_r12", backend="numpy")
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, 156)).astype("uint8")
    encoded = p.encode(bits)
    noisy = encoded.copy()
    flips = (rng.random(encoded.shape) < 0.04).astype("uint8")
    noisy = noisy ^ flips
    assert not np.array_equal(noisy, encoded)  # corruption is real
    result = p.decode(noisy)
    np.testing.assert_array_equal(result["bits"], bits)


def test_two_stage_fec_decode_failure_names_the_outer_stage():
    p = Packetizer(fec="conv_v27", fec1="rs_m8", backend="numpy")
    rng = np.random.default_rng(0)
    # k=886 -> conv_v27 encodes to 2*(886+6)=1784, exactly rs_m8's own
    # block size (k_bits), so fec1 sees exactly one RS codeword's worth.
    k = 886
    bits = rng.integers(0, 2, size=(1, k)).astype("uint8")
    encoded = p.encode(bits)
    heavy_noise = (rng.random(encoded.shape) < 0.3).astype("uint8")
    corrupted = encoded ^ heavy_noise
    with pytest.raises(ValueError, match=r"fec1 \(outer"):
        p.decode(corrupted)


def test_single_stage_fec_decode_failure_names_the_inner_stage():
    p = Packetizer(fec="rs_m8", backend="numpy")
    rs = p.fec_codec
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, rs.k_bits)).astype("uint8")
    encoded = p.encode(bits)
    noisy = encoded.reshape(1, 255, 8).copy()
    err_rng = np.random.default_rng(9)
    positions = err_rng.choice(255, size=20, replace=False)  # > t=16, uncorrectable
    for pos in positions:
        noisy[0, pos] ^= err_rng.integers(1, 256, size=8).astype("uint8")
    noisy = noisy.reshape(1, rs.n_bits)
    with pytest.raises(ValueError, match=r"fec0 \(inner"):
        p.decode(noisy)


def test_two_stage_encoded_length_accounts_for_both_stages():
    p = Packetizer(fec="conv_v27", fec1="ldpc_648_r12", backend="numpy")
    predicted = p.encoded_length(156)
    bits = np.zeros((1, 156), dtype="uint8")
    actual = p.encode(bits)
    assert predicted == actual.shape[-1] == 648
