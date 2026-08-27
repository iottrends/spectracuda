import numpy as np
import pytest

from spectracuda.fec import FEC
from spectracuda.fec.ldpc import LDPCCode, _gf2_inverse
from spectracuda.fec.ldpc_tables import BASE_MATRICES

ALL_VARIANTS = sorted(BASE_MATRICES)

# Published IEEE 802.11n (n, k) pairs -- rate * n, per variant.
_EXPECTED_NK = {
    "ldpc_648_r12": (648, 324),
    "ldpc_648_r23": (648, 432),
    "ldpc_648_r34": (648, 486),
    "ldpc_648_r56": (648, 540),
    "ldpc_1296_r12": (1296, 648),
    "ldpc_1296_r23": (1296, 864),
    "ldpc_1296_r34": (1296, 972),
    "ldpc_1296_r56": (1296, 1080),
    "ldpc_1944_r12": (1944, 972),
    "ldpc_1944_r23": (1944, 1296),
    "ldpc_1944_r34": (1944, 1458),
    "ldpc_1944_r56": (1944, 1620),
}


def _gf2_rank(matrix: np.ndarray) -> int:
    """Independent (from _gf2_inverse) GF(2) rank computation, used
    purely as a structural cross-check on the sourced base matrices --
    deliberately not reusing ldpc.py's own elimination code."""
    m = matrix.copy().astype(np.uint8) % 2
    rows, cols = m.shape
    rank = 0
    for col in range(cols):
        pivot = None
        for r in range(rank, rows):
            if m[r, col]:
                pivot = r
                break
        if pivot is None:
            continue
        m[[rank, pivot]] = m[[pivot, rank]]
        for r in range(rows):
            if r != rank and m[r, col]:
                m[r] ^= m[rank]
        rank += 1
        if rank == rows:
            break
    return rank


def _build_dense_H(variant: str) -> np.ndarray:
    spec = BASE_MATRICES[variant]
    Z, base = spec["Z"], spec["base"]
    mb, nb = len(base), len(base[0])
    H = np.zeros((mb * Z, nb * Z), dtype=np.uint8)
    for br, row in enumerate(base):
        for bc, shift in enumerate(row):
            if shift < 0:
                continue
            for z in range(Z):
                H[br * Z + z, bc * Z + (z + shift) % Z] = 1
    return H


# --- structural sanity (run first, cheapest, highest signal) --------------


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_dimensions_match_published_802_11n_values(variant):
    spec = BASE_MATRICES[variant]
    n = spec["n"]
    mb = len(spec["base"])
    k = n - mb * spec["Z"]
    assert (n, k) == _EXPECTED_NK[variant]


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_parity_submatrix_is_full_rank_over_gf2(variant):
    """The property the systematic encoder relies on (H_p = H's last
    mb*Z columns must be invertible) -- checked directly against the
    sourced base matrix, independent of ldpc.py's own _gf2_inverse
    (which would otherwise just raise ValueError on the same data if
    this were false -- this test isolates the check itself)."""
    H = _build_dense_H(variant)
    mb_z = H.shape[0]
    H_p = H[:, -mb_z:]
    assert _gf2_rank(H_p) == mb_z


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_constructs_without_raising(variant):
    LDPCCode(variant, backend="numpy")


def test_gf2_inverse_matches_identity_round_trip():
    rng = np.random.default_rng(0)
    n = 30
    while True:
        m = rng.integers(0, 2, size=(n, n)).astype(np.uint8)
        if _gf2_rank(m) == n:
            break
    inv = _gf2_inverse(m)
    product = (m.astype("int64") @ inv.astype("int64")) % 2
    np.testing.assert_array_equal(product, np.eye(n, dtype="int64"))


def test_gf2_inverse_raises_on_singular_matrix():
    singular = np.zeros((4, 4), dtype=np.uint8)
    singular[0] = [1, 0, 0, 0]
    singular[1] = [1, 0, 0, 0]  # duplicate row -> singular
    with pytest.raises(ValueError):
        _gf2_inverse(singular)


# --- round-trip, all 12 variants ------------------------------------------


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_clean_round_trip_and_zero_syndrome(variant):
    code = LDPCCode(variant, backend="numpy")
    rng = np.random.default_rng(0)
    msg = rng.integers(0, 2, size=(2, code.k)).astype("uint8")
    codeword = code.encode(msg)
    assert codeword.shape == (2, code.n)
    np.testing.assert_array_equal(codeword[:, : code.k], msg)  # systematic

    H = _build_dense_H(variant)
    syndrome = (np.asarray(codeword).astype("int64") @ H.T.astype("int64")) % 2
    np.testing.assert_array_equal(syndrome, np.zeros_like(syndrome))

    decoded = code.decode(codeword)
    np.testing.assert_array_equal(decoded, msg)


def test_wrong_message_length_raises():
    """10 used to be the "wrong length" example here -- it no longer is:
    shortened LDPC (see ldpc.py's encode()/decode() docstrings, same
    technique as ReedSolomonCode's own shortened-block support) now
    accepts any 1..k, so the genuinely out-of-range boundary moved to 0
    and >k."""
    code = LDPCCode("ldpc_648_r12", backend="numpy")
    with pytest.raises(ValueError):
        code.encode(np.zeros((1, 0), dtype="uint8"))
    with pytest.raises(ValueError):
        code.encode(np.zeros((1, code.k + 1), dtype="uint8"))


def test_wrong_codeword_length_raises():
    """Same shift as above -- a decode() length is only invalid now if
    the implied real_k (length - n_checks) falls outside 1..k."""
    code = LDPCCode("ldpc_648_r12", backend="numpy")
    n_checks = code.n - code.k
    with pytest.raises(ValueError):
        code.decode(np.zeros((1, n_checks), dtype="uint8"))  # real_k would be 0
    with pytest.raises(ValueError):
        code.decode(np.zeros((1, code.n + 1), dtype="uint8"))  # real_k would be k+1


def test_shortened_round_trip_various_lengths():
    """The actual new behavior: a message genuinely shorter than k
    round-trips, and the transmitted codeword is real_k + n_checks --
    NOT padded up to the full n, which is the entire point (see
    docs/mac.md's writeup of the identical rs_m8 bug this mirrors)."""
    code = LDPCCode("ldpc_648_r12", backend="numpy")
    n_checks = code.n - code.k
    rng = np.random.default_rng(3)
    for real_k in (1, 13, 104, code.k // 2, code.k - 1, code.k):
        msg = rng.integers(0, 2, size=(2, real_k)).astype("uint8")
        codeword = code.encode(msg)
        assert codeword.shape[-1] == real_k + n_checks
        np.testing.assert_array_equal(code.decode(codeword), msg)


def test_shortened_still_corrects_injected_errors():
    """Real error-correction (BP convergence) isn't broken by
    shortening -- proven with real injected bit flips, not assumed from
    the clean encode/decode round trip alone."""
    code = LDPCCode("ldpc_648_r12", backend="numpy")
    rng = np.random.default_rng(4)
    real_k = 104  # e.g. Mac's own bind-request PDU size
    msg = rng.integers(0, 2, size=(1, real_k)).astype("uint8")
    codeword = code.encode(msg)
    corrupted = codeword.copy()
    flip_idx = rng.choice(codeword.shape[-1], size=3, replace=False)
    corrupted[0, flip_idx] ^= 1
    np.testing.assert_array_equal(code.decode(corrupted, p=0.02), msg)


def test_process_is_alias_for_encode():
    code = LDPCCode("ldpc_648_r12", backend="numpy")
    msg = np.zeros((1, code.k), dtype="uint8")
    np.testing.assert_array_equal(code.process(msg), code.encode(msg))


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        LDPCCode("ldpc_9999_r12", backend="numpy")


# --- BER stress test (mirrors Viterbi's pattern) --------------------------


def test_stress_reduces_ber_substantially_at_moderate_noise():
    """Smallest/fastest variant, 648_12: random bit-flip injection at a
    moderate rate must be corrected to (near) zero BER, matching
    Viterbi's own stress-test pattern (test_fec_viterbi.py)."""
    code = LDPCCode("ldpc_648_r12", backend="numpy")
    rng = np.random.default_rng(0)
    n_batch = 6
    msg = rng.integers(0, 2, size=(n_batch, code.k)).astype("uint8")
    codeword = code.encode(msg)

    p_true = 0.03
    flips = (rng.random(codeword.shape) < p_true).astype("uint8")
    noisy = codeword ^ flips
    ber_before = np.mean(noisy != codeword)
    assert ber_before > 0.01  # corruption is real, not accidentally a no-op

    decoded = code.decode(noisy, p=p_true)
    ber_after = np.mean(decoded != msg)
    assert ber_after < 1e-6
    assert ber_after < ber_before


@pytest.mark.parametrize("variant", ["ldpc_1296_r34", "ldpc_1944_r23"])
def test_stress_reduces_ber_substantially_on_other_variants(variant):
    code = LDPCCode(variant, backend="numpy")
    rng = np.random.default_rng(1)
    msg = rng.integers(0, 2, size=(3, code.k)).astype("uint8")
    codeword = code.encode(msg)

    p_true = 0.02
    flips = (rng.random(codeword.shape) < p_true).astype("uint8")
    noisy = codeword ^ flips
    decoded = code.decode(noisy, p=p_true)
    np.testing.assert_array_equal(decoded, msg)


def test_over_capacity_noise_raises_not_silently_wrong():
    """Heavy noise must raise ValueError (BP fails to converge to a
    zero-syndrome codeword), not silently return a wrong result."""
    code = LDPCCode("ldpc_648_r12", backend="numpy")
    rng = np.random.default_rng(2)
    msg = rng.integers(0, 2, size=(2, code.k)).astype("uint8")
    codeword = code.encode(msg)
    p_true = 0.15
    flips = (rng.random(codeword.shape) < p_true).astype("uint8")
    noisy = codeword ^ flips
    with pytest.raises(ValueError):
        code.decode(noisy, p=p_true)


# --- through the FEC dispatcher -------------------------------------------


def test_fec_dispatcher_presents_uniform_bit_interface():
    fec = FEC("ldpc_648_r12", backend="numpy")
    assert fec.k_bits == 324
    assert fec.n_bits == 648
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(2, fec.k_bits)).astype("uint8")
    encoded = fec.encode(bits)
    assert encoded.shape == (2, fec.n_bits)
    decoded = fec.decode(encoded)
    np.testing.assert_array_equal(decoded, bits)


def test_fec_dispatcher_chunks_multiple_blocks_into_batch_dimension():
    fec = FEC("ldpc_648_r12", backend="numpy")
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(2, 3 * fec.k_bits)).astype("uint8")  # 3 blocks
    encoded = fec.encode(bits)
    assert encoded.shape == (2, 3 * fec.n_bits)
    decoded = fec.decode(encoded)
    np.testing.assert_array_equal(decoded, bits)


def test_fec_dispatcher_forwards_decode_kwargs():
    fec = FEC("ldpc_648_r12", backend="numpy")
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, fec.k_bits)).astype("uint8")
    encoded = fec.encode(bits)
    p_true = 0.03
    flips = (rng.random(encoded.shape) < p_true).astype("uint8")
    noisy = encoded ^ flips
    decoded = fec.decode(noisy, p=p_true)
    np.testing.assert_array_equal(decoded, bits)


def test_fec_dispatcher_encoded_decoded_length_helpers():
    """100 used to be the "not a multiple of k_bits" error example here
    -- it no longer is: FEC.encoded_length()/decoded_length() now
    support LDPC's own shortened-codeword leftover block, same as they
    already did for rs_m8 (see fec.py's encoded_length() docstring)."""
    fec = FEC("ldpc_648_r12", backend="numpy")
    assert fec.encoded_length(324) == 648
    assert fec.decoded_length(648) == 324
    n_checks = fec.n_bits - fec.k_bits
    assert fec.encoded_length(100) == 100 + n_checks  # shortened leftover, no full blocks
    assert fec.decoded_length(100 + n_checks) == 100
    assert fec.accepts_partial_block is True
