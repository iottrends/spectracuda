"""Standalone tests for the four interleaver strategies (docs/todo.md
#1.12) -- zero Packetizer/Ofdm involvement, matching the same
standalone-first rigor already applied to HeaderCodec/Packetizer
(tests/test_framing_header.py/test_framing_packetizer.py)."""
import numpy as np
import pytest

from spectracuda.interleaver import (
    BlockInterleaver,
    ConvolutionalInterleaver,
    LiquidInterleaver,
    PermutationInterleaver,
)

ALL_CLASSES_KWARGS = [
    (BlockInterleaver, {}),
    (PermutationInterleaver, {"seed": 7}),
    (ConvolutionalInterleaver, {"branches": 4, "base_delay": 3}),
]
SIZES = [8, 17, 40, 64, 100, 255, 324]


# --- generic properties, all three bit-granular classes -------------------


@pytest.mark.parametrize("cls,kwargs", ALL_CLASSES_KWARGS)
@pytest.mark.parametrize("n_bits", SIZES)
def test_permutation_is_a_true_bijection(cls, kwargs, n_bits):
    obj = cls(n_bits, backend="numpy", **kwargs)
    perm = np.asarray(obj._perm)
    assert sorted(perm.tolist()) == list(range(n_bits))


@pytest.mark.parametrize("cls,kwargs", ALL_CLASSES_KWARGS)
@pytest.mark.parametrize("n_bits", SIZES)
def test_round_trips_batched_with_different_content_per_item(cls, kwargs, n_bits):
    obj = cls(n_bits, backend="numpy", **kwargs)
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(4, n_bits)).astype("uint8")
    encoded = obj.encode(bits)
    assert encoded.shape == bits.shape
    decoded = obj.decode(encoded)
    np.testing.assert_array_equal(decoded, bits)


@pytest.mark.parametrize("cls,kwargs", ALL_CLASSES_KWARGS)
def test_genuinely_permutes_not_a_silent_identity(cls, kwargs):
    """Plumbing check mirroring FEC's own
    test_fec_is_genuinely_applied_not_a_silent_no_op: encode() must
    actually reorder bits, not silently pass them through."""
    obj = cls(64, backend="numpy", **kwargs)
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, 64)).astype("uint8")
    encoded = obj.encode(bits)
    assert not np.array_equal(encoded, bits)


@pytest.mark.parametrize("cls,kwargs", ALL_CLASSES_KWARGS)
def test_process_is_alias_for_encode(cls, kwargs):
    obj = cls(32, backend="numpy", **kwargs)
    bits = np.zeros((1, 32), dtype="uint8")
    bits[0, ::3] = 1
    np.testing.assert_array_equal(obj.process(bits), obj.encode(bits))


@pytest.mark.parametrize("cls,kwargs", ALL_CLASSES_KWARGS)
def test_wrong_length_raises(cls, kwargs):
    obj = cls(32, backend="numpy", **kwargs)
    with pytest.raises(ValueError):
        obj.encode(np.zeros((1, 10), dtype="uint8"))
    with pytest.raises(ValueError):
        obj.decode(np.zeros((1, 10), dtype="uint8"))


# --- burst-spreading property: the actual reason these classes exist ------


def _max_cluster_in_window(new_positions: np.ndarray, window: int) -> int:
    """Max number of a burst's new positions falling within any single
    `window`-wide slice of the output -- a direct, simple measure of
    "did interleaving actually spread this burst out" (low = spread
    out well, close to len(new_positions) = barely spread at all)."""
    order = np.sort(new_positions)
    best = 1
    for i in range(len(order)):
        j = i
        while j + 1 < len(order) and order[j + 1] - order[i] < window:
            j += 1
        best = max(best, j - i + 1)
    return best


@pytest.mark.parametrize("cls,kwargs", ALL_CLASSES_KWARGS)
def test_a_contiguous_burst_is_spread_out_after_interleaving(cls, kwargs):
    """The actual property an interleaver exists to provide: L
    consecutive ORIGINAL positions (what a real burst error would hit)
    must land scattered across the ENCODED stream, not still clustered
    together -- checked directly against the permutation array, not
    just "it round-trips" (which every bijection does trivially,
    including a no-op identity map)."""
    n_bits = 256
    obj = cls(n_bits, backend="numpy", **kwargs)
    burst_len = 16
    burst_start = 40
    original_positions = np.arange(burst_start, burst_start + burst_len)

    perm = np.asarray(obj._perm)
    # perm[out_pos] = original input position -> invert to find where
    # each ORIGINAL position landed in the output.
    inverse = np.argsort(perm)
    new_positions = inverse[original_positions]

    cluster = _max_cluster_in_window(new_positions, window=burst_len)
    assert cluster < burst_len  # genuinely spread, not still one contiguous clump


# --- LiquidInterleaver-specific -------------------------------------------


def _reference_grid_dims(n):
    M = 1 + int(np.floor(np.sqrt(n)))
    N = n // M
    while n >= M * N:
        N += 1
    return M, N


def _reference_permute_bytes(x, n, M, N):
    """Fully independent, self-contained port of interleaver_permute()
    -- operates on byte VALUES directly via a locally-defined index
    walk (deliberately NOT importing anything from spectracuda.
    interleaver.liquid: reusing that module's own internals wouldn't
    actually check anything independently, and reusing its _index_walk
    specifically is exactly what would have hidden the real n2-vs-
    n_bytes bug found during development -- this reference has its own,
    separately-typed-in walk loop)."""
    x = list(x)
    n2 = n // 2
    m = 0
    nn = n // 3
    for i in range(n2):
        while True:
            j = m * N + nn
            m += 1
            if m == M:
                nn = (nn + 1) % N
                m = 0
            if j < n2:
                break
        x[2 * j + 1], x[2 * i + 0] = x[2 * i + 0], x[2 * j + 1]
    return x


def _reference_permute_bytes_mask(x, n, M, N, mask):
    """Independent port of interleaver_permute_mask() -- see
    _reference_permute_bytes()'s docstring."""
    x = list(x)
    n2 = n // 2
    m = 0
    nn = n // 3
    for i in range(n2):
        while True:
            j = m * N + nn
            m += 1
            if m == M:
                nn = (nn + 1) % N
                m = 0
            if j < n2:
                break
        a, b = x[2 * i + 0], x[2 * j + 1]
        x[2 * i + 0] = (a & (~mask & 0xFF)) | (b & mask)
        x[2 * j + 1] = (a & mask) | (b & (~mask & 0xFF))
    return x


def _reference_liquid_interleave_bytes(x, depth=4):
    """Independent re-implementation of the literal C swap-loop
    algorithm, operating on byte VALUES directly -- LiquidInterleaver's
    permutation array is cross-checked against this below, so this test
    doesn't just check the class against itself."""
    n = len(x)
    M, N = _reference_grid_dims(n)
    x = list(x)
    if depth > 0:
        x = _reference_permute_bytes(x, n, M, N)
    if depth > 1:
        x = _reference_permute_bytes_mask(x, n, M, N + 2, 0x0F)
    if depth > 2:
        x = _reference_permute_bytes_mask(x, n, M, N + 4, 0x55)
    if depth > 3:
        x = _reference_permute_bytes_mask(x, n, M, N + 8, 0x33)
    return x


@pytest.mark.parametrize("n_bytes", [8, 16, 37, 38, 39, 45, 64, 100, 223, 255, 324])
def test_liquid_interleaver_matches_independent_reference_implementation(n_bytes):
    """The core correctness proof: this class's permutation-array
    result must match a SEPARATE, independently-structured
    reimplementation of liquid-dsp's literal swap-loop algorithm --
    covers exactly the sizes (39, 45, 255) that exposed the n2-vs-
    n_bytes derivation bug found during development (255 is RS(255,223)'s
    own codeword size -- this would have been a silent, serious
    correctness bug for the single most relevant real-world size)."""
    rng = np.random.default_rng(1)
    data_bytes = rng.integers(0, 256, size=n_bytes).astype("uint8")
    expected = np.array(_reference_liquid_interleave_bytes(list(data_bytes), depth=4), dtype="uint8")

    li = LiquidInterleaver(n_bytes * 8, depth=4, backend="numpy")
    data_bits = np.unpackbits(data_bytes)[None, :]
    encoded_bits = li.encode(data_bits)
    encoded_bytes = np.packbits(encoded_bits[0])
    np.testing.assert_array_equal(encoded_bytes, expected)


@pytest.mark.parametrize("n_bytes", [8, 37, 255])
def test_liquid_interleaver_round_trips(n_bytes):
    li = LiquidInterleaver(n_bytes * 8, backend="numpy")
    rng = np.random.default_rng(2)
    bits = rng.integers(0, 2, size=(3, n_bytes * 8)).astype("uint8")
    encoded = li.encode(bits)
    decoded = li.decode(encoded)
    np.testing.assert_array_equal(decoded, bits)


def test_liquid_interleaver_requires_byte_aligned_n_bits():
    with pytest.raises(ValueError):
        LiquidInterleaver(20, backend="numpy")


@pytest.mark.parametrize("depth", [0, 1, 2, 3, 4])
def test_liquid_interleaver_depth_controls_how_much_mixing_happens(depth):
    """depth=0 is a documented liquid-dsp no-op (identity permutation);
    each higher depth should still round-trip correctly."""
    li = LiquidInterleaver(64, depth=depth, backend="numpy")
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, 64)).astype("uint8")
    encoded = li.encode(bits)
    if depth == 0:
        np.testing.assert_array_equal(encoded, bits)
    else:
        assert not np.array_equal(encoded, bits)
    np.testing.assert_array_equal(li.decode(encoded), bits)


def test_liquid_interleaver_invalid_depth_raises():
    with pytest.raises(ValueError):
        LiquidInterleaver(64, depth=5, backend="numpy")


# --- per-class constructor validation --------------------------------------


def test_block_interleaver_invalid_rows_raises():
    with pytest.raises(ValueError):
        BlockInterleaver(64, rows=0, backend="numpy")


def test_convolutional_interleaver_invalid_branches_raises():
    with pytest.raises(ValueError):
        ConvolutionalInterleaver(64, branches=0, backend="numpy")


def test_permutation_interleaver_different_seeds_differ():
    a = PermutationInterleaver(64, seed=1, backend="numpy")
    b = PermutationInterleaver(64, seed=2, backend="numpy")
    assert not np.array_equal(np.asarray(a._perm), np.asarray(b._perm))


def test_permutation_interleaver_same_seed_reproducible():
    a = PermutationInterleaver(64, seed=99, backend="numpy")
    b = PermutationInterleaver(64, seed=99, backend="numpy")
    np.testing.assert_array_equal(np.asarray(a._perm), np.asarray(b._perm))


# --- unit_bits: found to matter for real correctness, not a convenience --


@pytest.mark.parametrize("cls,kwargs", ALL_CLASSES_KWARGS)
def test_unit_bits_requires_n_bits_be_a_multiple(cls, kwargs):
    with pytest.raises(ValueError):
        cls(20, unit_bits=8, backend="numpy", **kwargs)


@pytest.mark.parametrize("cls,kwargs", ALL_CLASSES_KWARGS)
def test_unit_bits_moves_whole_bytes_together_not_individual_bits(cls, kwargs):
    """The actual property unit_bits=8 exists for: every bit within one
    byte must land at the SAME new byte position as its 7 neighbors --
    i.e. the permutation, viewed as byte positions, is itself a genuine
    permutation of BYTES (each output byte's 8 bits all trace back to
    the SAME original byte), not a bit-level shuffle that happens to
    move some bits of a byte one way and others another way."""
    n_bytes = 32
    obj = cls(n_bytes * 8, unit_bits=8, backend="numpy", **kwargs)
    perm = np.asarray(obj._perm).reshape(n_bytes, 8)
    source_byte = perm // 8
    # every row (one output byte) must come from a single source byte
    assert np.all(source_byte == source_byte[:, [0]])


@pytest.mark.parametrize("cls,kwargs", ALL_CLASSES_KWARGS)
def test_unit_bits_round_trips(cls, kwargs):
    obj = cls(256, unit_bits=8, backend="numpy", **kwargs)
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(2, 256)).astype("uint8")
    encoded = obj.encode(bits)
    decoded = obj.decode(encoded)
    np.testing.assert_array_equal(decoded, bits)


def test_byte_granularity_genuinely_helps_rs_where_bit_granularity_hurts_it():
    """The concrete finding that motivated unit_bits existing at all,
    pinned as a permanent regression test: a contiguous Viterbi decode-
    error burst that already fits comfortably within Reed-Solomon's
    t=16-byte-per-codeword budget WITHOUT any interleaving gets
    fragmented into MORE distinct byte errors by a BIT-granularity
    interleaver (confirmed: turns ~7 concentrated byte errors in one
    codeword into 32+20 scattered byte errors across two, exceeding
    budget on both) -- while the SAME scenario with unit_bits=8 keeps
    both codewords comfortably under budget. Uses Packetizer directly
    (fec0=rs_m8, fec1=conv_v27 -- the CORRECT liquid-dsp assignment for
    this scenario: fec1 is decoded FIRST, facing the channel directly,
    so Viterbi must be fec1 and RS fec0 for RS to clean up Viterbi's
    residual -- see docs/todo.md #1.2/#1.12)."""
    from spectracuda.framing import Packetizer

    rng = np.random.default_rng(0)
    raw_bits = rng.integers(0, 2, size=(1, 2 * 1784)).astype("uint8")  # 2 rs_m8 codewords

    p_bitlevel = Packetizer(fec="rs_m8", fec1="conv_v27", interleaver="block", backend="numpy")
    p_bytelevel = Packetizer(
        fec="rs_m8", fec1="conv_v27", interleaver="block",
        interleaver_kwargs={"unit_bits": 8}, backend="numpy",
    )
    p_none = Packetizer(fec="rs_m8", fec1="conv_v27", interleaver="none", backend="numpy")

    def _corrupt_and_decode(p):
        encoded = p.encode(raw_bits)
        n = encoded.shape[-1]
        corrupted = encoded.copy()
        corrupted[0, n // 2 : n // 2 + 100] ^= 1  # same burst, same position, every case
        try:
            result = p.decode(corrupted)
            return np.array_equal(result["bits"], raw_bits)
        except ValueError:
            return False

    # Both "no interleaving" and "byte-granularity interleaving" succeed
    # here (the burst is small enough that even without help, one of the
    # two codewords absorbs it fine) -- the point being demonstrated is
    # specifically that BIT-granularity interleaving makes this SAME
    # scenario WORSE, not that byte-granularity is strictly required for
    # every possible burst.
    assert _corrupt_and_decode(p_none) is True
    assert _corrupt_and_decode(p_bytelevel) is True
    assert _corrupt_and_decode(p_bitlevel) is False
