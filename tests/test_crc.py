import numpy as np
import pytest

from spectracuda.fec import CRC


def test_matches_liquid_dsp_testvectors():
    """Byte-for-byte against liquid-dsp's own crc_autotest.c test
    vectors (data=bytes(range(256))): crc8->0x53, crc16->0x6fc6,
    crc24->0x10c59b, crc32->0x29058c73."""
    data = np.arange(256, dtype=np.uint8)[None, :]
    expected = {"crc8": 0x53, "crc16": 0x6FC6, "crc24": 0x10C59B, "crc32": 0x29058C73}
    for scheme, key in expected.items():
        crc = CRC(scheme, backend="numpy")
        got = crc.generate_key(data)[0]
        assert int(got) == key, f"{scheme}: got {hex(int(got))}, expected {hex(key)}"


def test_crc32_matches_zlib():
    """liquid's crc32 happens to be bit-for-bit the standard IEEE-802.3/
    zlib CRC-32 (confirmed separately -- see crc.py's module docstring);
    cross-check against Python's stdlib zlib as an independent oracle,
    not just liquid's own test vector."""
    import zlib

    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, size=(5, 137)).astype(np.uint8)
    crc = CRC("crc32", backend="numpy")
    got = crc.generate_key(data)
    for i in range(5):
        assert int(got[i]) == zlib.crc32(data[i].tobytes())


def test_key_length_bytes():
    expected = {"none": 0, "checksum": 1, "crc8": 1, "crc16": 2, "crc24": 3, "crc32": 4}
    for scheme, length in expected.items():
        assert CRC(scheme, backend="numpy").key_length == length


def test_checksum_matches_twos_complement_byte_sum():
    data = np.array([[1, 2, 3, 4, 250]], dtype=np.uint8)
    crc = CRC("checksum", backend="numpy")
    key = int(crc.generate_key(data)[0])
    total = int(data.sum()) & 0xFF
    expected = ((~total) + 1) & 0xFF
    assert key == expected


def test_append_then_check_key_round_trips_clean_message():
    rng = np.random.default_rng(1)
    for scheme in ["checksum", "crc8", "crc16", "crc24", "crc32"]:
        crc = CRC(scheme, backend="numpy")
        msg = rng.integers(0, 256, size=(4, 64)).astype(np.uint8)
        with_key = crc.append_key(msg)
        assert with_key.shape == (4, 64 + crc.key_length)
        np.testing.assert_array_equal(with_key[:, :64], msg)
        assert crc.check_key(with_key).all()


@pytest.mark.parametrize("scheme", ["checksum", "crc8", "crc16", "crc24", "crc32"])
def test_check_key_detects_every_single_bit_flip(scheme):
    """Mirrors liquid's own testbench_crc autotest: flip each bit of the
    message individually and confirm the check fails every time."""
    rng = np.random.default_rng(2)
    crc = CRC(scheme, backend="numpy")
    msg = rng.integers(0, 256, size=(1, 16)).astype(np.uint8)
    with_key = crc.append_key(msg)
    assert crc.check_key(with_key)[0]

    for byte_idx in range(16):
        for bit in range(8):
            corrupted = with_key.copy()
            corrupted[0, byte_idx] ^= 1 << bit
            assert not crc.check_key(corrupted)[0], f"{scheme}: missed flip byte={byte_idx} bit={bit}"


def test_scheme_none_key_length_zero_and_always_valid():
    crc = CRC("none", backend="numpy")
    msg = np.array([[1, 2, 3]], dtype=np.uint8)
    with_key = crc.append_key(msg)
    np.testing.assert_array_equal(with_key, msg)  # no-op
    assert crc.check_key(with_key)[0]
    corrupted = msg.copy()
    corrupted[0, 0] ^= 0xFF
    assert crc.check_key(corrupted)[0]  # "none" never detects anything -- matches liquid


def test_batch_with_different_content_per_item():
    rng = np.random.default_rng(3)
    crc = CRC("crc32", backend="numpy")
    msg = rng.integers(0, 256, size=(6, 40)).astype(np.uint8)
    with_key = crc.append_key(msg)
    valid = crc.check_key(with_key)
    assert valid.all()

    # corrupt only item 2
    corrupted = with_key.copy()
    corrupted[2, 5] ^= 0x01
    valid2 = crc.check_key(corrupted)
    expected = np.array([True, True, False, True, True, True])
    np.testing.assert_array_equal(valid2, expected)


def test_unknown_scheme_raises():
    with pytest.raises(ValueError):
        CRC("crc99", backend="numpy")


def test_process_is_alias_for_append_key():
    crc = CRC("crc16", backend="numpy")
    msg = np.array([[9, 8, 7]], dtype=np.uint8)
    np.testing.assert_array_equal(crc.process(msg), crc.append_key(msg))


def test_check_key_raises_on_message_shorter_than_key_length():
    crc = CRC("crc32", backend="numpy")
    with pytest.raises(ValueError):
        crc.check_key(np.zeros((1, 2), dtype=np.uint8))
