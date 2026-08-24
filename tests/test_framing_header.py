"""HeaderCodec: standalone tests proving header encode/decode is usable
independent of any Ofdm/OFDM machinery -- the actual gap this class
closes (see docs/todo.md #1.1)."""
import numpy as np
import pytest

from spectracuda.framing import HeaderCodec
from spectracuda.framing.header import CRC_SCHEME_CODES, FEC_SCHEME_CODES, MOD_SCHEME_CODES


def test_round_trips_all_fields_with_no_ofdm_object_involved():
    codec = HeaderCodec()
    bits = codec.encode_bits(1234, "qam16", "conv_v27", b"ABCDEFGH", "crc32")
    decoded = codec.decode_bits(bits)
    assert decoded["payload_len_bits"] == 1234
    assert decoded["mod_scheme"] == "qam16"
    assert decoded["fec0"] == "conv_v27"
    assert decoded["crc"] == "crc32"
    assert decoded["user_data"] == b"ABCDEFGH"
    assert decoded["fec1"] == "none"
    assert decoded["protocol_version"] == HeaderCodec.PROTOCOL_VERSION


def test_default_user_data_is_none_bytes():
    codec = HeaderCodec()
    bits = codec.encode_bits(80, "qpsk", "none", None)
    decoded = codec.decode_bits(bits)
    assert decoded["user_data"] == bytes(8)


def test_two_independent_instances_with_same_seed_agree():
    """Same scramble_seed -> interchangeable codecs, matching how Ofdm's
    own HeaderCodec(scramble_seed=42) is meant to be reconstructible
    independent of any particular Ofdm instance."""
    a = HeaderCodec(scramble_seed=7)
    b = HeaderCodec(scramble_seed=7)
    bits = a.encode_bits(42, "bpsk", "rs_m8", None, "crc16")
    decoded = b.decode_bits(bits)
    assert decoded["payload_len_bits"] == 42
    assert decoded["fec0"] == "rs_m8"
    assert decoded["crc"] == "crc16"


def test_different_seeds_produce_different_wire_bits():
    a = HeaderCodec(scramble_seed=1)
    b = HeaderCodec(scramble_seed=2)
    bits_a = a.encode_bits(80, "qpsk", "none", None)
    bits_b = b.encode_bits(80, "qpsk", "none", None)
    assert not np.array_equal(bits_a, bits_b)


def test_ldpc_variants_all_have_header_codes():
    codec = HeaderCodec()
    from spectracuda.fec.ldpc_tables import BASE_MATRICES

    for variant in BASE_MATRICES:
        bits = codec.encode_bits(80, "qpsk", variant, None)
        decoded = codec.decode_bits(bits)
        assert decoded["fec0"] == variant


@pytest.mark.parametrize("bad_field,value", [("mod_scheme", "16psk"), ("fec0", "turbo"), ("crc0", "md5")])
def test_unknown_field_values_raise(bad_field, value):
    codec = HeaderCodec()
    kwargs = {"mod_scheme": "qpsk", "fec0": "none", "user_data": None, "crc0": "none"}
    kwargs[{"mod_scheme": "mod_scheme", "fec0": "fec0", "crc0": "crc0"}[bad_field]] = value
    with pytest.raises(ValueError):
        codec.encode_bits(80, **kwargs)


def test_user_data_must_be_exactly_8_bytes():
    codec = HeaderCodec()
    with pytest.raises(ValueError):
        codec.encode_bits(80, "qpsk", "none", b"short")


def test_payload_len_bits_out_of_range_raises():
    codec = HeaderCodec()
    with pytest.raises(ValueError):
        codec.encode_bits(2 ** 16, "qpsk", "none", None)


def test_field_code_tables_are_internally_consistent():
    assert MOD_SCHEME_CODES["qpsk"] != MOD_SCHEME_CODES["bpsk"]
    assert CRC_SCHEME_CODES["none"] not in (0,)  # LIQUID_CRC_UNKNOWN=0 reserved
    assert 0 not in FEC_SCHEME_CODES.values() or FEC_SCHEME_CODES["none"] == 0
    assert len(set(FEC_SCHEME_CODES.values())) == len(FEC_SCHEME_CODES)  # no code collisions
