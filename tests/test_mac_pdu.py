import numpy as np
import pytest

from spectracuda.mac.pdu import (
    HEADER_LEN_BITS,
    SI_FIRST,
    SI_FULL,
    SN_MODULUS,
    TYPE_DATA,
    TYPE_STATUS,
    decode_header,
    encode_header,
    sn_add,
    sn_precedes,
)


def test_header_round_trip():
    h = encode_header(pdu_type=TYPE_DATA, si=SI_FIRST, sn=513, so=4000)
    assert h.shape == (HEADER_LEN_BITS,)
    d = decode_header(h)
    assert d == {"pdu_type": TYPE_DATA, "si": SI_FIRST, "sn": 513, "so": 4000}


def test_header_round_trip_at_field_boundaries():
    h = encode_header(pdu_type=TYPE_STATUS, si=SI_FULL, sn=SN_MODULUS - 1, so=(1 << 16) - 1)
    d = decode_header(h)
    assert d == {"pdu_type": TYPE_STATUS, "si": SI_FULL, "sn": SN_MODULUS - 1, "so": (1 << 16) - 1}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pdu_type": 8, "si": 0, "sn": 0, "so": 0},  # 8 > _MAX_TYPE=7 (3-bit field)
        {"pdu_type": 0, "si": 4, "sn": 0, "so": 0},
        {"pdu_type": 0, "si": 0, "sn": SN_MODULUS, "so": 0},
        {"pdu_type": 0, "si": 0, "sn": 0, "so": 1 << 16},
    ],
)
def test_encode_header_rejects_out_of_range_fields(kwargs):
    with pytest.raises(ValueError):
        encode_header(**kwargs)


def test_decode_header_rejects_wrong_length():
    with pytest.raises(ValueError):
        decode_header(np.zeros(HEADER_LEN_BITS - 1, dtype="uint8"))


def test_sn_add_wraps():
    assert sn_add(SN_MODULUS - 1, 1) == 0
    assert sn_add(SN_MODULUS - 1, 2) == 1
    assert sn_add(5, 0) == 5


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (0, 1, True),
        (1, 0, False),
        (0, 0, False),
        (SN_MODULUS - 1, 0, True),  # wraparound: 1023 precedes 0
        (0, SN_MODULUS - 1, False),
        (500, 500 + 100, True),
    ],
)
def test_sn_precedes_handles_wraparound(a, b, expected):
    assert sn_precedes(a, b % SN_MODULUS) == expected
