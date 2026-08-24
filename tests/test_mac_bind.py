import numpy as np
import pytest

from spectracuda.mac.bind import (
    decode_bind_request,
    decode_bind_response,
    encode_bind_request,
    encode_bind_response,
    evaluate_bind_request,
)


def test_bind_request_round_trip():
    pdu = encode_bind_request("am", 5000, 32, 4)
    d = decode_bind_request(pdu)
    assert d == {"mode": "am", "max_segment_bits": 5000, "window_size": 32, "max_retries": 4}


def test_bind_request_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown mode"):
        encode_bind_request("xyz", 5000, 32, 4)


def test_decode_bind_request_rejects_wrong_pdu_type():
    status_like = encode_bind_request("um", 100, 8, 4)  # a BIND_REQUEST
    # feed a BIND_RESPONSE where a BIND_REQUEST is expected
    response = encode_bind_response(evaluate_bind_request(decode_bind_request(status_like), 1000))
    with pytest.raises(ValueError, match="expected a BIND_REQUEST"):
        decode_bind_request(response)


def test_evaluate_accepts_a_compatible_request():
    """The real accept case: an independently-chosen local capacity that
    genuinely accommodates the request."""
    request = {"mode": "um", "max_segment_bits": 4000, "window_size": 16, "max_retries": 3}
    decision = evaluate_bind_request(request, local_max_segment_bits=6000)
    assert decision["accepted"] is True
    assert decision["reason"] == "none"
    assert decision["max_segment_bits"] == 4000  # echoed back


def test_evaluate_rejects_a_request_exceeding_local_capacity():
    """The real, meaningful test: two INDEPENDENTLY chosen configs, one
    genuinely too large for the other -- not self-consistent plumbing."""
    request = {"mode": "am", "max_segment_bits": 9000, "window_size": 16, "max_retries": 3}
    decision = evaluate_bind_request(request, local_max_segment_bits=4000)
    assert decision["accepted"] is False
    assert decision["reason"] == "segment_too_large"


def test_evaluate_accepts_exactly_at_the_boundary():
    request = {"mode": "tm", "max_segment_bits": 4000, "window_size": 1, "max_retries": 0}
    decision = evaluate_bind_request(request, local_max_segment_bits=4000)
    assert decision["accepted"] is True


def test_evaluate_rejects_unknown_mode():
    request = {"mode": "bogus", "max_segment_bits": 10, "window_size": 1, "max_retries": 1}
    decision = evaluate_bind_request(request, local_max_segment_bits=1_000_000)
    assert decision["accepted"] is False
    assert decision["reason"] == "unknown_mode"


@pytest.mark.parametrize("accepted", [True, False])
def test_bind_response_round_trip(accepted):
    request = {"mode": "um", "max_segment_bits": 2000, "window_size": 8, "max_retries": 2}
    local_capacity = 2000 if accepted else 100
    decision = evaluate_bind_request(request, local_max_segment_bits=local_capacity)
    assert decision["accepted"] is accepted

    pdu = encode_bind_response(decision)
    d = decode_bind_response(pdu)
    assert d["accepted"] is accepted
    assert d["reason"] == decision["reason"]
    assert d["mode"] == "um"


def test_decode_bind_response_rejects_wrong_pdu_type():
    req_pdu = encode_bind_request("um", 100, 8, 4)
    with pytest.raises(ValueError, match="expected a BIND_RESPONSE"):
        decode_bind_response(req_pdu)


def test_decode_bind_response_rejects_corrupted_reason_code():
    request = {"mode": "um", "max_segment_bits": 100, "window_size": 8, "max_retries": 2}
    decision = evaluate_bind_request(request, local_max_segment_bits=1000)
    pdu = encode_bind_response(decision)
    corrupted = pdu.copy()
    # reason byte is payload byte 1 -> bits [HEADER_LEN_BITS+8 : HEADER_LEN_BITS+16]
    from spectracuda.mac.pdu import HEADER_LEN_BITS

    corrupted[HEADER_LEN_BITS + 8 : HEADER_LEN_BITS + 16] = 1  # 0xFF -- not a known reason code
    with pytest.raises(ValueError, match="not known"):
        decode_bind_response(corrupted)
