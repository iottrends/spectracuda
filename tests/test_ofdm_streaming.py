"""Ofdm.rx_streaming(): a real, chunked streaming receiver (docs/mac.md-
adjacent, but this is a Layer 3 Ofdm feature, not MAC). Additive --
rx_process() is unchanged (verified by the full existing suite passing
unchanged after the internal _decode_header_from_sync()/
_decode_payload_from_header() extraction rx_streaming() reuses). See
Ofdm.rx_streaming()'s own docstring for the design (checked against
liquid-dsp's actual ofdmframesync_execute() state machine before writing
this, not invented from assumption).
"""
import numpy as np
import pytest

from spectracuda.pipeline import Ofdm


def _make_ofdm(**overrides):
    kwargs = dict(
        fft_size=256, n_pilot=8, n_data=216, cp_len=32, modem="qpsk",
        fec="conv_v27", crc="crc16", sync="schmidl_cox", cfo="schmidl_cox",
        n_training_symbols=2, backend="numpy",
    )
    kwargs.update(overrides)
    return Ofdm(**kwargs)


def _stream_one_frame(ofdm, tx_iq, chunk_size):
    """Feed tx_iq through rx_streaming() in chunk_size pieces, return the
    first non-None result (or None if the whole frame didn't complete)."""
    for i in range(0, len(tx_iq), chunk_size):
        r = ofdm.rx_streaming(tx_iq[i : i + chunk_size])
        if r is not None:
            return r
    return None


@pytest.mark.parametrize("chunk_size", [64, 128, 256, 1024, 37])
def test_streaming_matches_rx_process_bit_identical(chunk_size):
    """chunk_size=37 is deliberately not a multiple of anything (fft_size,
    slot_len, header/preamble lengths) -- proves no hidden alignment
    assumption anywhere in the accumulation logic."""
    ofdm = _make_ofdm()
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, 64)).astype("uint8")
    tx_iq = ofdm.generate_frame(bits)[0]

    ref = ofdm.rx_process(tx_iq[None, :])

    ofdm.reset_stream()
    result = _stream_one_frame(ofdm, tx_iq, chunk_size)

    assert result is not None
    np.testing.assert_array_equal(result["bits"], bits)
    np.testing.assert_array_equal(result["bits"], ref["bits"])
    assert result["crc_valid"] is not None and bool(np.asarray(result["crc_valid"])[0])
    assert result["evm"] == pytest.approx(float(np.asarray(ref["evm"])[0]), abs=1e-6)


def test_returns_none_until_the_frame_actually_completes():
    ofdm = _make_ofdm()
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, size=(1, 64)).astype("uint8")
    tx_iq = ofdm.generate_frame(bits)[0]

    ofdm.reset_stream()
    results = [ofdm.rx_streaming(tx_iq[i : i + 64]) for i in range(0, len(tx_iq), 64)]
    non_none = [r for r in results if r is not None]
    assert len(non_none) == 1  # exactly one completion, not zero, not more than one
    assert results[-1] is not None  # completes on (at latest) the final chunk


def test_multiple_frames_back_to_back_in_one_continuous_stream():
    """Real usage: a continuous stream carries MANY frames over time, not
    just one. Confirms state correctly resets between frames and both
    are found, in order, with a silence gap between them (so the second
    frame's own preamble genuinely has to be re-discovered, not just
    happen to already be at buffer position 0)."""
    ofdm = _make_ofdm()
    rng = np.random.default_rng(2)
    bits_a = rng.integers(0, 2, size=(1, 64)).astype("uint8")
    bits_b = rng.integers(0, 2, size=(1, 64)).astype("uint8")
    frame_a = ofdm.generate_frame(bits_a)[0]
    frame_b = ofdm.generate_frame(bits_b)[0]
    gap = np.zeros(50, dtype="complex64")
    stream = np.concatenate([frame_a, gap, frame_b])

    ofdm.reset_stream()
    delivered = []
    for i in range(0, len(stream), 90):  # deliberately not symbol-aligned
        r = ofdm.rx_streaming(stream[i : i + 90])
        if r is not None:
            delivered.append(r)

    assert len(delivered) == 2
    np.testing.assert_array_equal(delivered[0]["bits"], bits_a)
    np.testing.assert_array_equal(delivered[1]["bits"], bits_b)


def test_pure_noise_never_produces_a_false_complete_result():
    ofdm = _make_ofdm()
    rng = np.random.default_rng(3)
    noise = (rng.standard_normal(20000) + 1j * rng.standard_normal(20000)).astype("complex64") * 0.1

    ofdm.reset_stream()
    for i in range(0, len(noise), 128):
        r = ofdm.rx_streaming(noise[i : i + 128])
        assert r is None


def test_search_buffer_stays_bounded_on_a_long_noise_only_stream():
    """The STREAM_SEARCH_WINDOW_SYMBOLS cap must actually engage -- a long
    silent/noisy stream with no frame in it must not grow the
    accumulation buffer without bound."""
    ofdm = _make_ofdm()
    rng = np.random.default_rng(4)
    noise = (rng.standard_normal(50000) + 1j * rng.standard_normal(50000)).astype("complex64") * 0.1

    ofdm.reset_stream()
    for i in range(0, len(noise), 256):
        ofdm.rx_streaming(noise[i : i + 256])

    cap = ofdm.STREAM_SEARCH_WINDOW_SYMBOLS * ofdm.fft_size
    assert ofdm._stream_buffer.shape[-1] <= cap


def test_recovers_after_a_false_positive_or_corrupted_frame():
    """A real frame, corrupted heavily enough to fail header/payload
    decode, followed by noise, followed by a genuine clean frame --
    confirms one bad detection doesn't kill the receiver's ability to
    find a later real frame (the actual behavioral point of NOT raising
    from rx_streaming() on decode failure, unlike rx_process())."""
    ofdm = _make_ofdm()
    rng = np.random.default_rng(5)
    bits_bad = rng.integers(0, 2, size=(1, 64)).astype("uint8")
    bad_frame = ofdm.generate_frame(bits_bad)[0].copy()
    # Corrupt the payload region heavily -- past the point sync/header
    # still succeed, but FEC/CRC should not.
    bad_frame[900:1100] += (rng.standard_normal(200) + 1j * rng.standard_normal(200)).astype("complex64") * 2.0

    bits_good = rng.integers(0, 2, size=(1, 64)).astype("uint8")
    good_frame = ofdm.generate_frame(bits_good)[0]

    gap = np.zeros(60, dtype="complex64")
    stream = np.concatenate([bad_frame, gap, good_frame])

    ofdm.reset_stream()
    delivered = []
    for i in range(0, len(stream), 128):
        r = ofdm.rx_streaming(stream[i : i + 128])
        if r is not None:
            delivered.append(r)

    # The bad frame must NOT have produced a wrong "success" -- and the
    # good one that follows must still be found correctly.
    assert len(delivered) <= 1  # zero (bad frame just discarded) or one, never a wrong extra
    assert len(delivered) == 1
    np.testing.assert_array_equal(delivered[0]["bits"], bits_good)


def test_reset_stream_abandons_in_flight_state():
    ofdm = _make_ofdm()
    rng = np.random.default_rng(6)
    bits = rng.integers(0, 2, size=(1, 64)).astype("uint8")
    tx_iq = ofdm.generate_frame(bits)[0]

    ofdm.reset_stream()
    # 400 samples: past fft_size=256 (enough to have attempted and
    # succeeded at sync detection) but short of the full ~1408-sample
    # frame -- genuinely partway into decoding, not just "not enough to
    # try yet".
    ofdm.rx_streaming(tx_iq[:400])
    assert ofdm._stream_state != "SEEKING"

    ofdm.reset_stream()
    assert ofdm._stream_state == "SEEKING"
    assert ofdm._stream_buffer.shape[-1] == 0


def test_rx_streaming_is_single_stream_only():
    ofdm = _make_ofdm()
    with pytest.raises(ValueError, match="single-stream only"):
        ofdm.rx_streaming(np.zeros((2, 100), dtype="complex64"))
