import numpy as np
import pytest

from spectracuda.pipeline import Ofdm


def _make_ofdm(**overrides):
    kwargs = dict(fft_size=64, n_pilot=8, n_data=40, cp_len=16, modem="qpsk", backend="numpy")
    kwargs.update(overrides)
    return Ofdm(**kwargs)


def _payload_start(ofdm):
    """Sample offset where the payload region begins in generate_frame()'s
    output -- right after preamble + training(s) + header symbol(s)."""
    return ofdm.fft_size + ofdm.n_training_symbols * ofdm.slot_len + ofdm.num_symbols_header * ofdm.slot_len


def _corrupt_first_payload_symbol_bits(ofdm, tx_iq, first_symbol_bits, bit_positions_to_flip):
    """Rebuild the FIRST payload OFDM symbol with specific bits flipped
    (bit_positions_to_flip indexes into that symbol's own
    bits_per_ofdm_symbol-wide slice), using only public modem/grid/mod
    attributes, and splice it back into tx_iq -- gives exact, known
    control over which/how many bits get corrupted, unlike a channel
    impairment (AWGN/multipath), where the resulting bit-error pattern
    is only statistically controllable, not exact."""
    corrupted = first_symbol_bits.copy()
    corrupted[:, bit_positions_to_flip] ^= 1
    pilots_batch = ofdm.pilot_values[None, :]
    data_symbols = ofdm.modem.modulate(corrupted)
    freq = ofdm.grid.scatter(np, pilots_batch, data_symbols)
    time_symbol = ofdm.mod.process(freq)

    start = _payload_start(ofdm)
    out = tx_iq.copy()
    out[:, start : start + ofdm.slot_len] = time_symbol
    return out


def test_generate_frame_rejects_over_max_payload_symbols():
    """MAX_PAYLOAD_SYMBOLS=128 is a coherence-time limit, not a wire-
    format one: beyond roughly this many OFDM symbols, the RF channel
    has likely changed enough that the single channel estimate from the
    training symbol(s) no longer applies."""
    ofdm = _make_ofdm()
    too_many_bits = (Ofdm.MAX_PAYLOAD_SYMBOLS + 1) * ofdm.bits_per_ofdm_symbol
    with pytest.raises(ValueError, match="MAX_PAYLOAD_SYMBOLS"):
        ofdm.generate_frame(np.zeros((1, too_many_bits), dtype="uint8"))


def test_generate_frame_allows_exactly_max_payload_symbols():
    ofdm = _make_ofdm()
    exactly_max_bits = Ofdm.MAX_PAYLOAD_SYMBOLS * ofdm.bits_per_ofdm_symbol
    tx_iq = ofdm.generate_frame(np.zeros((1, exactly_max_bits), dtype="uint8"))
    assert tx_iq.shape[-1] > 0  # just needs to not raise


def test_rx_process_rejects_explicit_override_over_max_payload_symbols():
    ofdm = _make_ofdm()
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, ofdm.bits_per_ofdm_symbol)).astype("uint8")
    tx_iq = ofdm.generate_frame(bits)
    with pytest.raises(ValueError, match="MAX_PAYLOAD_SYMBOLS"):
        ofdm.rx_process(tx_iq, n_payload_symbols=Ofdm.MAX_PAYLOAD_SYMBOLS + 1)


def test_corrupted_header_can_produce_over_max_payload_symbols_and_gets_rejected():
    """Regression test for the actual failure mode found during
    development: a corrupted header decode returning a huge garbage
    symbol count used to crash deep inside slot extraction with a
    cryptic array-broadcast error. Two things confirmed together: (1) a
    corrupted payload_len_bits field (the same corruption technique as
    test_decoded_unknown_crc_code_raises_value_error, applied to a
    different byte) really does decode to a value implying more symbols
    than MAX_PAYLOAD_SYMBOLS; (2) rx_process() really does reject such a
    value once it encounters one (test_rx_process_rejects_explicit_
    override_over_max_payload_symbols exercises that same guard code
    path directly)."""
    ofdm = _make_ofdm()
    bits = ofdm._encode_header_bits(ofdm.bits_per_ofdm_symbol, "qpsk", "none", None)
    unscrambled = bits ^ ofdm._header_scramble_mask
    header_bytes = bytearray(np.packbits(unscrambled).tobytes())
    header_bytes[1] = 0xFF  # payload_len_bits high byte -> huge bogus length
    header_bytes[2] = 0xFF
    corrupted_bits = np.unpackbits(np.frombuffer(bytes(header_bytes), dtype=np.uint8)) ^ ofdm._header_scramble_mask

    decoded = ofdm._decode_header_bits(corrupted_bits)
    bits_per_symbol_payload = ofdm.grid.n_data * ofdm.modem.bits_per_symbol
    implied_n_payload_symbols = decoded["payload_len_bits"] // bits_per_symbol_payload
    assert implied_n_payload_symbols > Ofdm.MAX_PAYLOAD_SYMBOLS


def test_header_carries_full_liquid_dsp_style_fields():
    """The header now matches liquid-dsp's field layout (112 bits: 8
    protocol/version + 16 payload_len_bits + 8 mod_scheme + 8 crc/fec0 +
    8 fec1 + 64 user_data), not just a trimmed-down length field. Confirm
    every field round-trips through generate_frame -> rx_process."""
    ofdm = _make_ofdm(modem="qam16")
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, ofdm.bits_per_ofdm_symbol)).astype("uint8")
    tx_iq = ofdm.generate_frame(bits, user_data=b"ABCDEFGH")
    result = ofdm.rx_process(tx_iq)

    header = result["header"]
    assert header["protocol_version"] == Ofdm.PROTOCOL_VERSION
    assert header["payload_len_bits"] == bits.shape[-1]
    assert header["mod_scheme"] == "qam16"
    assert header["crc"] == "none"
    assert header["fec0"] == "none"
    assert header["fec1"] == "none"
    assert header["user_data"] == b"ABCDEFGH"
    np.testing.assert_array_equal(result["bits"], bits)


def test_rx_process_resolves_mod_scheme_from_header_not_self_modem():
    """The real correctness fix: a receiver is a separate device that
    never saw the transmitter's Ofdm(...) call, so rx_process() must not
    assume self.modem matches what's actually in the signal -- it has to
    read mod_scheme from the decoded header and use that. Proven here by
    decoding with an Ofdm object whose own modem= is deliberately WRONG
    (qam64) relative to what was actually transmitted (qpsk); decoding
    must still succeed because the header, not self.modem, drives it."""
    tx_ofdm = _make_ofdm(modem="qpsk")
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, tx_ofdm.bits_per_ofdm_symbol)).astype("uint8")
    tx_iq = tx_ofdm.generate_frame(bits)

    rx_ofdm = _make_ofdm(modem="qam64")  # deliberately mismatched
    result = rx_ofdm.rx_process(tx_iq)

    assert result["header"]["mod_scheme"] == "qpsk"  # correctly read from the wire, not "qam64"
    np.testing.assert_array_equal(result["bits"], bits)


def test_fec_conv_v27_end_to_end_identity_channel():
    ofdm = _make_ofdm(fec="conv_v27")
    # conv_v27 is a streaming code (any k works); pick k so the encoded
    # length divides bits_per_ofdm_symbol exactly: 2*(k+6) == 80 -> k=34.
    rng = np.random.default_rng(0)
    raw_bits = rng.integers(0, 2, size=(1, 34)).astype("uint8")
    tx_iq = ofdm.generate_frame(raw_bits)
    result = ofdm.rx_process(tx_iq)
    assert result["header"]["fec0"] == "conv_v27"
    assert result["header"]["payload_len_bits"] == 34  # RAW bit count, not the encoded one
    np.testing.assert_array_equal(result["bits"], raw_bits)


def test_fec_rs_m8_end_to_end_identity_channel():
    # rs_m8's block size (1784 raw bits) needs n_data*bits_per_symbol to
    # divide its encoded length (2040 bits) evenly -- n_data=204 at QPSK
    # gives bits_per_ofdm_symbol=408, and 2040/408=5 exactly.
    ofdm = Ofdm(fft_size=256, n_pilot=6, n_data=204, cp_len=32, modem="qpsk", fec="rs_m8", backend="numpy")
    rng = np.random.default_rng(0)
    raw_bits = rng.integers(0, 2, size=(1, ofdm.fec_codec.k_bits)).astype("uint8")
    tx_iq = ofdm.generate_frame(raw_bits)
    result = ofdm.rx_process(tx_iq)
    assert result["header"]["fec0"] == "rs_m8"
    assert result["header"]["payload_len_bits"] == ofdm.fec_codec.k_bits
    np.testing.assert_array_equal(result["bits"], raw_bits)


def test_rx_process_resolves_fec0_from_header_not_self_fec_codec():
    """Same correctness property as mod_scheme resolution, now for FEC:
    a receiver is a separate device that never saw the transmitter's
    Ofdm(...) call, so rx_process() must read fec0 from the decoded
    header, not assume self.fec_codec matches. Proven by decoding a
    conv_v27-encoded frame with a locally-mismatched fec="none" object."""
    tx_ofdm = _make_ofdm(fec="conv_v27")
    rng = np.random.default_rng(0)
    raw_bits = rng.integers(0, 2, size=(1, 34)).astype("uint8")
    tx_iq = tx_ofdm.generate_frame(raw_bits)

    rx_ofdm = _make_ofdm(fec="none")  # deliberately mismatched
    result = rx_ofdm.rx_process(tx_iq)

    assert result["header"]["fec0"] == "conv_v27"  # correctly read from the wire, not "none"
    np.testing.assert_array_equal(result["bits"], raw_bits)


def test_rx_process_resolves_rs_m8_fec0_from_header_not_self_fec_codec():
    """Same as above, for rs_m8 specifically -- not just conv_v27."""
    tx_ofdm = Ofdm(fft_size=256, n_pilot=6, n_data=204, cp_len=32, modem="qpsk", fec="rs_m8", backend="numpy")
    rng = np.random.default_rng(0)
    raw_bits = rng.integers(0, 2, size=(1, tx_ofdm.fec_codec.k_bits)).astype("uint8")
    tx_iq = tx_ofdm.generate_frame(raw_bits)

    rx_ofdm = Ofdm(fft_size=256, n_pilot=6, n_data=204, cp_len=32, modem="qpsk", fec="none", backend="numpy")
    result = rx_ofdm.rx_process(tx_iq)

    assert result["header"]["fec0"] == "rs_m8"
    np.testing.assert_array_equal(result["bits"], raw_bits)


def test_fec_ldpc_end_to_end_identity_channel_and_dynamic_resolution():
    """LDPC through the full Ofdm pipeline: clean round-trip AND the
    same dynamic-fec0-resolution correctness property already proven
    for conv_v27/rs_m8 -- a receiver that never saw fec="ldpc_648_r12"
    at its own construction must still decode correctly by reading it
    from the header. n_data=216 at bpsk gives bits_per_ofdm_symbol=216,
    which divides ldpc_648_r12's n=648 exactly (3 OFDM symbols)."""
    tx_ofdm = Ofdm(fft_size=256, n_pilot=8, n_data=216, cp_len=32, modem="bpsk",
                   fec="ldpc_648_r12", backend="numpy")
    rng = np.random.default_rng(0)
    raw_bits = rng.integers(0, 2, size=(1, tx_ofdm.fec_codec.k_bits)).astype("uint8")
    tx_iq = tx_ofdm.generate_frame(raw_bits)

    rx_ofdm = Ofdm(fft_size=256, n_pilot=8, n_data=216, cp_len=32, modem="bpsk", fec="none", backend="numpy")
    result = rx_ofdm.rx_process(tx_iq)

    assert result["header"]["fec0"] == "ldpc_648_r12"
    assert result["header"]["payload_len_bits"] == tx_ofdm.fec_codec.k_bits
    np.testing.assert_array_equal(result["bits"], raw_bits)


def test_two_stage_fec_end_to_end_identity_channel_and_dynamic_resolution():
    """fec1 (the outer/second FEC stage, docs/todo.md #1.2) through the
    full Ofdm pipeline: clean round-trip AND dynamic fec0/fec1
    resolution from the header, exactly like the single-stage tests
    above -- a receiver constructed with fec="none" (no FEC at all) must
    still decode correctly, reading BOTH stages from the wire. k=156 for
    conv_v27 (inner) encodes to 2*(156+6)=324... doubled to 648 bits by
    using 2 conv_v27 codewords back to back is NOT what happens here --
    n_data=216 at bpsk actually needs 648 encoded bits (3 symbols), and
    ldpc_648_r12 (outer)'s own k=324 must divide the inner's OWN output;
    2*(156+6)=324 exactly, so ldpc_648_r12 sees exactly one inner
    codeword's worth and produces exactly one 648-bit outer codeword."""
    tx_ofdm = Ofdm(fft_size=256, n_pilot=8, n_data=216, cp_len=32, modem="bpsk",
                   fec="conv_v27", fec1="ldpc_648_r12", backend="numpy")
    rng = np.random.default_rng(0)
    raw_bits = rng.integers(0, 2, size=(1, 156)).astype("uint8")
    tx_iq = tx_ofdm.generate_frame(raw_bits)

    rx_ofdm = Ofdm(fft_size=256, n_pilot=8, n_data=216, cp_len=32, modem="bpsk", fec="none", backend="numpy")
    result = rx_ofdm.rx_process(tx_iq)

    assert result["header"]["fec0"] == "conv_v27"
    assert result["header"]["fec1"] == "ldpc_648_r12"
    assert result["header"]["payload_len_bits"] == 156
    np.testing.assert_array_equal(result["bits"], raw_bits)


def test_two_stage_fec_corrects_errors_that_would_defeat_inner_alone():
    """Real coding-gain check inside the full pipeline, not just
    plumbing: deterministic bit errors injected into the payload region
    (confined to the first payload OFDM symbol, using only public
    modem/grid/mod attributes -- same reconstruction technique as the
    single-stage FEC tests above), at exactly the same 14-bit-flip
    pattern confirmed separately (FEC("conv_v27").decode() on the same
    156-bit message with the same positions flipped) to defeat conv_v27
    ALONE -- must be fully corrected once ldpc_648_r12 (outer) mops up
    after conv_v27 (inner) decodes, proving the outer stage is doing
    genuine work here, not redundant with what the inner stage would
    have handled unaided."""
    ofdm = Ofdm(fft_size=256, n_pilot=8, n_data=216, cp_len=32, modem="bpsk",
                fec="conv_v27", fec1="ldpc_648_r12", backend="numpy")
    rng = np.random.default_rng(0)
    raw_bits = rng.integers(0, 2, size=(1, 156)).astype("uint8")
    bit_positions_to_flip = np.arange(0, 40, 3)

    # Self-verifying, not just documented: confirm conv_v27 ALONE really
    # does fail this exact message/corruption pattern before trusting
    # the two-stage pipeline's success as meaningful.
    from spectracuda.fec import FEC

    conv_only = FEC("conv_v27", backend="numpy")
    inner_encoded = conv_only.encode(raw_bits)
    inner_corrupted = inner_encoded.copy()
    inner_corrupted[0, bit_positions_to_flip] ^= 1
    inner_only_decoded = conv_only.decode(inner_corrupted)
    assert not np.array_equal(inner_only_decoded, raw_bits)

    tx_iq = ofdm.generate_frame(raw_bits)
    encoded_bits = ofdm.packetizer.encode(raw_bits)
    grouped = encoded_bits.reshape(1, -1, ofdm.bits_per_ofdm_symbol)
    first_symbol_bits = grouped[:, 0, :]
    corrupted_tx = _corrupt_first_payload_symbol_bits(
        ofdm, tx_iq, first_symbol_bits, bit_positions_to_flip=bit_positions_to_flip
    )

    result = ofdm.rx_process(corrupted_tx)
    np.testing.assert_array_equal(result["bits"], raw_bits)


@pytest.mark.parametrize("n_batch", [3, 5])
def test_fec_conv_v27_batched_with_different_content_per_item(n_batch):
    ofdm = _make_ofdm(fec="conv_v27")
    rng = np.random.default_rng(0)
    raw_bits = rng.integers(0, 2, size=(n_batch, 34)).astype("uint8")
    tx_iq = ofdm.generate_frame(raw_bits)
    result = ofdm.rx_process(tx_iq)
    assert result["bits"].shape == (n_batch, 34)
    np.testing.assert_array_equal(result["bits"], raw_bits)


def test_fec_rs_m8_batched_with_different_content_per_item():
    ofdm = Ofdm(fft_size=256, n_pilot=6, n_data=204, cp_len=32, modem="qpsk", fec="rs_m8", backend="numpy")
    rng = np.random.default_rng(0)
    n_batch = 4
    raw_bits = rng.integers(0, 2, size=(n_batch, ofdm.fec_codec.k_bits)).astype("uint8")
    tx_iq = ofdm.generate_frame(raw_bits)
    result = ofdm.rx_process(tx_iq)
    assert result["bits"].shape == (n_batch, ofdm.fec_codec.k_bits)
    np.testing.assert_array_equal(result["bits"], raw_bits)


def test_fec_conv_v27_corrects_deterministic_bit_errors_within_ofdm_pipeline():
    """Not a channel-based/statistical test (fragile -- see
    test_fec_is_genuinely_applied_not_a_silent_no_op's docstring for why
    that approach was abandoned) -- deterministic AWGN added ONLY to the
    payload region (preamble/training/header stay pristine, so sync and
    header decode are 100% reliable), at a noise level empirically
    confirmed to (a) cause real bit errors without FEC and (b) be fully
    corrected with conv_v27, isolating exactly what needs proving."""
    rng = np.random.default_rng(0)
    noise_std = 0.08

    def _run(fec_scheme, k):
        ofdm = _make_ofdm(fec=fec_scheme)
        raw_bits = rng.integers(0, 2, size=(1, k)).astype("uint8")
        tx_iq = ofdm.generate_frame(raw_bits)
        start = _payload_start(ofdm)
        payload_len = tx_iq.shape[-1] - start
        noise = (rng.standard_normal(payload_len) + 1j * rng.standard_normal(payload_len)) * noise_std
        corrupted = tx_iq.copy()
        corrupted[0, start:] += noise
        result = ofdm.rx_process(corrupted)
        return raw_bits, result

    raw_none, result_none = _run("none", 80)
    ber_none = np.mean(result_none["bits"] != raw_none)
    assert ber_none > 0  # confirms the noise is real, not negligible

    raw_conv, result_conv = _run("conv_v27", 34)
    np.testing.assert_array_equal(result_conv["bits"], raw_conv)  # fully corrected


def test_fec_rs_m8_corrects_deterministic_confined_byte_errors_within_ofdm_pipeline():
    """Same idea as the conv_v27 test above, but RS needs a differently-
    shaped corruption to test meaningfully: it corrects up to t=16
    *confined* byte errors, not scattered bit errors across the whole
    payload (RS is a poor match for uniformly-scattered noise -- that's
    what conv_v27 is for). Corrupts exactly 10 bytes (well within t=16),
    confined to the first payload OFDM symbol, using only the object's
    public modem/grid/mod attributes to reconstruct that one symbol --
    not a channel impairment, so the error count is exact, not
    statistical."""
    ofdm = Ofdm(fft_size=256, n_pilot=6, n_data=204, cp_len=32, modem="qpsk", fec="rs_m8", backend="numpy")
    rng = np.random.default_rng(0)
    raw_bits = rng.integers(0, 2, size=(1, ofdm.fec_codec.k_bits)).astype("uint8")
    tx_iq = ofdm.generate_frame(raw_bits)

    encoded_bits = ofdm.fec_codec.encode(raw_bits)
    grouped = encoded_bits.reshape(1, -1, ofdm.bits_per_ofdm_symbol)
    first_symbol_bits = grouped[:, 0, :]
    corrupted_tx = _corrupt_first_payload_symbol_bits(
        ofdm, tx_iq, first_symbol_bits, bit_positions_to_flip=np.arange(40, 120)  # bytes 5-14, 10 bytes
    )

    result = ofdm.rx_process(corrupted_tx)
    np.testing.assert_array_equal(result["bits"], raw_bits)  # fully corrected

    # same 80-bit corruption with fec="none" should show real errors,
    # confirming the corruption is meaningful, not confirming nothing
    ofdm_none = Ofdm(fft_size=256, n_pilot=6, n_data=204, cp_len=32, modem="qpsk", fec="none", backend="numpy")
    raw_bits_none = rng.integers(0, 2, size=(1, ofdm_none.bits_per_ofdm_symbol * grouped.shape[1])).astype("uint8")
    tx_iq_none = ofdm_none.generate_frame(raw_bits_none)
    grouped_none = raw_bits_none.reshape(1, -1, ofdm_none.bits_per_ofdm_symbol)
    corrupted_tx_none = _corrupt_first_payload_symbol_bits(
        ofdm_none, tx_iq_none, grouped_none[:, 0, :], bit_positions_to_flip=np.arange(40, 120)
    )
    result_none = ofdm_none.rx_process(corrupted_tx_none)
    assert np.mean(result_none["bits"] != raw_bits_none) > 0


def test_fec_is_genuinely_applied_not_a_silent_no_op():
    """Plumbing check: with fec="conv_v27" configured, generate_frame()
    must actually FEC-encode the raw payload before modulating it (not
    silently pass it through) -- confirmed by checking the transmitted
    IQ differs from what the identical raw bits would produce with
    fec="none" (same modem/grid/preamble/training, only fec differs).
    Coding gain itself (that FEC measurably reduces BER under noise) is
    already rigorously proven in isolation, at a controlled operating
    point, by test_fec_viterbi.py's test_stress_reduces_ber_
    substantially_at_moderate_noise and test_fec_reed_solomon.py's
    test_corrects_up_to_t_symbol_errors -- re-deriving that statistic
    here under the full stack's compounding multipath+CFO+header-
    reliability randomness turned out to be too fragile to make a
    reliable, non-flaky assertion across arbitrary random seeds (tried;
    see git history/session notes) without extensive per-scenario SNR
    tuning that isn't this test's job."""
    ofdm_none = _make_ofdm(fec="none")
    ofdm_conv = _make_ofdm(fec="conv_v27")
    rng = np.random.default_rng(0)
    raw_bits = rng.integers(0, 2, size=(1, 34)).astype("uint8")

    # fec="none" needs a full symbol's worth (80 bits) to avoid the
    # separate "must be an exact multiple" gap -- pad with the same 34
    # bits repeated, irrelevant to what's being checked here (whether
    # the two paths produce different signals, not bit-exact content).
    padded_bits = np.tile(raw_bits, (1, 80 // 34 + 1))[:, :80]
    tx_none = ofdm_none.generate_frame(padded_bits)
    tx_conv = ofdm_conv.generate_frame(raw_bits)

    assert tx_none.shape == tx_conv.shape  # both: 1 preamble + training + header + 1 payload symbol
    assert not np.allclose(tx_none, tx_conv)  # genuinely different signal, not a no-op


def test_crc_end_to_end_identity_channel():
    """crc32 alone (fec="none"): 48 raw bits + 32-bit key = 80 bits,
    exactly one bits_per_ofdm_symbol at _make_ofdm's default qpsk/n_data=40."""
    ofdm = _make_ofdm(crc="crc32")
    rng = np.random.default_rng(0)
    raw_bits = rng.integers(0, 2, size=(1, 48)).astype("uint8")
    tx_iq = ofdm.generate_frame(raw_bits)
    result = ofdm.rx_process(tx_iq)
    assert result["header"]["crc"] == "crc32"
    assert result["header"]["payload_len_bits"] == 48  # RAW count, excluding the CRC key
    np.testing.assert_array_equal(result["bits"], raw_bits)
    np.testing.assert_array_equal(result["crc_valid"], [True])


def test_crc_none_returns_crc_valid_none():
    """Default crc="none": no key is appended, and rx_process() returns
    crc_valid=None rather than an all-True array -- there's genuinely
    nothing checked, which is a different thing from "checked and fine"."""
    ofdm = _make_ofdm()
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, ofdm.bits_per_ofdm_symbol)).astype("uint8")
    tx_iq = ofdm.generate_frame(bits)
    result = ofdm.rx_process(tx_iq)
    assert result["crc_valid"] is None


def test_rx_process_resolves_crc_from_header_not_self_crc_codec():
    """Same correctness property already proven for mod_scheme/fec0: a
    receiver is a separate device that never saw the transmitter's
    Ofdm(...) call, so rx_process() must read crc from the decoded
    header, not assume self.crc_codec matches. Proven by decoding a
    crc32-protected frame with a locally-mismatched crc="none" object."""
    tx_ofdm = _make_ofdm(crc="crc32")
    rng = np.random.default_rng(0)
    raw_bits = rng.integers(0, 2, size=(1, 48)).astype("uint8")
    tx_iq = tx_ofdm.generate_frame(raw_bits)

    rx_ofdm = _make_ofdm(crc="none")  # deliberately mismatched
    result = rx_ofdm.rx_process(tx_iq)

    assert result["header"]["crc"] == "crc32"  # correctly read from the wire, not "none"
    np.testing.assert_array_equal(result["bits"], raw_bits)
    np.testing.assert_array_equal(result["crc_valid"], [True])


@pytest.mark.parametrize("n_batch", [3, 5])
def test_crc_batched_with_different_content_per_item(n_batch):
    ofdm = _make_ofdm(crc="crc32")
    rng = np.random.default_rng(0)
    raw_bits = rng.integers(0, 2, size=(n_batch, 48)).astype("uint8")
    tx_iq = ofdm.generate_frame(raw_bits)
    result = ofdm.rx_process(tx_iq)
    assert result["bits"].shape == (n_batch, 48)
    np.testing.assert_array_equal(result["bits"], raw_bits)
    np.testing.assert_array_equal(result["crc_valid"], [True] * n_batch)


def test_crc_detects_deterministic_corruption_within_ofdm_pipeline():
    """The core proof this exists for: with fec="none" (so a flipped bit
    survives uncorrected all the way to the CRC check), an exact, known
    single-bit corruption of the payload -- built via the same
    reconstruct-the-symbol-from-public-attributes technique used for the
    FEC deterministic tests, not a channel roll of the dice -- must flip
    crc_valid to False, while an identical clean transmission (same
    content, no corruption) reports True. This is the actual gap CRC
    fills: conv_v27 alone has no way to signal a wrong decode."""
    ofdm = _make_ofdm(crc="crc32")
    rng = np.random.default_rng(0)
    raw_bits = rng.integers(0, 2, size=(1, 48)).astype("uint8")

    tx_clean = ofdm.generate_frame(raw_bits)
    result_clean = ofdm.rx_process(tx_clean)
    assert result_clean["crc_valid"][0]
    np.testing.assert_array_equal(result_clean["bits"], raw_bits)

    bits_with_crc = ofdm._bytes_to_bits(ofdm.crc_codec.append_key(ofdm._bits_to_bytes(raw_bits)))
    first_symbol_bits = np.asarray(bits_with_crc).reshape(1, -1, ofdm.bits_per_ofdm_symbol)[:, 0, :]
    corrupted_tx = _corrupt_first_payload_symbol_bits(
        ofdm, tx_clean, first_symbol_bits, bit_positions_to_flip=np.array([5])
    )
    result_corrupted = ofdm.rx_process(corrupted_tx)
    assert not result_corrupted["crc_valid"][0]


def test_crc_combined_with_fec_conv_v27_end_to_end():
    """crc + fec together, in the liquid-dsp order (crc appended first,
    then fec-encoded; see generate_frame()'s docstring/comments): 1 raw
    byte (8 bits) + crc8's 8-bit key = 16 bits into conv_v27 ->
    2*(16+6)=44 bits, exactly one bpsk OFDM symbol at n_data=44."""
    ofdm = Ofdm(fft_size=64, n_pilot=8, n_data=44, cp_len=16, modem="bpsk",
                fec="conv_v27", crc="crc8", backend="numpy")
    rng = np.random.default_rng(0)
    raw_bits = rng.integers(0, 2, size=(1, 8)).astype("uint8")
    tx_iq = ofdm.generate_frame(raw_bits)
    result = ofdm.rx_process(tx_iq)
    assert result["header"]["crc"] == "crc8"
    assert result["header"]["fec0"] == "conv_v27"
    assert result["header"]["payload_len_bits"] == 8  # RAW count, excluding both crc key and fec tail/rate
    np.testing.assert_array_equal(result["bits"], raw_bits)
    np.testing.assert_array_equal(result["crc_valid"], [True])


def test_crc_combined_with_iq_dtype_float16_identity_channel():
    """Checked directly, matching the same standard already applied to
    FEC: crc still round-trips exactly (key computed on requantized
    bytes at both ends, since generate_frame/rx_process only quantize
    the IQ samples, not the bits) with 16-bit ADC/DAC boundary
    quantization active."""
    ofdm = _make_ofdm(crc="crc32", iq_dtype="float16")
    rng = np.random.default_rng(0)
    raw_bits = rng.integers(0, 2, size=(1, 48)).astype("uint8")
    tx_iq = ofdm.generate_frame(raw_bits)
    assert tx_iq.dtype == np.complex64
    result = ofdm.rx_process(tx_iq)
    np.testing.assert_array_equal(result["bits"], raw_bits)
    np.testing.assert_array_equal(result["crc_valid"], [True])


def test_crc_requires_byte_aligned_payload():
    """crc is byte-oriented, matching liquid-dsp's own crc/packetizer
    interface (see crc.py's module docstring) -- a non-byte-aligned raw
    payload must raise a clear ValueError, not a confusing failure
    downstream in packbits."""
    ofdm = _make_ofdm(crc="crc32")
    with pytest.raises(ValueError, match="multiple of 8"):
        ofdm.generate_frame(np.zeros((1, 45), dtype="uint8"))


def test_unknown_crc_scheme_raises_value_error():
    with pytest.raises(ValueError):
        _make_ofdm(crc="crc99")


def test_decoded_unknown_crc_code_raises_value_error():
    """crc_code=0 (LIQUID_CRC_UNKNOWN) is deliberately unrepresented in
    _CRC_SCHEME_NAMES -- a decoded header carrying it must raise
    ValueError (likely corruption), not NotImplementedError: all six of
    CRC's real schemes are genuinely supported now, so 0 specifically is
    an invalid/reserved code, not a real-but-unsupported one."""
    ofdm = _make_ofdm()
    bits = ofdm._encode_header_bits(80, "qpsk", "none", None, "none")
    unscrambled = bits ^ ofdm._header_scramble_mask
    header_bytes = bytearray(np.packbits(unscrambled).tobytes())
    header_bytes[4] &= 0x1F  # clear crc_code (top 3 bits of byte 4) to 0 = LIQUID_CRC_UNKNOWN
    corrupted_bits = np.unpackbits(np.frombuffer(bytes(header_bytes), dtype=np.uint8)) ^ ofdm._header_scramble_mask
    with pytest.raises(ValueError):
        ofdm._decode_header_bits(corrupted_bits)


def test_decoded_crc_schemes_are_genuinely_supported():
    """The flip side of the above: all six of CRC's real schemes must
    round-trip through the header cleanly, not raise."""
    ofdm = _make_ofdm()
    for scheme in ("none", "checksum", "crc8", "crc16", "crc24", "crc32"):
        bits = ofdm._encode_header_bits(80, "qpsk", "none", None, scheme)
        decoded = ofdm._decode_header_bits(bits)
        assert decoded["crc"] == scheme


def test_decoded_fec0_conv_v27_and_rs_m8_are_genuinely_supported():
    """The flip side of the above: fec0 codes for conv_v27/rs_m8 must
    NOT raise -- they're real, working schemes now, not placeholders."""
    ofdm = _make_ofdm()
    for scheme in ("none", "conv_v27", "rs_m8"):
        bits = ofdm._encode_header_bits(80, "qpsk", scheme, None)
        decoded = ofdm._decode_header_bits(bits)
        assert decoded["fec0"] == scheme


def test_decoded_fec1_schemes_are_genuinely_supported():
    """fec1 (the outer/second FEC stage, see docs/todo.md #1.2) used to
    be forced to "none" and raise NotImplementedError otherwise -- now
    that two-stage FEC is wired in (spectracuda.framing.Packetizer), any
    real FEC scheme is a genuinely accepted fec1 value, not corruption."""
    ofdm = _make_ofdm()
    for scheme in ("none", "conv_v27", "rs_m8", "ldpc_648_r12"):
        bits = ofdm.header_codec.encode_bits(80, "qpsk", "none", None, fec1=scheme)
        decoded = ofdm._decode_header_bits(bits)
        assert decoded["fec1"] == scheme


def test_decoded_unknown_fec1_code_raises_value_error():
    """An out-of-range fec1 code (not one of the 15 assigned codes,
    0-14) must raise ValueError -- likely header corruption, not a
    real-but-unsupported scheme."""
    ofdm = _make_ofdm()
    bits = ofdm._encode_header_bits(80, "qpsk", "none", None)
    unscrambled = bits ^ ofdm._header_scramble_mask
    header_bytes = bytearray(np.packbits(unscrambled).tobytes())
    header_bytes[5] = 30  # unassigned fec1 code (only 0-14 are real)
    corrupted_bits = np.unpackbits(np.frombuffer(bytes(header_bytes), dtype=np.uint8)) ^ ofdm._header_scramble_mask
    with pytest.raises(ValueError):
        ofdm._decode_header_bits(corrupted_bits)


def test_header_papr_is_not_pathological():
    """Regression test for the real root cause found during development:
    unscrambled, mostly-repeated header content constructively interferes
    into a massive time-domain peak (measured: 181x peak/average power,
    vs ~5x for ordinary payload content). That spike, combined with real
    multipath + any timing imperfection, corrupted the header specifically
    while payload (naturally varied content) decoded fine under the
    identical channel -- confirmed by direct comparison, not assumption.
    Fixed by scrambling the header bits with a fixed mask before
    modulating (plus random filler for any unused capacity). This test
    checks the fix stays in place: PAPR must stay in the normal OFDM
    range, not blow up back toward the unscrambled ~181x."""
    ofdm = Ofdm(fft_size=256, n_pilot=6, n_data=200, cp_len=32, modem="qpsk", backend="numpy")
    header_time = ofdm._build_header_symbols(80, "qpsk", "none", None, n_batch=1)[0]
    papr = float(np.abs(header_time).max() ** 2 / np.mean(np.abs(header_time) ** 2))
    assert papr < 20.0  # normal OFDM PAPR is single-digit-to-low-double-digit; 181x was the bug


def test_header_survives_realistic_multipath_in_most_cases():
    """The header decodes correctly across several random multipath
    realizations at a realistic SNR, now that the PAPR bug is fixed --
    not a guarantee for every possible channel (an unprotected header
    can still occasionally lose a bit to a channel realization whose
    null happens to land on one of the spread bit positions; see class
    docstring -- real FEC/CRC, not built yet, is what closes that gap
    completely), but it should no longer fail systematically."""
    from spectracuda.sim import Channel

    fft_size, cp_len = 256, 32
    successes = 0
    n_seeds = 5
    for seed in range(n_seeds):
        ofdm = Ofdm(fft_size=fft_size, n_pilot=6, n_data=200, cp_len=cp_len, modem="qpsk",
                    n_training_symbols=2, backend="numpy")
        rng = np.random.default_rng(0)
        tx_bits = rng.integers(0, 2, size=(1, ofdm.grid.n_data * ofdm.modem.bits_per_symbol)).astype("uint8")
        tx_iq = ofdm.generate_frame(tx_bits)[0]

        taps = Channel.random_multipath_taps(3, seed=seed * 1000 + 7)
        channel = Channel(snr_db=25.0, multipath_taps=taps, cfo=0.15, cfo_fft_size=fft_size,
                           seed=seed * 1000 + 7, backend="numpy")
        padded = np.concatenate([np.zeros(40, dtype="complex64"), tx_iq, np.zeros(40, dtype="complex64")])
        rx_iq = channel.process(padded)[0]

        expected_bits = tx_bits.shape[-1]
        try:
            result = ofdm.rx_process(rx_iq)
            if result["header"]["payload_len_bits"] == expected_bits:
                successes += 1
        except (ValueError, NotImplementedError):
            pass  # a bad-luck null hitting the header is a known, documented residual risk

    assert successes >= n_seeds - 1  # at most one bad-luck failure expected out of 5


def test_identity_channel_round_trip_batched():
    ofdm = _make_ofdm()
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(3, ofdm.grid.n_data * ofdm.modem.bits_per_symbol)).astype("uint8")

    tx_iq = ofdm.generate_frame(bits)
    result = ofdm.rx_process(tx_iq)

    np.testing.assert_array_equal(result["bits"], bits)
    np.testing.assert_array_equal(result["start_index"], np.zeros(3, dtype=result["start_index"].dtype))
    np.testing.assert_allclose(result["cfo_estimate"], np.zeros(3), atol=1e-6)


def test_generate_frame_output_dtype_is_complex64_by_default():
    ofdm = _make_ofdm()
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, ofdm.grid.n_data * ofdm.modem.bits_per_symbol)).astype("uint8")
    tx_iq = ofdm.generate_frame(bits)
    assert tx_iq.dtype == np.complex64  # regression guard for the fft.py complex128 bug


def test_iq_dtype_float32_default_is_a_no_op():
    ofdm_default = _make_ofdm()
    ofdm_explicit = _make_ofdm(iq_dtype="float32")
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, ofdm_default.grid.n_data * ofdm_default.modem.bits_per_symbol)).astype("uint8")
    np.testing.assert_array_equal(ofdm_default.generate_frame(bits), ofdm_explicit.generate_frame(bits))


def test_iq_dtype_float16_quantizes_but_still_round_trips():
    ofdm = _make_ofdm(iq_dtype="float16")
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, ofdm.grid.n_data * ofdm.modem.bits_per_symbol)).astype("uint8")

    tx_default = _make_ofdm().generate_frame(bits)
    tx_quantized = ofdm.generate_frame(bits)

    assert tx_quantized.dtype == np.complex64  # still complex64 -- boundary quantization, not float16 compute
    assert not np.array_equal(tx_default, tx_quantized)  # quantization actually changed something
    np.testing.assert_allclose(tx_default, tx_quantized, atol=2e-3)  # but only by ~float16 resolution

    result = ofdm.rx_process(tx_quantized)
    np.testing.assert_array_equal(result["bits"], bits)  # identity channel: still recovers exactly


def test_invalid_iq_dtype_raises():
    with pytest.raises(ValueError):
        _make_ofdm(iq_dtype="float64")
    with pytest.raises(ValueError):
        _make_ofdm(iq_dtype="int16")


@pytest.mark.parametrize("fec_scheme", ["conv_v27", "rs_m8"])
def test_fec_combined_with_iq_dtype_float16_identity_channel(fec_scheme):
    """The FEC integration work didn't originally get exercised under
    iq_dtype="float16" at all -- checked directly on request, not
    assumed to be fine by extension. Both schemes still round-trip
    exactly on an identity channel with boundary quantization active."""
    if fec_scheme == "rs_m8":
        ofdm = Ofdm(fft_size=256, n_pilot=6, n_data=204, cp_len=32, modem="qpsk",
                    fec=fec_scheme, backend="numpy", iq_dtype="float16")
        k = ofdm.fec_codec.k_bits
    else:
        ofdm = Ofdm(fft_size=64, n_pilot=8, n_data=40, cp_len=16, modem="qpsk",
                    fec=fec_scheme, backend="numpy", iq_dtype="float16")
        k = 34  # 2*(34+6)=80 divides bits_per_ofdm_symbol=80 exactly

    rng = np.random.default_rng(0)
    raw_bits = rng.integers(0, 2, size=(1, k)).astype("uint8")
    tx_iq = ofdm.generate_frame(raw_bits)
    assert tx_iq.dtype == np.complex64  # still complex64 -- boundary quantization, not float16 compute

    result = ofdm.rx_process(tx_iq)
    assert result["header"]["fec0"] == fec_scheme
    np.testing.assert_array_equal(result["bits"], raw_bits)


def test_fec_combined_with_iq_dtype_float16_under_real_multipath_channel():
    """Same combination under a real noisy channel (not just an
    identity one) -- where a subtle interaction between quantization
    and FEC decoding would actually be more likely to show up."""
    from spectracuda.sim import Channel

    def _run(iq_dtype):
        ofdm = Ofdm(fft_size=256, n_pilot=6, n_data=200, cp_len=32, modem="qpsk", fec="conv_v27",
                    sync="schmidl_cox", cfo="schmidl_cox", n_training_symbols=2,
                    backend="numpy", iq_dtype=iq_dtype)
        rng = np.random.default_rng(0)
        raw_bits = rng.integers(0, 2, size=(1, 194)).astype("uint8")  # 2*(194+6)=400 exactly
        tx_iq = ofdm.generate_frame(raw_bits)
        taps = Channel.random_multipath_taps(3, seed=0)
        channel = Channel(snr_db=20.0, multipath_taps=taps, cfo=0.1, cfo_fft_size=256, seed=0, backend="numpy")
        padded = np.concatenate([np.zeros(40, dtype="complex64"), tx_iq[0], np.zeros(40, dtype="complex64")])
        rx_iq = channel.process(padded)[0]
        result = ofdm.rx_process(rx_iq)
        return raw_bits, result

    raw_bits_32, result_32 = _run("float32")
    raw_bits_16, result_16 = _run("float16")
    np.testing.assert_array_equal(result_32["bits"], raw_bits_32)
    np.testing.assert_array_equal(result_16["bits"], raw_bits_16)


def test_fec_conv_v27_and_rs_m8_are_now_accepted():
    """fec="conv_v27"/"rs_m8" used to raise NotImplementedError -- now
    that FEC is wired in, both construct successfully."""
    _make_ofdm(fec="conv_v27")
    _make_ofdm(fec="rs_m8")


def test_fec_ldpc_variants_are_accepted():
    """All 12 IEEE 802.11n LDPC variants construct successfully as
    fec= -- a deliberate scope expansion beyond liquid-dsp parity, not
    a liquid-dsp scheme (see spectracuda.fec's module docstring)."""
    from spectracuda.fec.ldpc_tables import BASE_MATRICES

    for variant in BASE_MATRICES:
        _make_ofdm(fec=variant)


def test_unknown_fec_scheme_raises_value_error():
    """"polar" (not "ldpc"): LDPC is a real, accepted scheme family now
    (see test_fec_ldpc_variants_are_accepted) -- "ldpc" bare (no size/
    rate suffix) would still correctly raise too, but naming a
    genuinely nonexistent scheme here avoids any ambiguity about which
    of the two this test is actually checking."""
    with pytest.raises(ValueError):
        _make_ofdm(fec="polar")


def test_n_training_symbols_must_be_positive():
    with pytest.raises(ValueError):
        _make_ofdm(n_training_symbols=0)


def test_generate_frame_automatically_pads_a_partial_last_symbol():
    """docs/todo.md #1.10: an arbitrary raw bit count (not a multiple of
    n_data*bits_per_symbol) used to raise -- generate_frame() now pads
    the partial last payload symbol automatically, and rx_process()
    strips the padding back off using only the header's existing
    payload_len_bits (no new wire field needed) -- verified with a
    genuinely non-aligned count (5, not a divisor or multiple of
    bits_per_ofdm_symbol=80)."""
    ofdm = _make_ofdm()
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, 5)).astype("uint8")
    tx_iq = ofdm.generate_frame(bits)
    assert tx_iq.shape[-1] > 0  # didn't raise
    result = ofdm.rx_process(tx_iq)
    assert result["n_payload_symbols"] == 1  # 5 bits padded up to exactly 1 symbol
    np.testing.assert_array_equal(result["bits"], bits)


def test_generate_frame_padding_is_a_true_no_op_when_already_aligned():
    """Backward-compatible: a bit count that already divides evenly
    needs no padding at all (padding_bits=0), unchanged from before."""
    ofdm = _make_ofdm()
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, ofdm.bits_per_ofdm_symbol)).astype("uint8")
    tx_iq = ofdm.generate_frame(bits)
    result = ofdm.rx_process(tx_iq)
    assert result["n_payload_symbols"] == 1
    np.testing.assert_array_equal(result["bits"], bits)


def test_generate_frame_padding_combined_with_crc_and_fec():
    """The exact combination that originally surfaced this gap (crc +
    fec at an arbitrary raw bit count) -- confirmed working end to end,
    not just the no-crc/no-fec case above."""
    ofdm = _make_ofdm(crc="crc16", fec="conv_v27")
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, 40)).astype("uint8")  # arbitrary, byte-aligned for crc16
    tx_iq = ofdm.generate_frame(bits)
    result = ofdm.rx_process(tx_iq)
    np.testing.assert_array_equal(result["bits"], bits)
    assert result["crc_valid"][0]


def test_generate_frame_padding_across_multiple_symbols():
    """Padding must also work when the payload already spans several
    full symbols plus one partial one, not just a single partial symbol."""
    ofdm = _make_ofdm()
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, 3 * ofdm.bits_per_ofdm_symbol + 17)).astype("uint8")
    tx_iq = ofdm.generate_frame(bits)
    result = ofdm.rx_process(tx_iq)
    assert result["n_payload_symbols"] == 4
    np.testing.assert_array_equal(result["bits"], bits)


def test_max_payload_symbols_still_enforced_after_padding():
    """Padding must not be usable to sneak an over-long frame past
    MAX_PAYLOAD_SYMBOLS -- the ceiling-division symbol count (post-
    padding) is exactly what's checked against the cap, same as before."""
    ofdm = _make_ofdm()
    too_many_bits = Ofdm.MAX_PAYLOAD_SYMBOLS * ofdm.bits_per_ofdm_symbol + 1  # forces one extra symbol
    with pytest.raises(ValueError, match="MAX_PAYLOAD_SYMBOLS"):
        ofdm.generate_frame(np.zeros((1, too_many_bits), dtype="uint8"))


def test_generate_frame_batched_padding_with_different_content_per_item():
    ofdm = _make_ofdm()
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(3, 5)).astype("uint8")
    tx_iq = ofdm.generate_frame(bits)
    result = ofdm.rx_process(tx_iq)
    np.testing.assert_array_equal(result["bits"], bits)


def test_sync_without_generate_preamble_rejected():
    from spectracuda.block import Block

    class _NoPreambleSync(Block):
        def process(self, batch, **kwargs):
            return {"start_index": np.zeros(batch.shape[0], dtype=int)}

    with pytest.raises(TypeError):
        _make_ofdm(sync=_NoPreambleSync(backend="numpy"))


@pytest.mark.parametrize("iq_dtype", ["float32", "float16"])
def test_full_256_subcarrier_scenario_matches_manual_demo(iq_dtype):
    """Same scenario as examples/ofdm_256_schmidl_cox_demo.py, but driven
    entirely through the Ofdm class instead of manual slot arithmetic --
    with 2 training-symbol repetitions averaged for a better channel
    estimate (BER should improve over the single-training-symbol demo's
    ~0.01, not just match it). Parametrized over iq_dtype so the ADC/DAC
    boundary quantization is exercised under a realistic noisy/multipath/
    CFO scenario, not just the trivial identity-channel round trip in
    test_iq_dtype_float16_quantizes_but_still_round_trips -- at 25dB SNR
    the added float16 quantization noise turns out to be negligible next
    to the AWGN already present (confirmed empirically: identical BER
    both ways), so both dtypes share the same BER threshold below."""
    ofdm = Ofdm(
        fft_size=256, n_pilot=6, n_data=200, cp_len=32, modem="qpsk",
        n_training_symbols=2, backend="numpy", iq_dtype=iq_dtype,
    )
    assert ofdm.grid.n_data == 200
    assert ofdm.grid.n_pilot == 6
    assert len(ofdm.grid.null_indices) == 50

    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(2, ofdm.grid.n_data * ofdm.modem.bits_per_symbol)).astype("uint8")
    tx_iq = ofdm.generate_frame(bits)

    imp_rng = np.random.default_rng(7)
    n_taps = 3
    pad_before, pad_after = 40, 40
    eps_true = 0.15
    snr_db = 25.0

    rx_list = []
    for b in range(tx_iq.shape[0]):
        taps = (imp_rng.standard_normal(n_taps) + 1j * imp_rng.standard_normal(n_taps)) / np.sqrt(2 * n_taps)
        padded = np.concatenate(
            [np.zeros(pad_before, dtype="complex64"), tx_iq[b], np.zeros(pad_after, dtype="complex64")]
        )
        channeled = np.convolve(padded, taps)[: len(padded)].astype("complex64")
        sig_power = float(np.mean(np.abs(channeled) ** 2))
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = (
            imp_rng.standard_normal(len(channeled)) + 1j * imp_rng.standard_normal(len(channeled))
        ) * np.sqrt(noise_power / 2)
        n = np.arange(len(channeled))
        cfo_ramp = np.exp(1j * 2 * np.pi * eps_true * n / ofdm.fft_size)
        rx_list.append(((channeled + noise) * cfo_ramp).astype("complex64"))
    rx_iq = np.stack(rx_list, axis=0)

    result = ofdm.rx_process(rx_iq)
    true_start = pad_before

    np.testing.assert_allclose(result["start_index"], [true_start] * 2, atol=5)
    np.testing.assert_allclose(result["cfo_estimate"], [eps_true] * 2, atol=0.01)
    ber = np.mean(result["bits"] != bits)
    assert ber < 0.01  # better than the 1-training-symbol demo's ~0.01


def test_sync_zc_is_a_genuine_drop_in_replacement_for_schmidl_cox():
    """ZadoffChuSync must be swappable behind sync= exactly like
    SchmidlCoxSync -- proven with a real generate_frame/rx_process round
    trip (identity channel), not just standalone sync-detection tests.
    Paired with cfo="pilot_based", NOT "schmidl_cox": SchmidlCoxCFO
    depends on the preamble's repeated-halves shape, which a Zadoff-Chu
    preamble doesn't have (confirmed during development -- pairing them
    produces a garbage CFO estimate that corrupts the whole frame; see
    cfo/pilot_based.py's module docstring). n_training_symbols=2:
    PilotBasedCFO needs at least two repeats for its phase-slope
    estimate (see its docstring)."""
    ofdm = _make_ofdm(sync="zc", cfo="pilot_based", n_training_symbols=2)
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, ofdm.bits_per_ofdm_symbol)).astype("uint8")
    tx_iq = ofdm.generate_frame(bits)
    result = ofdm.rx_process(tx_iq)
    np.testing.assert_array_equal(result["bits"], bits)


def test_sync_zc_under_real_multipath_and_awgn_channel():
    """n_pilot=32 (not the usual 6) and snr_db=40 (not the usual 20-25)
    here are deliberate, not arbitrary: checked directly (not assumed),
    PilotBasedCFO's phase-slope estimate is genuinely noisier than
    SchmidlCoxCFO's at typical OFDM test SNRs, because it only has
    n_pilot complex samples per repeat to average over, vs Schmidl-Cox's
    correlation across half an entire preamble's worth of time samples
    (fft_size//2, far more averaging) -- confirmed empirically during
    development: n_pilot=6 at snr_db=20-25 (this suite's usual scenario
    for the schmidl_cox pairing) gave wildly noisy CFO estimates and
    fully broke decode more often than not, even at eps as small as
    0.02-0.05, purely from AWGN (not sync timing or multipath group
    delay, both checked separately and ruled out). This is a real,
    inherent trade-off of pilot-count-limited phase-slope CFO tracking,
    not a bug -- see cfo/pilot_based.py's module docstring."""
    from spectracuda.sim import Channel

    ofdm = Ofdm(fft_size=256, n_pilot=32, n_data=223, cp_len=32, modem="qpsk",
                sync="zc", cfo="pilot_based", n_training_symbols=2, backend="numpy")
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, ofdm.bits_per_ofdm_symbol)).astype("uint8")
    tx_iq = ofdm.generate_frame(bits)

    taps = Channel.random_multipath_taps(3, seed=1)
    channel = Channel(snr_db=40.0, multipath_taps=taps, cfo=0.03, cfo_fft_size=256, seed=0, backend="numpy")
    padded = np.concatenate([np.zeros(40, dtype="complex64"), tx_iq[0], np.zeros(40, dtype="complex64")])
    rx_iq = channel.process(padded)[0][None, :]

    result = ofdm.rx_process(rx_iq)
    np.testing.assert_array_equal(result["bits"], bits)


def test_cfo_pilot_based_is_also_compatible_with_schmidl_cox_sync():
    """The flip side: PilotBasedCFO has no dependency on the preamble
    shape at all, so it must ALSO work paired with SchmidlCoxSync (the
    default sync=), not just with ZadoffChuSync."""
    ofdm = _make_ofdm(sync="schmidl_cox", cfo="pilot_based", n_training_symbols=2)
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, ofdm.bits_per_ofdm_symbol)).astype("uint8")
    tx_iq = ofdm.generate_frame(bits)
    result = ofdm.rx_process(tx_iq)
    np.testing.assert_array_equal(result["bits"], bits)


def test_channel_estimator_mmse_is_a_genuine_drop_in_replacement_for_ls():
    """MMSEChannelEstimator must be swappable behind channel_estimator=
    exactly like LSChannelEstimator -- proven with a real
    generate_frame/rx_process round trip (identity channel)."""
    ofdm = _make_ofdm(channel_estimator="mmse")
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, ofdm.bits_per_ofdm_symbol)).astype("uint8")
    tx_iq = ofdm.generate_frame(bits)
    result = ofdm.rx_process(tx_iq)
    np.testing.assert_array_equal(result["bits"], bits)


def test_channel_estimator_mmse_under_real_multipath_and_awgn_channel():
    """Real-channel integration check, same scenario style as the
    existing SchmidlCox/LS full-256 test -- confirms MMSE's channel
    estimate is genuinely usable inside the full pipeline, not just in
    isolation. A BER threshold, not bit-exact equality, is the right bar
    here: checked directly (not assumed), the default noise_var=1e-3 /
    max_delay=cp_len assumption doesn't automatically beat plain LS+
    interpolation against this scenario's specific (sparse, 3-tap)
    channel realization -- a generic uniform-PDP model can be a real
    mismatch against an actual sparse delay profile, a known,
    documented characteristic of parametric channel estimators (see
    channel/mmse.py's module docstring), not a bug. The standalone
    tests in tests/test_channel_mmse.py already rigorously prove MMSE
    beats naive LS-interpolation when properly configured (matched
    noise_var, well-determined pilot count) -- this test's job is only
    to prove channel_estimator="mmse" is a genuine, working drop-in
    swap end to end, not to re-litigate that comparison here."""
    from spectracuda.sim import Channel

    ofdm = Ofdm(fft_size=256, n_pilot=6, n_data=200, cp_len=32, modem="qpsk",
                channel_estimator="mmse", n_training_symbols=2, backend="numpy")
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, ofdm.bits_per_ofdm_symbol)).astype("uint8")
    tx_iq = ofdm.generate_frame(bits)

    taps = Channel.random_multipath_taps(3, seed=0)
    channel = Channel(snr_db=25.0, multipath_taps=taps, cfo=0.1, cfo_fft_size=256, seed=0, backend="numpy")
    padded = np.concatenate([np.zeros(40, dtype="complex64"), tx_iq[0], np.zeros(40, dtype="complex64")])
    rx_iq = channel.process(padded)[0][None, :]

    result = ofdm.rx_process(rx_iq)
    ber = np.mean(result["bits"] != bits)
    assert ber < 0.05


_STABLE_RESULT_KEYS = {
    "frame_found", "start_index", "sync_metric", "rssi_db", "cfo_estimate",
    "channel_estimate", "header", "n_payload_symbols", "bits", "crc_valid", "evm",
}


def test_rx_process_returns_the_full_stable_key_set_on_success():
    """The actual gap this closes (docs/todo.md #1.1 #2): rx_process()
    used to return "whatever fields happened to be convenient to add
    during development" -- now every key is always present."""
    ofdm = _make_ofdm()
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, ofdm.bits_per_ofdm_symbol)).astype("uint8")
    tx_iq = ofdm.generate_frame(bits)
    result = ofdm.rx_process(tx_iq)
    assert set(result.keys()) == _STABLE_RESULT_KEYS
    assert result["frame_found"] is True
    np.testing.assert_array_equal(result["bits"], bits)


def test_rx_process_reports_low_evm_on_a_clean_channel():
    """EVM (previously entirely missing -- see docs/todo.md #1.1 #2)
    must be small on an identity channel: the equalized symbols should
    sit right on their own hard-decision points."""
    ofdm = _make_ofdm()
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, ofdm.bits_per_ofdm_symbol)).astype("uint8")
    tx_iq = ofdm.generate_frame(bits)
    result = ofdm.rx_process(tx_iq)
    assert result["evm"][0] < 0.05


def test_rx_process_reports_rssi_regardless_of_frame_content():
    ofdm = _make_ofdm()
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, ofdm.bits_per_ofdm_symbol)).astype("uint8")
    tx_iq = ofdm.generate_frame(bits)
    result = ofdm.rx_process(tx_iq)
    assert np.isfinite(result["rssi_db"][0])


def test_rx_process_reports_frame_not_found_on_pure_noise():
    """The sharpest of the three gaps (docs/todo.md #1.1 #3): sync
    always returns SOME start_index/metric, even for pure noise with no
    preamble in it at all -- rx_process() must turn a weak best-
    candidate score into a clean, well-defined frame_found=False result
    (with every other field None) rather than marching on to decode a
    header that isn't there."""
    ofdm = _make_ofdm()
    rng = np.random.default_rng(0)
    noise = (
        (rng.standard_normal(2000) + 1j * rng.standard_normal(2000)) / np.sqrt(2)
    ).astype("complex64")[None, :]

    result = ofdm.rx_process(noise)
    assert set(result.keys()) == _STABLE_RESULT_KEYS
    assert result["frame_found"] is False
    assert result["bits"] is None
    assert result["header"] is None
    assert result["crc_valid"] is None
    assert result["cfo_estimate"] is None
    assert result["channel_estimate"] is None
    assert result["n_payload_symbols"] is None
    assert result["evm"] is None
    # still-defined, best-effort fields even with no real frame present:
    assert result["start_index"] is not None
    assert result["sync_metric"] is not None
    assert np.isfinite(result["rssi_db"][0])


def test_sync_threshold_is_genuinely_wired_in_not_a_no_op():
    """Proven both directions, not just that a low-metric noise buffer
    gets rejected at the default: a permissive threshold=0.0 must make
    rx_process() at least ATTEMPT to decode the very same noise buffer,
    rather than cleanly returning frame_found=False before ever calling
    CFO/header decode. Since there's no real frame in pure noise, that
    attempt is expected to fail loudly downstream (a header-corruption
    ValueError from one of the existing sanity checks) -- exactly the
    pre-existing "gets lucky and an unrelated sanity check catches it"
    failure mode this whole feature exists to avoid at the default
    threshold. Confirms sync_threshold= genuinely gates whether decoding
    is attempted at all, not a cosmetic parameter."""
    rng = np.random.default_rng(3)
    noise = (
        (rng.standard_normal(2000) + 1j * rng.standard_normal(2000)) / np.sqrt(2)
    ).astype("complex64")[None, :]

    ofdm_default = _make_ofdm()
    result_default = ofdm_default.rx_process(noise)
    assert result_default["frame_found"] is False

    ofdm_permissive = _make_ofdm(sync_threshold=0.0)
    with pytest.raises(ValueError):
        ofdm_permissive.rx_process(noise)


def test_default_sync_threshold_matches_documented_calibration():
    assert Ofdm.DEFAULT_SYNC_THRESHOLD == pytest.approx(0.3)
