"""MAC-level combination matrix closing a real gap in tests/test_ofdm_
combination_matrix.py: that file sweeps fft_size x modem x fec/rate for
conv_v27-alone and the 3 LDPC rates, but never fec="rs_m8" -- in
particular, never the two-stage fec="rs_m8" (inner) + fec1="conv_v27"
(outer) pairing this project's own benchmark/optimization work
(examples/benchmark_x86_stages_v2.py) was built around throughout. This
file exists specifically to close that: is rs_m8+conv_v27 actually
correct across fft_size, modem, training-symbol count, every valid
sync/cfo pairing, and every channel_estimator x equalizer combination --
not just fft_size=256 (the only config it had ANY automated coverage at
before this file).

Axes (all independently swept, free cross product except sync/cfo):
  fft_size:              128, 256
  modem:                 qpsk, qam16, qam64
  n_training_symbols:    1, 2
  sync/cfo:              only the 3 VALID pairings (schmidl_cox/
                         schmidl_cox, schmidl_cox/pilot_based, zc/
                         pilot_based) -- NOT a free 2x2. (zc,
                         schmidl_cox) is a known-incompatible pairing
                         (SchmidlCoxCFO depends on the preamble's
                         repeated-halves shape ZadoffChuSync doesn't
                         have), already covered as a standing negative-
                         case regression in test_ofdm_combination_
                         matrix.py -- not repeated here.
  channel_estimator x
  equalizer:             free 2x2 (ls/mmse x zf/mmse) -- no documented
                         incompatibility between these two axes, and
                         MMSEChannelEstimator's own module docstring
                         explicitly flags the mmse+mmse pairing as "an
                         open combination worth a dedicated test, not
                         yet done" -- this closes that too, not just the
                         rs_m8 gap.

A THIRD constraint, found the hard way (not initially known when this
file was first written -- an actual test run caught it, 96/288 failures,
every single one exactly cfo="pilot_based" AND n_training_symbols=1,
zero exceptions in either direction): PilotBasedCFO structurally
requires n_repeats=n_training_symbols >= 2 (a phase-SLOPE estimate needs
at least two known-symbol occurrences to measure a slope between -- see
cfo/pilot_based.py's own docstring/code, which raises a clean
ValueError for n_repeats<2, not a silent wrong answer). So
n_training_symbols=1 is only paired with cfo="schmidl_cox" below, NOT
cfo="pilot_based" -- exactly the same shape of constraint as the sync/
cfo pairing restriction above, just discovered one level later. See
test_pilot_based_cfo_with_one_training_symbol_is_a_known_invalid_
combination below for a standing regression proof of this (mirroring
test_ofdm_combination_matrix.py's own zc+schmidl_cox negative case).

Fixed, not swept: fec="rs_m8" (inner) + fec1="conv_v27" (outer) -- this
file's whole reason to exist, so it's the one constant, not a variable.
crc="crc16" (Mac(ofdm_kwargs=...) requires crc != "none").

2 x 3 x 2 x 3 x 2 x 2 = 144 combinations, x2 (batch, streaming
transport) = 288 test cases total.

Test level: Mac (send_iq()/receive_iq()), NOT raw Ofdm.generate_frame()/
rx_process() -- per explicit request: Mac's own PDU header/segmentation/
reassembly is real behavior a pure-Ofdm test wouldn't exercise at all.
Both Mac instances are manually .bound = True -- no real bind handshake
(out of scope for a PHY/reassembly correctness matrix; bind itself is
covered elsewhere, e.g. tests/test_mac_bind.py).

Streaming path: Mac has NO built-in streaming method (checked directly
in spectracuda/mac/mac.py, not assumed) -- reproduces the same manual
pattern examples/drone_air_unit.py already uses for real hardware: chunk
the IQ, feed Ofdm.rx_streaming(chunk), route each completed decode
through Mac.receive() (the SAME UM reassembly the batch path uses
internally via receive_iq()). STREAM_CHUNK_SIZE is deliberately NOT
drone_air_unit.py's CHUNK_SIZE=64 -- a real, measured finding from this
same project's work: rx_streaming()'s per-call cost is ~40-150us and
nearly fixed regardless of chunk size, so a tiny chunk size would make
this file needlessly slow (288 cases) for zero correctness benefit --
this is a pure in-memory test, not real hardware IQ pacing, so there's
no reason to mimic a small real-world chunk size here.

Runtime note: fec="rs_m8"/fec1="conv_v27" still run the REAL, pure-
Python/NumPy ConvolutionalCode/ReedSolomonCode in spectracuda/fec/ (the
libcorrect/Numba-accelerated paths from this session's optimization work
are examples/-only, not yet promoted into the library -- see that work's
own notes). SDU_BITS is kept small (400 bits, not the 4000-24000 used in
the throughput benchmarks) specifically to keep this correctness matrix's
total runtime reasonable despite that -- this file is about correctness
across configurations, not throughput (that's what
examples/benchmark_x86_stages_v2.py is for).
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from spectracuda.mac import Mac

FFT_CONFIGS = {
    128: dict(n_pilot=8, n_data=112, cp_len=32),
    256: dict(n_pilot=8, n_data=216, cp_len=32),
}
MODEMS = ["qpsk", "qam16", "qam64"]
N_TRAINING_SYMBOLS = [1, 2]
VALID_SYNC_CFO_PAIRS = [
    ("schmidl_cox", "schmidl_cox"),
    ("schmidl_cox", "pilot_based"),
    ("zc", "pilot_based"),
]
CHANNEL_ESTIMATORS = ["ls", "mmse"]
EQUALIZERS = ["zf", "mmse"]

SDU_BITS = 400  # multiple of 8 -- rs_m8 requires the CRC-appended, pre-FEC
                # bit count to be byte-aligned (see fec/fec.py's own check);
                # HEADER_LEN_BITS(32) + SDU_BITS + crc16's 16 key bits must
                # land on a byte boundary, so SDU_BITS itself must be a
                # multiple of 8 (32 and 16 already are)
STREAM_CHUNK_SIZE = 1024  # see module docstring -- NOT a real-hardware chunk size


def _make_mac_pair(fft_size, modem, n_train, sync, cfo, chanest, eq):
    cfg = FFT_CONFIGS[fft_size]
    ofdm_kwargs = dict(
        fft_size=fft_size, n_pilot=cfg["n_pilot"], n_data=cfg["n_data"], cp_len=cfg["cp_len"],
        modem=modem, fec="rs_m8", fec1="conv_v27", crc="crc16",
        sync=sync, cfo=cfo, n_training_symbols=n_train,
        channel_estimator=chanest, equalizer=eq, backend="numpy",
    )
    tx_mac = Mac(mode="um", ofdm_kwargs=ofdm_kwargs)
    rx_mac = Mac(mode="um", ofdm_kwargs=ofdm_kwargs)
    tx_mac.bound = True
    rx_mac.bound = True
    assert SDU_BITS <= tx_mac.max_segment_bits, (
        f"SDU_BITS={SDU_BITS} exceeds max_segment_bits={tx_mac.max_segment_bits} "
        f"for this config -- test assumption (single-PDU SDU) violated"
    )
    return tx_mac, rx_mac


def _stream_decode_all(rx_mac, iq_frame, chunk_size):
    """One generate_frame()-shaped IQ frame -> every SDU Mac.receive()
    delivers from it, driven through Ofdm.rx_streaming() chunk by chunk
    instead of one batch rx_process() call. Mirrors examples/drone_air_
    unit.py's _recv_one_chunk_and_stream_decode()+_handle_decoded_pdu()
    dispatch, simplified to DATA-only (no bind/link-quality pdu types
    relevant here)."""
    samples = np.asarray(iq_frame)[0]  # (1, N) -> (N,)
    delivered = []
    for i in range(0, len(samples), chunk_size):
        chunk = samples[i : i + chunk_size]
        result = rx_mac.ofdm.rx_streaming(chunk)
        if result is None:
            continue
        crc_valid = result["crc_valid"]
        if crc_valid is not None and not bool(np.asarray(crc_valid)[0]):
            continue  # matches receive_iq()'s own "don't hand a corrupted decode to reassembly" behavior
        bits = np.asarray(result["bits"])[0].astype("uint8")
        delivered.extend(rx_mac.receive(bits))
    return delivered


_CASES = [
    (fft_size, modem, n_train, sync, cfo, chanest, eq)
    for fft_size, modem, n_train, (sync, cfo), chanest, eq in itertools.product(
        FFT_CONFIGS, MODEMS, N_TRAINING_SYMBOLS, VALID_SYNC_CFO_PAIRS, CHANNEL_ESTIMATORS, EQUALIZERS
    )
    if not (cfo == "pilot_based" and n_train < 2)  # see module docstring's "THIRD constraint"
]
_IDS = [
    f"fft{f}-{m}-nt{nt}-{s}+{c}-{ce}+{eq}"
    for f, m, nt, s, c, ce, eq in _CASES
]


@pytest.mark.parametrize("fft_size,modem,n_train,sync,cfo,chanest,eq", _CASES, ids=_IDS)
def test_batch_rs_viterbi(fft_size, modem, n_train, sync, cfo, chanest, eq):
    """Mac.send_iq() -> Mac.receive_iq() -- the batch transport, one
    rx_process() call per frame under the hood."""
    tx_mac, rx_mac = _make_mac_pair(fft_size, modem, n_train, sync, cfo, chanest, eq)
    rng = np.random.default_rng(0)
    sdu = rng.integers(0, 2, size=SDU_BITS).astype("uint8")

    frames = tx_mac.send_iq(sdu)
    assert len(frames) == 1, f"expected 1 PDU/frame for this SDU size, got {len(frames)}"

    delivered = rx_mac.receive_iq(frames[0])
    assert len(delivered) == 1, f"expected exactly 1 SDU delivered, got {len(delivered)}"
    np.testing.assert_array_equal(delivered[0], sdu)


@pytest.mark.parametrize("fft_size,modem,n_train,sync,cfo,chanest,eq", _CASES, ids=_IDS)
def test_streaming_rs_viterbi(fft_size, modem, n_train, sync, cfo, chanest, eq):
    """Mac.send_iq() -> Ofdm.rx_streaming() chunk-by-chunk -> Mac.receive()
    -- the streaming transport, same SDU/config as the batch case above,
    proving both transports agree, not just that one of them works."""
    tx_mac, rx_mac = _make_mac_pair(fft_size, modem, n_train, sync, cfo, chanest, eq)
    rx_mac.ofdm.reset_stream()
    rng = np.random.default_rng(0)
    sdu = rng.integers(0, 2, size=SDU_BITS).astype("uint8")

    frames = tx_mac.send_iq(sdu)
    assert len(frames) == 1, f"expected 1 PDU/frame for this SDU size, got {len(frames)}"

    delivered = _stream_decode_all(rx_mac, frames[0], STREAM_CHUNK_SIZE)
    assert len(delivered) == 1, f"expected exactly 1 SDU delivered via streaming, got {len(delivered)}"
    np.testing.assert_array_equal(delivered[0], sdu)


def test_pilot_based_cfo_with_one_training_symbol_is_a_known_invalid_combination():
    """Standing regression proof of the constraint documented in this
    file's module docstring ("A THIRD constraint...") -- found by an
    actual full matrix run failing 96/288 cases before it was known,
    not assumed in advance. Mirrors test_ofdm_combination_matrix.py's
    own test_zc_sync_with_schmidl_cox_cfo_is_a_known_incompatible_pairing:
    proves the failure mode is exactly what's expected at THIS
    abstraction level, not an uncontrolled crash, and stays excluded
    from _CASES above for a documented reason rather than an
    unexplained gap in the matrix.

    NOTE this is deliberately NOT pytest.raises(ValueError) -- Ofdm.
    rx_process() itself does raise the ValueError cfo/pilot_based.py
    documents, but Mac._rx_one_frame() (which receive_iq() calls)
    deliberately catches (ValueError, NotImplementedError) and returns
    None/empty instead -- a bad frame is just non-delivery at this
    level, not a raised exception. A first version of this test asserted
    pytest.raises() here and failed with "DID NOT RAISE" -- a real
    reminder that Mac's contract at ITS level is silent non-delivery,
    confirmed by reading _rx_one_frame() directly rather than assumed
    from the lower-level cfo/pilot_based.py behavior."""
    tx_mac, rx_mac = _make_mac_pair(256, "qpsk", 1, "schmidl_cox", "pilot_based", "ls", "zf")
    sdu = np.zeros(SDU_BITS, dtype="uint8")
    frames = tx_mac.send_iq(sdu)
    delivered = rx_mac.receive_iq(frames[0])
    assert delivered == [], f"expected no SDU delivered (Mac swallows the ValueError), got {delivered}"
