"""Combination-matrix coverage for Ofdm.rx_streaming() -- the SAME axes
tests/test_ofdm_combination_matrix.py already exhaustively covers for
rx_process() (fft_size x modem x FEC-scheme/code-rate, and sync x cfo
valid pairings), but fed through chunked rx_streaming() instead of one
whole-buffer rx_process() call. A real, previously-open gap: every test
in tests/test_ofdm_streaming.py used a single fixed Ofdm config
throughout (fft_size=256/sync=schmidl_cox/cfo=schmidl_cox/etc.) -- zero
variation across sync strategy, CFO strategy, fft_size, cp_len,
channel_estimator, or equalizer had actually been exercised through the
streaming receiver specifically, even though the non-streaming rx_process()
path already had this matrix. rx_streaming() reuses rx_process()'s own
_decode_header_from_sync()/_decode_payload_from_header() helpers
internally, so most PER-SCHEME algorithmic correctness is already proven
by the existing matrix -- but the accumulation/state-machine logic itself
(chunk-boundary handling, the buffer-growth CFO-recorrection fix, the
frame_start/pos index bookkeeping) is NOT exercised by that matrix at
all, and IS what could plausibly behave differently per fft_size/sync/cfo
combination (e.g. different slot_len changes exactly where symbol
boundaries fall relative to arbitrary chunk boundaries). This file closes
that gap directly rather than assuming the non-streaming matrix implies
the streaming one is also fine.

Axis definitions (FFT_CONFIGS/MODEMS/FEC_CHOICES/VALID_SYNC_CFO_PAIRS)
are duplicated verbatim from test_ofdm_combination_matrix.py rather than
cross-imported, matching this project's existing convention of every test
file being self-contained -- keep them in sync if either file's axes
change.
"""
import itertools

import numpy as np
import pytest

from spectracuda.pipeline import Ofdm
from spectracuda.sim import Channel

FFT_CONFIGS = {
    64: dict(n_pilot=6, n_data=40, cp_len=16),
    128: dict(n_pilot=7, n_data=90, cp_len=16),
    256: dict(n_pilot=6, n_data=200, cp_len=32),
}
MODEMS = ["qpsk", "qam16", "qam64"]

FEC_CHOICES = {
    "conv_v27_r12": ("conv_v27", 64),
    "ldpc_648_r12": ("ldpc_648_r12", 324),
    "ldpc_648_r23": ("ldpc_648_r23", 432),
    "ldpc_648_r34": ("ldpc_648_r34", 486),
}

VALID_SYNC_CFO_PAIRS = [
    ("schmidl_cox", "schmidl_cox"),
    ("schmidl_cox", "pilot_based"),
    ("zc", "pilot_based"),
]

# Deliberately not a divisor of fft_size, cp_len, or slot_len for ANY
# config in FFT_CONFIGS -- proves no hidden chunk-alignment assumption
# across the whole matrix, not just the one config test_ofdm_streaming.py
# already checked this for.
CHUNK_SIZE = 97


def _make_ofdm(fft_size, modem, fec, sync="schmidl_cox", cfo="schmidl_cox", n_training_symbols=1):
    cfg = FFT_CONFIGS[fft_size]
    return Ofdm(
        fft_size=fft_size,
        modem=modem,
        fec=fec,
        sync=sync,
        cfo=cfo,
        n_training_symbols=n_training_symbols,
        backend="numpy",
        **cfg,
    )


def _stream_frame(ofdm, tx_iq, chunk_size=CHUNK_SIZE):
    """tx_iq: (1, N) from generate_frame(). Feeds it through
    ofdm.rx_streaming() in chunk_size pieces, returns the first non-None
    result (or None if the whole frame never completed)."""
    ofdm.reset_stream()
    samples = tx_iq[0]
    for i in range(0, samples.shape[-1], chunk_size):
        result = ofdm.rx_streaming(samples[i : i + chunk_size])
        if result is not None:
            return result
    return None


# --- matrix 1: fft_size x modem x fec/rate (36 cases), streamed ----------

_MATRIX_1_CASES = [
    (fft_size, modem, fec_name)
    for fft_size, modem, fec_name in itertools.product(FFT_CONFIGS, MODEMS, FEC_CHOICES)
]


@pytest.mark.parametrize(
    "fft_size,modem,fec_name",
    _MATRIX_1_CASES,
    ids=[f"fft{f}-{m}-{n}" for f, m, n in _MATRIX_1_CASES],
)
def test_fft_modem_rate_matrix_streaming(fft_size, modem, fec_name):
    fec_scheme, k = FEC_CHOICES[fec_name]
    ofdm = _make_ofdm(fft_size, modem, fec_scheme)
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, k)).astype("uint8")
    tx_iq = ofdm.generate_frame(bits)

    result = _stream_frame(ofdm, tx_iq)
    assert result is not None
    assert result["frame_found"] is True
    assert result["header"]["mod_scheme"] == modem
    assert result["header"]["fec0"] == fec_scheme
    np.testing.assert_array_equal(result["bits"], bits)


# --- matrix 2: sync x cfo (valid pairs) x fft_size (9 cases), streamed ----

_MATRIX_2_CASES = [
    (fft_size, sync, cfo)
    for fft_size, (sync, cfo) in itertools.product(FFT_CONFIGS, VALID_SYNC_CFO_PAIRS)
]


@pytest.mark.parametrize(
    "fft_size,sync,cfo",
    _MATRIX_2_CASES,
    ids=[f"fft{f}-{s}-{c}" for f, s, c in _MATRIX_2_CASES],
)
def test_sync_cfo_pairing_matrix_streaming(fft_size, sync, cfo):
    ofdm = _make_ofdm(fft_size, "qpsk", "conv_v27", sync=sync, cfo=cfo, n_training_symbols=2)
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, 64)).astype("uint8")
    tx_iq = ofdm.generate_frame(bits)

    result = _stream_frame(ofdm, tx_iq)
    assert result is not None
    assert result["frame_found"] is True
    np.testing.assert_array_equal(result["bits"], bits)


def test_zc_sync_with_schmidl_cox_cfo_never_completes_via_streaming():
    """Streaming's mirror of test_ofdm_combination_matrix.py's own
    negative-case test -- but a genuinely different code path: rx_process()
    raises ValueError on this known-incompatible pairing; rx_streaming()
    is designed to NEVER raise from a bad detection (see its own
    docstring) -- it should just keep discarding failed attempts and
    return None forever, not crash. Confirmed directly, not assumed from
    rx_process()'s already-proven behavior."""
    ofdm = _make_ofdm(64, "qpsk", "conv_v27", sync="zc", cfo="schmidl_cox")
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, 64)).astype("uint8")
    tx_iq = ofdm.generate_frame(bits)

    result = _stream_frame(ofdm, tx_iq)
    assert result is None  # never completes -- must not raise, must not fabricate a result


# --- real-channel stress subset, streamed ---------------------------------

_REAL_CHANNEL_CASES = [
    (64, "qpsk", "conv_v27_r12"),
    (128, "qam16", "ldpc_648_r23"),
    (256, "qam64", "ldpc_648_r34"),
]


@pytest.mark.parametrize(
    "fft_size,modem,fec_name",
    _REAL_CHANNEL_CASES,
    ids=[f"fft{f}-{m}-{n}" for f, m, n in _REAL_CHANNEL_CASES],
)
def test_fft_modem_rate_matrix_streaming_under_real_multipath_awgn_cfo(fft_size, modem, fec_name):
    fec_scheme, k = FEC_CHOICES[fec_name]
    ofdm = _make_ofdm(fft_size, modem, fec_scheme, n_training_symbols=2)
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, k)).astype("uint8")
    tx_iq = ofdm.generate_frame(bits)

    taps = Channel.random_multipath_taps(3, seed=0)
    channel = Channel(snr_db=20.0, multipath_taps=taps, cfo=0.05, cfo_fft_size=fft_size, seed=0, backend="numpy")
    pad = fft_size // 2
    padded = np.concatenate([np.zeros(pad, dtype="complex64"), tx_iq[0], np.zeros(pad, dtype="complex64")])
    rx_iq = channel.process(padded)[0][None, :]

    result = _stream_frame(ofdm, rx_iq)
    assert result is not None
    assert result["frame_found"] is True
    np.testing.assert_array_equal(result["bits"], bits)
