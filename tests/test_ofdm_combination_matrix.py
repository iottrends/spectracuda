"""Combination-matrix coverage for the Ofdm PHY chain: fft_size x modem x
FEC-scheme/code-rate x sync x cfo. Requested explicitly (not previously
tracked as its own systematic matrix in docs/todo.md) -- existing tests
elsewhere in this suite exercise these parameters individually or in
scattered specific combinations, but never as one deliberate cross-
product. Also relevant to (but NOT a resolution of, and NOT the same
regime as) docs/todo.md #1.11 / docs/mac.md's "Bugs found" #4: that open
question is specifically about LONG multi-symbol frames (~19 OFDM
symbols) degrading at small fft_size; every case here uses a SHORT
payload (a single FEC block, a handful of OFDM symbols at most), which
#1.11 itself already found to be reliable across all three fft_size
values -- consistent with, not contradicting, this matrix's 36/36 clean
result. This matrix broadens modem/rate coverage in the SHORT-frame
regime #1.11 already validated; it doesn't probe the long-frame
degradation itself.

Scope is the Ofdm PHY chain directly (`spectracuda.pipeline.Ofdm`), not
`spectracuda.mac`/`MacLink`: every axis requested (fft_size, modem,
code rate, FEC family, sync, cfo) is an `Ofdm(...)` constructor
parameter, not a MAC-layer concept -- MacLink wraps one Ofdm and adds
segmentation/ARQ on top, which would confound a per-combination PHY
correctness question with MAC retry/window behavior. If MAC-level
coverage of this same matrix is wanted, that's a separate, follow-on
task (MacLink's own capacity accounting already reuses
Packetizer.encoded_length(), so it should compose with any of these
Ofdm configs directly -- see session.py's module docstring).

## Combination space, listed out

Requested axes:
  - fft_size:  64, 128, 256
  - modem:     qpsk, qam16, qam64
  - code rate: 1/2, 2/3, 3/4
  - FEC family: viterbi (conv_v27), ldpc
  - sync:      schmidl_cox, zc (ZadoffChuSync)
  - cfo:       schmidl_cox (SchmidlCoxCFO), pilot_based (PilotBasedCFO)

Rate x FEC-family resolves to four concrete schemes (not a full 3x2
cross product): `conv_v27` is liquid-dsp's one rate-1/2 K=7 code with no
rate-2/3 or rate-3/4 variant implemented in this codebase (see
spectracuda.fec's module docstring) -- rate only genuinely varies for
LDPC, which has real rate 1/2/2/3/3/4/5/6 variants at three codeword
lengths each. rate=5/6 wasn't requested, so it's left out; codeword
length is fixed at 648 (the smallest/fastest of the three) for all LDPC
cases here, since generate_frame()'s automatic partial-symbol padding
(docs/todo.md #1.10) means codeword length no longer needs to divide
n_data*bits_per_symbol -- any fft_size/modem pairs with any codeword
length now:

    FEC_CHOICES = {
        "conv_v27":     rate 1/2 (viterbi),
        "ldpc_648_r12": rate 1/2 (ldpc),
        "ldpc_648_r23": rate 2/3 (ldpc),
        "ldpc_648_r34": rate 3/4 (ldpc),
    }

sync x cfo is NOT a free 2x2 cross product either -- SchmidlCoxCFO
depends on the preamble's repeated-halves shape (see
cfo/schmidl_cox.py's docstring) and is confirmed incompatible with
ZadoffChuSync's contiguous (non-repeated-halves) preamble (verified
directly: pairing them corrupts the whole frame -- see
cfo/pilot_based.py's module docstring for the original finding, and
`test_zc_sync_with_schmidl_cox_cfo_is_a_known_incompatible_pairing`
below for a standing regression proof). The three VALID pairs:

    (schmidl_cox, schmidl_cox)  -- the original, default pairing
    (schmidl_cox, pilot_based)  -- pilot_based has no preamble-shape
                                    dependency, works with either sync
    (zc, pilot_based)           -- ZC's only working cfo pairing

Full cross product if every axis were independent: 3 (fft) x 3 (modem)
x 4 (fec/rate) x 3 (valid sync/cfo pairs) = 108. Structured here as TWO
matrices instead of one 108-cell grid, to isolate what each axis
combination is actually testing (and keep the suite's runtime/failure-
localization sane) -- both are still exhaustive over their own stated
axes, nothing within them is spot-checked/sampled:

  1. fft_size x modem x fec/rate (3x3x4 = 36 cases), at the DEFAULT
     sync=schmidl_cox/cfo=schmidl_cox pairing, identity (lossless)
     channel -- this is the literal "fft64/128/256, qpsk/16qam/64qam,
     rate 1/2/2/3/3/4, viterbi+ldpc" grid.
  2. sync x cfo (3 valid pairs) x fft_size (3x3 = 9 cases), at a FIXED
     representative modem=qpsk/fec=conv_v27 -- isolates sync/cfo
     correctness from the modem/rate axis (already covered by matrix 1
     at the default sync/cfo pairing, so multiplying every sync/cfo pair
     by every modem/rate cell too would mostly re-test the same modem/
     rate correctness already proven, not new interaction risk).

Plus: one standing negative-case regression test for the known-invalid
(zc, schmidl_cox) pairing, and a small real-channel (multipath+AWGN+CFO)
stress subset (one modem/fec combination per fft_size, chosen to spread
coverage across different cells of matrix 1 rather than repeating one
combination three times) -- not the full 36-cell matrix under a real
channel, which would be substantial runtime for proportionally little
extra signal beyond what's already established elsewhere in this suite
about FEC/channel-estimator/equalizer behavior under noise.

crc= is deliberately left at "none" throughout this file -- not one of
the requested axes, and adding it would multiply every cell without
being asked for; CRC's own correctness is already covered exhaustively
in tests/test_crc.py and tests/test_ofdm_class.py's crc-specific tests.
"""
import itertools

import numpy as np
import pytest

from spectracuda.pipeline import Ofdm
from spectracuda.sim import Channel

# --- axis definitions --------------------------------------------------

FFT_CONFIGS = {
    64: dict(n_pilot=6, n_data=40, cp_len=16),
    128: dict(n_pilot=7, n_data=90, cp_len=16),
    256: dict(n_pilot=6, n_data=200, cp_len=32),
}
MODEMS = ["qpsk", "qam16", "qam64"]

# name -> (fec scheme string, a raw bit count that's an exact multiple of
# that scheme's own block size k -- ldpc needs an EXACT multiple (no crc
# here to add a key first); conv_v27 is a streaming code and accepts any
# k, so 64 is just a convenient, arbitrary choice).
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


# --- matrix 1: fft_size x modem x fec/rate (36 cases), identity channel --

_MATRIX_1_CASES = [
    (fft_size, modem, fec_name)
    for fft_size, modem, fec_name in itertools.product(FFT_CONFIGS, MODEMS, FEC_CHOICES)
]


@pytest.mark.parametrize(
    "fft_size,modem,fec_name",
    _MATRIX_1_CASES,
    ids=[f"fft{f}-{m}-{n}" for f, m, n in _MATRIX_1_CASES],
)
def test_fft_modem_rate_matrix_identity_channel(fft_size, modem, fec_name):
    fec_scheme, k = FEC_CHOICES[fec_name]
    ofdm = _make_ofdm(fft_size, modem, fec_scheme)
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, k)).astype("uint8")
    tx_iq = ofdm.generate_frame(bits)
    result = ofdm.rx_process(tx_iq)
    assert result["frame_found"] is True
    assert result["header"]["mod_scheme"] == modem
    assert result["header"]["fec0"] == fec_scheme
    np.testing.assert_array_equal(result["bits"], bits)


# --- matrix 2: sync x cfo (valid pairs) x fft_size (9 cases) --------------

_MATRIX_2_CASES = [
    (fft_size, sync, cfo)
    for fft_size, (sync, cfo) in itertools.product(FFT_CONFIGS, VALID_SYNC_CFO_PAIRS)
]


@pytest.mark.parametrize(
    "fft_size,sync,cfo",
    _MATRIX_2_CASES,
    ids=[f"fft{f}-{s}-{c}" for f, s, c in _MATRIX_2_CASES],
)
def test_sync_cfo_pairing_matrix_identity_channel(fft_size, sync, cfo):
    """Fixed modem=qpsk/fec=conv_v27 -- isolates sync/cfo correctness
    from the modem/rate axis (already covered by matrix 1's default
    sync/cfo pairing)."""
    ofdm = _make_ofdm(fft_size, "qpsk", "conv_v27", sync=sync, cfo=cfo, n_training_symbols=2)
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, 64)).astype("uint8")
    tx_iq = ofdm.generate_frame(bits)
    result = ofdm.rx_process(tx_iq)
    assert result["frame_found"] is True
    np.testing.assert_array_equal(result["bits"], bits)


def test_zc_sync_with_schmidl_cox_cfo_is_a_known_incompatible_pairing():
    """Standing regression proof for why (zc, schmidl_cox) is excluded
    from the valid-pairs list above: SchmidlCoxCFO's estimate depends on
    the preamble's repeated-halves shape, which ZadoffChuSync's
    contiguous preamble doesn't have -- confirmed here to still corrupt
    the frame (a decode exception, not a clean result), not silently
    "just work" if ever re-tried."""
    ofdm = _make_ofdm(64, "qpsk", "conv_v27", sync="zc", cfo="schmidl_cox")
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, 64)).astype("uint8")
    tx_iq = ofdm.generate_frame(bits)
    with pytest.raises(ValueError):
        ofdm.rx_process(tx_iq)


# --- real-channel stress subset: one modem/fec combo per fft_size ---------

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
def test_fft_modem_rate_matrix_under_real_multipath_awgn_cfo(fft_size, modem, fec_name):
    """A representative subset of matrix 1 (one cell per fft_size, chosen
    to spread coverage across different modem/fec cells rather than
    repeating one combination three times), now under a real multipath +
    AWGN(20dB) + CFO(0.05) channel -- not the full 36-cell grid, which
    would add substantial runtime for proportionally little additional
    signal beyond what's already established about FEC/channel-estimator/
    equalizer behavior under noise elsewhere in this suite."""
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

    result = ofdm.rx_process(rx_iq)
    assert result["frame_found"] is True
    np.testing.assert_array_equal(result["bits"], bits)
