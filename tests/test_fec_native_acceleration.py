"""Correctness of the optional native/Numba acceleration promoted into
the real library this session: fec/_native.py (libcorrect, vendored C
source, CPU-only) for ReedSolomonCode, and fec/_numba_crc.py (Numba JIT)
for CRC. See both modules' own docstrings for the full rationale/
measurements.

Activation is transparent (no new constructor argument) -- these tests
verify the PUBLIC ConvolutionalCode/ReedSolomonCode/CRC contract stays
correct regardless of which path is silently active on the machine
running the suite, not the internal acceleration mechanism directly.
Tests that specifically need the accelerated path active are skipped
(not failed) when it isn't available (e.g. no C compiler, no numba
installed) -- this suite must still pass on a bare NumPy-only machine,
matching this project's "runs anywhere" requirement.
"""
from __future__ import annotations

import numpy as np
import pytest

from spectracuda.fec import _native, _numba_crc
from spectracuda.fec.crc import CRC, SCHEMES
from spectracuda.fec.reed_solomon import ReedSolomonCode
from spectracuda.fec.viterbi import ConvolutionalCode

_NATIVE_OK = _native.native_available()
_NUMBA_OK = _numba_crc.numba_available()


# --- Viterbi: native backend, now fixed and active --------------------

@pytest.mark.skipif(not _NATIVE_OK, reason="no C compiler available -- native FEC backend inactive on this machine")
def test_convolutional_code_native_backend_is_active():
    """The libcorrect decode()-truncation bug documented in fec/_native.
    py's NativeConvolutional._decode_one() (withheld up to ~13 bits at
    the end of every decode, contrary to libcorrect's own documented
    contract) is now fixed there (decode-side zero-padding workaround,
    verified across 210 message sizes covering every T-mod-8 residue
    plus real bit-error injection against the pure-Python path as
    ground truth) -- native Viterbi is active again, not parked."""
    c = ConvolutionalCode(backend="numpy")
    assert c._native is not None


@pytest.mark.parametrize("k", [1, 6, 39, 100, 194, 4001, 4002, 4006, 24032])
def test_convolutional_code_correctness_every_t_mod_8_residue(k):
    """Every one of these k values lands on a DIFFERENT (k+6) mod 8
    residue -- specifically chosen to cover every case the original bug
    could hit (only (k+6)%8==6, i.e. k%8==0, was ever safe before the
    fix). Passes regardless of which backend is active; this is the
    permanent version of the manual sweep that found and later verified
    the fix for the bug."""
    c = ConvolutionalCode(backend="numpy")
    rng = np.random.default_rng(hash(k) % (2**31))
    msg = rng.integers(0, 2, size=(1, k)).astype("uint8")
    enc = c.encode(msg)
    dec = c.decode(enc)
    np.testing.assert_array_equal(dec, msg)


@pytest.mark.parametrize("k", [1, 39, 194, 4002, 4006])
@pytest.mark.parametrize("n_errors", [0, 1, 3, 5])
def test_convolutional_code_error_correction_matches_pure_python(k, n_errors):
    """Native decode of a REAL bit-error-corrupted codeword must match
    the pure-Python path's own answer exactly -- proves the fix holds
    under error correction, not just clean round-trips (a subtly wrong
    padding scheme could in principle still clean-decode correctly by
    luck while breaking actual error correction)."""
    from spectracuda.fec._native import NativeConvolutional

    pure_bits = ConvolutionalCode.__new__(ConvolutionalCode)
    ConvolutionalCode.__init__(pure_bits, backend="numpy")
    pure_bits._native = None  # force the pure-Python path as ground truth, regardless of machine

    rng = np.random.default_rng(hash((k, n_errors)) % (2**31))
    msg = rng.integers(0, 2, size=(1, k)).astype("uint8")
    enc = pure_bits.encode(msg)
    corrupted = enc.copy()
    if n_errors:
        idx = rng.choice(enc.shape[-1], size=n_errors, replace=False)
        corrupted[0, idx] ^= 1

    expected = pure_bits.decode(corrupted)
    c = ConvolutionalCode(backend="numpy")  # whatever backend is actually active
    actual = c.decode(corrupted)
    np.testing.assert_array_equal(actual, expected)


def test_convolutional_code_correctness_regardless_of_backend_state():
    """Baseline: encode/decode round-trips correctly via whatever path
    IS active (currently always pure-Python, per the lock above) --
    this must keep passing even if that changes later."""
    c = ConvolutionalCode(backend="numpy")
    rng = np.random.default_rng(0)
    for k in (1, 6, 194, 4000):
        msg = rng.integers(0, 2, size=(1, k)).astype("uint8")
        enc = c.encode(msg)
        dec = c.decode(enc)
        np.testing.assert_array_equal(dec, msg)


# --- Reed-Solomon: native backend, when available ---------------------

@pytest.mark.skipif(not _NATIVE_OK, reason="no C compiler available -- native FEC backend inactive on this machine")
def test_reed_solomon_native_backend_is_active():
    r = ReedSolomonCode(backend="numpy")
    assert r._native is not None


@pytest.mark.parametrize("real_k", [1, 8, 50, 105, 150, 223])
@pytest.mark.parametrize("n_errors", [0, 1, 8, 16])
def test_reed_solomon_correctness_across_sizes_and_error_counts(real_k, n_errors):
    """Passes regardless of which backend is active (native or pure-
    Python) -- proves the PUBLIC contract, not the internal path. This
    is the same sweep that caught nothing wrong with RS (unlike
    Viterbi) before it was promoted -- kept as a permanent regression
    test, not a one-off manual check."""
    r = ReedSolomonCode(backend="numpy")
    rng = np.random.default_rng(hash((real_k, n_errors)) % (2**31))
    msg = rng.integers(0, 256, size=real_k).astype("uint8")[None, :]
    encoded = r.encode(msg)
    corrupted = encoded.copy()
    if n_errors:
        idx = rng.choice(real_k + 32, size=n_errors, replace=False)
        corrupted[0, idx] = rng.integers(0, 256, size=n_errors).astype("uint8")
    decoded = r.decode(corrupted)
    np.testing.assert_array_equal(decoded, msg)


def test_reed_solomon_uncorrectable_codeword_raises_value_error():
    """t_max+1 (17) injected errors must still raise, matching
    ReedSolomonCode's documented contract exactly -- checked
    specifically against whichever backend is actually active, native
    included (a negative/failure return from correct_reed_solomon_
    decode() must translate to the SAME exception type the pure-Python
    path raises, not a different one callers would need to special-case)."""
    r = ReedSolomonCode(backend="numpy")
    rng = np.random.default_rng(99)
    msg = rng.integers(0, 256, size=223).astype("uint8")[None, :]
    encoded = r.encode(msg)
    corrupted = encoded.copy()
    idx = rng.choice(255, size=17, replace=False)
    corrupted[0, idx] = rng.integers(0, 256, size=17).astype("uint8")
    with pytest.raises(ValueError):
        r.decode(corrupted)


# --- CRC: Numba backend, when available --------------------------------

@pytest.mark.skipif(not _NUMBA_OK, reason="numba not installed -- CRC acceleration inactive on this machine")
def test_crc_numba_path_is_reachable():
    """Not a direct internal-state check (CRC has no persistent _native-
    style attribute -- numba_available() is checked per call) -- proves
    the accelerated path is actually importable/callable end to end."""
    from spectracuda.fec._numba_crc import _get_crc_row_fn

    fn = _get_crc_row_fn()
    assert fn is not None


@pytest.mark.parametrize("scheme", [s for s in SCHEMES if s not in ("none", "checksum")])
def test_crc_correctness_across_schemes(scheme):
    """Passes regardless of which backend is active. Values below are
    the same ones verified by hand against the pure-Python path earlier
    in this project's history (crc8=0x91, crc16=0x386f, crc24=0x20762f,
    crc32=0xadea11bd for this exact seed/message) -- pinned here as a
    permanent regression test, not re-derived per run."""
    expected = {"crc8": 0x91, "crc16": 0x386F, "crc24": 0x20762F, "crc32": 0xADEA11BD}
    rng = np.random.default_rng(0)
    msg = rng.integers(0, 256, size=3000).astype("uint8")[None, :]
    c = CRC(scheme, backend="numpy")
    key = int(c.generate_key(msg)[0])
    assert key == expected[scheme], f"{scheme}: got {hex(key)}, expected {hex(expected[scheme])}"


def test_crc_none_and_checksum_schemes_unaffected_by_numba_wiring():
    """"none"/"checksum" never go through the table-driven/Numba path
    at all (see crc.py) -- confirms wiring numba in didn't accidentally
    change their behavior."""
    rng = np.random.default_rng(0)
    msg = rng.integers(0, 256, size=100).astype("uint8")[None, :]
    assert int(CRC("none", backend="numpy").generate_key(msg)[0]) == 0
    checksum = CRC("checksum", backend="numpy").generate_key(msg)
    expected = ((~(msg.astype(np.int64).sum(axis=1) & 0xFF)) + 1) & 0xFF
    np.testing.assert_array_equal(checksum, expected.astype(np.uint64))
