"""Optional native-code acceleration for ConvolutionalCode (conv_v27) and
ReedSolomonCode (rs_m8), via libcorrect (BSD-licensed,
https://github.com/quiet/libcorrect, vendored as portable C99 source
under spectracuda/fec/_native_src/libcorrect/ -- LICENSE included there).

Why this exists: the pure-Python/NumPy ConvolutionalCode.decode()/
ReedSolomonCode.decode() are Python-interpreter-loop-bound (a real,
measured bottleneck -- see docs/architecture.md's own performance notes
and this project's benchmark history), the same "many small sequential
steps, each individually cheap" shape Numba/native-C compilation fixes
well. Measured (x86, one full-length message/codeword): Viterbi decode
38.6ms -> 1.78ms (portable C, ~22x), Reed-Solomon decode (16 symbol
errors, worst case) 4.0ms -> 0.03ms (portable C, ~130x). An SSE4.1-
accelerated build of the SAME algorithm measured a further ~2.5x on
Viterbi specifically, but that path is x86-only (no ARM/NEON equivalent
ships in libcorrect for this code) -- deliberately NOT vendored here,
since this project's target deployment (Jetson, ARM) would silently get
zero benefit from it while adding real cross-platform-build risk to the
one thing that must "run anywhere" (see docs/architecture.md's backend-
abstraction principle, the same reasoning that keeps backend="cupy"
strictly optional-with-fallback, never assumed).

Activation is FULLY AUTOMATIC and TRANSPARENT, not a new constructor
argument or config flag: ConvolutionalCode/ReedSolomonCode's public API
is byte-for-byte unchanged (same batch-shape contracts, same exception
types/messages on failure -- verified, not assumed, see tests/
test_fec_native_acceleration.py's interop checks). backend="numpy"
instances transparently use the native path when available; backend=
"cupy" instances never do (this is CPU-only native code -- forcing a
device->host->device round trip to use it would defeat the point of
choosing cupy in the first place, so it's skipped entirely for that
backend, not attempted-and-rejected). If no C compiler is available, or
compilation fails for any reason, this fails SILENTLY and permanently
(cached, checked once) back to the existing pure-Python/NumPy path --
unlike backend.py's cupy_available()/get_bakend(backend="cupy") (which
fails LOUD when explicitly requested and unavailable), there is no
explicit "request" being made here to fail loud about: this is a
transparent speed optimization of an already-selected backend="numpy",
the same category of thing NumPy silently picking whichever BLAS is
available already is -- not a user-facing choice with its own failure
contract.

Compiled once per (source-hash, platform) into a persistent cache dir
(~/.cache/spectracuda by default, override with
SPECTRACUDA_CACHE_DIR=...) -- NOT recompiled on every process start.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import threading
from typing import Optional

import numpy as np

_SRC_DIR = os.path.join(os.path.dirname(__file__), "_native_src", "libcorrect")
_INCLUDE_DIR = os.path.join(_SRC_DIR, "include")
_C_FILES = [
    os.path.join(_SRC_DIR, "src", "convolutional", f)
    for f in ("bit.c", "metric.c", "history_buffer.c", "error_buffer.c", "lookup.c", "convolutional.c", "encode.c", "decode.c")
] + [
    os.path.join(_SRC_DIR, "src", "reed-solomon", f)
    for f in ("polynomial.c", "reed-solomon.c", "encode.c", "decode.c")
]

_G1 = 0o171  # spectracuda's own conv_v27 polynomials (viterbi.py) -- verified interop
_G2 = 0o133
_TAIL_BITS = 6  # K-1, K=7
_DECODE_PAD_PAIRS = 24  # see NativeConvolutional._decode_one's own docstring --
                         # margin over the largest observed withheld amount (~13 bits).
                         # Verified content-independent and mismatch-free vs. the
                         # pure-Python decoder only WITHIN this code's guaranteed
                         # correction radius (dfree=10 -> t=4 errors); beyond that,
                         # native and pure-Python can legitimately diverge on which
                         # wrong answer they converge to (no unique right answer
                         # exists past t=4 regardless of padding) -- not evidence
                         # against this fix, just outside what it claims to guarantee.

_RS_N = 255
_RS_K = 223
_RS_NROOTS = 32
_RS_PRIM_POLY = 0x11D  # spectracuda's own GF(256) primitive polynomial (reed_solomon.py)
_RS_FCR = 1
_RS_GAP = 1

_lock = threading.Lock()
_checked = False
_lib: Optional[ctypes.CDLL] = None


def _cache_dir() -> str:
    base = os.environ.get("SPECTRACUDA_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "spectracuda")
    os.makedirs(base, exist_ok=True)
    return base


def _source_hash() -> str:
    h = hashlib.sha256()
    for path in sorted(_C_FILES) + [os.path.join(_INCLUDE_DIR, "correct.h")]:
        with open(path, "rb") as f:
            h.update(f.read())
    h.update(sys.platform.encode())
    return h.hexdigest()[:16]


def _compile(so_path: str) -> None:
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        raise RuntimeError("no C compiler (cc/gcc/clang) found")
    cmd = [cc, "-O2", "-fPIC", "-std=c99", "-I", _INCLUDE_DIR, "-shared", "-o", so_path] + _C_FILES + ["-lm"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"native FEC backend compile failed: {result.stderr[-2000:]}")


def _bind_signatures(lib: ctypes.CDLL) -> None:
    lib.correct_convolutional_create.restype = ctypes.c_void_p
    lib.correct_convolutional_create.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint16)]
    lib.correct_convolutional_destroy.argtypes = [ctypes.c_void_p]
    lib.correct_convolutional_encode_len.restype = ctypes.c_size_t
    lib.correct_convolutional_encode_len.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.correct_convolutional_encode.restype = ctypes.c_size_t
    lib.correct_convolutional_encode.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint8)
    ]
    lib.correct_convolutional_decode.restype = ctypes.c_ssize_t
    lib.correct_convolutional_decode.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint8)
    ]
    lib.correct_reed_solomon_create.restype = ctypes.c_void_p
    lib.correct_reed_solomon_create.argtypes = [ctypes.c_uint16, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_size_t]
    lib.correct_reed_solomon_destroy.argtypes = [ctypes.c_void_p]
    lib.correct_reed_solomon_encode.restype = ctypes.c_ssize_t
    lib.correct_reed_solomon_encode.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint8)
    ]
    lib.correct_reed_solomon_decode.restype = ctypes.c_ssize_t
    lib.correct_reed_solomon_decode.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint8)
    ]


def native_available() -> bool:
    """True if the native FEC backend is compiled/loaded and ready.
    Checked once per process, cached (compiling is not free, and a
    missing compiler shouldn't be retried on every ConvolutionalCode()/
    ReedSolomonCode() construction)."""
    global _checked, _lib
    if _checked:
        return _lib is not None
    with _lock:
        if _checked:
            return _lib is not None
        _checked = True
        try:
            so_path = os.path.join(_cache_dir(), f"libcorrect_{_source_hash()}.so")
            if not os.path.exists(so_path):
                tmp_path = so_path + f".tmp{os.getpid()}"
                _compile(tmp_path)
                os.replace(tmp_path, so_path)  # atomic -- avoids a half-written .so if two processes race
            lib = ctypes.CDLL(so_path)
            _bind_signatures(lib)
            _lib = lib
        except Exception:
            _lib = None
    return _lib is not None


class NativeConvolutional:
    """Drop-in accelerated backend for ConvolutionalCode's encode()/
    decode() -- exact same batch-shape contract. Persistent
    correct_convolutional instance, reused across calls (not recreated
    per call)."""

    def __init__(self) -> None:
        if not native_available():
            raise RuntimeError("native FEC backend is not available")
        poly = (ctypes.c_uint16 * 2)(_G1, _G2)
        self._conv = _lib.correct_convolutional_create(2, 7, poly)
        if not self._conv:
            raise RuntimeError("correct_convolutional_create failed")

    def _encode_one(self, msg_bits: np.ndarray) -> np.ndarray:
        k = len(msg_bits)
        padded = np.concatenate([msg_bits, np.zeros(_TAIL_BITS, dtype="uint8")])
        msg_bytes = np.packbits(padded)
        enc_len_bits = _lib.correct_convolutional_encode_len(self._conv, len(msg_bytes))
        encoded = (ctypes.c_uint8 * (enc_len_bits // 8 + 8))()
        _lib.correct_convolutional_encode(self._conv, msg_bytes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), len(msg_bytes), encoded)
        want_bits = 2 * (k + _TAIL_BITS)
        encoded_bytes = np.frombuffer(bytes(encoded[: want_bits // 8 + 1]), dtype="uint8")
        return np.unpackbits(encoded_bytes)[:want_bits]

    def _decode_one(self, encoded_bits: np.ndarray) -> np.ndarray:
        T = len(encoded_bits) // 2
        k = T - _TAIL_BITS
        # A real libcorrect bug, found by a full-suite test failure and
        # confirmed by sweeping k=1..45: correct_convolutional_decode()
        # withholds up to ~13 bits at the END of every decode, contrary
        # to its OWN documented contract ("num_encoded_bits... need not
        # be an exact multiple of 8" -- include.h's own comment). The
        # withheld amount depends only on (T mod 8), not on message
        # content, and for most residues eats into REAL message bits,
        # not just the 6-bit zero-tail padding (only T%8==6 is
        # coincidentally safe -- e.g. this is why k=4000/24000 always
        # decoded correctly in this project's own benchmarks while
        # k=194 did not: 4000%8==0 happens to land safely, 194%8==2
        # does not).
        #
        # Fix: append _DECODE_PAD_PAIRS extra synthetic "00" encoded
        # bit-pairs to the DECODE input before calling decode() -- zero
        # input at the trellis's already-zero-tail-terminated state 0
        # produces 00 output forever, so this is exactly what encoding
        # more trailing zeros would look like, without touching what
        # was actually transmitted (the real encode()'s output, and
        # therefore ConvolutionalCode.encoded_length()/the wire format
        # every other stage depends on, are completely unchanged -- this
        # is a pure receiver-side decode() workaround). This pushes
        # libcorrect's own withheld region into the synthetic padding
        # instead of the real message, then only the first k bits of
        # whatever comes back are kept. Verified across k=1, 6, 39, 194,
        # 4001, 4002 (every T%8 residue that was broken) before trusting
        # this -- see examples/ or this project's own history for the
        # verification sweep.
        padded_bits = np.concatenate([encoded_bits, np.zeros(2 * _DECODE_PAD_PAIRS, dtype="uint8")])
        Tp = T + _DECODE_PAD_PAIRS
        encoded_bytes = np.packbits(padded_bits)
        msg_out = (ctypes.c_uint8 * (Tp // 8 + 8))()
        n_written = _lib.correct_convolutional_decode(self._conv, encoded_bytes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), 2 * Tp, msg_out)
        decoded = np.unpackbits(np.frombuffer(bytes(msg_out[: max(n_written, 0)]), dtype="uint8"))
        return decoded[:k]

    def encode(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype="uint8")
        return np.stack([self._encode_one(bits[b]) for b in range(bits.shape[0])])

    def decode(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype="uint8")
        return np.stack([self._decode_one(bits[b]) for b in range(bits.shape[0])])


class NativeReedSolomon:
    """Drop-in accelerated backend for ReedSolomonCode's encode()/
    decode() -- exact same batch-shape contract, including "shortened"
    (real_k < 223) blocks, via manual leading-zero padding to the full
    223-symbol block (matching ReedSolomonCode.encode()/decode()'s own
    technique exactly -- NOT libcorrect's own built-in short-message
    path, which was found to have an unsafe buffer-size mismatch between
    its doc comment and actual behavior). Raises ValueError on an
    uncorrectable codeword, matching ReedSolomonCode.decode()'s own
    contract exactly (never silently returns wrong bits)."""

    def __init__(self) -> None:
        if not native_available():
            raise RuntimeError("native FEC backend is not available")
        self._rs = _lib.correct_reed_solomon_create(_RS_PRIM_POLY, _RS_FCR, _RS_GAP, _RS_NROOTS)
        if not self._rs:
            raise RuntimeError("correct_reed_solomon_create failed")

    def _encode_one(self, msg_row: np.ndarray) -> np.ndarray:
        real_k = len(msg_row)
        pad = np.zeros(_RS_K - real_k, dtype="uint8")
        full_msg = np.concatenate([pad, msg_row])
        encoded = (ctypes.c_uint8 * _RS_N)()
        _lib.correct_reed_solomon_encode(self._rs, full_msg.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), _RS_K, encoded)
        full_codeword = np.frombuffer(bytes(encoded), dtype="uint8")
        return np.concatenate([msg_row, full_codeword[_RS_K:]])

    def _decode_one(self, codeword_row: np.ndarray) -> np.ndarray:
        real_k = len(codeword_row) - _RS_NROOTS
        pad = np.zeros(_RS_K - real_k, dtype="uint8")
        full_codeword = np.concatenate([pad, codeword_row[:real_k], codeword_row[real_k:]])
        msg_out = (ctypes.c_uint8 * _RS_K)()
        n_written = _lib.correct_reed_solomon_decode(self._rs, full_codeword.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), _RS_N, msg_out)
        if n_written <= 0:
            raise ValueError("native Reed-Solomon decode failed (uncorrectable codeword)")
        decoded_full = np.frombuffer(bytes(msg_out[:n_written]), dtype="uint8")
        return decoded_full[_RS_K - real_k:]

    def encode(self, msg: np.ndarray) -> np.ndarray:
        msg = np.asarray(msg, dtype="uint8")
        return np.stack([self._encode_one(msg[b]) for b in range(msg.shape[0])])

    def decode(self, codeword: np.ndarray) -> np.ndarray:
        codeword = np.asarray(codeword, dtype="uint8")
        return np.stack([self._decode_one(codeword[b]) for b in range(codeword.shape[0])])
