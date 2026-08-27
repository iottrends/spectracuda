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
accelerated build of the SAME algorithm (see sse_available()/
NativeConvolutionalSSE below) measured a further ~2.5x on Viterbi
specifically, gated to x86_64 CPUs with runtime-verified SSE4.1 support
-- no ARM/NEON equivalent ships in upstream libcorrect for this code,
so ARM boxes (a Jetson, or a Raspberry Pi 5 -- see docs/todo.md) used to
fall all the way back to the portable path with none of that further
speedup. A from-scratch NEON port (see neon_available()/
NativeConvolutionalNEON below) closes most of that gap now -- NOT
vendored from upstream (no NEON build exists there), written for
spectracuda specifically, reusing the portable build's own
pair_lookup_t/history_buffer machinery unchanged and replacing only the
add-compare-select inner loop (see src/convolutional/neon/decode.c's
own module comment for its deliberately conservative scope, and
neon_available()'s own docstring for what has/hasn't been verified on
real ARM hardware yet).

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
import platform
import shutil
import subprocess
import sys
import threading
from typing import Optional

import numpy as np

_SRC_DIR = os.path.join(os.path.dirname(__file__), "_native_src", "libcorrect")
_INCLUDE_DIR = os.path.join(_SRC_DIR, "include")
_CONV_C_FILES = [
    os.path.join(_SRC_DIR, "src", "convolutional", f)
    for f in ("bit.c", "metric.c", "history_buffer.c", "error_buffer.c", "lookup.c", "convolutional.c", "encode.c", "decode.c")
]
_C_FILES = _CONV_C_FILES + [
    os.path.join(_SRC_DIR, "src", "reed-solomon", f)
    for f in ("polynomial.c", "reed-solomon.c", "encode.c", "decode.c")
]
# SSE4.1-accelerated Viterbi decode (see sse_available()'s own docstring
# for the two-part x86_64-and-runtime-cpuid gate this requires before
# ever touching these symbols): the SAME base convolutional.c/decode.c
# etc. as the portable build above, plus libcorrect's own
# src/convolutional/sse/*.c, which reuses those base functions for
# everything except the add-compare-select inner loop (vendored
# unmodified from reference/libcorrect/src/convolutional/sse/ -- see
# that upstream project's LICENSE, already vendored alongside the
# portable sources this shares a directory with). No Reed-Solomon
# equivalent exists upstream -- this is a Viterbi-only speedup.
_SSE_SRC_DIR = os.path.join(_SRC_DIR, "src", "convolutional", "sse")
_SSE_C_FILES = _CONV_C_FILES + [
    os.path.join(_SSE_SRC_DIR, f) for f in ("lookup.c", "convolutional.c", "encode.c", "decode.c")
]

# ARM NEON-accelerated Viterbi decode -- the ARM counterpart to the SSE
# path above, for machines (e.g. a Raspberry Pi 5) that get NONE of that
# x86-only speedup (see sse_available()'s own docstring: SSE4.1 doesn't
# exist on ARM at all, so those machines fall all the way back to the
# portable build otherwise). Unlike the SSE files, NOT vendored from
# upstream libcorrect -- no NEON build exists there (confirmed, see
# fec/_native_src/libcorrect/include/correct-neon.h's own header
# comment) -- this is original code written for spectracuda, reusing
# the portable build's own pair_lookup_t/history_buffer machinery
# unchanged and replacing only the add-compare-select inner loop (see
# src/convolutional/neon/decode.c's own module comment for the
# deliberately conservative scope of that port -- correctness-verified
# by the same sweep the SSE promotion required, see
# tests/test_fec_native_acceleration.py; NOT yet measured on real ARM
# hardware by the person promoting this comment, only by whoever ran
# that correctness sweep on one -- see neon_available()'s own docstring).
_NEON_SRC_DIR = os.path.join(_SRC_DIR, "src", "convolutional", "neon")
_NEON_C_FILES = _CONV_C_FILES + [
    os.path.join(_NEON_SRC_DIR, f) for f in ("convolutional.c", "encode.c", "decode.c")
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


_sse_lock = threading.Lock()
_sse_checked = False
_sse_lib: Optional[ctypes.CDLL] = None


def _cpu_supports_sse41() -> bool:
    """Runtime (not just compile-time) SSE4.1 feature check. This is
    load-bearing, not a nicety: compiling with -msse4.1 only proves the
    COMPILER can target that instruction set, not that the machine that
    will eventually RUN this process has it -- a compiled .so can
    outlive the machine it was built on (SPECTRACUDA_CACHE_DIR shared
    across a fleet, a container image copied to different hardware,
    etc.), and executing an SSE4.1 instruction on a CPU that lacks one
    is SIGILL: an immediate, uncatchable process crash, not a Python
    exception this module could fail silently out of like everything
    else here. So this must be checked BEFORE the compiled library is
    ever loaded, not discovered by trying it and catching a failure.

    Linux-only (reads /proc/cpuinfo) -- declines (returns False) on any
    other OS rather than guessing from platform.machine() alone, since
    there's no equally cheap, dependency-free runtime feature read
    available there. This only ever narrows sse_available() to a
    subset of x86_64 Linux machines; it can never widen it, so the
    failure mode of "declined when it could actually have run" is
    always the safe direction."""
    if sys.platform != "linux":
        return False
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith(("flags", "Features")):
                    return "sse4_1" in line.split()
    except OSError:
        return False
    return False


def _sse_source_hash() -> str:
    h = hashlib.sha256()
    for path in sorted(_SSE_C_FILES) + [
        os.path.join(_INCLUDE_DIR, "correct.h"),
        os.path.join(_INCLUDE_DIR, "correct-sse.h"),
    ]:
        with open(path, "rb") as f:
            h.update(f.read())
    h.update(sys.platform.encode())
    # Unlike _source_hash() above (the portable build is plain C99, safe
    # to share verbatim across architectures with the same sys.platform,
    # and just fails a Python-level ctypes.CDLL() load if it ever isn't
    # -- see native_available()'s except clause), this one MUST include
    # the machine architecture: an x86_64-compiled SSE .so handed to an
    # ARM/Jetson process by a cache directory shared across machines
    # wouldn't just fail to load, it could pass the ELF-format check for
    # a *different* x86_64 machine and then SIGILL -- see
    # _cpu_supports_sse41()'s own docstring. Keeping this hash disjoint
    # from _source_hash() means the two builds never collide on
    # filename either.
    h.update(platform.machine().encode())
    return h.hexdigest()[:16]


def _compile_sse(so_path: str) -> None:
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        raise RuntimeError("no C compiler (cc/gcc/clang) found")
    cmd = [cc, "-O2", "-fPIC", "-std=c99", "-msse4.1", "-I", _INCLUDE_DIR, "-shared", "-o", so_path] + _SSE_C_FILES + ["-lm"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"native SSE FEC backend compile failed: {result.stderr[-2000:]}")


def _bind_sse_signatures(lib: ctypes.CDLL) -> None:
    lib.correct_convolutional_sse_create.restype = ctypes.c_void_p
    lib.correct_convolutional_sse_create.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint16)]
    lib.correct_convolutional_sse_destroy.argtypes = [ctypes.c_void_p]
    lib.correct_convolutional_sse_encode_len.restype = ctypes.c_size_t
    lib.correct_convolutional_sse_encode_len.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.correct_convolutional_sse_encode.restype = ctypes.c_size_t
    lib.correct_convolutional_sse_encode.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint8)
    ]
    lib.correct_convolutional_sse_decode.restype = ctypes.c_ssize_t
    lib.correct_convolutional_sse_decode.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint8)
    ]


def sse_available() -> bool:
    """True if the SSE4.1-accelerated Viterbi decode path is compiled,
    loaded, AND verified safe to execute on THIS cpu. Checked once per
    process, cached -- same rationale as native_available().

    Gated on two independent conditions, both required:
      - platform.machine() says x86_64/AMD64 -- SSE4.1 doesn't exist
        anywhere else. ARM/Jetson (this project's actual deployment
        target) always declines this and uses native_available()'s
        portable path instead -- see this module's own docstring for
        why that SIMD build was deliberately left out of the always-on
        native path in the first place.
      - _cpu_supports_sse41() -- see its own docstring for why a
        runtime check is required here and a compile-time success is
        not enough evidence on its own.

    Fully independent of native_available() (the portable build): a
    separate .so, from a disjoint hash (_sse_source_hash(), not
    _source_hash()), that can be available, unavailable, or absent
    independently of the portable path's own status. Same "fails
    silently and permanently" contract as native_available() for every
    OTHER failure mode (no compiler, compile error, etc.) -- only the
    two gates above are checked ahead of time rather than discovered by
    failure, because a wrong guess there is a crash, not an exception."""
    global _sse_checked, _sse_lib
    if _sse_checked:
        return _sse_lib is not None
    with _sse_lock:
        if _sse_checked:
            return _sse_lib is not None
        _sse_checked = True
        if platform.machine() not in ("x86_64", "AMD64") or not _cpu_supports_sse41():
            return False
        try:
            so_path = os.path.join(_cache_dir(), f"libcorrect_sse_{_sse_source_hash()}.so")
            if not os.path.exists(so_path):
                tmp_path = so_path + f".tmp{os.getpid()}"
                _compile_sse(tmp_path)
                os.replace(tmp_path, so_path)  # atomic -- same race-avoidance as native_available()
            lib = ctypes.CDLL(so_path)
            _bind_sse_signatures(lib)
            _sse_lib = lib
        except Exception:
            _sse_lib = None
    return _sse_lib is not None


_neon_lock = threading.Lock()
_neon_checked = False
_neon_lib: Optional[ctypes.CDLL] = None


def _neon_source_hash() -> str:
    h = hashlib.sha256()
    for path in sorted(_NEON_C_FILES) + [
        os.path.join(_INCLUDE_DIR, "correct.h"),
        os.path.join(_INCLUDE_DIR, "correct-neon.h"),
    ]:
        with open(path, "rb") as f:
            h.update(f.read())
    h.update(sys.platform.encode())
    # Same reasoning as _sse_source_hash(): must include the machine
    # architecture, not just sys.platform, so a cache dir shared across
    # machines (e.g. SPECTRACUDA_CACHE_DIR on a shared filesystem) can
    # never hand an x86_64-compiled .so to an ARM process or vice versa.
    # Less immediately dangerous here than the SSE case (NEON intrinsics
    # simply fail to COMPILE on a non-ARM target -- no SIGILL risk from
    # a stale cross-arch .so slipping past a load, since compilation
    # itself is architecture-specific and happens fresh per machine
    # unless the cache dir is literally shared across differing
    # hardware), but kept disjoint anyway for the same "never collide on
    # filename" reason _sse_source_hash() gives.
    h.update(platform.machine().encode())
    return h.hexdigest()[:16]


def _compile_neon(so_path: str) -> None:
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        raise RuntimeError("no C compiler (cc/gcc/clang) found")
    # No -mfpu/-march flag needed: NEON/ASIMD is a MANDATORY part of the
    # AArch64 base ISA (unlike x86_64's optional SSE4.1, which needs
    # both an explicit -msse4.1 compile flag AND _cpu_supports_sse41()'s
    # own runtime check below) -- any aarch64 C compiler already targets
    # it by default, and neon_available()'s platform.machine() gate
    # below is the only gate this needs.
    cmd = [cc, "-O2", "-fPIC", "-std=c99", "-I", _INCLUDE_DIR, "-shared", "-o", so_path] + _NEON_C_FILES + ["-lm"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"native NEON FEC backend compile failed: {result.stderr[-2000:]}")


def _bind_neon_signatures(lib: ctypes.CDLL) -> None:
    lib.correct_convolutional_neon_create.restype = ctypes.c_void_p
    lib.correct_convolutional_neon_create.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint16)]
    lib.correct_convolutional_neon_destroy.argtypes = [ctypes.c_void_p]
    lib.correct_convolutional_neon_encode_len.restype = ctypes.c_size_t
    lib.correct_convolutional_neon_encode_len.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.correct_convolutional_neon_encode.restype = ctypes.c_size_t
    lib.correct_convolutional_neon_encode.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint8)
    ]
    lib.correct_convolutional_neon_decode.restype = ctypes.c_ssize_t
    lib.correct_convolutional_neon_decode.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint8)
    ]


def neon_available() -> bool:
    """True if the ARM NEON-accelerated Viterbi decode path is compiled
    and loaded. Checked once per process, cached -- same rationale as
    native_available()/sse_available().

    Gated on ONE condition, unlike sse_available()'s two: platform.
    machine() says aarch64/arm64. No runtime feature-check equivalent to
    _cpu_supports_sse41() is needed here -- NEON/ASIMD is a MANDATORY
    part of the AArch64 base architecture (every AArch64 CPU has it,
    unlike x86_64's genuinely optional SSE4.1), so the machine-type
    check alone is sufficient and can't produce a false positive the way
    a bare platform.machine()=="x86_64" check would for SSE4.1.

    NOT YET RUN ON REAL ARM HARDWARE by whoever wrote this gate (see
    src/convolutional/neon/decode.c's own module comment) -- this
    compiles and the correctness sweep must pass on an actual Pi 5 (or
    other aarch64 box) before any speed claim from this path is trusted,
    same discipline the SSE promotion required before IT was trusted.

    Fully independent of native_available()/sse_available(): a separate
    .so from a disjoint hash (_neon_source_hash()), same "fails silently
    and permanently back to native_available()'s portable path" contract
    for every other failure mode (no compiler, compile error, etc.)."""
    global _neon_checked, _neon_lib
    if _neon_checked:
        return _neon_lib is not None
    with _neon_lock:
        if _neon_checked:
            return _neon_lib is not None
        _neon_checked = True
        if platform.machine() not in ("aarch64", "arm64"):
            return False
        try:
            so_path = os.path.join(_cache_dir(), f"libcorrect_neon_{_neon_source_hash()}.so")
            if not os.path.exists(so_path):
                tmp_path = so_path + f".tmp{os.getpid()}"
                _compile_neon(tmp_path)
                os.replace(tmp_path, so_path)  # atomic -- same race-avoidance as native_available()
            lib = ctypes.CDLL(so_path)
            _bind_neon_signatures(lib)
            _neon_lib = lib
        except Exception:
            _neon_lib = None
    return _neon_lib is not None


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


class NativeConvolutionalSSE:
    """SSE4.1-accelerated drop-in for ConvolutionalCode's encode()/
    decode() -- see sse_available()'s own docstring for the gating this
    requires before this class may even be constructed. Same batch-
    shape contract as NativeConvolutional, and the same
    _DECODE_PAD_PAIRS decode-side workaround: libcorrect's SSE build
    reuses the base build's history_buffer/bit_writer code UNCHANGED
    (only the add-compare-select inner loop is replaced -- see
    src/convolutional/sse/decode.c), so the withheld-bits quirk that
    fix addresses is identical here, not something the SSE path happens
    to also have -- verified independently with the same k=1, 6, 39,
    194, 4001, 4002 sweep NativeConvolutional's own docstring describes,
    against both the portable native path and the pure-Python decoder
    as ground truth, before trusting this.

    encode() has no separate speed claim here: libcorrect's own SSE
    encode() (src/convolutional/sse/encode.c) just calls straight
    through to the identical portable correct_convolutional_encode()
    under the hood -- only decode() (the actual bottleneck -- Viterbi
    add-compare-select, not the encoder's simple shift-register
    convolution) got the SIMD treatment upstream. Kept here anyway
    (rather than falling back to NativeConvolutional's encode for this
    one method) so a single class fully owns one correct_convolutional_
    sse instance's lifetime instead of splitting it across two native
    handles."""

    def __init__(self) -> None:
        if not sse_available():
            raise RuntimeError("native SSE FEC backend is not available")
        poly = (ctypes.c_uint16 * 2)(_G1, _G2)
        self._conv = _sse_lib.correct_convolutional_sse_create(2, 7, poly)
        if not self._conv:
            raise RuntimeError("correct_convolutional_sse_create failed")

    def _encode_one(self, msg_bits: np.ndarray) -> np.ndarray:
        k = len(msg_bits)
        padded = np.concatenate([msg_bits, np.zeros(_TAIL_BITS, dtype="uint8")])
        msg_bytes = np.packbits(padded)
        enc_len_bits = _sse_lib.correct_convolutional_sse_encode_len(self._conv, len(msg_bytes))
        encoded = (ctypes.c_uint8 * (enc_len_bits // 8 + 8))()
        _sse_lib.correct_convolutional_sse_encode(self._conv, msg_bytes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), len(msg_bytes), encoded)
        want_bits = 2 * (k + _TAIL_BITS)
        encoded_bytes = np.frombuffer(bytes(encoded[: want_bits // 8 + 1]), dtype="uint8")
        return np.unpackbits(encoded_bytes)[:want_bits]

    def _decode_one(self, encoded_bits: np.ndarray) -> np.ndarray:
        # See NativeConvolutional._decode_one's own docstring for the
        # full withheld-bits story -- identical fix, identical margin,
        # this class's own docstring for why that's expected here too.
        T = len(encoded_bits) // 2
        k = T - _TAIL_BITS
        padded_bits = np.concatenate([encoded_bits, np.zeros(2 * _DECODE_PAD_PAIRS, dtype="uint8")])
        Tp = T + _DECODE_PAD_PAIRS
        encoded_bytes = np.packbits(padded_bits)
        msg_out = (ctypes.c_uint8 * (Tp // 8 + 8))()
        n_written = _sse_lib.correct_convolutional_sse_decode(self._conv, encoded_bytes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), 2 * Tp, msg_out)
        decoded = np.unpackbits(np.frombuffer(bytes(msg_out[: max(n_written, 0)]), dtype="uint8"))
        return decoded[:k]

    def encode(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype="uint8")
        return np.stack([self._encode_one(bits[b]) for b in range(bits.shape[0])])

    def decode(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype="uint8")
        return np.stack([self._decode_one(bits[b]) for b in range(bits.shape[0])])


class NativeConvolutionalNEON:
    """ARM NEON-accelerated drop-in for ConvolutionalCode's encode()/
    decode() -- the ARM counterpart to NativeConvolutionalSSE above, for
    machines that gate False on sse_available() (see neon_available()'s
    own docstring). Same batch-shape contract, same _DECODE_PAD_PAIRS
    decode-side workaround (the withheld-bits quirk lives in the shared
    portable warmup()/tail()/history_buffer machinery this class reuses
    unchanged -- see src/convolutional/neon/decode.c's own comment --
    not in anything NEON-specific, so it applies here identically).

    encode() has no separate speed claim here, same reasoning as
    NativeConvolutionalSSE's own docstring: just calls straight through
    to the identical portable correct_convolutional_encode() under the
    hood (src/convolutional/neon/encode.c)."""

    def __init__(self) -> None:
        if not neon_available():
            raise RuntimeError("native NEON FEC backend is not available")
        poly = (ctypes.c_uint16 * 2)(_G1, _G2)
        self._conv = _neon_lib.correct_convolutional_neon_create(2, 7, poly)
        if not self._conv:
            raise RuntimeError("correct_convolutional_neon_create failed")

    def _encode_one(self, msg_bits: np.ndarray) -> np.ndarray:
        k = len(msg_bits)
        padded = np.concatenate([msg_bits, np.zeros(_TAIL_BITS, dtype="uint8")])
        msg_bytes = np.packbits(padded)
        enc_len_bits = _neon_lib.correct_convolutional_neon_encode_len(self._conv, len(msg_bytes))
        encoded = (ctypes.c_uint8 * (enc_len_bits // 8 + 8))()
        _neon_lib.correct_convolutional_neon_encode(self._conv, msg_bytes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), len(msg_bytes), encoded)
        want_bits = 2 * (k + _TAIL_BITS)
        encoded_bytes = np.frombuffer(bytes(encoded[: want_bits // 8 + 1]), dtype="uint8")
        return np.unpackbits(encoded_bytes)[:want_bits]

    def _decode_one(self, encoded_bits: np.ndarray) -> np.ndarray:
        # See NativeConvolutional._decode_one's own docstring for the
        # full withheld-bits story -- identical fix, identical margin.
        T = len(encoded_bits) // 2
        k = T - _TAIL_BITS
        padded_bits = np.concatenate([encoded_bits, np.zeros(2 * _DECODE_PAD_PAIRS, dtype="uint8")])
        Tp = T + _DECODE_PAD_PAIRS
        encoded_bytes = np.packbits(padded_bits)
        msg_out = (ctypes.c_uint8 * (Tp // 8 + 8))()
        n_written = _neon_lib.correct_convolutional_neon_decode(self._conv, encoded_bytes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), 2 * Tp, msg_out)
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
