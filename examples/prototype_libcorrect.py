"""Prototype: does binding the existing, hand-optimized C library
`libcorrect` (https://github.com/quiet/libcorrect, BSD-licensed, a
declared drop-in substitute for Phil Karn's libfec) beat the Numba-JIT
prototypes (prototype_viterbi_numba.py / prototype_rs_numba.py) for
conv_v27/rs_m8 decode -- the next lever after "compile the existing
Python loop with Numba", per docs/todo.md's own framing: a real,
hand-tuned C implementation instead of a JIT'd version of this
project's own loop shape.

This is a standalone EXPERIMENT, not a change to spectracuda itself, same
posture as the two Numba prototypes: correctness is checked BEFORE any
timing number is trusted, and reuses spectracuda's own real encoders
wherever a genuine interop check is possible, rather than assuming a
round-trip against libcorrect's own encoder proves anything about
compatibility with this project's actual bitstream.

Build (one-time, not part of this script -- see reference/libcorrect):
    cd reference/libcorrect && mkdir -p build && gcc -O2 -fPIC -std=c99 \\
      -Iinclude -shared -o build/libcorrect.so \\
      src/convolutional/*.c src/reed-solomon/*.c -lm
(portable/non-SSE source files only -- no cmake available in this
environment, and the portable path is also what would actually run on
Jetson's ARM CPU, unlike the SSE-intrinsic path which is x86-only.)

Two correctness questions, answered separately, since they have
DIFFERENT answers:

1. Reed-Solomon: TRUE interop is possible and checked. libcorrect
   exposes correct_rs_primitive_polynomial_8_4_3_2_0 = 0x11d -- the
   EXACT SAME primitive polynomial spectracuda's own reed_solomon.py
   uses (see its module docstring) -- and its own docs say "sane
   values for first_consecutive_root and generator_root_gap are 1 and
   1", matching spectracuda's fcr=1/prim=1 exactly. So this script
   encodes with spectracuda's REAL ReedSolomonCode.encode(), injects
   real errors, and decodes with libcorrect -- a genuine cross-library
   bit-exact check, not just a self-consistency round-trip.

2. Convolutional: NOT assumed compatible, checked instead. libcorrect's
   own documented default K=7 rate-1/2 polynomial pair is (0161, 0127)
   octal -- DIFFERENT from spectracuda's NASA/CCSDS-standard (0171,
   0133) (viterbi.py's module docstring). Since correct_convolutional_
   create() accepts an arbitrary polynomial array (not just its
   predefined constants), this script tries passing spectracuda's own
   (0171, 0133) pair directly, encodes with spectracuda's REAL
   ConvolutionalCode.encode() (same zero-tail termination convention:
   K-1=6 zero bits appended before encoding), and checks whether
   libcorrect's decode recovers it. If that fails, it's almost
   certainly a bit-ordering/tap-numbering convention mismatch between
   the two libraries (a real, documented incompatibility, not a bug in
   either) -- falls back to timing libcorrect's OWN self-consistent
   round-trip (its own encode -> corrupt -> its own decode) instead,
   which still proves the library works and is fast, just not that
   it's a drop-in bitstream-compatible replacement.

Usage:
    python examples/prototype_libcorrect.py
"""
from __future__ import annotations

import ctypes
import os
import time

import numpy as np

from spectracuda.fec.reed_solomon import ReedSolomonCode
from spectracuda.fec.viterbi import ConvolutionalCode

_SO_PATH = os.path.join(
    os.path.dirname(__file__), "..", "reference", "libcorrect", "build", "libcorrect.so"
)

K_BITS = 4000  # matches prototype_viterbi_numba.py's K_BITS -- same comparison basis
RS_MSG_LEN = 223  # full-length rs_m8 block, matches prototype_rs_numba.py's K_BITS_SYMBOLS
N_ROUNDS = 30
N_WARMUP = 5


def _load_lib() -> ctypes.CDLL:
    if not os.path.exists(_SO_PATH):
        raise FileNotFoundError(
            f"{_SO_PATH} not found -- build libcorrect first (see this script's "
            f"module docstring for the one-time gcc build command)"
        )
    lib = ctypes.CDLL(_SO_PATH)

    lib.correct_convolutional_create.restype = ctypes.c_void_p
    lib.correct_convolutional_create.argtypes = [
        ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint16)
    ]
    lib.correct_convolutional_destroy.argtypes = [ctypes.c_void_p]
    lib.correct_convolutional_encode_len.restype = ctypes.c_size_t
    lib.correct_convolutional_encode_len.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.correct_convolutional_encode.restype = ctypes.c_size_t
    lib.correct_convolutional_encode.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.correct_convolutional_decode.restype = ctypes.c_ssize_t
    lib.correct_convolutional_decode.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint8),
    ]

    lib.correct_reed_solomon_create.restype = ctypes.c_void_p
    lib.correct_reed_solomon_create.argtypes = [
        ctypes.c_uint16, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_size_t
    ]
    lib.correct_reed_solomon_destroy.argtypes = [ctypes.c_void_p]
    lib.correct_reed_solomon_encode.restype = ctypes.c_ssize_t
    lib.correct_reed_solomon_encode.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.correct_reed_solomon_decode.restype = ctypes.c_ssize_t
    lib.correct_reed_solomon_decode.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint8),
    ]
    return lib


def _bits_to_bytes(bits: np.ndarray) -> np.ndarray:
    return np.packbits(bits.astype("uint8"))


def _bytes_to_bits(b: np.ndarray, n_bits: int) -> np.ndarray:
    return np.unpackbits(b)[:n_bits]


# --- Convolutional: interop attempt against spectracuda's own encoder ---

def _try_conv_interop(lib: ctypes.CDLL, rng: np.random.Generator) -> bool:
    poly_arr = (ctypes.c_uint16 * 2)(0o171, 0o133)  # spectracuda's exact G1/G2
    conv = lib.correct_convolutional_create(2, 7, poly_arr)
    assert conv, "correct_convolutional_create failed"

    sc_conv = ConvolutionalCode(backend="numpy")
    msg_bits = rng.integers(0, 2, size=K_BITS).astype("uint8")
    encoded_bits = np.asarray(sc_conv.encode(msg_bits[None, :]))[0]  # spectracuda's REAL encoder

    encoded_bytes = _bits_to_bytes(encoded_bits)
    msg_out = (ctypes.c_uint8 * (K_BITS // 8 + 8))()
    n_written = lib.correct_convolutional_decode(
        conv, encoded_bytes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        len(encoded_bits), msg_out,
    )
    lib.correct_convolutional_destroy(conv)
    if n_written <= 0:
        return False
    decoded_bits = np.unpackbits(np.frombuffer(bytes(msg_out[:n_written]), dtype="uint8"))[:K_BITS]
    return bool(np.array_equal(decoded_bits, msg_bits))


def _conv_self_roundtrip(lib: ctypes.CDLL, rng: np.random.Generator, n_errors: int) -> bool:
    """libcorrect's own encode -> inject n_errors bit flips -> libcorrect's own
    decode -- proves the library itself works correctly, independent of any
    interop question with spectracuda's bitstream."""
    poly_arr = (ctypes.c_uint16 * 2)(0o171, 0o133)
    conv = lib.correct_convolutional_create(2, 7, poly_arr)
    assert conv

    msg_bits = rng.integers(0, 2, size=K_BITS).astype("uint8")
    tail = np.zeros(6, dtype="uint8")
    padded = np.concatenate([msg_bits, tail])
    msg_bytes = _bits_to_bytes(padded)  # msg_len below is in BYTES, not bits -- see this
    # script's module docstring / the crash this fixed: encode_len()'s own doc says its
    # msg_len arg is bytes ("the number of bits in a msg_len of given size, in bytes"),
    # asymmetric with decode()'s explicitly-bit-counted num_encoded_bits -- confirmed
    # empirically (passing a bit-count here overflowed the encoded buffer by 8x and
    # corrupted the heap) before trusting the docs' ambiguous wording.

    enc_len_bits = lib.correct_convolutional_encode_len(conv, len(msg_bytes))
    encoded = (ctypes.c_uint8 * (enc_len_bits // 8 + 8))()
    n_encoded_bits = lib.correct_convolutional_encode(
        conv, msg_bytes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), len(msg_bytes), encoded,
    )
    encoded_bytes = np.frombuffer(bytes(encoded[: (n_encoded_bits + 7) // 8]), dtype="uint8").copy()
    encoded_bits = np.unpackbits(encoded_bytes)[:n_encoded_bits]

    flip_idx = rng.choice(n_encoded_bits, size=n_errors, replace=False)
    corrupted_bits = encoded_bits.copy()
    corrupted_bits[flip_idx] ^= 1
    corrupted_bytes = _bits_to_bytes(corrupted_bits)

    msg_out = (ctypes.c_uint8 * (len(padded) // 8 + 8))()
    n_written = lib.correct_convolutional_decode(
        conv, corrupted_bytes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        n_encoded_bits, msg_out,
    )
    lib.correct_convolutional_destroy(conv)
    decoded_bits = np.unpackbits(np.frombuffer(bytes(msg_out[:n_written]), dtype="uint8"))[: len(padded)]
    return bool(np.array_equal(decoded_bits, padded))


# --- Reed-Solomon: real interop against spectracuda's own encoder ---

def _rs_interop_check(lib: ctypes.CDLL, rng: np.random.Generator, n_symbol_errors: int) -> bool:
    rs = lib.correct_reed_solomon_create(0x11D, 1, 1, 32)  # matches reed_solomon.py exactly
    assert rs, "correct_reed_solomon_create failed"

    sc_rs = ReedSolomonCode(backend="numpy")
    msg = rng.integers(0, 256, size=RS_MSG_LEN).astype("uint8")
    encoded = np.asarray(sc_rs.encode(msg[None, :]))[0]  # spectracuda's REAL RS encoder, (255,) symbols

    corrupted = encoded.copy()
    err_idx = rng.choice(255, size=n_symbol_errors, replace=False)
    corrupted[err_idx] = rng.integers(0, 256, size=n_symbol_errors).astype("uint8")

    msg_out = (ctypes.c_uint8 * 255)()
    n_written = lib.correct_reed_solomon_decode(
        rs, corrupted.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), 255, msg_out,
    )
    lib.correct_reed_solomon_destroy(rs)
    if n_written <= 0:
        return False
    decoded = np.frombuffer(bytes(msg_out[:n_written]), dtype="uint8")
    return bool(np.array_equal(decoded, msg))


def run() -> None:
    rng = np.random.default_rng(0)
    lib = _load_lib()

    print("=== Correctness ===")
    conv_interop_ok = _try_conv_interop(lib, rng)
    print(f"Viterbi/conv_v27: libcorrect decodes spectracuda's OWN encoded bitstream "
          f"(same 0171/0133 poly passed explicitly): {'YES -- true interop' if conv_interop_ok else 'NO -- bit-order/convention mismatch (see below)'}")

    conv_self_ok_5 = _conv_self_roundtrip(lib, rng, n_errors=5)
    conv_self_ok_20 = _conv_self_roundtrip(lib, rng, n_errors=20)
    print(f"Viterbi/conv_v27: libcorrect's own encode->corrupt(5 bit errors)->decode round-trip: "
          f"{'PASS' if conv_self_ok_5 else 'FAIL'}")
    print(f"Viterbi/conv_v27: libcorrect's own encode->corrupt(20 bit errors)->decode round-trip: "
          f"{'PASS' if conv_self_ok_20 else 'FAIL'}")

    rs_ok_1 = _rs_interop_check(lib, rng, n_symbol_errors=1)
    rs_ok_16 = _rs_interop_check(lib, rng, n_symbol_errors=16)  # t_max for rs_m8
    print(f"Reed-Solomon/rs_m8: libcorrect decodes spectracuda's OWN encoded codeword "
          f"(1 injected symbol error): {'PASS -- true interop' if rs_ok_1 else 'FAIL'}")
    print(f"Reed-Solomon/rs_m8: same, at t_max=16 injected symbol errors: "
          f"{'PASS -- true interop' if rs_ok_16 else 'FAIL'}")

    if not (rs_ok_1 and rs_ok_16):
        print("\nRS interop failed -- refusing to report timing numbers as if they were")
        print("proven interoperable with spectracuda's own rs_m8 bitstream.")
        return

    print("\n=== Timing (libcorrect's own encode/decode calls, x86, single-threaded) ===")

    # Viterbi: time whichever decode path is actually correctness-verified above.
    poly_arr = (ctypes.c_uint16 * 2)(0o171, 0o133)
    conv = lib.correct_convolutional_create(2, 7, poly_arr)
    msg_bits = rng.integers(0, 2, size=K_BITS).astype("uint8")
    tail = np.zeros(6, dtype="uint8")
    padded = np.concatenate([msg_bits, tail])
    msg_bytes = _bits_to_bytes(padded)  # msg_len is BYTES here, see _conv_self_roundtrip's comment
    enc_len_bits = lib.correct_convolutional_encode_len(conv, len(msg_bytes))
    encoded = (ctypes.c_uint8 * (enc_len_bits // 8 + 8))()
    n_encoded_bits = lib.correct_convolutional_encode(
        conv, msg_bytes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), len(msg_bytes), encoded,
    )
    encoded_bytes = np.frombuffer(bytes(encoded[: (n_encoded_bits + 7) // 8]), dtype="uint8").copy()
    msg_out = (ctypes.c_uint8 * (len(padded) // 8 + 8))()

    for _ in range(N_WARMUP):
        lib.correct_convolutional_decode(
            conv, encoded_bytes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), n_encoded_bits, msg_out,
        )
    start = time.perf_counter()
    for _ in range(N_ROUNDS):
        lib.correct_convolutional_decode(
            conv, encoded_bytes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), n_encoded_bits, msg_out,
        )
    conv_time = (time.perf_counter() - start) / N_ROUNDS
    lib.correct_convolutional_destroy(conv)
    print(f"Viterbi decode ({K_BITS} msg bits, libcorrect): {conv_time * 1000:.4f} ms/call "
          f"-- interop-verified: {conv_interop_ok}")

    # Reed-Solomon
    rs = lib.correct_reed_solomon_create(0x11D, 1, 1, 32)
    msg = rng.integers(0, 256, size=RS_MSG_LEN).astype("uint8")
    encoded_arr = (ctypes.c_uint8 * 255)()
    lib.correct_reed_solomon_encode(
        rs, msg.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), RS_MSG_LEN, encoded_arr,
    )
    corrupted = np.frombuffer(bytes(encoded_arr), dtype="uint8").copy()
    err_idx = rng.choice(255, size=16, replace=False)
    corrupted[err_idx] = rng.integers(0, 256, size=16).astype("uint8")
    msg_out_rs = (ctypes.c_uint8 * 255)()

    for _ in range(N_WARMUP):
        lib.correct_reed_solomon_decode(
            rs, corrupted.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), 255, msg_out_rs,
        )
    start = time.perf_counter()
    for _ in range(N_ROUNDS):
        lib.correct_reed_solomon_decode(
            rs, corrupted.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), 255, msg_out_rs,
        )
    rs_time = (time.perf_counter() - start) / N_ROUNDS
    lib.correct_reed_solomon_destroy(rs)
    print(f"Reed-Solomon decode (1 full 255-byte block, 16 symbol errors, libcorrect): "
          f"{rs_time * 1000:.4f} ms/call -- interop-verified: True")


if __name__ == "__main__":
    run()
