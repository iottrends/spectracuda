"""Prototype: does a Numba-JIT-compiled Viterbi decode loop actually beat
the current pure-NumPy implementation (spectracuda/fec/viterbi.py) on
x86, as predicted from the earlier per-stage benchmark (99ms of the
~109ms RX chain is Viterbi decode, and the cost is Python/NumPy
per-iteration call overhead over ~300 sequential trellis steps, not the
actual 64-state math)?

This is a standalone EXPERIMENT, not a change to spectracuda itself --
it reuses ConvolutionalCode's own precomputed trellis tables directly
(pred_a/pred_b/out1_a/out2_a/out1_b/out2_b/input_bit_for_ns -- the exact
same tables the real decode() uses, not re-derived here) so the two
implementations are guaranteed to be decoding against the identical
trellis structure, not two independently-built (and possibly subtly
different) ones.

Correctness is checked BEFORE any timing number is trusted: a real
message is encoded through the real ConvolutionalCode.encode(), then
decoded both ways, and the two outputs must match bit-for-bit (this
matters more than usual here -- feeding random bits as if they were a
real received codeword wouldn't actually test decode() correctly, since
Viterbi assumes the input is a noisy version of a genuine trellis-
consistent codeword, not arbitrary bits).

Requires numba (dev-only prototype tool -- not added to pyproject.toml
as a real dependency unless/until this is promoted into the library):
    pip install numba

Usage:
    python examples/prototype_viterbi_numba.py
"""
from __future__ import annotations

import time

import numba
import numpy as np

from spectracuda.fec.viterbi import ConvolutionalCode

K_BITS = 4000  # matches benchmark_x86_stages.py's SDU_BITS scenario
N_ROUNDS = 30
N_WARMUP = 5


@numba.njit(cache=True)
def _viterbi_decode_numba(r1, r2, pred_a, pred_b, out1_a, out2_a, out1_b, out2_b, input_bit_for_ns, tail_bits):
    """Same exact algorithm as ConvolutionalCode.decode() -- add-compare-
    select forward pass, then traceback starting from state 0 (valid
    specifically because zero-tail termination guarantees the real
    encoder always ends there too, same as the original implementation
    relies on) -- written as plain nested loops (the natural, safest
    Numba style) rather than the original's vectorized-over-states
    numpy form."""
    n_batch, T = r1.shape
    n_states = pred_a.shape[0]

    path_metric = np.full((n_batch, n_states), np.float32(np.inf), dtype=np.float32)
    for b in range(n_batch):
        path_metric[b, 0] = 0.0

    survivor = np.empty((T, n_batch, n_states), dtype=np.int64)

    for t in range(T):
        new_metric = np.empty((n_batch, n_states), dtype=np.float32)
        for b in range(n_batch):
            r1_t = r1[b, t]
            r2_t = r2[b, t]
            for s in range(n_states):
                pa = pred_a[s]
                pb = pred_b[s]
                bm_a = (np.float32(0.0) if out1_a[s] == r1_t else np.float32(1.0)) + (
                    np.float32(0.0) if out2_a[s] == r2_t else np.float32(1.0)
                )
                bm_b = (np.float32(0.0) if out1_b[s] == r1_t else np.float32(1.0)) + (
                    np.float32(0.0) if out2_b[s] == r2_t else np.float32(1.0)
                )
                cand_a = path_metric[b, pa] + bm_a
                cand_b = path_metric[b, pb] + bm_b
                if cand_b < cand_a:
                    new_metric[b, s] = cand_b
                    survivor[t, b, s] = pb
                else:
                    new_metric[b, s] = cand_a
                    survivor[t, b, s] = pa
        path_metric = new_metric

    decoded_full = np.empty((n_batch, T), dtype=np.uint8)
    for b in range(n_batch):
        state = 0
        for t in range(T - 1, -1, -1):
            decoded_full[b, t] = input_bit_for_ns[state]
            state = survivor[t, b, state]

    return decoded_full[:, : T - tail_bits]


def _numba_decode(code: ConvolutionalCode, bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits)
    if bits.ndim == 1:
        bits = bits[None, :]
    r1 = bits[:, 0::2].astype("float32")
    r2 = bits[:, 1::2].astype("float32")
    return _viterbi_decode_numba(
        r1, r2,
        np.asarray(code._pred_a), np.asarray(code._pred_b),
        np.asarray(code._out1_a), np.asarray(code._out2_a),
        np.asarray(code._out1_b), np.asarray(code._out2_b),
        np.asarray(code._input_bit_for_ns), code.tail_bits,
    )


def run() -> None:
    code = ConvolutionalCode(backend="numpy")
    rng = np.random.default_rng(0)
    msg_bits = rng.integers(0, 2, size=(1, K_BITS)).astype("uint8")
    encoded = code.encode(msg_bits)  # a REAL, trellis-consistent codeword

    # -- correctness first: both decoders must agree with each other AND
    # with the real transmitted message, on a genuine encoded codeword --
    numpy_decoded = code.decode(encoded)
    numba_decoded = _numba_decode(code, encoded)
    numpy_correct = np.array_equal(numpy_decoded, msg_bits)
    numba_correct = np.array_equal(numba_decoded, msg_bits)
    agree = np.array_equal(numpy_decoded, numba_decoded)
    print(f"correctness: numpy decode matches original message: {numpy_correct}")
    print(f"correctness: numba decode matches original message: {numba_correct}")
    print(f"correctness: numpy and numba decode outputs agree exactly: {agree}")
    if not (numpy_correct and numba_correct and agree):
        print("\nCORRECTNESS FAILURE -- refusing to trust timing numbers below.")
        return

    # -- Numba's first call includes JIT compilation (not real decode
    # cost) -- warm up before timing, same requirement as the cupy
    # benchmarks' warm-up rounds, different underlying reason (JIT
    # compile vs cuFFT plan cache) but same fix. --
    for _ in range(N_WARMUP):
        code.decode(encoded)
        _numba_decode(code, encoded)

    start = time.perf_counter()
    for _ in range(N_ROUNDS):
        code.decode(encoded)
    numpy_time = (time.perf_counter() - start) / N_ROUNDS

    start = time.perf_counter()
    for _ in range(N_ROUNDS):
        _numba_decode(code, encoded)
    numba_time = (time.perf_counter() - start) / N_ROUNDS

    print(f"\nk_bits={K_BITS}, {N_ROUNDS} rounds each (after {N_WARMUP} warm-up rounds):")
    print(f"  current pure-numpy decode: {numpy_time * 1000:.4f} ms")
    print(f"  numba-jit decode:          {numba_time * 1000:.4f} ms")
    print(f"  speedup:                   {numpy_time / numba_time:.1f}x")


if __name__ == "__main__":
    run()
