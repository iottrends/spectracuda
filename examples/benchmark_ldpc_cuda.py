"""LDPC decode: numpy (CPU) vs cupy (CUDA), across a batch-size sweep.

The question this answers: examples/benchmark_x86_stages_v4.py found
LDPC decode taking ~350-800ms on this project's x86 dev machine (numpy,
batch size 12-28 codewords) -- roughly 2-3 orders of magnitude slower
than the already-optimized Viterbi/RS paths it's meant to replace. Not
a bug (verified: 50 is a normal, deliberate belief-propagation
iteration count, not a runaway loop) -- LDPC's min-sum decode is
STRUCTURALLY shaped for GPU batch parallelism (see fec/ldpc.py's own
module docstring: "the first FEC codec in this codebase whose
iterative decode core is genuinely GPU-parallel across the batch,
using only self.xp gather + masked reductions"), and a batch of a few
dozen codewords on a CPU gives that shape nothing real to parallelize
against. This script asks the real question directly: does batching
many codewords together and running on an actual CUDA GPU turn that
liability into the advantage it was designed to be?

This project's own dev machine has no GPU (verified: no nvidia-smi, no
cupy installed) -- this script is meant to run somewhere that DOES,
e.g. Google Colab with a GPU runtime selected (Runtime -> Change
runtime type -> T4 GPU or better).

Colab setup:
    !git clone <this repo's URL>
    %cd spectracuda
    !pip install -e ".[cuda]"     # installs cupy-cuda12x -- see pyproject.toml's
                                   # own [project.optional-dependencies] "cuda" extra;
                                   # if Colab's CUDA toolkit version doesn't match
                                   # cupy-cuda12x, check `!nvcc --version` and swap
                                   # in the matching cupy-cudaXXx package instead
    !python examples/benchmark_ldpc_cuda.py

Same correctness protections as examples/benchmark_stages_numpy_vs_cupy.py
(this project's own established pattern for a fair numpy-vs-cupy
comparison, reused here rather than reinvented): warm-up iterations
before timing (cupy's first call pays a one-time JIT/kernel-compile
cost that has nothing to do with steady-state throughput), and
cupy.cuda.Stream.null.synchronize() before stopping the clock (cupy
ops are asynchronous -- skipping this would undercount cupy's real
time and report a fake speedup). A real encode->decode->bit-exact-
match round trip is verified for EVERY batch size before its timing is
trusted, on both backends -- never assumed just because a smaller
batch size already passed.

Usage:
    python examples/benchmark_ldpc_cuda.py
    python examples/benchmark_ldpc_cuda.py ldpc_1944_r12
    python examples/benchmark_ldpc_cuda.py ldpc_648_r56 1,8,64,512,4096
"""
from __future__ import annotations

import sys
import time

import numpy as np

from spectracuda.backend import cupy_available
from spectracuda.fec.ldpc import LDPCCode

DEFAULT_VARIANT = "ldpc_1944_r12"
DEFAULT_BATCH_SIZES = [1, 8, 32, 128, 512, 2048, 8192]
N_ITERS = 10
N_WARMUP = 3
BIT_ERROR_RATE = 0.01  # synthetic channel crossover probability fed to decode()'s p= -- see fec/ldpc.py's own
                        # docstring for why this is needed at all (no soft-LLR pathway from Modem/demod yet)


def _parse_args():
    variant = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VARIANT
    batch_sizes = (
        [int(b) for b in sys.argv[2].split(",")] if len(sys.argv) > 2 else DEFAULT_BATCH_SIZES
    )
    return variant, batch_sizes


def _sync(backend: str) -> None:
    if backend == "cupy":
        import cupy

        cupy.cuda.Stream.null.synchronize()


def _time_decode(ldpc: LDPCCode, encoded, backend: str, n_iters: int) -> float:
    """Returns seconds/call, decode() only (not encode -- decode is the
    expensive, iterative direction; encode is one batched matmul,
    already cheap on either backend)."""
    for _ in range(N_WARMUP):
        ldpc.decode(encoded, p=BIT_ERROR_RATE)
    _sync(backend)
    start = time.perf_counter()
    for _ in range(n_iters):
        ldpc.decode(encoded, p=BIT_ERROR_RATE)
    _sync(backend)
    return (time.perf_counter() - start) / n_iters


def run() -> None:
    if not cupy_available():
        print("No working CuPy/CUDA runtime detected on this machine -- nothing to compare "
              "against. Run this on a GPU machine (e.g. Colab with a GPU runtime) instead -- "
              "see this script's own module docstring for the exact setup steps.")
        return
    import cupy

    variant, batch_sizes = _parse_args()
    print(f"=== LDPC decode: numpy vs cupy, variant={variant!r}, "
          f"batch sizes={batch_sizes} ===")
    print("    NOTE: decoding a CLEAN (error-free) codeword each time, matching "
          "benchmark_x86_stages_v4.py's own no-channel-noise methodology (this repo's "
          "benchmarks are digital-passthrough, no Channel/noise model) -- so these are "
          "BEST-CASE convergence numbers (early-exit once the syndrome hits zero), "
          "directly comparable to v4's own CPU numbers on that basis. A genuinely noisy "
          "channel would need more belief-propagation iterations per codeword and would "
          "be slower on EITHER backend, not just cupy -- not what this script is isolating.")

    numpy_ldpc = LDPCCode(variant, backend="numpy")
    cupy_ldpc = LDPCCode(variant, backend="cupy")
    k_bits, n_bits = numpy_ldpc.k, numpy_ldpc.n  # LDPCCode's own attribute names -- see fec/ldpc.py
    print(f"    k_bits={k_bits} n_bits={n_bits} (rate {k_bits/n_bits:.3f})")

    rng = np.random.default_rng(0)
    print(f"\n{'batch':>7} | {'numpy ms/call':>13} | {'cupy ms/call':>12} | {'speedup':>8} | "
          f"{'numpy Mbps':>10} | {'cupy Mbps':>10} | correct")
    print("-" * 90)
    for batch in batch_sizes:
        msg = rng.integers(0, 2, size=(batch, k_bits)).astype("uint8")
        # Same original message encoded independently on EACH backend
        # (not one backend's output re-encoded on the other) -- encode()
        # takes self.xp.asarray(msg) internally regardless of backend,
        # so a plain numpy msg array works as input either way.
        encoded_np = numpy_ldpc.encode(msg)
        encoded_cp = cupy_ldpc.encode(msg)

        # Real round trip verified for EVERY batch size, both backends,
        # before trusting its timing -- see module docstring. cupy's
        # decode() result is pulled to host via cupy.asnumpy() -- same
        # idiom fec/fec.py's own _to_host() uses -- exactly once, not
        # re-run.
        decoded_np = np.asarray(numpy_ldpc.decode(encoded_np, p=BIT_ERROR_RATE))
        decoded_cp = cupy.asnumpy(cupy_ldpc.decode(encoded_cp, p=BIT_ERROR_RATE))
        correct = np.array_equal(decoded_np, msg) and np.array_equal(decoded_cp, msg)

        numpy_s = _time_decode(numpy_ldpc, encoded_np, "numpy", N_ITERS)
        cupy_s = _time_decode(cupy_ldpc, encoded_cp, "cupy", N_ITERS)
        speedup = numpy_s / cupy_s
        total_bits = batch * k_bits
        numpy_mbps = total_bits / numpy_s / 1e6
        cupy_mbps = total_bits / cupy_s / 1e6
        print(f"{batch:>7} | {numpy_s*1000:>13.4f} | {cupy_s*1000:>12.4f} | {speedup:>7.2f}x | "
              f"{numpy_mbps:>10.2f} | {cupy_mbps:>10.2f} | {'OK' if correct else 'MISMATCH!!'}")

    print("\nRead this as: numpy ms/call stays roughly FLAT as batch size grows (CPU has no "
          "batch parallelism to exploit -- see examples/benchmark_x86_stages_v4.py's own finding), "
          "while cupy ms/call should grow much more slowly than batch size, so cupy Mbps should "
          "climb steeply -- that gap IS the answer to whether GPU batching turns LDPC's numpy "
          "weakness into a real advantage for this project's FEC bottleneck.")


if __name__ == "__main__":
    run()
