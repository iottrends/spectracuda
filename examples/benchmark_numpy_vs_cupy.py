"""Real numpy-vs-cupy timing comparison, same Ofdm config, same machine
-- answers "is this actually faster on the GPU" with measured numbers,
not assumption. Run this on the SAME machine for both backends (e.g.
Colab with a GPU runtime) so it's a fair comparison -- numpy here means
"the CPU cores on that GPU machine", not a different machine entirely.

Usage:
    python examples/benchmark_numpy_vs_cupy.py

Requires a working CUDA/CuPy runtime to compare against (see
spectracuda.backend.cupy_available()) -- exits with a clear message if
none is present, since there's nothing to compare on this machine then.

Design notes, both real correctness requirements for a fair number, not
style:
  - Warm-up calls before timing: cupy's FIRST kernel launch per op
    includes JIT/plan-cache compilation (cuFFT plans, etc.) that never
    recurs -- timing that in would make cupy look artificially slow.
  - cupy.cuda.Stream.null.synchronize() before stopping the clock: cupy
    ops are asynchronous by default (the Python call returns before the
    GPU actually finishes) -- an unsynchronized wall-clock timer would
    UNDER-count cupy's real time, making it look faster than it is.
  - Batch size is swept, not fixed at 1: this project's whole batch-
    first design (docs/architecture.md) exists because "kernel-launch
    and Python/CuPy dispatch overhead dominates at batch=1" -- a
    batch=1 number alone would be a misleading, worst-case comparison.
"""
from __future__ import annotations

import time

import numpy as np

from spectracuda.backend import cupy_available
from spectracuda.pipeline import Ofdm

FFT_SIZE = 256
CP_LEN = 32
N_DATA = 216
N_PILOT = 8
BATCH_SIZES = [1, 8, 32, 128, 512]
N_ITERS = 20  # timed iterations per (backend, batch_size) after warm-up
N_WARMUP = 3


def _make_ofdm(backend: str) -> Ofdm:
    return Ofdm(
        fft_size=FFT_SIZE, n_pilot=N_PILOT, n_data=N_DATA, cp_len=CP_LEN,
        modem="qpsk", fec="conv_v27", crc="crc32",
        sync="schmidl_cox", cfo="schmidl_cox",
        channel_estimator="ls", equalizer="mmse",
        backend=backend,
    )


def _sync(backend: str) -> None:
    """Block until the GPU has actually finished pending work -- required
    before stopping the clock, since cupy calls return before the device
    is done (see module docstring)."""
    if backend == "cupy":
        import cupy

        cupy.cuda.Stream.null.synchronize()


def _time_round_trip(ofdm: Ofdm, backend: str, batch_size: int, n_iters: int) -> float:
    bits_per_frame = ofdm.grid.n_data * ofdm.modem.bits_per_symbol
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(batch_size, bits_per_frame)).astype("uint8")

    for _ in range(N_WARMUP):
        tx_iq = ofdm.generate_frame(bits)
        ofdm.rx_process(tx_iq)
    _sync(backend)

    start = time.perf_counter()
    for _ in range(n_iters):
        tx_iq = ofdm.generate_frame(bits)
        ofdm.rx_process(tx_iq)
    _sync(backend)
    elapsed = time.perf_counter() - start
    return elapsed / n_iters  # seconds per (generate_frame + rx_process) round trip


def run() -> None:
    if not cupy_available():
        print("No working CuPy/CUDA runtime detected on this machine -- "
              "nothing to compare against. Run this on a GPU machine "
              "(e.g. Colab with a GPU runtime) instead.")
        return

    print(f"{'batch':>6} | {'numpy (ms/round-trip)':>22} | {'cupy (ms/round-trip)':>21} | {'speedup':>8}")
    print("-" * 66)
    numpy_ofdm = _make_ofdm("numpy")
    cupy_ofdm = _make_ofdm("cupy")
    for batch_size in BATCH_SIZES:
        numpy_s = _time_round_trip(numpy_ofdm, "numpy", batch_size, N_ITERS)
        cupy_s = _time_round_trip(cupy_ofdm, "cupy", batch_size, N_ITERS)
        speedup = numpy_s / cupy_s
        print(f"{batch_size:>6} | {numpy_s * 1000:>22.3f} | {cupy_s * 1000:>21.3f} | {speedup:>7.2f}x")


if __name__ == "__main__":
    run()
