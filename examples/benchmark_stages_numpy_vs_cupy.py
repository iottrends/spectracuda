"""Per-stage numpy-vs-cupy timing -- not just the whole-pipeline number
examples/benchmark_numpy_vs_cupy.py already gives, but each individual
block, so the "which stages are actually CUDA-resident vs stuck on ARM
cores" analysis (docs/todo.md, the mac.py/session.py cupy fixes' own
docstrings) gets real measured numbers behind it instead of just code
inspection.

Every stage benchmarked here is a real block pulled directly off one
fully-constructed Ofdm instance (ofdm.modem, ofdm.sync, ofdm.cfo, etc.)
-- not reconstructed by hand -- so each one is exactly as it's really
configured/used inside generate_frame()/rx_process(), same rationale as
mac/capacity.py's own "derive from the real object, don't re-derive
independently" precedent.

Two stages are EXPECTED to show ~no cupy benefit (or even a cupy
regression), predicted from code inspection before this script ever
ran, not discovered by it -- worth checking whether the numbers agree:
  - "rs_m8" FEC and CRC: both explicitly convert to host NumPy
    internally regardless of backend= (crc.py's _to_host_bytes(),
    reed_solomon.py's encode()/decode()) -- see docs/todo.md's cupy
    findings.
  - CFO estimation ("schmidl_cox"/"pilot_based" cfo=): both have a
    for-loop-over-batch-items with a host round-trip per item
    (int(start_index[b]), float(...) at the end) -- the real,
    unflagged-until-measured inefficiency noted when the mac.py cupy
    bug was fixed.

Same two correctness requirements as benchmark_numpy_vs_cupy.py: warm-up
iterations before timing (one-time JIT/cuFFT-plan compilation on cupy's
first call), and cupy.cuda.Stream.null.synchronize() before stopping the
clock (cupy ops are asynchronous -- skipping this would undercount
cupy's real time and report a fake speedup).

Usage:
    python examples/benchmark_stages_numpy_vs_cupy.py
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from spectracuda.backend import cupy_available
from spectracuda.pipeline import Ofdm

FFT_SIZE = 256
CP_LEN = 32
N_DATA = 216
N_PILOT = 8
BATCH_SIZES = [1, 32, 256]
N_ITERS = 30
N_WARMUP = 5


def _sync(backend: str) -> None:
    if backend == "cupy":
        import cupy

        cupy.cuda.Stream.null.synchronize()


def _time_call(fn: Callable[[], None], backend: str, n_iters: int) -> float:
    for _ in range(N_WARMUP):
        fn()
    _sync(backend)
    start = time.perf_counter()
    for _ in range(n_iters):
        fn()
    _sync(backend)
    return (time.perf_counter() - start) / n_iters


def _make_ofdm(backend: str, fec: str) -> Ofdm:
    return Ofdm(
        fft_size=FFT_SIZE, n_pilot=N_PILOT, n_data=N_DATA, cp_len=CP_LEN,
        modem="qpsk", fec=fec, crc="crc32",
        sync="schmidl_cox", cfo="schmidl_cox",
        channel_estimator="ls", equalizer="mmse",
        backend=backend,
    )


def _stage_benchmarks(ofdm: Ofdm, batch_size: int):
    """Returns {stage_name: zero-arg callable} -- each one real block,
    real method, synthetic input of that block's own documented
    batch-shape contract (see each class's own batch_shape_doc)."""
    xp = ofdm.xp
    rng = np.random.default_rng(0)
    bits_per_symbol = ofdm.modem.bits_per_symbol
    n_data_bits = N_DATA * bits_per_symbol

    data_bits = xp.asarray(rng.integers(0, 2, size=(batch_size, n_data_bits)).astype("uint8"))
    data_symbols = ofdm.modem.modulate(data_bits)
    freq_grid = xp.asarray(
        (rng.standard_normal((batch_size, FFT_SIZE)) + 1j * rng.standard_normal((batch_size, FFT_SIZE))).astype(
            "complex64"
        )
    )
    time_domain = ofdm.mod.process(freq_grid)
    noisy_stream = xp.asarray(
        (rng.standard_normal((batch_size, FFT_SIZE * 6)) + 1j * rng.standard_normal((batch_size, FFT_SIZE * 6))).astype(
            "complex64"
        )
    )
    start_index = xp.zeros((batch_size,), dtype="int64")
    # ofdm.channel_estimator is built from the training symbol's full set
    # of non-null subcarriers (fft_size - n_null), not just the n_pilot
    # ones -- see ch04's "training symbol" note -- so its own tx_pilots
    # has that width; match it exactly rather than guess n_pilot.
    n_known = ofdm.channel_estimator.tx_pilots.shape[-1]
    rx_pilots = xp.asarray(
        (rng.standard_normal((batch_size, n_known)) + 1j * rng.standard_normal((batch_size, n_known))).astype(
            "complex64"
        )
    )
    channel_est = xp.asarray(
        (rng.standard_normal((batch_size, N_DATA)) + 1j * rng.standard_normal((batch_size, N_DATA))).astype(
            "complex64"
        )
    )

    fec_codec = ofdm.packetizer.fec_codec  # fec0 -- whatever `fec=` was given
    # conv_v27 has no fixed k_bits (streaming code, any length + tail_bits)
    # -- self.k_bits only exists on the fixed-block schemes (rs_m8, ldpc_*).
    fec_k = getattr(fec_codec, "k_bits", n_data_bits)
    fec_bits = xp.asarray(rng.integers(0, 2, size=(batch_size, fec_k)).astype("uint8"))
    fec_encoded = fec_codec.encode(fec_bits)

    crc_codec = ofdm.packetizer.crc_codec
    crc_bytes = np.asarray(rng.integers(0, 256, size=(batch_size, 32)).astype("uint8"))  # always host, see module docstring

    stages = {
        "modem.modulate": lambda: ofdm.modem.modulate(data_bits),
        "modem.demodulate": lambda: ofdm.modem.demodulate(data_symbols),
        "fft.ifft+cp (mod)": lambda: ofdm.mod.process(freq_grid),
        "fft.fft+strip (demod)": lambda: ofdm.demod.process(time_domain),
        "sync": lambda: ofdm.sync.process(noisy_stream),
        "cfo.process": lambda: ofdm.cfo.process(noisy_stream, start_index=start_index),
        "channel_estimator": lambda: ofdm.channel_estimator.process(rx_pilots),
        "equalizer": lambda: ofdm.equalizer.process(data_symbols, channel_est=channel_est),
        f"fec[{ofdm.fec}].encode": lambda: fec_codec.encode(fec_bits),
        f"fec[{ofdm.fec}].decode": lambda: fec_codec.decode(fec_encoded),
        "crc.append_key": lambda: crc_codec.append_key(crc_bytes),
    }
    return stages


def run() -> None:
    if not cupy_available():
        print("No working CuPy/CUDA runtime detected on this machine -- "
              "nothing to compare against. Run this on a GPU machine "
              "(e.g. Colab with a GPU runtime) instead.")
        return

    for fec in ["conv_v27", "rs_m8"]:
        print(f"\n=== fec={fec!r} ===")
        numpy_ofdm = _make_ofdm("numpy", fec)
        cupy_ofdm = _make_ofdm("cupy", fec)
        for batch_size in BATCH_SIZES:
            print(f"\n-- batch_size={batch_size} --")
            print(f"{'stage':>24} | {'numpy (ms)':>10} | {'cupy (ms)':>10} | {'speedup':>8}")
            print("-" * 64)
            numpy_stages = _stage_benchmarks(numpy_ofdm, batch_size)
            cupy_stages = _stage_benchmarks(cupy_ofdm, batch_size)
            for name in numpy_stages:
                numpy_s = _time_call(numpy_stages[name], "numpy", N_ITERS)
                cupy_s = _time_call(cupy_stages[name], "cupy", N_ITERS)
                speedup = numpy_s / cupy_s
                print(f"{name:>24} | {numpy_s * 1000:>10.4f} | {cupy_s * 1000:>10.4f} | {speedup:>7.2f}x")


if __name__ == "__main__":
    run()
