"""Batched OFDM modulation/demodulation: IFFT+CP (tx) and CP-removal+FFT
(rx). Layer 1 fixed infrastructure -- a thin wrapper directly on
numpy.fft/cupy.fft (see docs/architecture.md, "thin layer over CuPy, not
a CUDA kernel library"), not a registry-driven strategy.

Uses numpy/cupy's standard (non-unitary) fft/ifft convention -- ifft
already includes the 1/N term, so ifft() then fft() is an exact round
trip with no extra scaling needed.

Explicit `.astype("complex64")` after every fft/ifft call -- a real bug
found in development: numpy.fft has no single-precision code path at all
and always computes internally in double precision, returning
complex128 regardless of input dtype (this is numpy's own documented
behavior, not a numpy version quirk). Left alone, every OFDM symbol on
the numpy backend would silently run at double precision -- costly on
Jetson-class GPUs if this pattern were ever copied into GPU code, and
worse, it made the numpy and cupy backends produce genuinely different
precision (cupy's FFT, backed by cuFFT, does preserve complex64) for the
identical configuration. The explicit cast pins both backends to the
complex64 this project is built around (see docs/architecture.md on
avoiding float64/double-precision compute).

Batch-shape contract: OfdmModulator.process() takes
(n_batch, fft_size) complex frequency-domain symbols ->
(n_batch, fft_size + cp_len) complex64 time-domain samples.
OfdmDemodulator.process() is the exact inverse.
"""
from __future__ import annotations

from typing import Any

from ..block import Block


class OfdmModulator(Block):
    def __init__(self, fft_size: int, cp_len: int, *, backend=None) -> None:
        super().__init__(backend=backend)
        if cp_len < 0 or cp_len >= fft_size:
            raise ValueError("cp_len must be in [0, fft_size)")
        self.fft_size = fft_size
        self.cp_len = cp_len
        self.batch_shape_doc = (
            f"(n_batch, {fft_size}) complex in -> "
            f"(n_batch, {fft_size + cp_len}) complex out"
        )

    def process(self, freq_domain: Any, **kwargs: Any) -> Any:
        xp = self.xp
        freq_domain = xp.asarray(freq_domain)
        if freq_domain.shape[-1] != self.fft_size:
            raise ValueError(
                f"expected last axis size {self.fft_size}, got "
                f"{freq_domain.shape[-1]}"
            )
        time_domain = xp.fft.ifft(freq_domain, axis=-1).astype("complex64")
        if self.cp_len:
            cp = time_domain[..., -self.cp_len :]
            return xp.concatenate([cp, time_domain], axis=-1)
        return time_domain


class OfdmDemodulator(Block):
    def __init__(self, fft_size: int, cp_len: int, *, backend=None) -> None:
        super().__init__(backend=backend)
        if cp_len < 0 or cp_len >= fft_size:
            raise ValueError("cp_len must be in [0, fft_size)")
        self.fft_size = fft_size
        self.cp_len = cp_len
        self.batch_shape_doc = (
            f"(n_batch, {fft_size + cp_len}) complex in -> "
            f"(n_batch, {fft_size}) complex out"
        )

    def process(self, time_domain: Any, **kwargs: Any) -> Any:
        xp = self.xp
        time_domain = xp.asarray(time_domain)
        expected = self.fft_size + self.cp_len
        if time_domain.shape[-1] != expected:
            raise ValueError(
                f"expected last axis size {expected}, got "
                f"{time_domain.shape[-1]}"
            )
        no_cp = time_domain[..., self.cp_len :]
        return xp.fft.fft(no_cp, axis=-1).astype("complex64")
