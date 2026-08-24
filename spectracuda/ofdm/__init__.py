"""OFDM Layer 1 infra: FFT/IFFT + cyclic-prefix add/remove
(OfdmModulator/OfdmDemodulator), resource grid and pilot/data extraction
(ResourceGrid).

Fixed infrastructure, configured by params (fft_size=, cp_len=) -- not
registry/strategy-driven, since there's no competing algorithm to choose
between. See docs/architecture.md, Phase 1.
"""
from .fft import OfdmDemodulator, OfdmModulator
from .resource_grid import DATA, NULL, PILOT, ResourceGrid

__all__ = [
    "OfdmModulator",
    "OfdmDemodulator",
    "ResourceGrid",
    "NULL",
    "PILOT",
    "DATA",
]
