"""Layer 3: Ofdm -- the full OFDM tx+rx chain as one constructor-
configured class (see docs/architecture.md and ofdm.py's module
docstring for why this replaced the earlier OfdmRx/OfdmTx split).
"""
from .ofdm import Ofdm

__all__ = ["Ofdm"]
