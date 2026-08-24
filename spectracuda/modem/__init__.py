"""Modem: Modem(scheme) -- one class, scheme-name string, mirrors
liquid-dsp's modem_create(scheme) directly (BPSK/QPSK/16/64/256-QAM
implemented; liquid-dsp's other 48 schemes -- see
docs/liquid-dsp-api-inventory.md -- are out of scope for v1).
"""
from .mapper import Modem

__all__ = ["Modem"]
