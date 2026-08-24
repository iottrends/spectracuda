"""Sync strategies (Layer 2): SchmidlCoxSync and ZadoffChuSync both
implemented.

Algorithm references: reference/liquid-dsp's qdetector_create_linear (ZC
-- see zadoff_chu.py's module docstring for what's actually ported vs
re-derived for batch processing) and the S0/S1 preamble logic in
src/framing/src/ofdmframesync.c (Schmidl-Cox -- liquid-dsp doesn't
expose it as its own reusable block, so it's extracted from source).
"""
from .schmidl_cox import SchmidlCoxSync
from .zadoff_chu import ZadoffChuSync

__all__ = ["SchmidlCoxSync", "ZadoffChuSync"]
