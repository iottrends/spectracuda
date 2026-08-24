"""Equalizer strategies (Layer 2): ZFEqualizer, MMSEEqualizer.

liquid-dsp's eqlms/eqrls are sample-adaptive single-carrier equalizers,
not per-subcarrier frequency-domain ZF/MMSE -- no direct precedent, so
these are designed from reference, not ported (see
docs/liquid-dsp-api-inventory.md).
"""
from .mmse import MMSEEqualizer
from .zf import ZFEqualizer

__all__ = ["ZFEqualizer", "MMSEEqualizer"]
