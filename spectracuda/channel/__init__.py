"""Channel estimator strategies (Layer 2): LSChannelEstimator and
MMSEChannelEstimator both implemented.

No liquid-dsp precedent (see docs/liquid-dsp-api-inventory.md) -- designed
from standard pilot-based LS/MMSE reference derivations, not ported. Not
to be confused with MMSEEqualizer (spectracuda.equalizer) -- a different
pipeline stage; see mmse.py's module docstring.
"""
from .ls import LSChannelEstimator
from .mmse import MMSEChannelEstimator

__all__ = ["LSChannelEstimator", "MMSEChannelEstimator"]
