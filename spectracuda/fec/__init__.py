"""FEC: FEC(scheme) -- one class, scheme-name string, mirrors
liquid-dsp's fec_create(scheme) directly. liquid-dsp-parity schemes:
"conv_v27" (Viterbi) and "rs_m8" (Reed-Solomon RS(255,223)) -- exactly
liquid-dsp's own LIQUID_FEC_CONV_V27/LIQUID_FEC_RS_M8. Neither is
ported from liquid-dsp -- see viterbi.py's and reed_solomon.py's module
docstrings: liquid-dsp wraps Phil Karn's external `libfec` C library
for both, with no fallback when it's absent.

Deliberate scope expansion BEYOND liquid-dsp parity: the 12-variant
IEEE 802.11n QC-LDPC family ("ldpc_648_r12" ... "ldpc_1944_r56", see
ldpc.py's module docstring). liquid-dsp has no LDPC at all -- this
follows the same "no liquid-dsp precedent -> design from a standard
reference instead of deferring" reasoning already used for
LSChannelEstimator/ZFEqualizer/MMSEEqualizer, not a liquid-dsp port.

CRC: CRC(scheme) -- see crc.py's module docstring. Unlike conv_v27/
rs_m8, liquid-dsp's CRC module is entirely self-contained (no external
library), so this one IS a genuine byte-exact port, verified against
liquid's own crc_autotest.c test vectors.
"""
from .crc import CRC
from .fec import FEC

__all__ = ["FEC", "CRC"]
