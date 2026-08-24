"""Interleaver strategies (Layer 2): swappable via `interleaver=` on
`Packetizer`/`Ofdm`, four algorithms -- see docs/todo.md #1.12 for the
full write-up (why an interleaver is needed at all for concatenated
FEC, why LDPC doesn't need one the way conv_v27+rs_m8 does, and how
each of these four compares).

  - "block" (`BlockInterleaver`, RECOMMENDED default): the textbook
    matrix interleaver, pure reshape+transpose, best fit for this
    project's batch-first array-op style.
  - "permutation" (`PermutationInterleaver`): one fixed pseudo-random
    shuffle table, the same fixed-seed-randomization technique already
    used for the header scramble mask / payload filler bits.
  - "convolutional" (`ConvolutionalInterleaver`): a finite-block
    adaptation of the classic Forney/Ramsey-type interleaver CCSDS/
    DVB-S actually specify (see its own module docstring for the
    explicit, deliberate deviation from the true streaming construct).
  - "liquid" (`LiquidInterleaver`): a verified-correct port of
    liquid-dsp's own `interleaver.c` algorithm -- confirmed (by reading
    the source directly, not assumed) to be neither of the two
    textbook designs above, a bespoke multi-pass byte/bit-swap scheme.

Registered via spectracuda.registry (the same string-or-instance
pattern as sync=/cfo=/channel_estimator=/equalizer=), NOT the private
scheme-dict pattern FEC/CRC/Modem use -- unlike those, each interleaver
here has real tunable parameters (rows, seed, branches/base_delay,
depth) worth exposing as swappable-strategy config, not just a fixed
name-to-algorithm mapping.
"""
from .block import BlockInterleaver
from .convolutional import ConvolutionalInterleaver
from .liquid import LiquidInterleaver
from .permutation import PermutationInterleaver

__all__ = ["BlockInterleaver", "ConvolutionalInterleaver", "PermutationInterleaver", "LiquidInterleaver"]
