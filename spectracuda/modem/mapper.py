"""Modem: Modem(scheme) -- one class, scheme-name string, mirrors
liquid-dsp's modem_create(scheme) directly.

Implements standard Gray-coded square PSK/QAM constellations (the same
family liquid-dsp itself uses): BPSK, QPSK, 16-QAM, 64-QAM, 256-QAM.
Per-axis (I/Q) Gray-coded PAM with the well-known IEEE 802.11-style
average-power normalization constants (1/sqrt(2), 1/sqrt(10), 1/sqrt(42),
1/sqrt(170)) -- cross-checked against those known standard constants
since reference/liquid-dsp isn't buildable in this environment yet (no
autoconf/automake/libtool/cmake installed). A bit-exact cross-check
against liquid-dsp's own modem module (docs/liquid-dsp-api-inventory.md)
is still owed once build tooling is available -- in particular, the
convention used here (first half of each symbol's bits -> I axis, second
half -> Q axis) is this project's own choice and has NOT been verified to
match liquid-dsp's bit ordering.

Batch-shape contract: modulate() takes (n_batch, n_bits) bits ->
(n_batch, n_bits // bits_per_symbol) complex64. demodulate() (hard
decision, nearest constellation point) is the exact inverse.
"""
from __future__ import annotations

from typing import Any

from ..block import Block

_BITS_PER_SYMBOL = {
    "bpsk": 1,
    "qpsk": 2,
    "qam16": 4,
    "qam64": 6,
    "qam256": 8,
}


class Modem(Block):
    """Gray-coded PSK/QAM modulator-demodulator.

    Parameters
    ----------
    scheme:
        One of "bpsk", "qpsk", "qam16", "qam64", "qam256".
    """

    def __init__(self, scheme: str, *, backend=None) -> None:
        super().__init__(backend=backend)
        if scheme not in _BITS_PER_SYMBOL:
            raise ValueError(
                f"Unknown modem scheme {scheme!r}; expected one of "
                f"{sorted(_BITS_PER_SYMBOL)}"
            )
        self.scheme = scheme
        self.bits_per_symbol = _BITS_PER_SYMBOL[scheme]
        self.batch_shape_doc = (
            f"(n_batch, n_bits) bits in -> "
            f"(n_batch, n_bits // {self.bits_per_symbol}) complex64 out, "
            f"and the exact inverse for demodulate()"
        )
        # Precomputed once here, not on every modulate()/demodulate() call:
        # `half` (bits per I/Q axis) and therefore these weight/shift arrays
        # are fixed for this instance's whole lifetime. Re-deriving them
        # per call (per-symbol-batch, i.e. every OFDM symbol) was a real,
        # measured cost -- np.arange()+shift-table construction running on
        # the hot TX/RX path for no reason, since nothing in it ever
        # changes after __init__.
        half = 1 if scheme == "bpsk" else self.bits_per_symbol // 2
        xp = self.xp
        self._weights_half = 1 << xp.arange(half - 1, -1, -1, dtype="int64")
        self._shifts_half = xp.arange(half - 1, -1, -1, dtype="int64")
        self._norm = self._compute_norm_factor()

    # -- internal helpers ---------------------------------------------------

    def _gray_to_binary(self, g, nbits: int):
        """Parallel-prefix-XOR gray-to-binary, elementwise over an xp array."""
        b = g
        shift = 1
        while shift < nbits:
            b = b ^ (b >> shift)
            shift *= 2
        return b

    def _binary_to_gray(self, b):
        return b ^ (b >> 1)

    def _bits_to_int(self, bits):
        """Pack MSB-first bits along the last axis into integers. `bits`
        is always `half`-wide here (this instance's fixed I/Q axis width),
        so the weight table is the precomputed `self._weights_half`, not
        rebuilt per call."""
        return (bits.astype("int64") * self._weights_half).sum(axis=-1)

    def _int_to_bits(self, ints, nbits: int):
        """Unpack integers into MSB-first bits along a new last axis.
        `nbits` is always this instance's fixed `half`, so the shift
        table is the precomputed `self._shifts_half`, not rebuilt per
        call."""
        return ((ints[..., None] >> self._shifts_half) & 1).astype("uint8")

    def _pam_level(self, binary_idx, nbits: int):
        """Natural-binary index (0..2**nbits-1) -> symmetric odd PAM level
        (-(2**nbits-1), ..., -1, 1, ..., 2**nbits-1). float32, not
        float64 -- this runs for every symbol modulated/demodulated, and
        float64 elementwise math has drastically lower throughput than
        float32 on Jetson-class GPUs (this was a real bug: the array
        used to be built in float64 then immediately downcast, paying
        the double-precision cost for nothing)."""
        return 2 * binary_idx.astype("float32") - (2 ** nbits - 1)

    def _compute_norm_factor(self) -> float:
        """Average-symbol-power normalization (matches the well-known
        IEEE 802.11-style constants: 1, 1/sqrt(2), 1/sqrt(10), 1/sqrt(42),
        1/sqrt(170) for bpsk/qpsk/16/64/256-QAM respectively)."""
        if self.scheme == "bpsk":
            return 1.0
        half = self.bits_per_symbol // 2
        m = 2 ** half
        avg_energy_per_axis = (m * m - 1) / 3.0
        return 1.0 / (2 * avg_energy_per_axis) ** 0.5

    # -- public API -----------------------------------------------------------

    def modulate(self, bits: Any) -> Any:
        xp = self.xp
        bits = xp.asarray(bits)
        if bits.shape[-1] % self.bits_per_symbol != 0:
            raise ValueError(
                f"bit count {bits.shape[-1]} is not a multiple of "
                f"bits_per_symbol={self.bits_per_symbol}"
            )
        n_symbols = bits.shape[-1] // self.bits_per_symbol
        grouped = bits.reshape(bits.shape[0], n_symbols, self.bits_per_symbol)
        norm = self._norm

        if self.scheme == "bpsk":
            b = grouped[..., 0]
            level = 2 * b.astype("float32") - 1  # float32, not float64 -- see _pam_level
            return (level * norm).astype("complex64")

        half = self.bits_per_symbol // 2
        i_gray = self._bits_to_int(grouped[..., :half])
        q_gray = self._bits_to_int(grouped[..., half:])
        i_bin = self._gray_to_binary(i_gray, half)
        q_bin = self._gray_to_binary(q_gray, half)
        i_level = self._pam_level(i_bin, half)
        q_level = self._pam_level(q_bin, half)
        return ((i_level + 1j * q_level) * norm).astype("complex64")

    def demodulate(self, symbols: Any) -> Any:
        """Hard-decision demodulation (nearest constellation point)."""
        xp = self.xp
        symbols = xp.asarray(symbols)
        norm = self._norm
        descaled = symbols / norm

        if self.scheme == "bpsk":
            bits = (xp.real(descaled) >= 0).astype("uint8")
            return bits.reshape(symbols.shape[0], -1)

        half = self.bits_per_symbol // 2
        m = 2 ** half

        def _level_to_binary(level):
            b = xp.round((level + (m - 1)) / 2.0)
            return xp.clip(b, 0, m - 1).astype("int64")

        i_bin = _level_to_binary(xp.real(descaled))
        q_bin = _level_to_binary(xp.imag(descaled))
        i_bits = self._int_to_bits(self._binary_to_gray(i_bin), half)
        q_bits = self._int_to_bits(self._binary_to_gray(q_bin), half)
        bits = xp.concatenate([i_bits, q_bits], axis=-1)
        return bits.reshape(symbols.shape[0], -1)

    def process(self, batch: Any, **kwargs: Any) -> Any:
        """Alias for modulate() (bits -> symbols); call demodulate()
        explicitly for the inverse direction."""
        return self.modulate(batch)
