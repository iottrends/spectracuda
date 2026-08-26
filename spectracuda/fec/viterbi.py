"""ConvolutionalCode: rate-1/2, constraint-length-7 convolutional
encoder + hard-decision Viterbi decoder (liquid-dsp's "conv_v27" scheme).

Uses the standard NASA/CCSDS K=7 generator polynomials (171, 133 octal;
dfree=10) -- the same industry-standard code used in countless real
systems (802.11, DVB-S, GPS, deep-space telemetry). liquid-dsp itself
does NOT implement convolutional coding from scratch: `fec_conv.c` is
entirely `#if LIBFEC_ENABLED`, wrapping Phil Karn's external `libfec` C
library, with every function returning a "libfec not installed" error
when it's absent (confirmed by reading
reference/liquid-dsp/src/fec/src/fec_conv.c). So there's no liquid-dsp
reference implementation to port or bit-exact-validate against here,
unlike Modem/OFDM -- this is a from-scratch implementation of the same
well-documented standard algorithm, not a port.

Zero-tail termination: K-1=6 zero bits are appended to the input before
encoding (flushes the shift register back to state 0), and stripped from
the decoded output -- standard practice so the trellis has a known,
unambiguous start AND end state, needed for a well-defined traceback.

Batched across codewords, not within one trellis (see
docs/architecture.md): the add-compare-select recursion is run once per
encoded-bit-pair (a Python loop over T time steps -- inherent to Viterbi,
the sequential dependency can't be parallelized away), but every
operation *within* a time step is vectorized across the full
(n_batch, 64 states) at once, on whichever backend (self.xp) is active.
Every trellis state has exactly two predecessor states (a property of
this shift-register structure, not assumed) which lets the whole
add-compare-select step be plain array ops with no per-state Python
loop.

Batch-shape contract: encode(bits) takes (n_batch, k) bits ->
(n_batch, 2*(k+6)) bits. decode(bits) takes (n_batch, 2*(k+6)) received
bits -> (n_batch, k) decoded bits (hard-decision Viterbi).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..block import Block
from ._native import NativeConvolutional, native_available

_K = 7
_G1 = 0o171  # NASA/CCSDS standard rate-1/2 K=7 generator polynomials
_G2 = 0o133  # (dfree=10)
_N_STATES = 1 << (_K - 1)  # 64


def _parity(x: int) -> int:
    """Bitwise parity (XOR of all set bits) of a small int."""
    x ^= x >> 4
    x ^= x >> 2
    x ^= x >> 1
    return x & 1


class ConvolutionalCode(Block):
    """Rate-1/2, K=7 convolutional code (liquid-dsp's "conv_v27"):
    encode via direct shift-register convolution, decode via
    hard-decision Viterbi (add-compare-select + traceback)."""

    def __init__(self, *, backend=None) -> None:
        super().__init__(backend=backend)
        xp = self.xp
        self.rate = 0.5
        self.tail_bits = _K - 1
        self.batch_shape_doc = (
            "encode: (n_batch, k) bits -> (n_batch, 2*(k+6)) bits. "
            "decode: (n_batch, 2*(k+6)) bits -> (n_batch, k) bits "
            "(hard-decision Viterbi)"
        )

        # Transparent native-code acceleration (see fec/_native.py's own
        # module docstring for the full rationale/measurements) -- CPU-
        # only, so only ever attempted for backend="numpy"; backend=
        # "cupy" always uses the array-vectorized path below regardless
        # of whether the native backend happens to be available, since
        # routing cupy arrays through it would mean a device->host->
        # device round trip on every call, defeating the point of
        # choosing cupy.
        #
        # A real libcorrect decode() bug lived here until fixed --
        # correct_convolutional_decode() withholds up to ~13 bits at the
        # end of every decode, contrary to its own documented contract
        # (see NativeConvolutional._decode_one's own docstring in
        # fec/_native.py for the full story and the fix: padding the
        # DECODE-side input with synthetic zero-state trellis steps,
        # which does NOT change the encoded bit length/wire format).
        # This was found via this project's own full test suite (16
        # failures) after an earlier promotion attempt trusted a
        # narrower check (one message size) that never happened to
        # trigger it -- re-verified across 210 message sizes (every
        # T-mod-8 residue) plus real bit-error injection against the
        # pure-Python path as ground truth before re-enabling here.
        self._native = NativeConvolutional() if (self.backend == "numpy" and native_available()) else None

        # Forward transition table (plain numpy -- tiny, built once,
        # independent of backend): for each of the 64 states and each
        # input bit (0/1), the resulting next_state and the two output
        # bits.
        next_state = np.empty((_N_STATES, 2), dtype=np.int64)
        out1 = np.empty((_N_STATES, 2), dtype=np.uint8)
        out2 = np.empty((_N_STATES, 2), dtype=np.uint8)
        for s in range(_N_STATES):
            for b in (0, 1):
                reg = (s << 1) | b  # 7-bit register: bit0=current input b, bits1-6=state
                next_state[s, b] = reg & (_N_STATES - 1)
                out1[s, b] = _parity(reg & _G1)
                out2[s, b] = _parity(reg & _G2)

        self._next_state_xp = xp.asarray(next_state)
        self._out1_xp = xp.asarray(out1.astype("float32"))
        self._out2_xp = xp.asarray(out2.astype("float32"))

        # Reverse (butterfly) structure for Viterbi: next_state's LSB IS
        # the input bit that produced it (since reg's bit0 = b directly),
        # and its two predecessor states are (next_state >> 1) and
        # (next_state >> 1) + 32 -- a direct consequence of the shift
        # register only depending on the low 5 bits of the previous
        # state for the shifted-up portion.
        ns = np.arange(_N_STATES)
        input_bit_for_ns = (ns & 1).astype(np.uint8)
        pred_a = ns >> 1
        pred_b = pred_a + (_N_STATES // 2)

        self._pred_a = xp.asarray(pred_a)
        self._pred_b = xp.asarray(pred_b)
        self._input_bit_for_ns = xp.asarray(input_bit_for_ns)
        self._out1_a = xp.asarray(out1[pred_a, input_bit_for_ns].astype("float32"))
        self._out2_a = xp.asarray(out2[pred_a, input_bit_for_ns].astype("float32"))
        self._out1_b = xp.asarray(out1[pred_b, input_bit_for_ns].astype("float32"))
        self._out2_b = xp.asarray(out2[pred_b, input_bit_for_ns].astype("float32"))

    def encode(self, bits: Any) -> Any:
        xp = self.xp
        bits = xp.asarray(bits)
        if bits.ndim == 1:
            bits = bits[None, :]
        if self._native is not None:
            return xp.asarray(self._native.encode(bits))
        n_batch = bits.shape[0]
        tail = xp.zeros((n_batch, self.tail_bits), dtype=bits.dtype)
        padded = xp.concatenate([bits, tail], axis=-1)  # (n_batch, k+tail)
        T = padded.shape[-1]

        state = xp.zeros(n_batch, dtype="int64")
        batch_idx = xp.arange(n_batch)
        out1_seq = xp.empty((n_batch, T), dtype="uint8")
        out2_seq = xp.empty((n_batch, T), dtype="uint8")
        for t in range(T):
            b = padded[:, t].astype("int64")
            out1_seq[:, t] = self._out1_xp[state, b]
            out2_seq[:, t] = self._out2_xp[state, b]
            state = self._next_state_xp[state, b]

        encoded = xp.empty((n_batch, 2 * T), dtype="uint8")
        encoded[:, 0::2] = out1_seq
        encoded[:, 1::2] = out2_seq
        return encoded

    def decode(self, bits: Any) -> Any:
        xp = self.xp
        bits = xp.asarray(bits)
        if bits.ndim == 1:
            bits = bits[None, :]
        if self._native is not None:
            return xp.asarray(self._native.decode(bits))
        n_batch = bits.shape[0]
        T = bits.shape[-1] // 2
        r1 = bits[:, 0::2].astype("float32")
        r2 = bits[:, 1::2].astype("float32")

        path_metric = xp.full((n_batch, _N_STATES), xp.inf, dtype="float32")
        path_metric[:, 0] = 0.0  # encoder always starts at state 0
        survivor_prev_state = xp.empty((T, n_batch, _N_STATES), dtype="int64")

        for t in range(T):
            r1_t = r1[:, t : t + 1]
            r2_t = r2[:, t : t + 1]
            bm_a = (self._out1_a[None, :] != r1_t).astype("float32") + (
                self._out2_a[None, :] != r2_t
            ).astype("float32")
            bm_b = (self._out1_b[None, :] != r1_t).astype("float32") + (
                self._out2_b[None, :] != r2_t
            ).astype("float32")
            cand_a = path_metric[:, self._pred_a] + bm_a
            cand_b = path_metric[:, self._pred_b] + bm_b
            choose_b = cand_b < cand_a
            path_metric = xp.where(choose_b, cand_b, cand_a)
            survivor_prev_state[t] = xp.where(choose_b, self._pred_b[None, :], self._pred_a[None, :])

        decoded_full = xp.empty((n_batch, T), dtype="uint8")
        state = xp.zeros(n_batch, dtype="int64")
        batch_idx = xp.arange(n_batch)
        for t in range(T - 1, -1, -1):
            decoded_full[:, t] = self._input_bit_for_ns[state]
            state = survivor_prev_state[t, batch_idx, state]

        return decoded_full[:, : T - self.tail_bits]

    def process(self, batch: Any, **kwargs: Any) -> Any:
        """Alias for encode() -- required to satisfy Block's abstract
        process() contract, matching the same pattern used by Modem.
        Call decode() explicitly for the inverse direction."""
        return self.encode(batch)
