"""LDPCCode: the 12-variant IEEE 802.11n QC-LDPC family (4 rates x 3
codeword lengths), one class parameterized by `variant` (not 12
classes) -- matches liquid-dsp's own naming precedent
(`LIQUID_FEC_CONV_V27` -> one `ConvolutionalCode` class, not per-rate
subclasses).

Deliberate scope expansion BEYOND liquid-dsp parity, not a port: LDPC
doesn't exist in liquid-dsp at all (`docs/liquid-dsp-api-inventory.md`
and `docs/todo.md` both list it as a "confirmed non-gap" for exactly
that reason). It's added anyway, the same way `LSChannelEstimator`/
`ZFEqualizer`/`MMSEEqualizer` were added with no liquid-dsp precedent --
designed from the published IEEE 802.11n standard tables, reasoning made
explicit here rather than assumed to need a liquid-dsp equivalent first.

Base (shift) matrices: `spectracuda/fec/ldpc_tables.py` (see its module
docstring for sourcing/verification -- both the shift-direction
convention and a GF(2) full-rank structural check were independently
verified against the fetched reference before this class trusted them,
not assumed correct from the source alone).

GPU-batching precedent followed: Viterbi, not Reed-Solomon. RS's
Berlekamp-Massey is a genuine per-codeword Python loop (CPU-bound in
practice); Viterbi's add-compare-select vectorizes every array op
across `(n_batch, states)` via `self.xp`, looping only over the fixed,
data-independent number of trellis steps. LDPC's min-sum belief
propagation has the same shape (fixed iteration count, fixed sparsity
pattern) -- this is the first FEC codec in this codebase whose
iterative decode core is genuinely GPU-parallel across the batch, using
only `self.xp` gather + masked reductions (no custom kernels, no
scatter -- see the edge-index construction below for why gather alone
suffices).

Systematic encoding via one-time GF(2) elimination, not a shipped
generator matrix: split H = [H_m (first k columns) | H_p (last mb*Z
columns)]. `ldpc_tables.py`/its own tests confirm H_p is invertible
over GF(2) for all 12 variants (the standard's base matrices are
constructed to guarantee this). `G_parity = H_p^-1 @ H_m (mod 2)` is
computed once per instance via vectorized (row-XOR, not per-element
Python-loop) Gauss-Jordan elimination -- cheap, amortized, not part of
the per-call encode/decode hot path. `encode(bits)` is then one batched
matmul: `parity = (bits @ G_parity.T) % 2`, `codeword = [bits | parity]`.

Decode: normalized min-sum belief propagation over a fixed
`max_iterations`, mirroring Viterbi's "Python loop over the sequential/
iteration axis, vectorize everything within it" precedent. Messages are
kept as FLAT per-edge arrays (shape `(n_batch, n_edges)`), converted to
"grouped by check"/"grouped by variable" views and back via four static
index tables built once at construction
(`_check_slot_edges`/`_var_slot_edges` for the forward gather,
`_edge_owner_check`/`_edge_slot_in_check` and
`_edge_owner_var`/`_edge_slot_in_var` for the reverse gather) --
EVERY step is a gather (`array[:, index_table]`), never a scatter:
scatter-add semantics differ subtly between numpy and cupy and are
avoided entirely in this codebase's existing blocks (see
`pipeline/ofdm.py`'s own `_extract_slot` for the same gather-only
philosophy), and gather-only is possible here specifically because
each edge belongs to exactly one check-row-slot and exactly one
var-row-slot, making the reverse mapping a clean bijection, not a
reduction.

Known, explicit limitation (not silently papered over): every scheme in
this codebase's `FEC` interface is strictly hard-decision bits in/out --
there is no soft (LLR) pathway from `Modem`/demod today. A real min-sum
LDPC decoder normally performs best fed true channel LLRs; this class
instead assumes a binary symmetric channel with caller-supplied
crossover probability `p` (`decode(bits, p=0.02)`) and synthesizes
uniform-magnitude LLRs from the hard bits (`(1-2*bit) * log((1-p)/p)`).
Functionally correct, a legitimate way to run BP, but it leaves real
coding gain on the table versus a future soft-input pathway -- the same
spirit as Reed-Solomon's own documented Forney-vs-linear-solve
trade-off note.

Failure mode, mirroring RS's "fail loud" convention: after
`max_iterations`, the final syndrome (`H @ codeword mod 2`, one batched
matmul) is recomputed on the hard-decided result. Unlike RS's algebraic
decode, BP is an approximate iterative algorithm with no convergence
guarantee -- if any batch item's syndrome isn't all-zero, `decode()`
raises `ValueError` (same as RS's `_decode_one` raising on the first
uncorrectable item) rather than returning an unconverged, silently
wrong codeword.

Batch-shape contract: encode(bits) takes (n_batch, k) uint8 bits ->
(n_batch, n) uint8 bits. decode(bits, p=0.02, max_iterations=None)
takes (n_batch, n) received (possibly noisy) bits -> (n_batch, k)
decoded bits; may raise ValueError if BP doesn't converge to a
zero-syndrome codeword within max_iterations for any batch item.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..block import Block
from .ldpc_tables import BASE_MATRICES

_DEFAULT_MAX_ITERATIONS = 50
_MIN_SUM_ALPHA = 0.75  # standard normalized-min-sum scaling factor


def _expand_base_matrix(base: Tuple[Tuple[int, ...], ...], Z: int) -> Tuple[List[int], List[int]]:
    """Expand the (mb, 24) base matrix of circulant shifts into an edge
    list (check_indices, var_indices), one entry per edge, in a fixed
    deterministic (block_row, block_col, z) scan order.

    Shift convention (entry k>=0 -> Z x Z submatrix = RIGHT circular
    column shift of the identity by k, i.e. MATLAB's
    `circshift(eye(Z), [0, k])`) verified directly against
    `numpy.roll` before use -- see ldpc_tables.py's module docstring for
    the verification and the exact formula this mirrors:
        for z in range(Z): check = block_row*Z + z; var = block_col*Z + (z+k) % Z
    """
    checks: List[int] = []
    variables: List[int] = []
    for block_row, row in enumerate(base):
        for block_col, shift in enumerate(row):
            if shift < 0:
                continue
            for z in range(Z):
                checks.append(block_row * Z + z)
                variables.append(block_col * Z + (z + shift) % Z)
    return checks, variables


def _gf2_inverse(matrix: np.ndarray) -> np.ndarray:
    """Invert a square matrix over GF(2) via vectorized (row-XOR, not
    per-element Python loop) Gauss-Jordan elimination. Raises ValueError
    if singular -- for this class's actual use (inverting H_p), that
    would mean the sourced base matrix doesn't have the invertible-
    parity-submatrix property the 802.11n standard guarantees, i.e. a
    real data problem, caught here rather than silently proceeding."""
    n = matrix.shape[0]
    augmented = np.concatenate([matrix.astype(np.uint8) % 2, np.eye(n, dtype=np.uint8)], axis=1)
    row = 0
    for col in range(n):
        pivot = None
        for r in range(row, n):
            if augmented[r, col]:
                pivot = r
                break
        if pivot is None:
            raise ValueError(
                f"matrix is singular over GF(2) at column {col} -- not invertible "
                f"(the 802.11n base matrix's parity submatrix should always be "
                f"invertible; this indicates a data/transcription problem)"
            )
        if pivot != row:
            augmented[[row, pivot]] = augmented[[pivot, row]]
        eliminate_mask = augmented[:, col].astype(bool).copy()
        eliminate_mask[row] = False
        augmented[eliminate_mask] ^= augmented[row]
        row += 1
    return augmented[:, n:]


class LDPCCode(Block):
    """Parameters
    ----------
    variant:
        One of `ldpc_tables.BASE_MATRICES`'s keys, e.g. "ldpc_648_r12"
        (648-bit codeword, rate 1/2). See `spectracuda.fec.FEC` for the
        full list of 12 variant names as exposed through `FEC`.
    max_iterations:
        Default number of normalized min-sum BP iterations; overridable
        per `decode()` call.
    """

    def __init__(self, variant: str, *, max_iterations: int = _DEFAULT_MAX_ITERATIONS, backend=None) -> None:
        super().__init__(backend=backend)
        if variant not in BASE_MATRICES:
            raise ValueError(f"Unknown LDPC variant {variant!r}; expected one of {sorted(BASE_MATRICES)}")
        self.variant = variant
        self.max_iterations = max_iterations
        xp = self.xp

        spec = BASE_MATRICES[variant]
        Z = spec["Z"]
        base = spec["base"]
        mb = len(base)
        nb = len(base[0])
        self.Z = Z
        self.n = spec["n"]
        assert nb * Z == self.n
        self.mb = mb
        n_checks = mb * Z
        self.k = self.n - n_checks
        self.rate_str = spec["rate"]
        self.batch_shape_doc = (
            f"encode: (n_batch, {self.k}) bits -> (n_batch, {self.n}) bits. "
            f"decode: (n_batch, {self.n}) bits -> (n_batch, {self.k}) bits "
            f"(may raise ValueError on BP non-convergence)."
        )

        checks, variables = _expand_base_matrix(base, Z)
        n_edges = len(checks)
        self.n_edges = n_edges

        # Dense parity-check matrix (also used for encode's GF(2)
        # elimination and decode's post-BP syndrome check).
        H = np.zeros((n_checks, self.n), dtype=np.uint8)
        H[checks, variables] = 1
        self._H = xp.asarray(H)

        # --- systematic generator: G_parity = H_p^-1 @ H_m (mod 2) -----
        H_m = H[:, : self.k]
        H_p = H[:, self.k :]
        H_p_inv = _gf2_inverse(H_p)
        G_parity = (H_p_inv.astype(np.int64) @ H_m.astype(np.int64)) % 2
        self._G_parity = xp.asarray(G_parity.astype("int32"))  # (n_checks, k)

        # --- edge-index tables for BP (see module docstring) -----------
        check_degrees = [0] * n_checks
        var_degrees = [0] * self.n
        for c, v in zip(checks, variables):
            check_degrees[c] += 1
            var_degrees[v] += 1
        self.max_check_degree = max(check_degrees) if check_degrees else 0
        self.max_var_degree = max(var_degrees) if var_degrees else 0

        check_slot_edges = np.zeros((n_checks, self.max_check_degree), dtype=np.int64)
        check_slot_mask = np.zeros((n_checks, self.max_check_degree), dtype=bool)
        var_slot_edges = np.zeros((self.n, self.max_var_degree), dtype=np.int64)
        var_slot_mask = np.zeros((self.n, self.max_var_degree), dtype=bool)
        edge_owner_check = np.empty(n_edges, dtype=np.int64)
        edge_slot_in_check = np.empty(n_edges, dtype=np.int64)
        edge_owner_var = np.empty(n_edges, dtype=np.int64)
        edge_slot_in_var = np.empty(n_edges, dtype=np.int64)

        next_check_slot = [0] * n_checks
        next_var_slot = [0] * self.n
        for e, (c, v) in enumerate(zip(checks, variables)):
            jc = next_check_slot[c]
            check_slot_edges[c, jc] = e
            check_slot_mask[c, jc] = True
            edge_owner_check[e] = c
            edge_slot_in_check[e] = jc
            next_check_slot[c] += 1

            jv = next_var_slot[v]
            var_slot_edges[v, jv] = e
            var_slot_mask[v, jv] = True
            edge_owner_var[e] = v
            edge_slot_in_var[e] = jv
            next_var_slot[v] += 1

        self._check_slot_edges = xp.asarray(check_slot_edges)
        self._check_slot_mask = xp.asarray(check_slot_mask)
        self._var_slot_edges = xp.asarray(var_slot_edges)
        self._var_slot_mask = xp.asarray(var_slot_mask)
        self._edge_owner_check = xp.asarray(edge_owner_check)
        self._edge_slot_in_check = xp.asarray(edge_slot_in_check)
        self._edge_owner_var = xp.asarray(edge_owner_var)
        self._edge_slot_in_var = xp.asarray(edge_slot_in_var)

    # -- public API -----------------------------------------------------

    def encode(self, bits: Any) -> Any:
        xp = self.xp
        bits = xp.asarray(bits)
        if bits.ndim == 1:
            bits = bits[None, :]
        if bits.shape[-1] != self.k:
            raise ValueError(f"expected {self.k} message bits, got {bits.shape[-1]}")
        parity = (bits.astype("int32") @ self._G_parity.T) % 2
        return xp.concatenate([bits.astype("uint8"), parity.astype("uint8")], axis=-1)

    def decode(self, bits: Any, p: float = 0.02, max_iterations: Optional[int] = None) -> Any:
        xp = self.xp
        bits = xp.asarray(bits)
        if bits.ndim == 1:
            bits = bits[None, :]
        if bits.shape[-1] != self.n:
            raise ValueError(f"expected {self.n} codeword bits, got {bits.shape[-1]}")
        if not (0.0 < p < 0.5):
            raise ValueError(f"p={p} must be in (0, 0.5) (assumed BSC crossover probability)")
        n_iterations = self.max_iterations if max_iterations is None else max_iterations
        n_batch = bits.shape[0]

        llr_scale = float(math.log((1 - p) / p))
        channel_llr = (1 - 2 * bits.astype("float32")) * llr_scale  # (n_batch, n)

        pad_mag = xp.asarray(np.float32(1e6))
        var_mask_f = self._var_slot_mask.astype("float32")
        check_mask = self._check_slot_mask

        # Per-edge var-to-check messages, initialized to the channel LLR.
        q_flat = channel_llr[:, self._edge_owner_var]  # (n_batch, n_edges)
        r_flat = xp.zeros((n_batch, self.n_edges), dtype="float32")

        for _ in range(n_iterations):
            # --- check-to-var update ---
            q_by_check = q_flat[:, self._check_slot_edges]  # (n_batch, n_checks, max_check_deg)
            q_by_check = xp.where(check_mask[None, :, :], q_by_check, pad_mag)  # neutral magnitude
            sign = xp.where(q_by_check < 0, xp.asarray(-1.0, dtype="float32"), xp.asarray(1.0, dtype="float32"))
            mag = xp.abs(q_by_check)

            order = xp.argsort(mag, axis=-1)
            min1_idx = order[:, :, 0]
            min1 = xp.take_along_axis(mag, min1_idx[:, :, None], axis=-1)[:, :, 0]
            min2 = mag[
                xp.arange(n_batch)[:, None],
                xp.arange(self._check_slot_edges.shape[0])[None, :],
                order[:, :, 1],
            ]
            sign_product = xp.prod(sign, axis=-1)

            is_min1 = xp.arange(self._check_slot_edges.shape[1])[None, None, :] == min1_idx[:, :, None]
            out_mag = xp.where(is_min1, min2[:, :, None], min1[:, :, None])
            out_sign = sign_product[:, :, None] * sign
            r_by_check = _MIN_SUM_ALPHA * out_sign * out_mag

            r_flat = r_by_check[:, self._edge_owner_check, self._edge_slot_in_check]

            # --- var-to-check update ---
            r_by_var = r_flat[:, self._var_slot_edges] * var_mask_f[None, :, :]
            total = channel_llr + xp.sum(r_by_var, axis=-1)
            q_by_var = total[:, :, None] - r_by_var
            q_flat = q_by_var[:, self._edge_owner_var, self._edge_slot_in_var]

        r_by_var_final = r_flat[:, self._var_slot_edges] * var_mask_f[None, :, :]
        total_llr = channel_llr + xp.sum(r_by_var_final, axis=-1)
        hard_bits = (total_llr < 0).astype("uint8")  # (n_batch, n)

        syndrome = (hard_bits.astype("int32") @ self._H.T.astype("int32")) % 2
        bad = xp.asarray(xp.any(syndrome != 0, axis=-1))
        bad_host = np.asarray(bad if self.backend != "cupy" else self._to_host(bad))
        if bad_host.any():
            bad_items = np.nonzero(bad_host)[0].tolist()
            raise ValueError(
                f"LDPC decode failed to converge to a zero-syndrome codeword within "
                f"{n_iterations} iterations for batch item(s) {bad_items} -- too many "
                f"errors for variant {self.variant!r} at p={p}"
            )
        return hard_bits[:, : self.k]

    def process(self, batch: Any, **kwargs: Any) -> Any:
        """Alias for encode() -- required to satisfy Block's abstract
        process() contract, matching the same pattern used by
        Modem/FEC/ConvolutionalCode/ReedSolomonCode."""
        return self.encode(batch)

    def _to_host(self, arr: Any) -> np.ndarray:
        import cupy

        return cupy.asnumpy(arr)
