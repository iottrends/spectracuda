# LDPC FEC scheme — implementation plan

Status: **implemented.** See `spectracuda/fec/ldpc.py`,
`spectracuda/fec/ldpc_tables.py`, `tests/test_fec_ldpc.py`, and
`docs/todo.md` §1.8 for the final state (a few details ended up
deliberately different from this plan's literal text after direct
verification during implementation — notably the base-matrix edge-
expansion formula, corrected after checking against `numpy.roll` rather
than followed as originally written here; see `ldpc_tables.py`'s module
docstring). This plan is kept below for historical/design-rationale
context. Read `spectracuda/fec/fec.py`, `spectracuda/fec/viterbi.py`, and
`spectracuda/fec/reed_solomon.py` in full before starting — this plan
assumes and builds directly on those conventions rather than restating
them in full.

## Context

`docs/todo.md` and `docs/liquid-dsp-api-inventory.md` currently list LDPC
as a "confirmed non-gap" — liquid-dsp has no LDPC at all, so its absence
was being treated as parity rather than a missing feature. LDPC is being
added anyway, as a deliberate scope expansion beyond liquid-dsp parity —
consistent with the project's own established pattern for pieces with no
liquid-dsp precedent (`LSChannelEstimator`, `ZFEqualizer`/`MMSEEqualizer`):
designed from a standard reference, not ported, with the reasoning made
explicit in the module docstring.

Scope: the **full 802.11n LDPC table** — 4 rates (1/2, 2/3, 3/4, 5/6) x 3
codeword lengths (648, 1296, 1944) = **12 codes**, all quasi-cyclic
(QC-LDPC), the same family later reused in 802.11ac/ax.

## Key finding that governs the design

`FEC` does **not** use `registry.py`'s `register()`/`resolve()` — it has
its own private dispatch in `spectracuda/fec/fec.py`: `_SCHEMES =
{"conv_v27": ConvolutionalCode, "rs_m8": ReedSolomonCode}`, plus
`_SYMBOL_LEVEL_SCHEMES` to distinguish byte/symbol-block codes (RS) from
raw-bit-streamed codes (conv_v27). `FEC.encode`/`decode`/`encoded_length`/
`decoded_length` branch on these sets. LDPC is a third kind — a
**fixed-size bit-block** code (like RS's block-ness, but operating in raw
bits, not byte symbols) — so it needs its own set alongside
`_SYMBOL_LEVEL_SCHEMES`, following the exact chunking pattern
`_pack_bits_to_symbols`/`_unpack_symbols_to_bits` already establish (fold
extra blocks into the batch dimension; `FEC` handles any bit count that's
a multiple of the scheme's block size transparently).

GPU-batching precedent to follow: **Viterbi, not Reed-Solomon.** RS is
CPU-only in practice (a Python loop per codeword in the batch for
Berlekamp-Massey) despite accepting/returning `xp` arrays. Viterbi
genuinely vectorizes every array op across `(n_batch, ...)` via `self.xp`,
looping only over the fixed, data-independent number of steps. LDPC's
min-sum belief propagation has the same shape (fixed iteration count,
fixed sparsity pattern) — it should be the **first FEC codec in this
codebase whose iterative core is genuinely GPU-parallel across the
batch**, using only `self.xp` gather + masked reductions (no custom
kernels, matching how every other block in this codebase is built).

## Design

### New files

- `spectracuda/fec/ldpc_tables.py` — the 12 base (shift) matrices as data:
  for each variant, the lifting size `Z` (27/54/81), and the `(mb, nb=24)`
  base matrix of circulant shift values (`-1` = zero block, `0..Z-1` =
  shift amount), sourced from the IEEE 802.11 standard's published LDPC
  parity-check matrix tables.

  **Data-sourcing risk, called out explicitly.** These are ~2,100
  integers across 12 matrices. Transcription errors are the single
  biggest risk in this task — a wrong shift value produces a "code" that
  either fails to decode or silently isn't the real 802.11n code. Before
  writing any decoder code: source the matrices via `WebFetch` from a
  citable reference (the IEEE 802.11 standard text, or a well-known
  toolbox's published tables), and add automated structural self-checks
  as the *first* tests written, not last: each variant's parity
  submatrix (last `mb*Z` columns) must be full rank over GF(2) — a
  direct, cheap, high-signal check that catches most transcription
  mistakes immediately, independent of whether decoding works yet.

- `spectracuda/fec/ldpc.py` — `LDPCCode(Block)`, one class parameterized
  by variant (not 12 classes). Constructor does, once per instance
  (cheap, amortized — not per encode/decode call):

  1. Expand the base matrix + `Z` into an edge list: for each nonzero
     `(block_row, block_col, shift)` in the base matrix, generate `Z`
     edges via the standard circulant expansion (`check = block_row*Z +
     (z+shift)%Z`, `var = block_col*Z + z`).
  2. Build **dense generator data via one-time GF(2) elimination**, not a
     stored/shipped generator matrix: split `H = [H_m | H_p]` where `H_p`
     is the last `mb*Z` columns (802.11n's base matrices are constructed
     so this parity submatrix is invertible — this is exactly what the
     rank check above verifies). Compute `G_parity = H_p^-1 @ H_m (mod
     2)` once, via vectorized (not per-bit-Python-loop) GF(2) row
     reduction. This makes the code systematic by construction —
     codeword = `[message (k bits) | parity (n-k bits)]` — matching the
     standard's own layout, so no extra bit-reordering step is needed.
  3. Build padded, masked edge-index arrays for both directions
     (`var_edge_ids`/`var_edge_mask` shape `(n_vars, max_var_degree)`,
     `check_edge_ids`/`check_edge_mask` shape `(n_checks,
     max_check_degree)`) — 802.11n row/column weights aren't perfectly
     uniform, so padding + masking (not a ragged structure) is required;
     padded slots are neutral (magnitude `+inf`, sign `+1`) so they never
     affect a min/product-of-sign reduction.

  `encode(bits)`: `(n_batch, k)` uint8 -> `(n_batch, n)` uint8. Parity =
  `(bits @ G_parity.T) % 2`, a batched matmul via `self.xp` — genuinely
  GPU-parallel, matches the "vectorize across the batch" precedent.

  `decode(bits, p=0.02)`: `(n_batch, n)` uint8 -> `(n_batch, k)` uint8.
  `p` is an assumed bit-flip probability (this codebase's FEC interface
  is strictly hard-decision bits in/out everywhere — no LLR pathway
  exists from `Modem`/demod today, see flag below) used to turn hard
  input bits into uniform-magnitude channel LLRs (`(1-2*bit) *
  log((1-p)/p)`), i.e. decoding models a binary symmetric channel. Then
  runs **normalized min-sum belief propagation** for a fixed
  `max_iterations` (Python loop, matching Viterbi's "loop over the
  sequential/iteration axis, vectorize everything within it" precedent):

  - Var-to-check step: gather incoming check-to-var messages per
    variable via `var_edge_ids` (fancy-index **gather**, not
    scatter-add — gather is reliably fast/correct on both numpy and
    cupy, unlike scatter-add semantics, so scatter is avoided entirely),
    sum along the padded axis, add channel LLR, subtract each edge's own
    incoming message (extrinsic).
  - Check-to-var step: standard O(degree) "top-2 magnitude +
    sign-product" trick (not O(degree²) pairwise) to get each check's
    outgoing message to every edge at once, vectorized via gather +
    masked min/argmin/sign-product along the padded axis.
  - After the loop: final per-variable LLR = channel LLR + full incoming
    sum; hard-decide; take the first `k` bits as the decoded message.
  - **Failure mode, mirroring RS's "fail loud" convention.** Recompute
    the syndrome (`H @ codeword mod 2`, batched matmul) on the final
    hard decision. Unlike RS's algebraic decode, BP is an approximate
    iterative algorithm with no convergence guarantee — if the syndrome
    isn't all-zero for a batch item after `max_iterations`, raise
    `ValueError` (decode failure) rather than returning an unconverged,
    silently wrong codeword.

### Changed files

- `spectracuda/fec/fec.py`:
  - Add all 12 `"ldpc_<N>_r<rate>"` entries to `_SCHEMES` (each a
    `functools.partial(LDPCCode, variant=...)`, so the existing
    `_SCHEMES[scheme](backend=backend)` call site needs no change).
  - Add `_BLOCK_BIT_SCHEMES = {"ldpc_648_r12", ...}` (all 12), and extend
    `encode`/`decode`/`encoded_length`/`decoded_length` with a third
    branch alongside the existing symbol-level/streamed split — same
    "fold extra blocks into the batch dimension" chunking as
    `_pack_bits_to_symbols`, but without the byte-packing step (already
    bit-level).
  - Update the module docstring (currently states v1 scope is exactly
    `conv_v27`/`rs_m8` "per the earlier scope decision... LDPC/Polar
    don't exist in liquid-dsp either, so deferring them is parity, not a
    reduced subset") to describe LDPC as an intentional scope expansion
    beyond liquid-dsp parity, with the same reasoning pattern used by the
    channel-estimator/equalizer docstrings.
- `spectracuda/fec/__init__.py`: update docstring, same reasoning.
- `spectracuda/pipeline/ofdm.py`: add the 12 new codes to
  `_FEC_SCHEME_CODES` (values 3-14; field is 5 bits, room for up to 31,
  only 3 used today) — `_FEC_SCHEME_NAMES` updates automatically via the
  existing dict-comprehension reversal. No other structural change
  needed: header packing/validation, `generate_frame`, and `rx_process`
  are all already generic over this dict.
- `docs/architecture.md`: update the `fec:` bullet — no longer "LDPC/Polar
  don't exist... deferring them is parity"; note LDPC is now implemented
  as a deliberate non-liquid-dsp addition, framed like the channel
  estimator/equalizer bullets already are.
- `docs/todo.md`: move LDPC out of the "confirmed non-gaps" list (§1.8),
  note it's implemented, and add a note that it's the first genuinely
  GPU-batched-decode FEC scheme in the codebase.

### New tests — `tests/test_fec_ldpc.py`

Following the established RS/Viterbi conventions (hardcoded
`backend="numpy"`, `np.random.default_rng` for determinism):

- Structural sanity, run for **all 12 variants** first (cheapest,
  highest signal for catching a transcription error in
  `ldpc_tables.py`): parity submatrix full-rank over GF(2); `(n, k)`
  dimensions match the published 802.11n values for that variant.
- Round-trip with no noise, all 12 variants: `H @ encode(msg) mod 2 == 0`
  (checked directly against the sourced `H`, independent of the
  systematic-encoding derivation) and `decode(encode(msg)) == msg`.
- BER stress test (mirrors Viterbi's pattern), on the smallest/fastest
  variant (`ldpc_648_r12`) plus one or two spot-checks on others: random
  bit-flip injection at a moderate rate, assert decode substantially
  reduces BER vs. the raw injected error rate.
- Over-capacity failure test: heavy noise -> `pytest.raises(ValueError)`,
  not a silently wrong result.
- `test_process_is_alias_for_encode` (same convention as every other FEC
  test file).
- Update existing `tests/test_fec.py::test_unknown_scheme_raises` — it
  currently uses `"ldpc"` as the known-bad example string; switch it to
  a genuinely nonexistent scheme name now that LDPC schemes are real.

## Known trade-off to flag explicitly (not silently papered over)

The existing `FEC.decode` interface across every scheme is strictly
hard-decision bits in/out. A real LDPC min-sum decoder normally performs
best with true soft (LLR) input from the demodulator; that pathway
doesn't exist anywhere in this codebase's `Modem`/`FEC` interfaces today.
This plan keeps LDPC consistent with the existing uniform interface by
modeling a binary symmetric channel (fixed bit-flip probability `p`) to
synthesize LLRs from hard bits — functionally correct and a legitimate
way to run BP, but it leaves real performance on the table compared to a
future soft-input pathway. Document this in `ldpc.py`'s module docstring
as a known, explicit limitation, same spirit as RS's
Forney-vs-linear-solve tradeoff note.

## Verification

1. `pytest tests/test_fec_ldpc.py -v` — structural, round-trip,
   BER-stress, and failure-mode tests above.
2. `pytest tests/test_fec.py tests/test_fec_viterbi.py
   tests/test_fec_reed_solomon.py -v` — confirm no regression, including
   the updated `test_unknown_scheme_raises`.
3. `pytest` (full suite) — confirm `pipeline/ofdm.py`'s header round-trip
   tests (`tests/test_ofdm_class.py`) still pass with the new
   `_FEC_SCHEME_CODES` entries added.
4. Manual smoke check: construct `Ofdm(..., fec="ldpc_648_r12")`,
   `generate_frame()` -> `rx_process()` round-trip on a clean
   (no-channel-impairment) signal, confirming the new scheme plugs into
   the full pipeline end to end, not just the standalone `FEC` class.
