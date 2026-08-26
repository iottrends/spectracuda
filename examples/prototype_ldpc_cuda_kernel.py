"""Prototype: a single fused CUDA kernel for LDPC min-sum decode.

UNTESTED ON REAL HARDWARE -- written on a dev machine with no GPU (no
nvidia-smi, no cupy installed, confirmed earlier this session). This is
a prototype in the same sense examples/prototype_libcorrect.py and
examples/prototype_crc_numba.py were: a standalone, correctness-first
experiment to verify BEFORE trusting it, not something promoted into
fec/ldpc.py yet. Do not wire this into the real library until
verify_correctness() below has actually been run, on a real GPU, and
passed -- see this project's own history for why that promotion
discipline exists (fec/_native.py's Viterbi decode had a real bug that
only 210-message-size sweep + real bit-error-injection testing caught,
not a single happy-path check).

## Why this exists

fec/ldpc.py's decode() is a plain array-op implementation: 50 belief-
propagation iterations, each doing ~8 separate xp (numpy/cupy) calls.
On cupy that's ~400 separate GPU kernel launches per decode() call --
examples/benchmark_x86_stages_v4.py and examples/benchmark_ldpc_cuda.py
both measured this costing 200-800ms on CPU and still ~25-30x SLOWER
in absolute throughput than this project's already-optimized CPU
rs_m8+conv_v27, even with a real ~10x cupy-over-numpy speedup. Reading
NVIDIA's own cuPHY LDPC kernels (reference/aerial-cuda-accelerated-
ran-NOTES.md has the full writeup) found the two real reasons theirs is
fast: (1) ONE kernel launch decodes an entire batch's full iteration
loop, not hundreds of small launches, and (2) each codeword gets its
own CUDA thread block with its working data addressed directly rather
than re-derived through Python-level array gathers every iteration.

This prototype targets reason (1) directly -- one kernel launch per
decode() call, the whole 50-iteration loop inside it -- using the SAME
edge-index tables fec/ldpc.py's own LDPCCode.__init__ already builds
(check_slot_edges/var_slot_edges/etc.), not a from-scratch data layout.
It does NOT yet attempt cuPHY's reason (2) (shared-memory residency
per codeword) -- a rough estimate (see reference/aerial-cuda-
accelerated-ran-NOTES.md) puts the per-codeword message-array working
set at ~64KB for ldpc_1944_r12, already close to typical default
per-block shared memory budgets, so that's a real second step with its
own real engineering (dynamic shared memory opt-in, occupancy tuning),
deliberately deferred rather than attempted blind. This version keeps
q_flat/r_flat in GLOBAL memory (each codeword gets its own slice of a
(n_batch, n_edges) array) -- simpler, safer to reason about without
being able to test it, and should already recover most of reason (1)'s
win (eliminating ~400 launches down to 1) even without reason (2).

## Correctness approach, given no GPU to test on here

Every formula below is a direct, line-by-line translation of
fec/ldpc.py's decode() -- cross-checked term by term while writing
this, not just "looks similar". One real discrepancy WAS caught this
way before finalizing: the numpy version excludes the check-to-var
minimum by EDGE INDEX (`is_min1 = arange(...) == min1_idx`, from
argsort), not by comparing magnitude VALUES -- a naive `mag == min1`
translation would be wrong in the case of ties between more than one
edge (both would get excluded, not just the argsort-selected one).
Fixed here by tracking min1's own loop index (min1_j) explicitly and
comparing indices, matching the array version's actual semantics
exactly (see the analysis in this file's own history/commit message
for why, in the one case they'd differ, exact float ties, they'd
actually still agree -- fixed anyway to not have to rely on that
argument being airtight).

verify_correctness() below is the real gate: for several batch sizes,
it encodes random messages with the EXISTING, trusted LDPCCode.encode(),
decodes with BOTH the existing array-op LDPCCode.decode() (ground
truth) and this kernel, and asserts bit-exact agreement, plus a real
BSC bit-error-injection round trip (not just clean codewords) --
mirroring fec/ldpc.py's own module docstring's error-injection
verification discipline. This has NOT been run (no GPU here) -- running
it and it passing is the actual bar for trusting any of this.

## Not yet done, real gaps if this passes and gets promoted

- No shared-memory residency (reason (2) above) -- a real, separate
  next step once this baseline is validated.
- No handling of the ValueError-on-non-convergence contract
  fec/ldpc.py's decode() has -- this prototype always returns hard
  bits regardless of syndrome; a caller must check convergence itself
  (see verify_correctness() for the pattern) until that's added.
- No fp16/__half path -- float32 only, matching fec/ldpc.py's own
  dtype choice.

Usage (on a real CUDA machine, e.g. Colab with a GPU runtime):
    python examples/prototype_ldpc_cuda_kernel.py
"""
from __future__ import annotations

import time

import numpy as np

from spectracuda.backend import cupy_available
from spectracuda.fec.ldpc import LDPCCode, _MIN_SUM_ALPHA

# One fused kernel: init q_flat, run the whole belief-propagation loop,
# write final hard-decision bits -- all in one launch. One CUDA block
# per codeword (blockIdx.x = batch item); threads within a block are
# NOT one-per-edge/one-per-node structurally required to match Z or any
# other code parameter -- a plain grid-stride loop over checks (then
# variables) inside each phase, sized by a fixed BLOCK_SIZE chosen at
# launch time (see cuda_ldpc_decode()).
#
# Every array below is passed as a raw pointer; per-codeword arrays
# (channel_llr, q_flat, r_flat, hard_bits) are indexed by
# blockIdx.x * (that array's per-codeword width) -- everything else
# (check_slot_edges, check_degree, var_slot_edges, var_degree,
# edge_owner_var) is the SAME for every codeword in the batch (built
# once from the code's base matrix, exactly as fec/ldpc.py's own
# LDPCCode.__init__ already computes it) and is read-only, shared
# across all blocks.
_LDPC_MIN_SUM_KERNEL_SRC = r"""
extern "C" __global__
void ldpc_min_sum_decode(
    const float* __restrict__ channel_llr,        // (n_batch, n)
    float* __restrict__ q_flat,                    // (n_batch, n_edges) scratch
    float* __restrict__ r_flat,                    // (n_batch, n_edges) scratch
    const long long* __restrict__ check_slot_edges,// (n_checks, max_check_degree)
    const int* __restrict__ check_degree,           // (n_checks,)
    const long long* __restrict__ var_slot_edges,   // (n, max_var_degree)
    const int* __restrict__ var_degree,              // (n,)
    const long long* __restrict__ edge_owner_var,    // (n_edges,)
    unsigned char* __restrict__ hard_bits,          // (n_batch, n) output
    int n, int n_edges, int n_checks,
    int max_check_degree, int max_var_degree,
    int n_iterations, float alpha)
{
    int b = blockIdx.x;  // one block per codeword -- matches cuPHY's own
                          // "one thread block per codeword" grid shape
                          // (see reference/aerial-cuda-accelerated-ran-
                          // NOTES.md), even though this version doesn't
                          // yet put that block's data in shared memory.
    const float* my_llr = channel_llr + (size_t)b * n;
    float*       my_q   = q_flat      + (size_t)b * n_edges;
    float*       my_r   = r_flat      + (size_t)b * n_edges;
    unsigned char* my_out = hard_bits + (size_t)b * n;

    // --- init: q_flat[e] = channel_llr[edge_owner_var[e]] ---
    // Mirrors fec/ldpc.py decode(): "q_flat = channel_llr[:, self._edge_owner_var]"
    for (int e = threadIdx.x; e < n_edges; e += blockDim.x) {
        my_q[e] = my_llr[edge_owner_var[e]];
    }
    __syncthreads();

    for (int iter = 0; iter < n_iterations; ++iter) {
        // --- check-to-var update: one thread handles one check node's
        // full edge list (degree <= max_check_degree, typically small,
        // e.g. 8 for ldpc_1944_r12 -- see fec/ldpc.py's own
        // max_check_degree) -- mirrors the "--- check-to-var update ---"
        // block in fec/ldpc.py decode(): normalized min-sum, excluding
        // each edge's OWN contribution from its own min1/min2 (the
        // is_min1 / sign_product logic there).
        for (int c = threadIdx.x; c < n_checks; c += blockDim.x) {
            int deg = check_degree[c];
            const long long* edges = check_slot_edges + (size_t)c * max_check_degree;

            float min1 = 3.0e38f, min2 = 3.0e38f;
            int   min1_j = -1;
            float sign_product = 1.0f;
            for (int j = 0; j < deg; ++j) {
                float q = my_q[edges[j]];
                float mag = fabsf(q);
                float s = (q < 0.0f) ? -1.0f : 1.0f;
                sign_product *= s;
                if (mag < min1) { min2 = min1; min1 = mag; min1_j = j; }
                else if (mag < min2) { min2 = mag; }
            }
            for (int j = 0; j < deg; ++j) {
                long long e = edges[j];
                float q = my_q[e];
                float s = (q < 0.0f) ? -1.0f : 1.0f;
                // Exclude by EDGE INDEX (min1_j), not by re-comparing
                // magnitude values -- see this file's own module
                // docstring for why that distinction was checked before
                // trusting it (a tie between >1 edge's magnitude would
                // make a value-based check wrong).
                float out_mag  = (j == min1_j) ? min2 : min1;
                float out_sign = sign_product * s;  // == product of every OTHER edge's sign
                my_r[e] = alpha * out_sign * out_mag;
            }
        }
        __syncthreads();

        // --- var-to-check update: one thread handles one variable
        // node's full edge list -- mirrors the "--- var-to-check
        // update ---" block in fec/ldpc.py decode().
        for (int v = threadIdx.x; v < n; v += blockDim.x) {
            int deg = var_degree[v];
            const long long* edges = var_slot_edges + (size_t)v * max_var_degree;
            float total = my_llr[v];
            for (int j = 0; j < deg; ++j) {
                total += my_r[edges[j]];
            }
            for (int j = 0; j < deg; ++j) {
                long long e = edges[j];
                my_q[e] = total - my_r[e];
            }
        }
        __syncthreads();
    }

    // --- final hard decision, using the LAST iteration's check-to-var
    // messages (my_r as left by the loop above) -- mirrors fec/ldpc.py
    // decode()'s post-loop "r_by_var_final"/"total_llr"/"hard_bits"
    // block exactly (that block does NOT run one more var-to-check
    // update first -- neither does this).
    for (int v = threadIdx.x; v < n; v += blockDim.x) {
        int deg = var_degree[v];
        const long long* edges = var_slot_edges + (size_t)v * max_var_degree;
        float total = my_llr[v];
        for (int j = 0; j < deg; ++j) {
            total += my_r[edges[j]];
        }
        my_out[v] = (total < 0.0f) ? 1 : 0;
    }
}
"""

# Second kernel: the SAME algorithm as ldpc_min_sum_decode above -- every
# check-update/var-update formula is byte-for-byte identical, deliberately
# not re-derived -- with q_flat/r_flat/channel_llr moved from a
# (n_batch, ...) slice of GLOBAL memory into a per-block SHARED memory
# buffer, loaded once at the start and never touching global memory again
# until the final output write. This is cuPHY's second real technique
# (reference/aerial-cuda-accelerated-ran-NOTES.md) -- keeping each
# codeword's working set on-chip for the whole decode instead of round-
# tripping the whole batch through global memory every iteration, which
# examples/prototype_ldpc_cuda_kernel.py's own benchmark_speed() showed
# is the real ceiling the global-memory kernel hits (Mbps plateaus
# ~17-19 regardless of batch size from 32 upward -- a bandwidth ceiling,
# not a launch-count one, since launch-count is already fixed at 1
# either way).
#
# Shared memory needed = (n + 2*n_edges) * 4 bytes (float32). Varies by
# LDPC variant -- see this file's own module docstring / the commit that
# added this kernel for the exact per-variant numbers: ldpc_648_* needs
# only ~21KB (fits under every CUDA GPU's default 48KB per-block limit,
# no special opt-in needed -- the SAFE variant to validate this on
# first); ldpc_1296_* needs ~40-42KB (still under 48KB, should be safe
# on most GPUs); ldpc_1944_* needs ~58-63KB -- OVER the default limit,
# requires cupy's max_dynamic_shared_size_bytes opt-in (see
# cuda_ldpc_decode_shared() below), and may not even fit depending on
# the specific GPU's actual shared memory budget.
_LDPC_MIN_SUM_SHARED_KERNEL_SRC = r"""
extern "C" __global__
void ldpc_min_sum_decode_shared(
    const float* __restrict__ channel_llr_g,        // (n_batch, n) -- global, loaded into shared once
    const long long* __restrict__ check_slot_edges, // (n_checks, max_check_degree) -- global, read-only, shared across batch
    const int* __restrict__ check_degree,
    const long long* __restrict__ var_slot_edges,   // (n, max_var_degree) -- global, read-only, shared across batch
    const int* __restrict__ var_degree,
    const long long* __restrict__ edge_owner_var,
    unsigned char* __restrict__ hard_bits,          // (n_batch, n) output, global
    int n, int n_edges, int n_checks,
    int max_check_degree, int max_var_degree,
    int n_iterations, float alpha)
{
    extern __shared__ float smem[];
    float* my_llr = smem;                // n floats
    float* my_q   = smem + n;            // n_edges floats
    float* my_r   = smem + n + n_edges;  // n_edges floats

    int b = blockIdx.x;  // one block per codeword
    const float* llr_g = channel_llr_g + (size_t)b * n;
    unsigned char* my_out = hard_bits + (size_t)b * n;

    // --- load channel LLRs from global into shared memory, ONCE ---
    for (int v = threadIdx.x; v < n; v += blockDim.x) {
        my_llr[v] = llr_g[v];
    }
    __syncthreads();

    // --- init: q_flat[e] = channel_llr[edge_owner_var[e]] --- (shared-memory copy now)
    for (int e = threadIdx.x; e < n_edges; e += blockDim.x) {
        my_q[e] = my_llr[edge_owner_var[e]];
    }
    __syncthreads();

    for (int iter = 0; iter < n_iterations; ++iter) {
        // --- check-to-var update --- (identical math to ldpc_min_sum_decode above)
        for (int c = threadIdx.x; c < n_checks; c += blockDim.x) {
            int deg = check_degree[c];
            const long long* edges = check_slot_edges + (size_t)c * max_check_degree;

            float min1 = 3.0e38f, min2 = 3.0e38f;
            int   min1_j = -1;
            float sign_product = 1.0f;
            for (int j = 0; j < deg; ++j) {
                float q = my_q[edges[j]];
                float mag = fabsf(q);
                float s = (q < 0.0f) ? -1.0f : 1.0f;
                sign_product *= s;
                if (mag < min1) { min2 = min1; min1 = mag; min1_j = j; }
                else if (mag < min2) { min2 = mag; }
            }
            for (int j = 0; j < deg; ++j) {
                long long e = edges[j];
                float q = my_q[e];
                float s = (q < 0.0f) ? -1.0f : 1.0f;
                float out_mag  = (j == min1_j) ? min2 : min1;
                float out_sign = sign_product * s;
                my_r[e] = alpha * out_sign * out_mag;
            }
        }
        __syncthreads();

        // --- var-to-check update --- (identical math to ldpc_min_sum_decode above)
        for (int v = threadIdx.x; v < n; v += blockDim.x) {
            int deg = var_degree[v];
            const long long* edges = var_slot_edges + (size_t)v * max_var_degree;
            float total = my_llr[v];
            for (int j = 0; j < deg; ++j) {
                total += my_r[edges[j]];
            }
            for (int j = 0; j < deg; ++j) {
                long long e = edges[j];
                my_q[e] = total - my_r[e];
            }
        }
        __syncthreads();
    }

    // --- final hard decision --- (identical to ldpc_min_sum_decode above)
    for (int v = threadIdx.x; v < n; v += blockDim.x) {
        int deg = var_degree[v];
        const long long* edges = var_slot_edges + (size_t)v * max_var_degree;
        float total = my_llr[v];
        for (int j = 0; j < deg; ++j) {
            total += my_r[edges[j]];
        }
        my_out[v] = (total < 0.0f) ? 1 : 0;
    }
}
"""

_BLOCK_SIZE = 256  # threads/block -- a grid-stride loop inside the kernel handles
                    # n_checks/n both being larger (or smaller) than this


def _kernel_support_arrays(ldpc: LDPCCode):
    """Derives the few extra arrays this kernel needs (check_degree/
    var_degree as plain counts) from LDPCCode's OWN already-computed,
    already-trusted tables -- not re-deriving the edge/check/var
    structure from scratch, to keep this prototype's own surface area
    (and bug risk) as small as possible. Returns cupy int32 arrays."""
    import cupy

    check_degree = cupy.asnumpy(ldpc._check_slot_mask).sum(axis=-1).astype("int32")
    var_degree = cupy.asnumpy(ldpc._var_slot_mask).sum(axis=-1).astype("int32")
    return cupy.asarray(check_degree), cupy.asarray(var_degree)


def cuda_ldpc_decode(ldpc: LDPCCode, encoded_bits, p: float = 0.02, max_iterations=None):
    """Drop-in-shaped replacement for LDPCCode.decode() -- SAME
    batch-shape contract (n_batch, n) bits in -> (n_batch, k) bits out
    -- but via the single fused kernel above instead of ~400 chained
    array-op kernel launches. `ldpc` must be a backend="cupy"
    LDPCCode instance (its own precomputed edge tables are reused
    directly, not rebuilt). Does NOT check convergence/raise
    ValueError yet (see module docstring's "not yet done" list) --
    callers should verify the syndrome themselves for now (see
    verify_correctness() below for the pattern)."""
    import cupy

    if ldpc.backend != "cupy":
        raise ValueError("cuda_ldpc_decode requires a backend='cupy' LDPCCode instance")

    bits = cupy.asarray(encoded_bits)
    if bits.ndim == 1:
        bits = bits[None, :]
    if bits.shape[-1] != ldpc.n:
        raise ValueError(f"expected {ldpc.n} codeword bits, got {bits.shape[-1]}")
    n_batch = bits.shape[0]
    n_iterations = ldpc.max_iterations if max_iterations is None else max_iterations

    import math
    llr_scale = float(math.log((1 - p) / p))
    channel_llr = ((1 - 2 * bits.astype("float32")) * llr_scale).astype("float32")

    check_degree, var_degree = _kernel_support_arrays(ldpc)

    q_flat = cupy.empty((n_batch, ldpc.n_edges), dtype="float32")
    r_flat = cupy.empty((n_batch, ldpc.n_edges), dtype="float32")
    hard_bits = cupy.empty((n_batch, ldpc.n), dtype="uint8")

    kernel = cupy.RawKernel(_LDPC_MIN_SUM_KERNEL_SRC, "ldpc_min_sum_decode")
    kernel(
        (n_batch,), (_BLOCK_SIZE,),
        (
            channel_llr, q_flat, r_flat,
            ldpc._check_slot_edges.astype("int64"), check_degree,
            ldpc._var_slot_edges.astype("int64"), var_degree,
            ldpc._edge_owner_var.astype("int64"),
            hard_bits,
            np.int32(ldpc.n), np.int32(ldpc.n_edges), np.int32(ldpc.mb * ldpc.Z),
            np.int32(ldpc.max_check_degree), np.int32(ldpc.max_var_degree),
            np.int32(n_iterations), np.float32(_MIN_SUM_ALPHA),
        ),
    )
    return hard_bits[:, : ldpc.k]


def cuda_ldpc_decode_shared(ldpc: LDPCCode, encoded_bits, p: float = 0.02, max_iterations=None):
    """Same contract as cuda_ldpc_decode() above, via the shared-memory
    kernel instead -- see _LDPC_MIN_SUM_SHARED_KERNEL_SRC's own comment
    for the shared-memory budget per LDPC variant (ldpc_648_* is the
    safe one to try first: ~21KB, fits under every GPU's default 48KB
    limit with no special opt-in).

    Raises RuntimeError with a clear, actionable message (not a bare
    CUDA error) if this variant's shared-memory requirement exceeds
    what this specific GPU supports -- cupy's own
    max_dynamic_shared_size_bytes assignment is what surfaces that
    failure; caught and re-raised here with the actual numbers involved
    rather than a cryptic underlying CUDA error code."""
    import cupy

    if ldpc.backend != "cupy":
        raise ValueError("cuda_ldpc_decode_shared requires a backend='cupy' LDPCCode instance")

    bits = cupy.asarray(encoded_bits)
    if bits.ndim == 1:
        bits = bits[None, :]
    if bits.shape[-1] != ldpc.n:
        raise ValueError(f"expected {ldpc.n} codeword bits, got {bits.shape[-1]}")
    n_batch = bits.shape[0]
    n_iterations = ldpc.max_iterations if max_iterations is None else max_iterations

    import math
    llr_scale = float(math.log((1 - p) / p))
    channel_llr = ((1 - 2 * bits.astype("float32")) * llr_scale).astype("float32")

    check_degree, var_degree = _kernel_support_arrays(ldpc)
    hard_bits = cupy.empty((n_batch, ldpc.n), dtype="uint8")

    shared_bytes = (ldpc.n + 2 * ldpc.n_edges) * 4  # float32: channel_llr + q_flat + r_flat
    kernel = cupy.RawKernel(_LDPC_MIN_SUM_SHARED_KERNEL_SRC, "ldpc_min_sum_decode_shared")
    if shared_bytes > 48 * 1024:  # CUDA's default per-block limit -- above this NEEDS the opt-in below
        try:
            kernel.max_dynamic_shared_size_bytes = shared_bytes
        except Exception as exc:
            raise RuntimeError(
                f"variant {ldpc.variant!r} needs {shared_bytes} bytes ({shared_bytes/1024:.1f} KB) of "
                f"per-block shared memory, which this GPU could not provide (underlying error: {exc}). "
                f"Try a smaller variant instead -- e.g. ldpc_648_* needs only ~21KB, well under every "
                f"CUDA GPU's default 48KB limit."
            ) from exc

    kernel(
        (n_batch,), (_BLOCK_SIZE,),
        (
            channel_llr,
            ldpc._check_slot_edges.astype("int64"), check_degree,
            ldpc._var_slot_edges.astype("int64"), var_degree,
            ldpc._edge_owner_var.astype("int64"),
            hard_bits,
            np.int32(ldpc.n), np.int32(ldpc.n_edges), np.int32(ldpc.mb * ldpc.Z),
            np.int32(ldpc.max_check_degree), np.int32(ldpc.max_var_degree),
            np.int32(n_iterations), np.float32(_MIN_SUM_ALPHA),
        ),
        shared_mem=shared_bytes,
    )
    return hard_bits[:, : ldpc.k]


def verify_correctness(variant: str = "ldpc_1944_r12", decode_fn=None, label: str = "kernel") -> bool:
    """The real gate any of these kernels needs to clear before anyone
    trusts them. Encodes random messages with the EXISTING, trusted
    encode(), decodes with both the existing array-op decode() (ground
    truth) and `decode_fn` (defaults to the global-memory kernel,
    cuda_ldpc_decode -- pass cuda_ldpc_decode_shared to test the
    shared-memory one instead), for CLEAN codewords and for codewords
    with real injected bit errors (within the code's correction
    capability) -- same discipline fec/ldpc.py's own module docstring
    and fec/_native.py's Viterbi verification history both used: a
    single clean-codeword pass is not enough on its own."""
    if decode_fn is None:
        decode_fn = cuda_ldpc_decode
    if not cupy_available():
        print("No working CuPy/CUDA runtime detected -- run this on a GPU machine "
              "(e.g. Colab) instead. See this script's own module docstring.")
        return False

    numpy_ldpc = LDPCCode(variant, backend="numpy")
    cupy_ldpc = LDPCCode(variant, backend="cupy")
    rng = np.random.default_rng(0)
    all_ok = True

    print(f"--- verify_correctness: variant={variant!r}, {label} ---")
    for batch in [1, 4, 16]:
        for n_errors in [0, 1, 2]:  # 0 = clean; 1-2 = real injected BSC errors
            msg = rng.integers(0, 2, size=(batch, numpy_ldpc.k)).astype("uint8")
            encoded = np.asarray(numpy_ldpc.encode(msg))
            corrupted = encoded.copy()
            for b in range(batch):
                if n_errors:
                    flip = rng.choice(encoded.shape[-1], size=n_errors, replace=False)
                    corrupted[b, flip] ^= 1

            ref = np.asarray(numpy_ldpc.decode(corrupted, p=0.02))
            kernel_out = np.asarray(decode_fn(cupy_ldpc, corrupted, p=0.02).get())

            ok = np.array_equal(ref, msg) and np.array_equal(kernel_out, msg)
            all_ok &= ok
            print(f"batch={batch:3d} n_errors={n_errors}: ref_correct={np.array_equal(ref, msg)} "
                  f"kernel_correct={np.array_equal(kernel_out, msg)} "
                  f"kernel_matches_ref={np.array_equal(kernel_out, ref)} "
                  f"{'OK' if ok else 'MISMATCH !!'}")

    print(f"\n{'ALL PASSED' if all_ok else 'SOME FAILED -- do NOT trust/promote this kernel yet'} "
          f"({label}, variant={variant!r})")
    return all_ok


_SPEED_BATCH_SIZES = [1, 8, 32, 128, 512, 2048]
_N_ITERS = 10
_N_WARMUP = 3
_CPU_BASELINE_MBPS = 9.0  # this project's own measured rs_m8+conv_v27 CPU throughput,
                           # roughly -- see examples/benchmark_x86_stages_v3.py's own
                           # session history (typically ~8-11 Mbps) -- printed only as a
                           # reference point, not re-measured here.


def _sync() -> None:
    import cupy

    cupy.cuda.Stream.null.synchronize()


def _time_decode(decode_fn, ldpc, encoded) -> float:
    for _ in range(_N_WARMUP):
        decode_fn(ldpc, encoded, p=0.02) if decode_fn is not None else ldpc.decode(encoded, p=0.02)
    _sync()
    start = time.perf_counter()
    for _ in range(_N_ITERS):
        decode_fn(ldpc, encoded, p=0.02) if decode_fn is not None else ldpc.decode(encoded, p=0.02)
    _sync()
    return (time.perf_counter() - start) / _N_ITERS


def benchmark_speed(variant: str = "ldpc_1944_r12", extra_kernels=()) -> None:
    """Only meaningful to run AFTER verify_correctness() has passed for
    every kernel included -- a fast wrong answer is not a result. Same
    warmup + Stream.null.synchronize() protections as examples/
    benchmark_ldpc_cuda.py (this project's own established cupy-timing
    pattern), decoding a CLEAN codeword each call (same "best-case,
    matches the CPU-side methodology" reasoning documented in that
    script's own module docstring).

    `extra_kernels`: list of (label, decode_fn) pairs beyond the
    built-in "array-op" (LDPCCode.decode() itself) and "global-mem
    kernel" (cuda_ldpc_decode) columns -- pass
    [("shared-mem kernel", cuda_ldpc_decode_shared)] to add that
    comparison too. Any kernel that raises for this variant (e.g. the
    shared-memory one on a variant whose working set doesn't fit on
    this GPU) prints the error for that column and continues with the
    rest, rather than aborting the whole sweep."""
    if not cupy_available():
        print("No working CuPy/CUDA runtime detected -- run this on a GPU machine instead.")
        return
    import cupy

    columns = [("array-op", None), ("global-mem kernel", cuda_ldpc_decode), *extra_kernels]
    rng = np.random.default_rng(0)

    print(f"\n=== speed: {' vs '.join(label for label, _ in columns)}, variant={variant!r} ===")
    header = f"{'batch':>7} | " + " | ".join(f"{label + ' ms':>16}" for label, _ in columns)
    print(header)
    print("-" * len(header))
    for batch in _SPEED_BATCH_SIZES:
        ldpc = LDPCCode(variant, backend="cupy")
        msg = rng.integers(0, 2, size=(batch, ldpc.k)).astype("uint8")
        encoded = cupy.asarray(ldpc.encode(msg))
        total_bits = batch * ldpc.k

        times_ms = []
        for label, decode_fn in columns:
            try:
                # fresh LDPCCode instance per column -- no shared mutable state across kernels/columns
                col_ldpc = LDPCCode(variant, backend="cupy")
                t = _time_decode(decode_fn, col_ldpc, encoded)
                times_ms.append(f"{t*1000:>13.4f}")
            except Exception as exc:  # e.g. shared-memory kernel not fitting this GPU/variant
                times_ms.append("FAILED")
                print(f"  [{label} @ batch={batch}] {exc}")
        print(f"{batch:>7} | " + " | ".join(f"{t:>16}" for t in times_ms))

    print(f"\nFor reference, this project's own CPU rs_m8+conv_v27 (already-optimized, "
          f"real measured throughput this whole session): ~{_CPU_BASELINE_MBPS:.0f} Mbps.")


if __name__ == "__main__":
    global_ok = verify_correctness("ldpc_1944_r12", cuda_ldpc_decode, "global-mem kernel")

    # The shared-memory kernel: validate first against the SAFE variant
    # (~21KB, fits under every GPU's default 48KB limit, no opt-in
    # needed -- see _LDPC_MIN_SUM_SHARED_KERNEL_SRC's own comment) --
    # then attempt the real variant used throughout this project's own
    # benchmark history, which may or may not fit depending on this
    # specific GPU's shared memory budget.
    shared_ok_small = verify_correctness("ldpc_648_r12", cuda_ldpc_decode_shared, "shared-mem kernel")
    shared_ok_real = False
    if shared_ok_small:
        try:
            shared_ok_real = verify_correctness("ldpc_1944_r12", cuda_ldpc_decode_shared, "shared-mem kernel")
        except Exception as exc:
            print(f"\nshared-mem kernel on ldpc_1944_r12: FAILED to even launch -- {exc}")

    if global_ok:
        benchmark_speed("ldpc_1944_r12")
    if shared_ok_small:
        benchmark_speed("ldpc_648_r12", extra_kernels=[("shared-mem kernel", cuda_ldpc_decode_shared)])
    if shared_ok_real:
        benchmark_speed("ldpc_1944_r12", extra_kernels=[("shared-mem kernel", cuda_ldpc_decode_shared)])
    if not (global_ok or shared_ok_small):
        print("\nSkipping all speed benchmarks -- correctness failed, a fast wrong answer isn't useful.")
