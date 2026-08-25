"""Prototype: Numba-JIT Reed-Solomon decode, same experiment shape as
prototype_viterbi_numba.py -- RS decode is the new RX-chain bottleneck
once Viterbi got its own Numba fix (benchmark_x86_stages.py: RS ~6-9ms
vs Viterbi's ~1-2ms). ReedSolomonCode's own docstring already flags WHY:
Berlekamp-Massey/Chien search/the error-magnitude linear solve are all
plain Python loops over GF(256) lookup-table arithmetic -- the same
"per-iteration interpreter overhead over many small steps" shape as
Viterbi's trellis recursion, just with a heavier per-codeword algorithm
(syndromes + BM + Chien + Gaussian elimination, not one add-compare-
select step).

This reuses ReedSolomonCode's own GF(256) EXP/LOG tables directly
(imported, not rebuilt) so both implementations are guaranteed to agree
on the exact same field arithmetic, not two independently-built (and
possibly subtly different) ones -- same rationale as the Viterbi
prototype reusing ConvolutionalCode's own trellis tables.

Correctness is checked against REAL encoded codewords with actual
injected symbol errors (0, 1, 5, 10, 16 -- the exact counts
reed_solomon.py's own docstring says were verified for the original
implementation), not just a clean round trip -- decode's whole point is
correcting errors, so that's what has to be checked, matching the 16
(t_max) case specifically since that's where a subtle off-by-one in a
ported algorithm is most likely to surface.

Requires numba (dev-only prototype tool):
    pip install numba

Usage:
    python examples/prototype_rs_numba.py
"""
from __future__ import annotations

import time

import numba
import numpy as np

from spectracuda.fec.reed_solomon import _EXP, _LOG, ReedSolomonCode

_N = 255
_K = 223
_NROOTS = 32
_FCR = 1
_PRIM = 1

K_BITS_SYMBOLS = _K  # full-length codeword, matches the "unchanged original behavior" case
N_ROUNDS = 30
N_WARMUP = 5


@numba.njit(cache=True)
def _gf_mul_nb(a, b, EXP, LOG):
    if a == 0 or b == 0:
        return 0
    return EXP[LOG[a] + LOG[b]]


@numba.njit(cache=True)
def _gf_inv_nb(a, EXP, LOG):
    return EXP[255 - LOG[a]]


@numba.njit(cache=True)
def _gf_pow_nb(a, n, EXP, LOG):
    if a == 0:
        return 0
    e = (LOG[a] * (n % 255)) % 255
    return EXP[e]


@numba.njit(cache=True)
def _decode_one_nb(codeword_row, EXP, LOG):
    """Same algorithm as ReedSolomonCode._decode_one() exactly (syndromes
    -> Berlekamp-Massey -> Chien search -> direct GF(256) linear solve
    for error magnitudes, see that class's module docstring for why a
    linear solve replaces the textbook Forney shortcut here) -- fixed-
    size numpy arrays instead of Python lists (Numba's natural style),
    a status code returned instead of raising ValueError directly
    (Numba's nopython exceptions don't support the dynamic f-string
    messages the original raises -- the caller raises the real,
    informative ValueError instead, see _numba_decode() below)."""
    N, K, NROOTS, FCR, PRIM = _N, _K, _NROOTS, _FCR, _PRIM

    synd = np.zeros(NROOTS, dtype=np.int64)
    any_nonzero = False
    for i in range(NROOTS):
        root = _gf_pow_nb(2, FCR + i * PRIM, EXP, LOG)
        result = 0
        for c in range(N):
            result = _gf_mul_nb(result, root, EXP, LOG) ^ codeword_row[c]
        synd[i] = result
        if result != 0:
            any_nonzero = True

    if not any_nonzero:
        return codeword_row[:K].copy(), 0  # no errors detected

    # -- Berlekamp-Massey --
    C = np.zeros(NROOTS + 1, dtype=np.int64)
    B = np.zeros(NROOTS + 1, dtype=np.int64)
    C[0] = 1
    B[0] = 1
    L = 0
    m = 1
    b = 1
    for n in range(NROOTS):
        delta = synd[n]
        for i in range(1, L + 1):
            delta ^= _gf_mul_nb(C[i], synd[n - i], EXP, LOG)
        if delta == 0:
            m += 1
        elif 2 * L <= n:
            T = C.copy()
            coef = _gf_mul_nb(delta, _gf_inv_nb(b, EXP, LOG), EXP, LOG)
            for i in range(NROOTS + 1):
                if i + m < NROOTS + 1:
                    C[i + m] ^= _gf_mul_nb(coef, B[i], EXP, LOG)
            L, B, b, m = n + 1 - L, T, delta, 1
        else:
            coef = _gf_mul_nb(delta, _gf_inv_nb(b, EXP, LOG), EXP, LOG)
            for i in range(NROOTS + 1):
                if i + m < NROOTS + 1:
                    C[i + m] ^= _gf_mul_nb(coef, B[i], EXP, LOG)
            m += 1

    # -- Chien search --
    found_pos = np.zeros(NROOTS, dtype=np.int64)
    found_xl = np.zeros(NROOTS, dtype=np.int64)
    n_found = 0
    for j in range(N):
        root_test = _gf_pow_nb(2, -j, EXP, LOG)
        val = 0
        for deg in range(L + 1):
            val ^= _gf_mul_nb(C[deg], _gf_pow_nb(root_test, deg, EXP, LOG), EXP, LOG)
        if val == 0:
            if n_found < NROOTS:
                found_pos[n_found] = N - 1 - j
                found_xl[n_found] = _gf_pow_nb(2, j, EXP, LOG)
            n_found += 1

    if n_found != L:
        return codeword_row[:K].copy(), 1  # uncorrectable -- degree/root-count mismatch

    # -- direct GF(256) linear solve for error magnitudes (Gaussian elimination) --
    A = np.zeros((L, L), dtype=np.int64)
    rhs = np.zeros(L, dtype=np.int64)
    for i in range(L):
        for col in range(L):
            A[i, col] = _gf_pow_nb(found_xl[col], FCR + i, EXP, LOG)
        rhs[i] = synd[i]

    for col in range(L):
        pivot = -1
        for r in range(col, L):
            if A[r, col] != 0:
                pivot = r
                break
        if pivot == -1:
            return codeword_row[:K].copy(), 2  # singular -- more errors than correctable
        for c in range(L):
            A[col, c], A[pivot, c] = A[pivot, c], A[col, c]
        rhs[col], rhs[pivot] = rhs[pivot], rhs[col]
        inv = _gf_inv_nb(A[col, col], EXP, LOG)
        for c in range(L):
            A[col, c] = _gf_mul_nb(A[col, c], inv, EXP, LOG)
        rhs[col] = _gf_mul_nb(rhs[col], inv, EXP, LOG)
        for r in range(L):
            if r != col and A[r, col] != 0:
                factor = A[r, col]
                for c in range(L):
                    A[r, c] ^= _gf_mul_nb(factor, A[col, c], EXP, LOG)
                rhs[r] ^= _gf_mul_nb(factor, rhs[col], EXP, LOG)

    corrected = codeword_row.copy()
    for i in range(L):
        corrected[found_pos[i]] ^= rhs[i]
    return corrected[:K].copy(), 0


_STATUS_MESSAGES = {
    1: "RS decode failed: found error-location count doesn't match the "
       "error-locator degree -- more than t=16 symbol errors, uncorrectable",
    2: "RS decode failed: singular error-location matrix (more errors "
       "than the code can correct)",
}


def _numba_decode(codeword: np.ndarray) -> np.ndarray:
    """Batch wrapper, same shape contract as ReedSolomonCode.decode()
    for the full-length (real_k == K) case."""
    codeword = np.asarray(codeword, dtype="uint8")
    if codeword.ndim == 1:
        codeword = codeword[None, :]
    n_batch = codeword.shape[0]
    decoded = np.empty((n_batch, _K), dtype="uint8")
    for b in range(n_batch):
        result, status = _decode_one_nb(codeword[b], _EXP, _LOG)
        if status != 0:
            raise ValueError(_STATUS_MESSAGES[status])
        decoded[b] = result
    return decoded


def _inject_errors(codeword: np.ndarray, n_errors: int, rng: np.random.Generator) -> np.ndarray:
    corrupted = codeword.copy()
    positions = rng.choice(corrupted.shape[-1], size=n_errors, replace=False)
    for pos in positions:
        # flip to a different, deterministic-random symbol value -- never
        # the no-op "corrupt to the same value" case
        delta = rng.integers(1, 256)
        corrupted[..., pos] = (int(corrupted[..., pos]) + delta) % 256
    return corrupted


def run() -> None:
    code = ReedSolomonCode(backend="numpy")
    rng = np.random.default_rng(0)
    msg = rng.integers(0, 256, size=(1, K_BITS_SYMBOLS)).astype("uint8")
    codeword = code.encode(msg)

    print("correctness (real encoded codeword, injected errors):")
    all_ok = True
    for n_errors in [0, 1, 5, 10, 16]:
        corrupted = _inject_errors(codeword[0], n_errors, np.random.default_rng(100 + n_errors))[None, :]
        numpy_decoded = code.decode(corrupted)
        numba_decoded = _numba_decode(corrupted)
        numpy_ok = np.array_equal(numpy_decoded, msg)
        numba_ok = np.array_equal(numba_decoded, msg)
        agree = np.array_equal(numpy_decoded, numba_decoded)
        print(f"  {n_errors:>2} errors: numpy correct={numpy_ok}  numba correct={numba_ok}  agree={agree}")
        all_ok = all_ok and numpy_ok and numba_ok and agree

    # the over-capacity case: both must raise, not silently mis-decode
    over_capacity = _inject_errors(codeword[0], 17, np.random.default_rng(999))[None, :]
    numpy_raised = numba_raised = False
    try:
        code.decode(over_capacity)
    except ValueError:
        numpy_raised = True
    try:
        _numba_decode(over_capacity)
    except ValueError:
        numba_raised = True
    print(f"  17 errors (over capacity): numpy raises={numpy_raised}  numba raises={numba_raised}")
    all_ok = all_ok and numpy_raised and numba_raised

    if not all_ok:
        print("\nCORRECTNESS FAILURE -- refusing to trust timing numbers below.")
        return

    # 0 errors takes the early-exit fast path (syndrome computation only
    # -- the "all syndromes zero" check) and never runs Berlekamp-Massey/
    # Chien search/the linear solve at all. 16 errors (the maximum this
    # code corrects) exercises the FULL pipeline -- the realistic case a
    # genuinely noisy channel would actually hit, not just the clean-
    # channel shortcut -- so both are worth reporting, not just one.
    error_scenarios = {"0 errors (fast path)": 0, "16 errors (full pipeline, worst case)": 16}
    for label, n_errors in error_scenarios.items():
        test_codeword = (
            codeword if n_errors == 0
            else _inject_errors(codeword[0], n_errors, np.random.default_rng(100 + n_errors))[None, :]
        )

        for _ in range(N_WARMUP):
            code.decode(test_codeword)
            _numba_decode(test_codeword)

        start = time.perf_counter()
        for _ in range(N_ROUNDS):
            code.decode(test_codeword)
        numpy_time = (time.perf_counter() - start) / N_ROUNDS

        start = time.perf_counter()
        for _ in range(N_ROUNDS):
            _numba_decode(test_codeword)
        numba_time = (time.perf_counter() - start) / N_ROUNDS

        print(f"\nk={K_BITS_SYMBOLS} symbols, {N_ROUNDS} rounds each (after {N_WARMUP} warm-up rounds), {label}:")
        print(f"  current pure-python decode: {numpy_time * 1000:.4f} ms")
        print(f"  numba-jit decode:           {numba_time * 1000:.4f} ms")
        print(f"  speedup:                    {numpy_time / numba_time:.1f}x")


if __name__ == "__main__":
    run()
