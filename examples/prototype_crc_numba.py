"""Prototype: Numba-JIT for CRC.generate_key() -- the new dominant TX
cost once Viterbi/RS encode were fixed by libcorrect (benchmark_x86_
stages_v2.py: CRC key generation, 7.79ms of an 11.27ms TX total).

Why NOT libcorrect or a generic external CRC library (anycrc) here,
unlike Viterbi/RS: checked empirically first, not assumed. anycrc
matches spectracuda's crc32 exactly out of the box (both are the
well-known IEEE-802.3/zlib CRC-32 -- see crc.py's own docstring), but
crc8/16/24 do NOT match under any refin/refout/init/xorout combination
tried. Root cause, found by reading crc.py itself rather than guessing
further: liquid-dsp's crc.c (and this port of it) runs EVERY scheme's
shift-and-XOR recursion through a fixed 32-bit register (`_REG_MASK =
0xFFFFFFFF`, unconditionally) and only masks down to the scheme's own
width at the very end -- crc8/16/24 are NOT computed as genuine 8/16/24
bit CRCs internally, despite the name. Standard CRC catalog libraries
(anycrc included) assume register width == output width, so they
structurally cannot reproduce this quirk. crc32 "worked" by accident:
32 == 32, so the quirk is invisible for that one scheme specifically.

Given that, JIT-compiling THIS project's own generate_key() (same
approach as the Viterbi/RS Numba prototypes before libcorrect
superseded them) is the only path guaranteed correct by construction --
it's the same code, just compiled, not a reimplementation that has to
be independently re-verified against an undocumented internal quirk.

Usage:
    python examples/prototype_crc_numba.py
"""
from __future__ import annotations

import time

import numba
import numpy as np

from spectracuda.fec.crc import CRC, _POLY_WIDTH_BYTES, _REG_MASK, _TABLES, _WIDTH_MASK

N_BYTES = 3000  # ~ a 24000-bit payload's worth of PDU bytes, matching benchmark_x86_stages_v2.py
N_ROUNDS = 30
N_WARMUP = 5


@numba.njit(cache=True)
def _crc_row_numba(msg_row: np.ndarray, table: np.ndarray, reg_mask: np.uint64, width_mask: np.uint64) -> np.uint64:
    key = reg_mask
    for i in range(msg_row.shape[0]):
        idx = (key ^ np.uint64(msg_row[i])) & np.uint64(0xFF)
        key = ((key >> np.uint64(8)) ^ table[idx]) & reg_mask
    return (~key) & reg_mask & width_mask


def _numba_generate_key(scheme: str, msg_bytes: np.ndarray) -> np.ndarray:
    """Same batch contract as CRC.generate_key(): (n_batch, n) uint8 -> (n_batch,)."""
    if scheme in ("none", "checksum"):
        raise NotImplementedError("only table-driven crc8/16/24/32 need this fix")
    table = _TABLES[scheme]
    width_mask = _WIDTH_MASK[scheme]
    n_batch = msg_bytes.shape[0]
    out = np.empty(n_batch, dtype=np.uint64)
    for b in range(n_batch):
        out[b] = _crc_row_numba(msg_bytes[b], table, _REG_MASK, width_mask)
    return out


def run() -> None:
    rng = np.random.default_rng(0)
    msg = rng.integers(0, 256, size=N_BYTES).astype("uint8")[None, :]

    print("=== Correctness (all 4 table-driven schemes) ===")
    all_ok = True
    for scheme in ("crc8", "crc16", "crc24", "crc32"):
        sc = CRC(scheme, backend="numpy")
        expected = sc.generate_key(msg)
        got = _numba_generate_key(scheme, msg)
        ok = np.array_equal(expected, got)
        all_ok &= ok
        print(f"  {scheme}: expected={hex(int(expected[0]))} numba={hex(int(got[0]))} match={ok}")

    if not all_ok:
        print("\nCorrectness FAILED -- refusing to report timing.")
        return

    print("\n=== Timing (crc16, matching the benchmark's default crc=) ===")
    sc = CRC("crc16", backend="numpy")
    for _ in range(N_WARMUP):
        sc.generate_key(msg)
        _numba_generate_key("crc16", msg)

    start = time.perf_counter()
    for _ in range(N_ROUNDS):
        sc.generate_key(msg)
    py_time = (time.perf_counter() - start) / N_ROUNDS

    start = time.perf_counter()
    for _ in range(N_ROUNDS):
        _numba_generate_key("crc16", msg)
    nb_time = (time.perf_counter() - start) / N_ROUNDS

    print(f"  current pure-numpy: {py_time * 1000:.4f} ms")
    print(f"  numba-jit:          {nb_time * 1000:.4f} ms")
    print(f"  speedup:            {py_time / nb_time:.1f}x")


if __name__ == "__main__":
    run()
