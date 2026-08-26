"""Optional Numba-JIT acceleration for CRC.generate_key()'s table-driven
byte loop (crc8/16/24/32 -- "none"/"checksum" are already vectorized
numpy one-liners, not this loop, see crc.py). Measured (x86, ~3000-byte
message, crc16): 8.2ms -> 0.007ms, ~1100x -- almost entirely Python-
interpreter-loop overhead, the same shape Numba fixes well throughout
this project's own history.

Not libcorrect/native.py's C-compile-and-cache story: numba is a real,
ordinary pip package (optional extra, see pyproject.toml's "fast"
extra), so "available" here just means "importable" -- no compiler, no
vendored source, no build-and-cache step of our own (numba does its own
JIT caching internally via cache=True). Activation is the same
transparent, silent-fallback pattern as _native.py: no new constructor
argument, CRC.generate_key()'s public contract is unchanged, and a
missing numba install (or any import failure) falls back to the
existing pure-NumPy path with no error.
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np

_lock = threading.Lock()
_checked = False
_njit = None  # the numba.njit decorator, once imported -- None if unavailable
_TABLE_DRIVEN = {"crc8", "crc16", "crc24", "crc32"}


def numba_available() -> bool:
    global _checked, _njit
    if _checked:
        return _njit is not None
    with _lock:
        if _checked:
            return _njit is not None
        _checked = True
        try:
            import numba

            _njit = numba.njit
        except Exception:
            _njit = None
    return _njit is not None


_crc_row_numba: Optional[object] = None  # compiled lazily, only once numba_available() is confirmed True


def _get_crc_row_fn():
    """Compiles the JIT kernel on first real use, not at import time --
    keeps `import spectracuda` cheap even when numba IS installed but
    this particular function is never called (e.g. crc="none")."""
    global _crc_row_numba
    if _crc_row_numba is None:
        import numba

        @numba.njit(cache=True)
        def _crc_row(msg_row: np.ndarray, table: np.ndarray, reg_mask: np.uint64, width_mask: np.uint64) -> np.uint64:
            key = reg_mask
            for i in range(msg_row.shape[0]):
                idx = (key ^ np.uint64(msg_row[i])) & np.uint64(0xFF)
                key = ((key >> np.uint64(8)) ^ table[idx]) & reg_mask
            return (~key) & reg_mask & width_mask

        _crc_row_numba = _crc_row
    return _crc_row_numba


def numba_generate_key(scheme: str, msg: np.ndarray, table: np.ndarray, reg_mask: np.uint64, width_mask: np.uint64) -> np.ndarray:
    """Same contract as CRC.generate_key()'s table-driven branch: msg
    (n_batch, n) uint8 -> (n_batch,) uint64 key. Caller (crc.py) is
    responsible for only calling this when scheme is table-driven and
    numba_available() is True."""
    fn = _get_crc_row_fn()
    n_batch = msg.shape[0]
    out = np.empty(n_batch, dtype=np.uint64)
    for b in range(n_batch):
        out[b] = fn(msg[b], table, reg_mask, width_mask)
    return out
