"""Numba-JIT replacement for CRC.generate_key()'s table-driven byte loop
(crc8/16/24/32) -- see prototype_crc_numba.py's module docstring for why
this is Numba, not libcorrect or a generic external CRC library
(anycrc): liquid-dsp's crc.c runs every scheme through a fixed 32-bit
register regardless of the scheme's own width, a quirk standard CRC
catalog libraries can't reproduce, verified by direct comparison before
being ruled out. Bit-exact by construction here instead, since this is
the SAME code as crc.py's own generate_key(), just compiled.

"none"/"checksum" are NOT table-driven (see crc.py) and are left as the
original numpy path -- this only replaces the 4 real table-driven
schemes, which is what the byte-loop cost applies to.
"""
from __future__ import annotations

import numba
import numpy as np

from spectracuda.fec.crc import _REG_MASK, _TABLES, _WIDTH_MASK

_TABLE_DRIVEN = {"crc8", "crc16", "crc24", "crc32"}


@numba.njit(cache=True)
def _crc_row_numba(msg_row: np.ndarray, table: np.ndarray, reg_mask: np.uint64, width_mask: np.uint64) -> np.uint64:
    key = reg_mask
    for i in range(msg_row.shape[0]):
        idx = (key ^ np.uint64(msg_row[i])) & np.uint64(0xFF)
        key = ((key >> np.uint64(8)) ^ table[idx]) & reg_mask
    return (~key) & reg_mask & width_mask


def numba_generate_key(self, msg_bytes) -> np.ndarray:
    """Drop-in replacement for CRC.generate_key(self, msg_bytes) -- same
    (n_batch, n) uint8 -> (n_batch,) uint64 contract."""
    if self.scheme not in _TABLE_DRIVEN:
        return _ORIG_GENERATE_KEY(self, msg_bytes)
    msg = self._to_host_bytes(msg_bytes)
    table = _TABLES[self.scheme]
    width_mask = _WIDTH_MASK[self.scheme]
    n_batch = msg.shape[0]
    out = np.empty(n_batch, dtype=np.uint64)
    for b in range(n_batch):
        out[b] = _crc_row_numba(msg[b], table, _REG_MASK, width_mask)
    return out


from spectracuda.fec.crc import CRC  # noqa: E402

_ORIG_GENERATE_KEY = CRC.generate_key
