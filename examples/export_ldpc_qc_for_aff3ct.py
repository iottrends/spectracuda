"""Export one of spectracuda's own 802.11n QC-LDPC base matrices (see
``spectracuda/fec/ldpc_tables.py``) into AFF3CT's native ".qc" file
format, so AFF3CT's decoder can be benchmarked against the EXACT same
code this project uses elsewhere (CPU array-op decode in fec/ldpc.py,
the hand-written CUDA kernel in prototype_ldpc_cuda_kernel.py) -- not
just "a" 802.11n-shaped code of the same size.

Not a shipped runtime dependency -- this is a one-off benchmarking tool,
same "reference, not shipped" status as reference/aff3ct/ itself (see
.gitignore).

## The two format gotchas, derived by reading AFF3CT's own parser
(``reference/aff3ct/src/Tools/Code/LDPC/QC/QC.cpp``, function
``QC::_read``), not assumed:

1. **Row/column layout is transposed relative to spectracuda's base
   matrix.** AFF3CT's file header is ``N_red M_red Z``, followed by
   M_red lines of N_red values each. Tracing `_read`'s index math
   (`idxLgn = i*Z` becomes the post-transpose COLUMN/variable
   dimension, `idxCol = j*Z` becomes the post-transpose ROW/check
   dimension) shows: N_red must equal spectracuda's `mb` (base-matrix
   row count, i.e. check-block-rows), M_red must equal spectracuda's
   `nb` (base-matrix col count = 24, variable-block-columns) -- and
   each FILE LINE corresponds to one variable-block-column, with the
   N_red values on that line being the shifts for each check-block-row
   in order. That's spectracuda's base matrix transposed.

2. **Shift sign is inverted.** spectracuda's convention (see
   ldpc_tables.py docstring + fec/ldpc.py's edge expansion):
   ``var = (check + k) % Z`` for base-matrix entry k. AFF3CT's parser
   builds edges as ``check = (var + value) % Z``. Solving both for the
   same (check, var) relation: value = (Z - k) % Z, i.e. the shift
   must be NEGATED mod Z, not copied as-is -- confirmed algebraically
   from _read's `add_connection(idxLgn+k, idxCol+(k+value)%Z)` line,
   not by trial and error.

Usage:
    python examples/export_ldpc_qc_for_aff3ct.py ldpc_1944_r12 out.qc
"""

from __future__ import annotations

import sys

from spectracuda.fec.ldpc_tables import BASE_MATRICES


def export_qc(variant: str, out_path: str) -> None:
    spec = BASE_MATRICES[variant]
    Z = spec["Z"]
    base = spec["base"]
    mb = len(base)  # check-block-rows
    nb = len(base[0])  # variable-block-cols (always 24 for 802.11n)

    # Header order + body layout verified empirically against the real
    # aff3ct-4.7.0 binary (not just QC::_read's raw connectivity): the
    # Codec/Decoder_LDPC factory reads the *first* header number as the
    # codeword size (N_cw) axis, which must be nb*Z = n. That means the
    # header is "nb mb Z" and the body is mb LINES (one per spectracuda
    # base-matrix row / check-block-row) of nb values each -- i.e.
    # spectracuda's base matrix transcribed row-for-row, unswapped.
    lines = [f"{nb} {mb} {Z}"]
    for j in range(mb):  # one file-line per check-block-row (as-is)
        row_vals = []
        for i in range(nb):  # each line lists all var-block-col shifts
            k = base[j][i]
            if k == -1:
                row_vals.append(-1)
            else:
                row_vals.append((Z - k) % Z)
        lines.append(" ".join(str(v) for v in row_vals))

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    n = spec["n"]
    n_checks = mb * Z
    k_bits = n - n_checks
    print(
        f"wrote {out_path}: variant={variant} Z={Z} mb={mb} nb={nb} "
        f"n={n} k={k_bits} rate={spec['rate']}"
    )


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "ldpc_1944_r12"
    out_path = sys.argv[2] if len(sys.argv) > 2 else f"{variant}.qc"
    export_qc(variant, out_path)
