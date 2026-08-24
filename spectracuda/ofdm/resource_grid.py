"""ResourceGrid: subcarrier allocation (null/pilot/data) and pilot/data
extraction. Layer 1 fixed infrastructure -- configured by params, not
registry-driven (see docs/architecture.md).

Conceptually mirrors liquid-dsp's ofdmframe_init_default_sctype /
ofdmframe_init_sctype_range default allocation (guard-band nulls, a
DC null, pilots at regular spacing) -- see
docs/liquid-dsp-api-inventory.md -- but takes exact desired counts
(n_data, n_pilot) rather than liquid-dsp's fixed internal heuristic.

Subcarrier index 0 is DC (unshifted FFT bin order, matching
OfdmModulator/OfdmDemodulator). This matters for where guard bands go:
in this ordering the highest +/- frequencies (the Nyquist edge, where
real guard bands belong) sit at indices around fft_size // 2, NOT at
the array's literal start/end (index 0/fft_size-1 sit right next to DC).
An earlier version of this file nulled the array edges instead -- fixed
here.
"""
from __future__ import annotations

import numpy as np

NULL, PILOT, DATA = 0, 1, 2


class ResourceGrid:
    """Subcarrier allocation for one OFDM symbol (shared metadata across
    a batch -- this class holds plain numpy index arrays regardless of
    backend; callers convert to the active xp as needed).

    Parameters
    ----------
    fft_size:
        Number of subcarriers (FFT size), M.
    n_data:
        Exact number of data subcarriers.
    n_pilot:
        Exact number of pilot subcarriers.
    dc_null:
        Whether to null the DC (bin 0) subcarrier (standard OFDM
        practice; counts as one of the null subcarriers below).

    Everything not assigned to data or pilot becomes a null/guard
    subcarrier: `n_null = fft_size - n_data - n_pilot`. The guard band
    (n_null, minus 1 if dc_null) is placed as one contiguous block
    centered on the Nyquist bin (index fft_size // 2), and the remaining
    n_pilot pilots are spread evenly across what's left.
    """

    def __init__(
        self,
        fft_size: int,
        n_data: int,
        n_pilot: int,
        dc_null: bool = True,
    ) -> None:
        if fft_size < 4:
            raise ValueError("fft_size must be >= 4")
        n_used = n_data + n_pilot
        if n_used > fft_size:
            raise ValueError(
                f"n_data + n_pilot ({n_used}) exceeds fft_size ({fft_size})"
            )
        n_null_total = fft_size - n_used
        dc_cost = 1 if dc_null else 0
        n_guard = n_null_total - dc_cost
        if n_guard < 0:
            raise ValueError(
                "dc_null=True needs at least one null subcarrier; "
                "reduce n_data/n_pilot or set dc_null=False"
            )

        self.fft_size = fft_size
        sctype = np.full(fft_size, DATA, dtype=np.uint8)

        if dc_null:
            sctype[0] = NULL

        if n_guard > 0:
            center = fft_size // 2
            lo = center - n_guard // 2
            hi = lo + n_guard  # exclusive
            sctype[lo:hi] = NULL

        eligible = np.where(sctype == DATA)[0]
        if len(eligible) < n_pilot:
            raise ValueError(
                f"only {len(eligible)} subcarriers left for {n_pilot} pilots "
                f"after DC/guard nulling"
            )
        if n_pilot > 0:
            positions = np.linspace(0, len(eligible) - 1, n_pilot).round().astype(int)
            pilot_bins = eligible[np.unique(positions)]
            sctype[pilot_bins] = PILOT

        self.sctype = sctype
        self.null_indices = np.where(sctype == NULL)[0]
        self.pilot_indices = np.where(sctype == PILOT)[0]
        self.data_indices = np.where(sctype == DATA)[0]

        if len(self.pilot_indices) != n_pilot:
            raise ValueError(
                f"requested n_pilot={n_pilot} but linspace placement only "
                f"produced {len(self.pilot_indices)} distinct pilot bins "
                f"(too many pilots for the eligible pool) -- request fewer "
                f"pilots or a larger fft_size"
            )

    @property
    def n_pilot(self) -> int:
        return len(self.pilot_indices)

    @property
    def n_data(self) -> int:
        return len(self.data_indices)

    def extract_pilots(self, xp, grid_batch):
        """(n_batch, fft_size) -> (n_batch, n_pilot)."""
        return grid_batch[:, xp.asarray(self.pilot_indices)]

    def extract_data(self, xp, grid_batch):
        """(n_batch, fft_size) -> (n_batch, n_data)."""
        return grid_batch[:, xp.asarray(self.data_indices)]

    def scatter(self, xp, pilot_values, data_values):
        """Inverse of extract_pilots/extract_data: build a full
        (n_batch, fft_size) grid from separate pilot/data arrays (null
        subcarriers are zero)."""
        n_batch = data_values.shape[0]
        grid = xp.zeros((n_batch, self.fft_size), dtype=data_values.dtype)
        grid[:, xp.asarray(self.pilot_indices)] = pilot_values
        grid[:, xp.asarray(self.data_indices)] = data_values
        return grid
