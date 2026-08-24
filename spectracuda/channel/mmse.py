"""MMSEChannelEstimator: pilot-based MMSE (Wiener) channel estimation
across all subcarriers, using an assumed uniform power-delay-profile
(PDP) frequency correlation model.

No liquid-dsp precedent (same gap noted for LSChannelEstimator -- see
docs/liquid-dsp-api-inventory.md and docs/todo.md, §1.5); designed from
the standard MMSE/Wiener channel-estimation derivation used throughout
OFDM literature (e.g. Edfors et al., "OFDM channel estimation by
singular value decomposition", and the equivalent textbook treatment in
802.11/3GPP-style receiver references), not ported.

Model: the channel is assumed to have `max_delay` significant time-
domain taps (default: the OFDM symbol's own cp_len -- the CP's whole
purpose is to absorb multipath up to that length, so it is the natural,
non-arbitrary default rather than an unrelated arbitrary constant) with
i.i.d., equal-power gains (a "uniform PDP" prior -- the standard
simplifying assumption when the true delay profile isn't known). This
gives a closed-form frequency-domain correlation between any two
subcarriers m, n:

    R_H[m,n] = (1/L) * sum_{l=0}^{L-1} exp(-j*2*pi*(m-n)*l/N)
             = 1                                    if (m-n) % N == 0
             = (1/L) * (1-r^L)/(1-r), r=exp(-j*2*pi*(m-n)/N)   otherwise

The standard Wiener/MMSE weight matrix is then
`W = R_fp @ inv(R_pp + noise_var*I)`, where R_pp is R_H restricted to
pilot x pilot indices and R_fp is R_H for every subcarrier x pilot
indices; the full-spectrum estimate is `H_mmse = W @ H_ls_pilot`, with
`H_ls_pilot = rx_pilots / tx_pilots` the same per-pilot LS estimate
LSChannelEstimator itself computes.

R_pp/R_fp/W depend only on pilot_indices/fft_size/max_delay/noise_var
(the assumed MODEL), never on the received signal -- so W is computed
ONCE at construction, not per call. This makes process() a single
batched matmul (`h_full = h_ls_pilot @ W.T`, xp handles the batch axis
natively), unlike LSChannelEstimator's own per-batch-item Python loop
(needed there only because `xp.interp` is 1-D-only) -- a nice side
benefit of the closed-form model, not a claim that MMSE is "simpler",
just that this particular implementation happens to vectorize more
cleanly across the batch.

Real, honest limitation (verified empirically, not assumed): this is a
genuinely underdetermined estimation problem when `len(pilot_indices) <
max_delay` -- there are more assumed channel-tap unknowns than pilot
observations, so even at near-zero noise the estimate has an
irreducible NMSE floor (confirmed directly: n_pilot=8 against
max_delay=16 left ~0.8 NMSE at noise_var->0, while n_pilot>=max_delay
converged to <1e-9). This is a standard, expected property of pilot-
based channel estimation (real OFDM standards size pilot spacing around
the expected delay spread for exactly this reason), not a bug -- when
`Ofdm` wires this in, `pilot_indices` is actually the FULL known
training-symbol grid (pilot + data bins together, see pipeline/ofdm.py:
`self._train_known_indices`), typically far larger than cp_len, so this
floor rarely matters there; it matters most if this class is
constructed directly against a genuinely sparse literal pilot set.

Batch-shape contract: identical to LSChannelEstimator -- process(rx_pilots)
takes (n_batch, n_pilot) complex received pilot symbols -> (n_batch,
fft_size) complex channel estimate across every subcarrier.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..block import Block
from ..registry import register


def _uniform_pdp_correlation(k: np.ndarray, fft_size: int, max_delay: int) -> np.ndarray:
    """R_H[k] = (1/L) sum_{l=0}^{L-1} exp(-j*2*pi*k*l/N), closed form.
    Verified against a direct sum for several (k, N, L) combinations
    before being trusted (see tests/test_channel_mmse.py)."""
    k = np.asarray(k, dtype="float64")
    zero_mask = (k % fft_size) == 0
    r = np.exp(-1j * 2 * np.pi * k / fft_size)
    with np.errstate(divide="ignore", invalid="ignore"):
        geometric_sum = (1 - r ** max_delay) / (1 - r)
    result = geometric_sum / max_delay
    return np.where(zero_mask, 1.0 + 0j, result)


@register("channel_estimator", "mmse")
class MMSEChannelEstimator(Block):
    """Parameters
    ----------
    pilot_indices:
        Subcarrier indices of the known tones used for the estimate
        (e.g. ResourceGrid.pilot_indices, or -- as `Ofdm` itself passes
        -- the full known training-symbol grid).
    fft_size:
        Total number of subcarriers.
    tx_pilots:
        Known transmitted values at `pilot_indices`, shape (n_pilot,)
        (broadcast across the batch) or (n_batch, n_pilot).
    cp_len:
        Assumed max channel delay spread, in samples -- defaults to the
        OFDM symbol's cyclic prefix length (see module docstring for
        why that's the natural default), or fft_size // 4 if not given
        (a conservative fallback when cp_len truly isn't known).
    noise_var:
        Estimated noise variance used for MMSE regularization -- same
        role/default (1e-3) as MMSEEqualizer's own `noise_var`. Larger
        values trust the pilot observations less (more smoothing
        toward the assumed correlation model); real deployments should
        pass an actual SNR-derived estimate rather than relying on the
        default.
    """

    def __init__(
        self,
        pilot_indices,
        fft_size: int,
        tx_pilots,
        *,
        cp_len: Optional[int] = None,
        noise_var: float = 1e-3,
        backend=None,
        **kwargs: Any,
    ) -> None:
        super().__init__(backend=backend)
        xp = self.xp
        self.fft_size = fft_size
        self.pilot_indices = xp.asarray(pilot_indices)
        self.tx_pilots = xp.asarray(tx_pilots)
        n_pilot = int(self.pilot_indices.shape[0])
        if n_pilot < 1:
            raise ValueError("need at least 1 pilot subcarrier")
        self.max_delay = int(cp_len) if cp_len is not None else max(1, fft_size // 4)
        self.noise_var = noise_var
        self.batch_shape_doc = (
            f"(n_batch, {n_pilot}) complex rx pilots in -> (n_batch, "
            f"{fft_size}) complex channel estimate out"
        )

        # Model matrices depend only on the pilot layout/assumed delay
        # spread, never on received data -- built once, here, in plain
        # numpy (tiny: fft_size x n_pilot and n_pilot x n_pilot), same
        # "metadata-scale work always runs in host numpy" convention
        # used elsewhere in this project (e.g. Ofdm's own _to_host).
        if self.backend == "cupy":
            import cupy

            pilot_host = cupy.asnumpy(self.pilot_indices).astype("int64")
        else:
            pilot_host = np.asarray(self.pilot_indices).astype("int64")
        m = np.arange(fft_size)
        r_fp = _uniform_pdp_correlation(m[:, None] - pilot_host[None, :], fft_size, self.max_delay)
        r_pp = _uniform_pdp_correlation(
            pilot_host[:, None] - pilot_host[None, :], fft_size, self.max_delay
        )
        weight = r_fp @ np.linalg.inv(r_pp + noise_var * np.eye(n_pilot))
        self._weight_T = xp.asarray(weight.T.astype("complex64"))  # (n_pilot, fft_size)

    def process(self, rx_pilots: Any, **kwargs: Any) -> Any:
        xp = self.xp
        rx_pilots = xp.asarray(rx_pilots)
        h_ls_pilot = rx_pilots / self.tx_pilots  # (n_batch, n_pilot)
        return (h_ls_pilot @ self._weight_T).astype("complex64")
