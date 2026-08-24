"""Smoke test that every Phase 1 block actually composes into a working
chain -- bits -> QAM -> resource grid -> OfdmModulator -> OfdmDemodulator
-> grid extract -> equalize -> QAM demod -> bits -- before OfdmRx exists
to wire this declaratively (see docs/architecture.md).

Uses an identity channel (no distortion) since this test's job is to
catch plumbing/shape bugs across block boundaries, not re-verify
per-block algorithms already covered in their own unit tests.
"""
import numpy as np
import pytest

from spectracuda.channel import LSChannelEstimator
from spectracuda.equalizer import ZFEqualizer
from spectracuda.modem import Modem
from spectracuda.ofdm import OfdmDemodulator, OfdmModulator, ResourceGrid


@pytest.mark.parametrize("scheme", ["qpsk", "qam16", "qam64"])
def test_full_phase1_chain_identity_channel(scheme):
    rng = np.random.default_rng(42)
    fft_size, cp_len, n_batch = 64, 16, 5

    grid = ResourceGrid(fft_size=fft_size, n_data=48, n_pilot=8, dc_null=True)
    modem = Modem(scheme, backend="numpy")
    mod = OfdmModulator(fft_size, cp_len, backend="numpy")
    demod = OfdmDemodulator(fft_size, cp_len, backend="numpy")

    n_data_bits = grid.n_data * modem.bits_per_symbol
    tx_bits = rng.integers(0, 2, size=(n_batch, n_data_bits)).astype("uint8")
    tx_symbols = modem.modulate(tx_bits)  # (n_batch, n_data)

    tx_pilots = np.ones((grid.n_pilot,), dtype="complex64")
    tx_grid = grid.scatter(np, np.tile(tx_pilots, (n_batch, 1)), tx_symbols)

    time_domain = mod.process(tx_grid)
    # identity channel: no distortion, no noise
    rx_grid = demod.process(time_domain)

    rx_pilots = grid.extract_pilots(np, rx_grid)
    rx_data = grid.extract_data(np, rx_grid)

    est = LSChannelEstimator(grid.pilot_indices, grid.fft_size, tx_pilots, backend="numpy")
    h_hat = est.process(rx_pilots)
    h_hat_data = h_hat[:, grid.data_indices]

    eq = ZFEqualizer(backend="numpy")
    equalized = eq.process(rx_data, channel_est=h_hat_data)

    rx_bits = modem.demodulate(equalized)
    assert rx_bits.shape == tx_bits.shape
    np.testing.assert_array_equal(rx_bits, tx_bits)
