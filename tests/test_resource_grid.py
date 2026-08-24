import numpy as np
import pytest

from spectracuda.ofdm import DATA, NULL, ResourceGrid


def test_allocation_partitions_all_subcarriers():
    grid = ResourceGrid(fft_size=64, n_data=48, n_pilot=8, dc_null=True)
    assert grid.n_pilot == 8
    assert grid.n_data == 48
    assert len(grid.null_indices) == 64 - 48 - 8
    assert set(grid.pilot_indices) | set(grid.data_indices) | set(grid.null_indices) == set(
        range(64)
    )


def test_exact_user_scenario_256_subcarriers():
    """256 subcarriers, 6 pilots, 200 data, rest (50) guard/null."""
    grid = ResourceGrid(fft_size=256, n_data=200, n_pilot=6, dc_null=True)
    assert grid.n_data == 200
    assert grid.n_pilot == 6
    assert len(grid.null_indices) == 50


def test_guard_band_is_centered_on_nyquist_not_array_edges():
    """Guard/null subcarriers belong around fft_size//2 (the Nyquist bin
    in unshifted FFT order), not at the array's literal start/end -- those
    sit right next to DC, not at the spectrum's edges."""
    grid = ResourceGrid(fft_size=64, n_data=40, n_pilot=4, dc_null=True)
    center = 64 // 2
    guard_only = grid.null_indices[grid.null_indices != 0]  # exclude the DC null
    assert guard_only.min() < center <= guard_only.max()
    # the literal array edges should NOT be nulled by this placement
    assert 63 not in grid.null_indices
    assert 1 not in grid.null_indices


def test_dc_null_can_be_disabled():
    grid = ResourceGrid(fft_size=64, n_data=48, n_pilot=8, dc_null=False)
    assert grid.sctype[0] != NULL
    assert len(grid.null_indices) == 64 - 48 - 8


def test_small_fft_size_raises():
    with pytest.raises(ValueError):
        ResourceGrid(fft_size=2, n_data=0, n_pilot=0)


def test_over_allocation_raises():
    with pytest.raises(ValueError):
        ResourceGrid(fft_size=64, n_data=60, n_pilot=8)  # 68 > 64


def test_extract_and_scatter_round_trip():
    grid = ResourceGrid(fft_size=32, n_data=24, n_pilot=4, dc_null=True)
    rng = np.random.default_rng(0)
    full = (rng.standard_normal((3, 32)) + 1j * rng.standard_normal((3, 32))).astype(
        "complex64"
    )
    # zero out null bins so scatter(extract(...)) round-trips exactly
    full[:, grid.null_indices] = 0
    pilots = grid.extract_pilots(np, full)
    data = grid.extract_data(np, full)
    assert pilots.shape == (3, grid.n_pilot)
    assert data.shape == (3, grid.n_data)
    rebuilt = grid.scatter(np, pilots, data)
    np.testing.assert_allclose(rebuilt, full)
