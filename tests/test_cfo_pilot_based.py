import numpy as np
import pytest

from spectracuda.cfo import PilotBasedCFO


def _build_repeated_symbols(fft_size, cp_len, pilot_indices, n_repeats, rng, eps=0.0, preamble_len=None):
    """Build a (preamble-shaped padding) + n_repeats identical known OFDM
    symbols (random QPSK content, fixed across repeats), with an
    optional CFO phase ramp applied across the WHOLE signal (matching
    how Ofdm/Channel apply CFO -- a single continuous phase ramp, not
    reset per symbol). preamble_len defaults to fft_size, matching
    process()'s own assumption that start_index points at a no-CP
    preamble exactly fft_size samples long (see class docstring)."""
    if preamble_len is None:
        preamble_len = fft_size
    n_sub = fft_size
    alphabet = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype="complex64") / np.sqrt(2)
    grid = alphabet[rng.integers(0, 4, size=n_sub)]
    time_symbol = np.fft.ifft(grid).astype("complex64")
    slot = np.concatenate([time_symbol[-cp_len:], time_symbol]) if cp_len > 0 else time_symbol

    preamble = np.zeros(preamble_len, dtype="complex64")
    full = np.concatenate([preamble] + [slot] * n_repeats).astype("complex64")
    if eps != 0.0:
        n = np.arange(full.shape[-1])
        full = (full * np.exp(1j * 2 * np.pi * eps * n / fft_size)).astype("complex64")
    return full, np.asarray(pilot_indices)


def test_requires_start_index():
    cfo = PilotBasedCFO(64, 16, pilot_indices=[1, 10, 30], tx_pilots=[1, 1, 1], n_repeats=2, backend="numpy")
    with pytest.raises(ValueError):
        cfo.process(np.zeros((1, 200), dtype="complex64"))


def test_n_repeats_below_2_raises():
    cfo = PilotBasedCFO(64, 16, pilot_indices=[1, 10, 30], tx_pilots=[1, 1, 1], n_repeats=1, backend="numpy")
    with pytest.raises(ValueError):
        cfo.process(np.zeros((1, 200), dtype="complex64"), start_index=np.array([0]))


def test_estimate_matches_known_cfo_no_noise():
    fft_size, cp_len = 64, 16
    pilot_indices = [3, 15, 40, 55]
    rng = np.random.default_rng(0)
    eps_true = 0.15
    rx, pidx = _build_repeated_symbols(fft_size, cp_len, pilot_indices, n_repeats=3, rng=rng, eps=eps_true)

    cfo = PilotBasedCFO(fft_size, cp_len, pilot_indices=pidx, tx_pilots=np.ones(len(pidx)),
                         n_repeats=3, backend="numpy")
    est = cfo.process(rx[None, :], start_index=np.array([0]))
    assert est[0] == pytest.approx(eps_true, abs=1e-6)


def test_estimate_zero_when_no_cfo():
    fft_size, cp_len = 64, 16
    pilot_indices = [3, 15, 40, 55]
    rng = np.random.default_rng(1)
    rx, pidx = _build_repeated_symbols(fft_size, cp_len, pilot_indices, n_repeats=2, rng=rng, eps=0.0)

    cfo = PilotBasedCFO(fft_size, cp_len, pilot_indices=pidx, tx_pilots=np.ones(len(pidx)),
                         n_repeats=2, backend="numpy")
    est = cfo.process(rx[None, :], start_index=np.array([0]))
    assert est[0] == pytest.approx(0.0, abs=1e-6)


def test_start_index_accounts_for_preamble_offset():
    """process() is given start_index (the PREAMBLE's start), not the
    first repeated symbol's start -- it must skip exactly fft_size
    samples (the no-CP preamble) itself before reading repeats."""
    fft_size, cp_len = 64, 16
    pilot_indices = [3, 15, 40, 55]
    rng = np.random.default_rng(2)
    eps_true = -0.2
    true_offset = 25
    rx, pidx = _build_repeated_symbols(
        fft_size, cp_len, pilot_indices, n_repeats=2, rng=rng, eps=eps_true, preamble_len=fft_size
    )
    rx_embedded = np.concatenate([np.zeros(true_offset, dtype="complex64"), rx])

    cfo = PilotBasedCFO(fft_size, cp_len, pilot_indices=pidx, tx_pilots=np.ones(len(pidx)),
                         n_repeats=2, backend="numpy")
    est = cfo.process(rx_embedded[None, :], start_index=np.array([true_offset]))
    assert est[0] == pytest.approx(eps_true, abs=1e-6)


def test_batch_with_different_cfo_per_item():
    fft_size, cp_len = 64, 16
    pilot_indices = [3, 15, 40, 55]
    eps_list = [0.1, -0.15, 0.0]
    rows = []
    for i, eps in enumerate(eps_list):
        rng = np.random.default_rng(i)
        rx, pidx = _build_repeated_symbols(fft_size, cp_len, pilot_indices, n_repeats=2, rng=rng, eps=eps)
        rows.append(rx)
    rx_batch = np.stack(rows, axis=0)

    cfo = PilotBasedCFO(fft_size, cp_len, pilot_indices=pidx, tx_pilots=np.ones(len(pidx)),
                         n_repeats=2, backend="numpy")
    est = cfo.process(rx_batch, start_index=np.zeros(3, dtype=int))
    np.testing.assert_allclose(est, eps_list, atol=1e-6)


def test_correct_undoes_the_offset():
    fft_size, cp_len = 64, 16
    pilot_indices = [3, 15, 40, 55]
    rng = np.random.default_rng(0)
    eps_true = 0.12
    clean, pidx = _build_repeated_symbols(fft_size, cp_len, pilot_indices, n_repeats=1, rng=rng, eps=0.0)
    n = np.arange(clean.shape[-1])
    with_cfo = (clean * np.exp(1j * 2 * np.pi * eps_true * n / fft_size)).astype("complex64")

    cfo = PilotBasedCFO(fft_size, cp_len, pilot_indices=pidx, tx_pilots=np.ones(len(pidx)),
                         n_repeats=2, backend="numpy")
    corrected = cfo.correct(with_cfo[None, :], np.array([eps_true]))
    np.testing.assert_allclose(corrected[0], clean, atol=1e-4)


def test_insufficient_samples_raises():
    fft_size, cp_len = 64, 16
    cfo = PilotBasedCFO(fft_size, cp_len, pilot_indices=[1, 5], tx_pilots=[1, 1], n_repeats=2, backend="numpy")
    with pytest.raises(ValueError):
        cfo.process(np.zeros((1, 50), dtype="complex64"), start_index=np.array([0]))
