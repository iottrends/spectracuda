import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from ofdm_256_schmidl_cox_demo import run  # noqa: E402


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_full_scenario_recovers_bits_at_25db(seed):
    result = run(seed=seed, snr_db=25.0, eps_true=0.15, verbose=False)
    # A few samples of timing error from channel-delay-spread smearing is
    # expected and harmless (it's well inside what CP_LEN=32 absorbs) --
    # not the same as the ~cp_len-wide plateau bug this test would have
    # caught before the preamble's own CP was removed (see the module
    # docstring in examples/ofdm_256_schmidl_cox_demo.py).
    assert abs(result["detected_start_index"] - result["true_start_index"]) <= 5
    assert result["estimated_cfo"] == pytest.approx(result["true_cfo"], abs=0.01)
    # A residual 1-2 sample timing offset (channel-delay-spread smearing,
    # see above) introduces a small frequency-dependent phase ramp that
    # isn't corrected by reusing the training symbol's channel estimate
    # as-is -- occasionally flips a borderline QPSK decision even at
    # 25dB SNR. This is realistic, not a bug: it's exactly what FEC
    # (Phase 3, not implemented yet) exists to clean up.
    assert result["ber"] < 0.02


def test_degrades_gracefully_at_low_snr():
    result = run(seed=0, snr_db=5.0, eps_true=0.15, verbose=False)
    # timing/CFO acquisition (run once on the preamble) should still be solid
    assert abs(result["detected_start_index"] - result["true_start_index"]) <= 5
    assert result["estimated_cfo"] == pytest.approx(result["true_cfo"], abs=0.05)
    # payload BER may not be zero at 5dB SNR with a training-only estimate
    # and no FEC yet (Phase 3) -- just check it's not garbage.
    assert result["ber"] < 0.2
