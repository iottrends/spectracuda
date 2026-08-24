"""Permanent pytest coverage for the 4-Mac/4-Ofdm bidirectional AM
picture -- see examples/mac_bidirectional_am_batch_demo.py and
examples/mac_bidirectional_am_streaming_demo.py for the narrated
versions (module docstrings there explain the model and the
STATUS-routing wrinkle in full); this file just asserts the same
scenario across a handful of seeds, for both the batch (rx_process())
and streaming (rx_streaming()) receive paths.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

import mac_bidirectional_am_batch_demo as batch_demo  # noqa: E402
import mac_bidirectional_am_streaming_demo as streaming_demo  # noqa: E402


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_batch_bidirectional_am_round_trip(seed):
    result = batch_demo.run(seed=seed, verbose=False)
    assert result["n_distinct_ofdm_objects"] == 4
    assert result["forward_pdus"] >= 3  # real segmentation, not a single-PDU no-op
    assert result["forward_retransmits"] == 1  # exactly the one deliberately dropped PDU
    assert result["forward_correct"]
    assert result["reverse_retransmits"] == 0  # clean direction -- nothing to resend
    assert result["reverse_correct"]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_streaming_bidirectional_am_round_trip(seed):
    result = streaming_demo.run(seed=seed, verbose=False)
    assert result["n_distinct_ofdm_objects"] == 4
    assert result["forward_pdus"] >= 3
    assert result["forward_retransmits"] == 1
    assert result["forward_correct"]
    assert result["reverse_retransmits"] == 0
    assert result["reverse_correct"]


def test_batch_and_streaming_agree():
    """Same seed, same scenario, two different receive mechanisms --
    both must reach the identical outcome (same PDU/retransmit counts),
    same cross-check spirit as test_ofdm_streaming.py's own comparison
    against rx_process()."""
    batch = batch_demo.run(seed=7, verbose=False)
    streaming = streaming_demo.run(seed=7, verbose=False)
    assert batch["forward_pdus"] == streaming["forward_pdus"]
    assert batch["forward_retransmits"] == streaming["forward_retransmits"]
    assert batch["forward_correct"] and streaming["forward_correct"]
    assert batch["reverse_correct"] and streaming["reverse_correct"]
