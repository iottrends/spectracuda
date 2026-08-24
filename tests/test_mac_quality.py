import numpy as np
import pytest

from spectracuda.mac.quality import LinkQualityTracker, decode_quality_report, encode_quality_report


def test_tracker_running_stats():
    t = LinkQualityTracker()
    t.observe(rssi_db=-10.0, evm=0.1, delivered=True)
    t.observe(rssi_db=-20.0, evm=0.3, delivered=False)
    assert t.n_attempts == 2
    assert t.n_delivered == 1
    assert t.delivered_ratio == pytest.approx(0.5)
    assert t.mean_rssi_db == pytest.approx(-15.0)
    assert t.mean_evm == pytest.approx(0.2)


def test_tracker_ignores_missing_evm_in_the_mean():
    """A lost frame (frame_found=False) has no evm at all -- must not
    silently count as evm=0 and drag the mean down."""
    t = LinkQualityTracker()
    t.observe(rssi_db=-10.0, evm=0.1, delivered=True)
    t.observe(rssi_db=-40.0, evm=None, delivered=False)
    t.observe(rssi_db=-12.0, evm=0.3, delivered=True)
    assert t.mean_evm == pytest.approx(0.2)  # mean of 0.1/0.3 only, not /3
    assert t.mean_rssi_db == pytest.approx((-10.0 - 40.0 - 12.0) / 3)  # rssi always counted


def test_tracker_empty_reports_zero_not_nan_or_crash():
    t = LinkQualityTracker()
    assert t.delivered_ratio == 0.0
    assert t.mean_rssi_db == 0.0
    assert t.mean_evm == 0.0
    d = t.report_dict()
    encode_quality_report(d)  # must not raise on an empty tracker


def test_report_round_trip_realistic_values():
    t = LinkQualityTracker()
    rng = np.random.default_rng(0)
    for _ in range(50):
        rssi = float(rng.uniform(-40, -5))
        delivered = bool(rng.random() > 0.2)
        evm = float(rng.uniform(0.01, 0.6)) if delivered else None
        t.observe(rssi_db=rssi, evm=evm, delivered=delivered)

    pdu = encode_quality_report(t.report_dict())
    d = decode_quality_report(pdu)
    assert d["n_attempts"] == t.n_attempts
    assert d["n_delivered"] == t.n_delivered
    assert d["delivered_ratio"] == pytest.approx(t.delivered_ratio)
    assert d["mean_rssi_db"] == pytest.approx(t.mean_rssi_db, abs=0.02)  # fixed-point rounding
    assert d["mean_evm"] == pytest.approx(t.mean_evm, abs=1e-3)


def test_report_round_trip_negative_rssi_near_zero_and_far_negative():
    for rssi in [-0.5, -1.0, -60.0, -100.0]:
        t = LinkQualityTracker()
        t.observe(rssi_db=rssi, evm=0.05, delivered=True)
        pdu = encode_quality_report(t.report_dict())
        d = decode_quality_report(pdu)
        assert d["mean_rssi_db"] == pytest.approx(rssi, abs=0.02)


def test_decode_quality_report_rejects_wrong_pdu_type():
    from spectracuda.mac.bind import encode_bind_request

    other_pdu = encode_bind_request("um", 100, 8, 4)
    with pytest.raises(ValueError, match="expected a LINK_QUALITY"):
        decode_quality_report(other_pdu)


def test_encode_quality_report_clamps_extreme_counts_rather_than_raising():
    t = LinkQualityTracker()
    t.n_attempts = 1_000_000
    t.n_delivered = 999_999
    t._rssi_sum = -20.0 * t.n_attempts
    t._evm_sum = 0.1 * t.n_attempts
    t._evm_count = t.n_attempts
    pdu = encode_quality_report(t.report_dict())  # must not raise
    d = decode_quality_report(pdu)
    assert d["n_attempts"] == 65535  # clamped to u16 max, not wrapped/corrupted
