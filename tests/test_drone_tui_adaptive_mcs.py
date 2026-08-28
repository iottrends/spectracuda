"""examples/drone_tui/adaptive_mcs.py's McsController.

Two layers, matching how the user asked this to actually be exercised:

1. Pure state-machine tests against directly-constructed report dicts --
   fast, deterministic, no radio -- covering the delta/hysteresis/
   clamping logic in isolation.
2. A REAL end-to-end run: two actual Mac(mode="um", ofdm_kwargs=...)
   objects, the drone's own default PHY config, exchanging real
   LINK_QUALITY status PDUs as real IQ (Mac.build_quality_report()/
   handle_quality_report_iq(), the same primitives air_unit.py/
   ground_unit.py use) over an impaired spectracuda.sim.channel.Channel,
   round after round -- not a synthetic report dict fed straight to the
   controller. This is the "status messages exchanged in real time"
   case: LinkQualityTracker.observe() accumulates from genuine
   rx_process() outcomes, encode_quality_report()/decode_quality_report()
   do the real wire round-trip, and set_tx_scheme() is applied through
   the real Mac/Ofdm path -- proving the whole chain, not just the
   controller's arithmetic.

SNR thresholds below are measured, not guessed -- see adaptive_mcs.py's
own module docstring for the calibration sweep (same PHY_KWARGS as
drone_air_unit.py: fec=rs_m8, fec1=conv_v27, interleaver=block).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "drone_tui"))
from adaptive_mcs import McsController, MCS_TABLE  # noqa: E402

from spectracuda.mac import Mac  # noqa: E402
from spectracuda.sim.channel import Channel  # noqa: E402

# -- layer 1: pure state-machine tests -----------------------------------


def _report(n_attempts, n_delivered):
    return {"n_attempts": n_attempts, "n_delivered": n_delivered}


def test_first_report_only_establishes_a_baseline():
    c = McsController(start_index=1)
    assert c.on_quality_report(_report(100, 100)) is None
    assert c.index == 1


def test_bad_interval_steps_down_immediately_no_streak_needed():
    c = McsController(start_index=2)  # "qam16"
    c.on_quality_report(_report(100, 100))  # baseline
    result = c.on_quality_report(_report(120, 100))  # 20 new attempts, 0 delivered -> ratio 0.0
    assert result == "qpsk"
    assert c.index == 1


def test_already_at_floor_a_bad_interval_does_nothing():
    c = McsController(start_index=0)
    c.on_quality_report(_report(100, 100))
    assert c.on_quality_report(_report(120, 100)) is None
    assert c.index == 0


def test_up_step_requires_consecutive_good_intervals_not_just_one():
    c = McsController(start_index=0, up_streak_needed=3)
    c.on_quality_report(_report(0, 0))  # baseline
    assert c.on_quality_report(_report(20, 20)) is None  # streak=1
    assert c.on_quality_report(_report(40, 40)) is None  # streak=2
    result = c.on_quality_report(_report(60, 60))  # streak=3 -> climb
    assert result == "qpsk"
    assert c.index == 1


def test_hysteresis_band_neither_climbs_nor_falls_and_resets_streak():
    c = McsController(start_index=1, down_ratio=0.85, up_ratio=0.98, up_streak_needed=2)
    c.on_quality_report(_report(0, 0))
    c.on_quality_report(_report(20, 20))  # streak=1 (ratio 1.0)
    # a mid-band interval (0.90) resets the streak without moving the index
    assert c.on_quality_report(_report(40, 38)) is None
    assert c.index == 1
    # two more fully-good intervals are needed again from here, not just one
    assert c.on_quality_report(_report(60, 58)) is None
    assert c.on_quality_report(_report(80, 78)) == "qam16"


def test_top_of_table_a_good_interval_does_nothing_further():
    c = McsController(start_index=len(MCS_TABLE) - 1, up_streak_needed=1)
    c.on_quality_report(_report(0, 0))
    assert c.on_quality_report(_report(20, 20)) is None
    assert c.index == len(MCS_TABLE) - 1


def test_no_new_attempts_since_last_report_is_a_no_op():
    """Covers the saturated/clamped-counter case (see adaptive_mcs.py's
    docstring) as well as a genuinely quiet interval -- either way,
    d_attempts <= 0 must never be misread as a 0/0 "bad" ratio."""
    c = McsController(start_index=1)
    c.on_quality_report(_report(100, 100))
    assert c.on_quality_report(_report(100, 100)) is None
    assert c.index == 1


# -- layer 2: real Mac<->Mac LINK_QUALITY exchange over a real channel ---

_PHY_KWARGS = dict(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32,
    fec="rs_m8", fec1="conv_v27", interleaver="block", interleaver_kwargs={"unit_bits": 8},
    crc="crc16", sync="schmidl_cox", cfo="schmidl_cox", channel_estimator="ls", equalizer="mmse",
    n_training_symbols=2, backend="numpy",
)
_FRAMES_PER_ROUND = 30  # matches the calibration sweep in adaptive_mcs.py's docstring


def _bind(a: Mac, b: Mac) -> None:
    req = a.build_bind_request()
    resp = b.handle_bind_request_iq(req)
    assert a.handle_bind_response_iq(resp)
    assert a.bound and b.bound


def _run_one_round(sender: Mac, receiver: Mac, controller: McsController, channel: Channel, seed: int) -> None:
    """One LINK_QUALITY interval's worth of real traffic: `sender`
    transmits `_FRAMES_PER_ROUND` DATA frames through `channel` to
    `receiver` (feeding receiver.quality via the real receive_iq() path),
    then `receiver` reports what it observed back to `sender` as a real
    LINK_QUALITY PDU (clean/lossless return path -- only the forward
    direction is impaired here, deliberately, so the round only measures
    what it's meant to), and finally `sender` feeds that decoded report
    into `controller` and applies any resulting scheme change -- exactly
    the sequence air_unit.py/ground_unit.py's own _receive_loop would
    run on each arrived LINK_QUALITY pdu."""
    rng = np.random.default_rng(seed)
    for _ in range(_FRAMES_PER_ROUND):
        sdu = rng.integers(0, 2, size=800).astype("uint8")
        for pdu_iq in sender.send_iq(sdu):
            receiver.receive_iq(channel.process(pdu_iq))
    report_iq = receiver.build_quality_report()
    report = sender.handle_quality_report_iq(report_iq)
    changed = controller.on_quality_report(report)
    if changed is not None:
        sender.set_tx_scheme(modem=changed)


def test_real_time_status_exchange_falls_back_fast_and_climbs_slow():
    tx = Mac(mode="um", ofdm_kwargs=_PHY_KWARGS)
    rx = Mac(mode="um", ofdm_kwargs=_PHY_KWARGS)
    _bind(tx, rx)

    controller = McsController(start_index=len(MCS_TABLE) - 1)  # start at "qam64"
    assert controller.current_modem == "qam64"
    tx.set_tx_scheme(modem=controller.current_modem)  # align the real Ofdm with the controller's chosen start
    assert tx.ofdm.modem.scheme == "qam64"

    # Round 1: a channel that only qam64 can't survive (snr_db=14,
    # measured 2/30 for qam64 vs 27/30+ for everything else -- see
    # adaptive_mcs.py's calibration). One bad round is enough -- no
    # hysteresis on the way down.
    bad_channel = Channel(snr_db=14, seed=1, backend="numpy")
    _run_one_round(tx, rx, controller, bad_channel, seed=100)
    assert controller.current_modem == "qam16"
    assert tx.ofdm.modem.scheme == "qam16"  # the real Ofdm actually moved, not just the controller's own index

    # It should SETTLE there under continued marginal-for-the-top
    # conditions, not keep falling -- qam16 is well above down_ratio at
    # this SNR (measured 27/30 = 0.90 >= 0.85).
    _run_one_round(tx, rx, controller, bad_channel, seed=101)
    assert controller.current_modem == "qam16"

    # Round set 2: a clean channel (snr_db=30, measured 30/30 for every
    # level). Climbing needs up_streak_needed=3 consecutive good rounds
    # PER level -- run enough rounds to reach the top again, and confirm
    # it does not jump there in fewer rounds than that requires.
    good_channel = Channel(snr_db=30, seed=2, backend="numpy")
    for i in range(3):
        _run_one_round(tx, rx, controller, good_channel, seed=200 + i)
    assert controller.current_modem == "qam64"  # qam16 -> qam64 took exactly 3 good rounds

    # And a live, bit-exact DATA send at the new (climbed-back-to) scheme
    # actually round-trips -- proving the switch isn't just bookkeeping.
    sdu = np.random.default_rng(9).integers(0, 2, size=20000).astype("uint8")
    pdus = tx.send_iq(sdu)
    delivered = [out for pdu_iq in pdus for out in rx.receive_iq(pdu_iq)]
    assert len(delivered) == 1
    assert np.array_equal(delivered[0], sdu)


def test_real_time_status_exchange_never_falls_below_the_floor():
    tx = Mac(mode="um", ofdm_kwargs=_PHY_KWARGS)
    rx = Mac(mode="um", ofdm_kwargs=_PHY_KWARGS)
    _bind(tx, rx)

    controller = McsController(start_index=0)  # already "bpsk", the floor
    tx.set_tx_scheme(modem=controller.current_modem)  # align the real Ofdm with the controller's chosen start
    terrible_channel = Channel(snr_db=0, seed=3, backend="numpy")  # measured: total outage for every level
    for i in range(3):
        _run_one_round(tx, rx, controller, terrible_channel, seed=300 + i)
    assert controller.current_modem == "bpsk"
    assert tx.ofdm.modem.scheme == "bpsk"
