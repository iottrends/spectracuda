"""McsController: adaptive modulation, driven by the peer's own
LINK_QUALITY reports (see spectracuda/mac/quality.py) -- the same
reports the dashboard's "peer-reported quality" panel already displays,
here also fed into a decision loop that calls Mac.set_tx_scheme() to
change what modem THIS side transmits with next.

Why modem-only, not modem+fec: Ofdm.reconfigure_tx_scheme()/
Mac.set_tx_scheme() (spectracuda/pipeline/ofdm.py, spectracuda/mac/
mac.py) support changing fec/fec1 too, but drone_air_unit.py/
drone_ground_unit.py's real PHY_KWARGS run a concatenated code
(fec="rs_m8", fec1="conv_v27") THROUGH a block interleaver -- and
Packetizer's constructor raises ValueError for interleaver != "none"
with fec1 == "none" (spectracuda/framing/packetizer.py). Any MCS table
level that reset fec1 to "none" would crash reconfigure_tx_scheme() the
moment it was selected, for a link actually running the drone's default
config -- verified directly, not assumed (an interactive check against
Ofdm(fec1="conv_v27", interleaver="block", ...).reconfigure_tx_scheme
(fec1="none") does raise exactly that). Varying modem alone sidesteps
this entirely: fec/fec1/interleaver stay whatever the link was started
with, for its whole life, and only mod_scheme changes frame to frame --
still a real, useful adaptive-MCS axis (modulation order is the dominant
throughput/robustness lever anyway), just a narrower one than a full
modem+fec table would be.

Calibration (why down_ratio=0.85/up_ratio=0.98/up_streak_needed=3):
measured against the drone's actual PHY_KWARGS (fft_size=256, n_pilot=8,
n_data=216, cp_len=32, fec=rs_m8, fec1=conv_v27, interleaver=block,
crc=crc16) through spectracuda.sim.channel.Channel at a sweep of AWGN
SNRs, 30 frames/point:

    snr_db=30: bpsk/qpsk/qam16/qam64 all 30/30 (both ends of the table
               fully reliable -- a "climb all the way up" channel)
    snr_db=14: bpsk 30/30, qpsk 27/30, qam16 27/30, qam64 2/30 -- kills
               ONLY the most aggressive level, everything else stays
               comfortably above 0.85
    snr_db=12: bpsk 28/30, qpsk 21/30, qam16 17/30, qam64 0/30

Not reproduced as a permanent script -- rerun the equivalent sweep
yourself (Ofdm+Channel, no Mac/threads needed) if these thresholds are
ever revisited; the point is they were measured against this exact PHY
config, not guessed.

CRITICAL: LinkQualityTracker (quality.py) is an explicit LIFETIME
running average -- n_attempts/n_delivered only ever grow, never reset
(same reasoning stats.py's RateTracker docstring gives for why it needs
its OWN windowed tracker rather than reading LinkQualityTracker
directly). A decoded LINK_QUALITY report's own `delivered_ratio` field
is therefore a lifetime ratio -- using it directly here would mean one
bad patch gets diluted into whatever history came before it, and could
take arbitrarily long to ever cross a down_ratio threshold on a
long-running link. McsController instead diffs CONSECUTIVE reports'
raw n_attempts/n_delivered itself (`_last_n_attempts`/
`_last_n_delivered`) to recover the ratio for just the interval SINCE
the last report -- the actual "how did my last ~LINK_QUALITY_INTERVAL_S
worth of frames do" signal adaptive MCS needs. One known consequence of
reusing this wire format as-is, not fixed here: encode_quality_report()
clamps n_attempts/n_delivered to 65535 (uint16) -- on a link running
long enough to saturate that (LINK_QUALITY_INTERVAL_S=0.1s -> ~1.8
hours of continuous reporting), the decoded counts stop advancing and
every subsequent delta reads as d_attempts=0 ("no signal", see
on_quality_report() below), silently freezing the current MCS instead
of erroring. A real limitation of the existing wire format, not
something this controller works around.

Asymmetric by design (matches real link-adaptation schemes like ARF/
Minstrel, not a from-scratch invention): step DOWN immediately on any
bad interval (one bad report is enough -- a stuck-too-aggressive link is
actively losing throughput to failed frames, so there's no reason to
wait), step UP only after `up_streak_needed` CONSECUTIVE good intervals
(climbing wrongly costs a burst of failures the down-step logic then has
to notice and unwind, so probing up is deliberately cautious). No
separate cooldown timer needed beyond that: `_good_streak` resets to 0
on every change (up OR down) and on every non-good interval, so the
up-streak requirement alone already prevents hunting right at a
boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

# Ordered least- to most-aggressive. Index 0 is also this controller's
# floor -- a bad report at index 0 has nowhere lower to fall back to and
# is simply left there (see on_quality_report()).
MCS_TABLE: Sequence[str] = ("bpsk", "qpsk", "qam16", "qam64")


@dataclass
class McsController:
    """One instance per outgoing direction (so, one per unit -- feed it
    the PEER's LINK_QUALITY reports about how THIS unit's own frames have
    been landing, see air_unit.py/ground_unit.py's wiring). Call
    on_quality_report() once per arrived LINK_QUALITY report, in arrival
    order -- this is a sequential diff against the previous call, not a
    pure function of one report in isolation."""

    table: Sequence[str] = MCS_TABLE
    start_index: int = 1  # "qpsk" -- matches drone_air_unit.py's own PHY_KWARGS default, not the floor
    down_ratio: float = 0.85
    up_ratio: float = 0.98
    up_streak_needed: int = 3

    def __post_init__(self) -> None:
        if not (0 <= self.start_index < len(self.table)):
            raise ValueError(f"start_index={self.start_index} out of range for a {len(self.table)}-entry table")
        self.index = self.start_index
        self._good_streak = 0
        self._last_n_attempts = 0
        self._last_n_delivered = 0

    @property
    def current_modem(self) -> str:
        return self.table[self.index]

    def on_quality_report(self, report: Dict[str, Any]) -> Optional[str]:
        """report: a quality.decode_quality_report()-shaped dict (needs
        n_attempts/n_delivered -- the raw lifetime counters, not just the
        derived delivered_ratio, see class docstring for why). Returns
        the new modem scheme name if this call changed self.index, else
        None (including the very first call, which only has a baseline
        to record, nothing yet to diff against)."""
        n_attempts = int(report["n_attempts"])
        n_delivered = int(report["n_delivered"])
        d_attempts = n_attempts - self._last_n_attempts
        d_delivered = n_delivered - self._last_n_delivered
        self._last_n_attempts = n_attempts
        self._last_n_delivered = n_delivered
        if d_attempts <= 0:
            # No new attempts since the last report (or a clamped/
            # non-advancing counter, see class docstring) -- no fresh
            # signal to act on either way, leave state untouched.
            return None
        ratio = d_delivered / d_attempts

        if ratio < self.down_ratio:
            self._good_streak = 0
            if self.index == 0:
                return None  # already the floor -- nowhere lower to fall back to
            self.index -= 1
            return self.current_modem

        if ratio >= self.up_ratio:
            self._good_streak += 1
            if self._good_streak >= self.up_streak_needed and self.index < len(self.table) - 1:
                self.index += 1
                self._good_streak = 0
                return self.current_modem
            return None

        self._good_streak = 0  # in the hysteresis band -- fine, but not confidently good enough to climb
        return None
