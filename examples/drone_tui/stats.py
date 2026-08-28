"""RateTracker: sliding-window bits/sec throughput meter.

Why this exists separately from spectracuda.mac.quality.LinkQualityTracker
(which the dashboard's "attempts/delivered/failed" panels already read
straight from mac.quality): that tracker is a deliberate LIFETIME running
average (see its own docstring) -- exactly right for "how reliable has
this link been overall", wrong for "what's it doing right now". A live
Mbps number needs a windowed rate instead, which is a display-only
concern of this example, not something spectracuda's core Mac/quality
module needs to grow just for this dashboard.

record(n_bits) on every successfully-delivered DATA frame (see
air_unit.py/ground_unit.py's wiring -- RX side gates on pdu_type==
TYPE_DATA specifically, so LINK_QUALITY heartbeat traffic doesn't inflate
the "data rate" number; TX side's _send_message() only ever carries real
outgoing messages/traffic-gen payloads, never control frames, so no
gating is needed there). rate_bps() reports bits/sec averaged over the
trailing `window_s` seconds -- smooth enough to read, reacts to
/traffic start|stop within about one window.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Tuple


class RateTracker:
    def __init__(self, window_s: float = 2.0) -> None:
        self.window_s = window_s
        self._lock = threading.Lock()
        self._samples: Deque[Tuple[float, int]] = deque()  # (timestamp, n_bits), oldest first
        self.total_frames = 0
        self.total_bits = 0

    def record(self, n_bits: int) -> None:
        now = time.monotonic()
        with self._lock:
            self.total_frames += 1
            self.total_bits += n_bits
            self._samples.append((now, n_bits))
            self._prune(now)

    def rate_bps(self) -> float:
        """Bits/sec over the last window_s seconds -- always divides by
        the fixed window (not the actual span covered by samples so
        far), so this reads as a stable "recent rate" rather than
        spiking on the very first sample after a quiet period."""
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            bits = sum(b for _, b in self._samples)
        return bits / self.window_s

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()
