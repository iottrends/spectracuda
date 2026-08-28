"""TrafficGenerator: a background thread that sends synthetic payloads at
a controlled rate, driven by '/traffic' commands typed into either
unit's dashboard input line (see commands.py's dispatcher and air_unit.py/
ground_unit.py's wiring). Lets a link be stress-tested with a steady
stream of traffic instead of hand-typed one-off messages, while the
SAME local/peer LinkQualityTracker panels the dashboard already shows
respond to the added load in real time -- no separate stats UI needed
for "is traffic getting through", that's already what those panels are.

Payload content: `SEQ:{n:06d}|` followed by filler 'x' characters up to
--size bytes total. Deterministic, not random, so a human watching the
scrolling "received: ..." log can spot gaps/reordering at a glance.
A dedicated sequence-gap/loss detector was considered and skipped as
scope creep beyond what these dashboards already answer.

Pacing: fixed-period scheduling (`next_send += period` each iteration,
not `time.sleep(period)` after each send) so per-send jitter (encode +
ZMQ send time) doesn't accumulate into rate drift over a long run -- if
a send falls behind schedule, the next deadline is resynced to "now"
rather than firing a catch-up burst.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional


class TrafficGenerator:
    """One instance per unit (owned by CommandDispatcher), reused across
    start/stop cycles -- start() refuses to run a second generator
    concurrently rather than silently spawning two."""

    LOG_EVERY_N = 20

    def __init__(self, label: str, send_fn: Callable[[str], bool], log_fn: Callable[..., None]) -> None:
        self.label = label
        self._send_fn = send_fn  # -> True if actually sent (bound), False if dropped (not bound)
        self._log = log_fn
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.n_sent = 0
        self.n_dropped = 0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(
        self, rate_hz: float, size_bytes: int, count: Optional[int], duration_s: Optional[float]
    ) -> str:
        if self.running:
            return f"[{self.label}] traffic generator already running -- /traffic stop first"
        if rate_hz <= 0:
            return f"[{self.label}] rate must be > 0"
        if size_bytes < 8:
            return f"[{self.label}] size must be >= 8 bytes (needs room for the SEQ: prefix)"
        self.n_sent = 0
        self.n_dropped = 0
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, args=(rate_hz, size_bytes, count, duration_s), daemon=True
        )
        self._thread.start()
        limit = f"count={count}" if count is not None else (f"duration={duration_s}s" if duration_s else "unlimited")
        return f"[{self.label}] traffic generator started: rate={rate_hz}/s size={size_bytes}B {limit}"

    def stop(self) -> str:
        if not self.running:
            return f"[{self.label}] traffic generator is not running"
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        return f"[{self.label}] traffic generator stopped: {self.n_sent} sent, {self.n_dropped} dropped"

    def status(self) -> str:
        state = "running" if self.running else "stopped"
        return f"[{self.label}] traffic generator {state}: {self.n_sent} sent, {self.n_dropped} dropped so far"

    def _run(self, rate_hz: float, size_bytes: int, count: Optional[int], duration_s: Optional[float]) -> None:
        period = 1.0 / rate_hz
        deadline = time.monotonic() + duration_s if duration_s else None
        next_send = time.monotonic()
        i = 0
        while not self._stop_event.is_set():
            if count is not None and i >= count:
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
            prefix = f"SEQ:{i:06d}|".encode("ascii")
            payload = (prefix + b"x" * max(0, size_bytes - len(prefix)))[:size_bytes]
            if self._send_fn(payload.decode("ascii")):
                self.n_sent += 1
            else:
                self.n_dropped += 1
            i += 1
            if i % self.LOG_EVERY_N == 0:
                suffix = f"/{count}" if count is not None else ""
                dropped_note = f" ({self.n_dropped} dropped, not bound)" if self.n_dropped else ""
                self._log(f"[{self.label}] traffic: sent {self.n_sent}{suffix}{dropped_note}")
            next_send += period
            sleep_s = next_send - time.monotonic()
            if sleep_s > 0:
                self._stop_event.wait(sleep_s)
            else:
                next_send = time.monotonic()  # fell behind -- resync instead of a runaway catch-up burst
        self._log(f"[{self.label}] traffic generator finished: {self.n_sent} sent, {self.n_dropped} dropped")
