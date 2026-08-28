"""Dashboard: a thread-safe stdlib-`curses` front end shared by
air_unit.py and ground_unit.py in this same folder.

Why a shared module rather than duplicating curses code in both scripts:
same rationale as every other pair in this repo (drone_air_unit.py/
drone_ground_unit.py already share nothing directly, but pluto_air_unit.py/
pluto_ground_unit.py both import from the drone_*.py pair instead of
duplicating logic) -- one curses layout, used identically by both ends.

Deliberately stdlib-only (`curses`, in every CPython on Linux/macOS --
Windows needs `windows-curses`, not handled here, matching this repo's
existing Linux/Pi-first assumption elsewhere e.g. the NEON kernel work).
No new pip dependency beyond what drone_air_unit.py/drone_ground_unit.py
already need (pyzmq, see pyproject.toml's `examples` extra).

Thread-safety: air_unit.py/ground_unit.py run several background threads
(receive loop, quality-report loop, heartbeat watchdog, local-quality
poller) that all call into this dashboard concurrently -- log() and the
set_*() methods below all go through one `self._lock`, and the curses
draw call in run()'s loop takes the same lock for the whole redraw, so
no two threads ever write to the terminal at once. This is the exact
same category of bug drone_air_unit.py's own `push_lock` was written to
fix (see that file's _send_chunks() docstring) -- concurrent unlocked
writers corrupting shared, ordered state -- just here it's curses cells
on a terminal instead of bytes on a ZMQ socket.

Layout (redrawn every `refresh_s`, default 100ms):
    row 0:      title + BOUND/UNBOUND state
    rows 1-2:   local RX quality (this process's own mac.quality) +
                live RX data rate (DATA frames only, see stats.py)
    rows 3-4:   peer-reported quality (the peer's last LINK_QUALITY pdu)
    row 5:      live TX data rate (this process's own outgoing traffic)
                + current adaptive-MCS modem scheme, if enabled (see
                adaptive_mcs.py -- omitted entirely when disabled)
    row 6:      separator
    middle:     scrolling event/message log, most recent lines at the
                bottom, sized to whatever vertical space is left
    last row:   "> " + whatever's been typed so far (own input line)

Not a general-purpose curses widget toolkit -- one fixed layout, sized
to this one dashboard's fields. If a third field is ever needed, extend
_draw() directly rather than generalizing prematurely.
"""
from __future__ import annotations

import curses
import threading
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional

_ENTER_KEYS = (10, 13, curses.KEY_ENTER)
_BACKSPACE_KEYS = (curses.KEY_BACKSPACE, 127, 8)


def _fmt_quality(report: Optional[Dict[str, Any]]) -> str:
    if report is None:
        return "  (none received yet)"
    n_attempts = report["n_attempts"]
    n_delivered = report["n_delivered"]
    n_failed = n_attempts - n_delivered  # CRC-failed/undeliverable frames -- the "dropped" count
    ratio = report.get("delivered_ratio")
    if ratio is None:  # local mac.quality.report_dict() doesn't carry this key -- derive it
        ratio = n_delivered / n_attempts if n_attempts else 0.0
    return (
        f"  attempts={n_attempts} delivered={n_delivered} failed={n_failed} ({ratio:.1%}) "
        f"rssi={report['mean_rssi_db']:.1f}dB evm={report['mean_evm']:.4f}"
    )


def _fmt_rate(bps: float) -> str:
    if bps >= 1_000_000:
        return f"{bps / 1_000_000:.2f} Mbps"
    if bps >= 1_000:
        return f"{bps / 1_000:.2f} kbps"
    return f"{bps:.0f} bps"


class Dashboard:
    LOG_CAPACITY = 500  # scrollback kept in memory; only the tail that fits on screen is ever drawn

    def __init__(self, label: str) -> None:
        self.label = label
        self._lock = threading.Lock()
        self._log: Deque[str] = deque(maxlen=self.LOG_CAPACITY)
        self._bound = False
        self._local_quality: Optional[Dict[str, Any]] = None
        self._peer_quality: Optional[Dict[str, Any]] = None
        self._rx_rate_bps = 0.0  # DATA-frame-only goodput -- see stats.RateTracker's docstring for why LINK_QUALITY heartbeats are excluded
        self._tx_rate_bps = 0.0
        self._tx_scheme: Optional[str] = None  # current adaptive-MCS modem, or None if adaptive MCS is disabled
        self._quit = False

    # -- called from any thread --------------------------------------
    def log(self, *args: Any, **_ignored: Any) -> None:
        """Signature-compatible with builtin print(*args) (no sep/end
        support -- nothing in drone_air_unit.py/drone_ground_unit.py's
        reused code passes those). See air_unit.py/ground_unit.py's
        `air.print = dashboard.log` monkeypatch for why this needs to
        accept **_ignored rather than just *args: str.join fails loudly
        on an unexpected kwarg otherwise, and this is meant to be a
        drop-in print() replacement, not a stricter one."""
        line = " ".join(str(a) for a in args)
        with self._lock:
            self._log.append(line)

    def set_bound(self, bound: bool) -> None:
        with self._lock:
            self._bound = bound

    def set_local_quality(self, report: Dict[str, Any]) -> None:
        with self._lock:
            self._local_quality = report

    def set_peer_quality(self, report: Dict[str, Any]) -> None:
        with self._lock:
            self._peer_quality = report

    def set_rx_rate(self, bps: float) -> None:
        with self._lock:
            self._rx_rate_bps = bps

    def set_tx_rate(self, bps: float) -> None:
        with self._lock:
            self._tx_rate_bps = bps

    def set_tx_scheme(self, modem: str) -> None:
        with self._lock:
            self._tx_scheme = modem

    def request_quit(self) -> None:
        self._quit = True

    # -- main thread only ----------------------------------------------
    def run(self, on_submit: Callable[[str], None], refresh_s: float = 0.1) -> None:
        """Blocks until the user quits (Ctrl-C, which curses.wrapper lets
        propagate after it's already restored the terminal -- see module
        docstring) or request_quit() is called from another thread.
        Calls on_submit(line) once per Enter keypress with the composed
        input line (never empty -- blank Enter presses are swallowed)."""
        curses.wrapper(self._run, on_submit, refresh_s)

    def _run(self, stdscr: "curses._CursesWindow", on_submit: Callable[[str], None], refresh_s: float) -> None:
        curses.curs_set(1)
        stdscr.nodelay(True)
        stdscr.timeout(int(refresh_s * 1000))
        input_buf = ""
        while not self._quit:
            with self._lock:
                self._draw(stdscr, input_buf)
            key = stdscr.getch()
            if key == -1:
                continue
            if key in _ENTER_KEYS:
                if input_buf:
                    on_submit(input_buf)
                input_buf = ""
            elif key in _BACKSPACE_KEYS:
                input_buf = input_buf[:-1]
            elif 32 <= key <= 126:  # printable ASCII only -- no attempt at wide/unicode input editing here
                input_buf += chr(key)
            # anything else (arrow keys, KEY_RESIZE, ...) is ignored; the
            # next _draw() call re-reads stdscr.getmaxyx() itself, so a
            # terminal resize is picked up naturally on the next redraw

    def _draw(self, stdscr: "curses._CursesWindow", input_buf: str) -> None:
        """Caller must already hold self._lock. Uses addnstr (not addstr)
        everywhere so an over-long line is silently clipped to the
        terminal width instead of raising curses.error -- the one real
        curses gotcha this avoids (writing past the window's last cell
        otherwise raises, most commonly bottom-right corner writes)."""
        height, width = stdscr.getmaxyx()
        stdscr.erase()

        state = "BOUND" if self._bound else "UNBOUND"
        stdscr.addnstr(0, 0, f"spectracuda drone link -- {self.label} unit  [{state}]", width - 1, curses.A_BOLD)

        stdscr.addnstr(1, 0, "Local RX quality (frames received here):", width - 1)
        stdscr.addnstr(2, 0, _fmt_quality(self._local_quality) + f"   rx data rate={_fmt_rate(self._rx_rate_bps)}", width - 1)
        stdscr.addnstr(3, 0, "Peer-reported quality (peer's view of us):", width - 1)
        stdscr.addnstr(4, 0, _fmt_quality(self._peer_quality), width - 1)
        scheme_note = f"  mcs={self._tx_scheme}" if self._tx_scheme is not None else ""
        stdscr.addnstr(5, 0, f"Local TX: tx data rate={_fmt_rate(self._tx_rate_bps)}{scheme_note}", width - 1)
        stdscr.addnstr(6, 0, "-" * (width - 1), width - 1)

        log_top = 7
        log_bottom = height - 2  # leave the last row for the input line
        n_log_rows = max(0, log_bottom - log_top)
        tail = list(self._log)[-n_log_rows:] if n_log_rows else []
        for i, line in enumerate(tail):
            stdscr.addnstr(log_top + i, 0, line, width - 1)

        if height >= 1:
            try:
                stdscr.addnstr(height - 1, 0, f"> {input_buf}", width - 1)
            except curses.error:
                pass  # bottom-right cell write on some terminals -- purely cosmetic, safe to skip
        stdscr.refresh()
