"""CommandDispatcher: parses '/'-prefixed lines typed into either unit's
dashboard input line and routes them to a handler. Anything NOT starting
with '/' is not a command at all -- see air_unit.py/ground_unit.py's
on_submit, where a plain line is still just an outbound message, exactly
as it was before this feature existed. Shared by both units (like
dashboard.py) so the command grammar/parsing exists in exactly one place.

Currently one command group: /traffic, wrapping traffic.py's
TrafficGenerator. Structured so a second command group would just be
another `elif cmd == "...":` branch in dispatch(), not a redesign.
"""
from __future__ import annotations

import shlex
from typing import Callable, Dict, List

from traffic import TrafficGenerator

HELP_TEXT = (
    "commands: /traffic start [rate=N] [size=N] [count=N] [duration=Nsec]  |  "
    "/traffic stop  |  /traffic status  |  /help  -- anything else is sent as a message"
)


def _parse_kv(tokens: List[str]) -> Dict[str, str]:
    kv: Dict[str, str] = {}
    for tok in tokens:
        if "=" not in tok:
            raise ValueError(f"expected key=value, got {tok!r}")
        k, v = tok.split("=", 1)
        kv[k] = v
    return kv


class CommandDispatcher:
    def __init__(self, label: str, send_fn: Callable[[str], bool], log_fn: Callable[..., None]) -> None:
        self.label = label
        self._log = log_fn
        self.traffic = TrafficGenerator(label, send_fn, log_fn)

    def is_command(self, line: str) -> bool:
        return line.startswith("/")

    def dispatch(self, line: str) -> None:
        try:
            tokens = shlex.split(line[1:])
        except ValueError as exc:
            self._log(f"[{self.label}] bad command: {exc}")
            return
        if not tokens:
            self._log(f"[{self.label}] {HELP_TEXT}")
            return
        cmd, rest = tokens[0], tokens[1:]
        if cmd == "help":
            self._log(f"[{self.label}] {HELP_TEXT}")
        elif cmd == "traffic":
            self._dispatch_traffic(rest)
        else:
            self._log(f"[{self.label}] unknown command /{cmd} -- {HELP_TEXT}")

    def _dispatch_traffic(self, rest: List[str]) -> None:
        if not rest:
            self._log(f"[{self.label}] {HELP_TEXT}")
            return
        sub, args = rest[0], rest[1:]
        if sub == "start":
            try:
                kv = _parse_kv(args)
                rate = float(kv.get("rate", 5))
                size = int(kv.get("size", 64))
                count = int(kv["count"]) if "count" in kv else None
                duration = float(kv["duration"]) if "duration" in kv else None
            except ValueError as exc:
                self._log(f"[{self.label}] bad /traffic start args: {exc}")
                return
            self._log(self.traffic.start(rate_hz=rate, size_bytes=size, count=count, duration_s=duration))
        elif sub == "stop":
            self._log(self.traffic.stop())
        elif sub == "status":
            self._log(self.traffic.status())
        else:
            self._log(f"[{self.label}] unknown /traffic subcommand {sub!r} -- expected start/stop/status")
