"""Packet/signal quality stats, computed as small pure functions (not
Block subclasses -- these aren't swappable strategies, just arithmetic
on already-computed arrays, the same "plain function, xp passed
explicitly" style already used by e.g. ResourceGrid.scatter(xp, ...)).

liquid-dsp's closest equivalent is `ofdmflexframesync_get_framedatastats()`
-- a defined struct with EVM, RSSI, CFO, and payload valid/invalid
counters. `pipeline/ofdm.py`'s `rx_process()` now returns a result dict
with a defined, stable key set (see its own docstring) rather than
"whatever fields happened to be convenient to add during development" --
this module supplies the two readouts that were previously entirely
missing (EVM, RSSI); CFO and payload-valid (crc_valid) were already
present as ad hoc fields and are now part of the same defined contract.
"""
from __future__ import annotations

from typing import Any


def compute_evm(xp: Any, actual: Any, ideal: Any) -> Any:
    """Per-batch-item RMS EVM (fractional, standard normalized
    definition -- multiply by 100 for percent, or 20*log10(evm) for dB):
    sqrt(mean(|actual - ideal|^2) / mean(|ideal|^2)).

    `ideal` is normally the receiver's OWN hard-decision re-modulated
    symbols (the standard practical way to compute EVM without needing
    the original transmitted symbols as ground truth -- what a real EVM
    meter does too), not literally what was transmitted."""
    err_power = xp.mean(xp.abs(actual - ideal) ** 2, axis=-1)
    ref_power = xp.mean(xp.abs(ideal) ** 2, axis=-1)
    return xp.sqrt(err_power / ref_power)


def compute_rssi_db(xp: Any, iq: Any) -> Any:
    """Per-batch-item received signal power, in dB: 10*log10(mean(|iq|^2)).

    NOT a calibrated/absolute RSSI (dBm) -- this codebase's IQ samples
    have no real antenna/ADC gain reference (see pipeline/ofdm.py's
    class docstring on iq_dtype for the related point about what's
    actually simulated here). This is a relative, simulation-only power
    readout: useful for comparing frames/channel conditions against each
    other within this codebase, not as an absolute RF power measurement."""
    power = xp.mean(xp.abs(iq) ** 2, axis=-1)
    return 10.0 * xp.log10(power + 1e-20)
