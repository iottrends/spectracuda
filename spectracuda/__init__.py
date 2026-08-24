"""spectracuda: GPU-accelerated, liquid-dsp-inspired SDR PHY (+ MAC)
framework.

See README.md for a quick-start example and docs/architecture.md for the
full design (layered API model, the string-or-instance rule, backend
abstraction, batch-shape contract); docs/liquid-dsp-api-inventory.md for
the algorithm references this project is built against; docs/todo.md for
the concrete, itemized state of every piece.

Top-level entry points live in submodules, not re-exported here (import
`from spectracuda.pipeline import Ofdm`, `from spectracuda.mac import
MacLink`, `from spectracuda.fec import FEC, CRC`, etc. -- see README.md).
"""
from __future__ import annotations

__version__ = "0.0.1"

__all__ = ["__version__"]
