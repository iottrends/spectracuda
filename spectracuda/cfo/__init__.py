"""CFO strategies (Layer 2): SchmidlCoxCFO and PilotBasedCFO both
implemented.

Kept as its own swappable strategy, independent of which `sync` block is
used (decoupled deliberately -- see docs/architecture.md, "CFO
placement"). SchmidlCoxCFO pairs with SchmidlCoxSync specifically (it
depends on the preamble's repeated-halves shape); PilotBasedCFO has no
such dependency and is the correct pairing for ZadoffChuSync (or any
other differently-shaped preamble) -- see pilot_based.py's module
docstring for why.
"""
from .pilot_based import PilotBasedCFO
from .schmidl_cox import SchmidlCoxCFO

__all__ = ["SchmidlCoxCFO", "PilotBasedCFO"]
