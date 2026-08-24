"""String-or-instance resolution for every swappable (Layer 2) block param.

Every strategy constructor argument on OfdmRx/OfdmTx (sync=, cfo=,
channel_estimator=, equalizer=, modem=, fec=) accepts either a scheme
string (resolved here to a default-configured instance) or an
already-built instance (for custom params or subclassing) -- the
sklearn/Keras pattern. See docs/architecture.md, "The string-or-instance
rule".

Layer 1 blocks (FFT/CP, resource grid, framing) are never registered
here -- there's nothing to choose between, they're plain constructor
params on the blocks that use them.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Type

from .block import Block

# category -> {name -> class}, e.g. _REGISTRY["sync"]["zc"] = ZadoffChuSync
_REGISTRY: Dict[str, Dict[str, Type[Block]]] = {}


def register(category: str, name: str) -> Callable[[Type[Block]], Type[Block]]:
    """Class decorator registering a Block subclass under (category, name).

    Example
    -------
    >>> @register("sync", "zc")
    ... class ZadoffChuSync(Block):
    ...     ...
    """

    def decorator(cls: Type[Block]) -> Type[Block]:
        _REGISTRY.setdefault(category, {})[name] = cls
        return cls

    return decorator


def available(category: str) -> List[str]:
    """List registered scheme names for a category (e.g. "sync")."""
    return sorted(_REGISTRY.get(category, {}).keys())


def resolve(category: str, value: Any, **default_kwargs: Any) -> Block:
    """Resolve a strategy param to a Block instance.

    - If `value` is already a Block instance, it's returned unchanged
      (so custom-configured or subclassed instances always work).
    - If `value` is a string, it's looked up in the (category, name)
      registry and constructed with `default_kwargs`.
    - Anything else raises TypeError.
    """
    if isinstance(value, Block):
        return value
    if isinstance(value, str):
        try:
            cls = _REGISTRY[category][value]
        except KeyError as exc:
            known = available(category)
            raise KeyError(
                f"Unknown {category!r} scheme {value!r}. "
                f"Available: {known or '(none registered yet)'}"
            ) from exc
        return cls(**default_kwargs)
    raise TypeError(
        f"{category} expected a scheme string or a Block instance, "
        f"got {type(value).__name__}"
    )
