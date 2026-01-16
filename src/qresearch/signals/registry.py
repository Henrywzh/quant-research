from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional
import pandas as pd


SignalFn = Callable[..., pd.DataFrame]


@dataclass(frozen=True)
class SignalSpec:
    name: str
    fn: SignalFn
    description: str = ""
    # optional: default params, so you can call with no args
    defaults: Dict[str, Any] | None = None


_REGISTRY: Dict[str, SignalSpec] = {}


def register_signal(
    name: str,
    *,
    description: str = "",
    defaults: Optional[Dict[str, Any]] = None,
):
    """
    Decorator: @register_signal("mom", defaults={"lookback": 21})
    """
    def _wrap(fn: SignalFn) -> SignalFn:
        if name in _REGISTRY:
            raise ValueError(f"Signal already registered: {name}")
        _REGISTRY[name] = SignalSpec(
            name=name, fn=fn, description=description, defaults=defaults or {}
        )
        return fn
    return _wrap


def get_signal(name: str) -> SignalSpec:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown signal '{name}'. Available: {list_signals()}")
    return _REGISTRY[name]


def list_signals() -> list[str]:
    return sorted(_REGISTRY.keys())


def compute_signal(
    prices: pd.DataFrame,
    name: str,
    **params: Any,
) -> pd.DataFrame:
    """
    Compute a registered signal by name.
    Params override defaults.
    """
    spec = get_signal(name)
    merged = dict(spec.defaults or {})
    merged.update(params)

    out = spec.fn(prices, **merged)

    # basic contract checks
    out = out.reindex(index=prices.index, columns=prices.columns)
    return out
