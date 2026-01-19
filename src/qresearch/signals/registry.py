from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, Literal

import pandas as pd

from qresearch.data.types import MarketData


RequiredField = Literal["close", "open", "high", "low", "volume"]

# Signal signature: first arg is MarketData, returns DataFrame aligned to md.close
SignalFn = Callable[..., pd.DataFrame]


@dataclass(frozen=True)
class SignalSpec:
    name: str
    fn: SignalFn
    description: str = ""
    defaults: Dict[str, Any] | None = None
    requires: tuple[RequiredField, ...] = ("close",)


_REGISTRY: Dict[str, SignalSpec] = {}


def register_signal(
    name: str,
    *,
    description: str = "",
    defaults: Optional[Dict[str, Any]] = None,
    requires: tuple[RequiredField, ...] = ("close",),
):
    """
    Decorator:

        @register_signal("mom_ret", defaults={"lookback": 21, "skip": 0}, requires=("close",))
        def mom_ret(md: MarketData, lookback: int = 21, skip: int = 0) -> pd.DataFrame:
            ...

    Conventions:
    - fn(md: MarketData, **params) -> pd.DataFrame
    - output is reindexed to md.close.index / md.close.columns
    """
    def _wrap(fn: SignalFn) -> SignalFn:
        if name in _REGISTRY:
            raise ValueError(f"Signal already registered: {name}")
        _REGISTRY[name] = SignalSpec(
            name=name,
            fn=fn,
            description=description,
            defaults=defaults or {},
            requires=requires,
        )
        return fn

    return _wrap


def get_signal(name: str) -> SignalSpec:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown signal '{name}'. Available: {list_signals()}")
    return _REGISTRY[name]


def list_signals() -> list[str]:
    return sorted(_REGISTRY.keys())


def _validate_market_data(md: MarketData, spec: SignalSpec) -> None:
    # Make sure md.close exists (your whole framework anchors on it)
    if md.close is None or md.close.empty:
        raise ValueError("MarketData.close is required and cannot be empty")

    for field in spec.requires:
        val = getattr(md, field, None)
        if val is None:
            raise ValueError(f"Signal '{spec.name}' requires '{field}' but MarketData.{field} is None")


def compute_signal(
    md: MarketData,
    name: str,
    **params: Any,
) -> pd.DataFrame:
    """
    Compute a registered signal by name.
    Params override defaults.

    Returns:
      DataFrame aligned to md.close index/columns.
    """
    spec = get_signal(name)
    _validate_market_data(md, spec)

    merged = dict(spec.defaults or {})
    merged.update(params)

    out = spec.fn(md, **merged)

    # Basic contract checks: align to close grid
    out = out.reindex(index=md.close.index, columns=md.close.columns)
    return out
