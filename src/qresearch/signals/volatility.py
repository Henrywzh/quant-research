import numpy as np
import pandas as pd

from qresearch.data.types import MarketData
from qresearch.signals.registry import register_signal


# Assumption: 3M ≈ 63 trading days (21 * 3). Override via lookback if desired.


@register_signal(
    "vol_std_3M",
    description="3M volatility: rolling std of daily close-to-close returns over lookback (~63d).",
    defaults={"lookback": 63, "sign": 1},
    requires=("close",),
)
def vol_std_3M(md: MarketData, lookback: int = 63, sign: int = 1) -> pd.DataFrame:
    r = md.close.pct_change()
    return sign * r.rolling(lookback).std()


@register_signal(
    "vol_highlow_avg_3M",
    description="3M intraday range level: rolling mean of (high/low) over lookback (~63d).",
    defaults={"lookback": 63, "sign": 1},
    requires=("high", "low"),
)
def vol_highlow_avg_3M(md: MarketData, lookback: int = 63, sign: int = 1) -> pd.DataFrame:
    hl = _safe_div(md.high, md.low)
    return sign * hl.rolling(lookback).mean()


@register_signal(
    "vol_highlow_std_3M",
    description="3M intraday range dispersion: rolling std of (high/low) over lookback (~63d).",
    defaults={"lookback": 63, "sign": 1},
    requires=("high", "low"),
)
def vol_highlow_std_3M(md: MarketData, lookback: int = 63, sign: int = 1) -> pd.DataFrame:
    hl = _safe_div(md.high, md.low)
    return sign * hl.rolling(lookback).std()


@register_signal(
    "vol_upshadow_std_3M",
    description=(
        "3M upper-shadow std (normalized): std of (high - max(open, close)) / high over lookback (~63d)."
    ),
    defaults={"lookback": 63, "sign": 1},
    requires=("open", "close", "high"),
)
def vol_upshadow_std_3M(md: MarketData, lookback: int = 63, sign: int = 1) -> pd.DataFrame:
    body_top = np.maximum(md.open, md.close)
    up = _safe_div(md.high - body_top, md.high).clip(lower=0.0)
    return sign * up.rolling(lookback).std()


@register_signal(
    "vol_downshadow_std_3M",
    description=(
        "3M lower-shadow std (normalized): std of (min(open, close) - low) / low over lookback (~63d)."
    ),
    defaults={"lookback": 63, "sign": 1},
    requires=("open", "close", "low"),
)
def vol_downshadow_std_3M(md: MarketData, lookback: int = 63, sign: int = 1) -> pd.DataFrame:
    body_bot = np.minimum(md.open, md.close)
    down = _safe_div(body_bot - md.low, md.low).clip(lower=0.0)
    return sign * down.rolling(lookback).std()


@register_signal(
    "vol_w_upshadow_std_3M",
    description="3M Williams upper-shadow std: std of (high - close) / high over lookback (~63d).",
    defaults={"lookback": 63, "sign": 1},
    requires=("close", "high"),
)
def vol_w_upshadow_std_3M(md: MarketData, lookback: int = 63, sign: int = 1) -> pd.DataFrame:
    w_up = _safe_div(md.high - md.close, md.high).clip(lower=0.0)
    return sign * w_up.rolling(lookback).std()


@register_signal(
    "vol_w_downshadow_std_3M",
    description="3M Williams lower-shadow std: std of (close - low) / low over lookback (~63d).",
    defaults={"lookback": 63, "sign": 1},
    requires=("close", "low"),
)
def vol_w_downshadow_std_3M(md: MarketData, lookback: int = 63, sign: int = 1) -> pd.DataFrame:
    w_down = _safe_div(md.close - md.low, md.low).clip(lower=0.0)
    return sign * w_down.rolling(lookback).std()


def _safe_div(numer: pd.DataFrame, denom: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    denom2 = denom.where(denom.abs() > eps, np.nan)
    return numer / denom2
