from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .registry import register_signal


@register_signal("mom_ret", description="Return over lookback with optional skip", defaults={"lookback": 21, "skip": 0})
def mom_ret(prices: pd.DataFrame, lookback: int = 21, skip: int = 0) -> pd.DataFrame:
    p1 = prices.shift(skip)
    p0 = prices.shift(skip + lookback)
    return p1 / p0 - 1.0


@register_signal("mom_12_1", description="12-1 momentum (252 lookback, 21 skip)", defaults={"lookback": 252, "skip": 21})
def mom_12_1(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    p1 = prices.shift(skip)
    p0 = prices.shift(skip + lookback)
    return p1 / p0 - 1.0


@register_signal("ma_diff", description="ma and price difference", defaults={"lookback": 21, "skip": 0, 'sign': 1})
def ma_diff(prices: pd.DataFrame, lookback: int, skip: int = 0, sign: int = 1) -> pd.DataFrame:
    """
    mom_{L,skip}(t) = price(t-skip)/price(t-skip-L) - 1
    Computed using close[t] information only, no lookahead.
    """
    p1 = prices.shift(skip)
    p0 = prices.shift(skip + lookback)
    return sign * p1 / p0 - 1.0


SignalType = Literal[
    "mom_1m", "mom_3m", "mom_6m", "mom_12m",
    "mom_12_1",          # 12-month momentum skipping most recent 1 month
    "mom_ret_over_vol",  # risk-adjusted momentum
    "ma",
    "trend_annret_r2",
]


@dataclass(frozen=True)
class SignalConfig:
    signal: SignalType = "mom_1m"

    # trading-day horizons
    mom_1m: int = 21
    mom_3m: int = 63
    mom_6m: int = 126
    mom_12m: int = 252
    skip_1m: int = 21  # for mom_12_1

    # MA strength
    ma_lookback: int = 50

    # risk-adjusted momentum
    risk_mom_ret_lookback: int = 63
    risk_mom_vol_lookback: int = 63
    risk_mom_skip: int = 0

    # trend score (annret * R^2)
    trend_lookback: int = 126
    ann_factor: int = 252


def momentum(prices: pd.DataFrame, lookback: int, skip: int = 0) -> pd.DataFrame:
    """
    mom_{L,skip}(t) = price(t-skip)/price(t-skip-L) - 1
    Computed using close[t] information only, no lookahead.
    """
    p1 = prices.shift(skip)
    p0 = prices.shift(skip + lookback)
    return p1 / p0 - 1.0


def momentum_over_vol(
    prices: pd.DataFrame,
    ret_lookback: int,
    vol_lookback: int,
    skip: int = 0,
) -> pd.DataFrame:
    """
    Risk-adjusted momentum: (lookback return) / (realized vol).
    """
    mom = momentum(prices, lookback=ret_lookback, skip=skip)
    r = prices.pct_change(fill_method=None).shift(skip)
    vol = r.rolling(vol_lookback).std(ddof=0)
    return mom / vol.replace(0.0, np.nan)


def ma_strength(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    ma = prices.rolling(lookback).mean()
    return prices / ma - 1.0


def trend_score_annret_r2(
    prices: pd.DataFrame,
    lookback: int,
    ann_factor: int = 252,
) -> pd.DataFrame:
    """
    Trend score = annualised return * R^2 from linear regression of log(price) on time.
    Uses an OLS-equivalent closed form on rolling windows (fast for ETF universes).
    """
    px = prices.replace([np.inf, -np.inf], np.nan).ffill()
    y = np.log(px)

    L = int(lookback)
    if L < 2:
        raise ValueError("lookback must be >= 2")

    x = np.arange(L, dtype=float)
    x_mean = x.mean()
    var_x = ((x - x_mean) ** 2).sum()  # constant

    sum_y = y.rolling(L, min_periods=L).sum()
    sum_y2 = (y * y).rolling(L, min_periods=L).sum()

    def dot_x(arr: np.ndarray) -> float:
        return float(np.dot(x, arr))

    sum_xy = y.rolling(L, min_periods=L).apply(dot_x, raw=True)

    y_mean = sum_y / L
    cov_xy = sum_xy - L * x_mean * y_mean
    b = cov_xy / var_x  # slope per day

    var_y = (sum_y2 / L) - (y_mean ** 2)
    var_y = var_y.clip(lower=0.0)

    denom = var_x * (L * var_y)
    r2 = (cov_xy ** 2) / denom
    r2 = r2.clip(lower=0.0, upper=1.0)

    ann_ret = np.exp(b * ann_factor) - 1.0
    return ann_ret * r2


def compute_scores(prices: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    """
    Map prices -> cross-sectional scores (higher = better).

    Contract:
    - output has same index/columns as prices (aligned)
    - may contain NaNs during warmup
    """
    sig = cfg.signal

    if sig == "mom_1m":
        out = momentum(prices, lookback=cfg.mom_1m, skip=0)

    elif sig == "mom_3m":
        out = momentum(prices, lookback=cfg.mom_3m, skip=0)

    elif sig == "mom_6m":
        out = momentum(prices, lookback=cfg.mom_6m, skip=0)

    elif sig == "mom_12m":
        out = momentum(prices, lookback=cfg.mom_12m, skip=0)

    elif sig == "mom_12_1":
        out = momentum(prices, lookback=cfg.mom_12m, skip=cfg.skip_1m)

    elif sig == "mom_ret_over_vol":
        out = momentum_over_vol(
            prices,
            ret_lookback=cfg.risk_mom_ret_lookback,
            vol_lookback=cfg.risk_mom_vol_lookback,
            skip=cfg.risk_mom_skip,
        )

    elif sig == "ma":
        out = ma_strength(prices, lookback=cfg.ma_lookback)

    elif sig == "trend_annret_r2":
        out = trend_score_annret_r2(prices, lookback=cfg.trend_lookback, ann_factor=cfg.ann_factor)

    else:
        raise ValueError(f"Unknown signal: {sig}")

    out = out.replace([np.inf, -np.inf], np.nan)
    return out.reindex(index=prices.index, columns=prices.columns)
