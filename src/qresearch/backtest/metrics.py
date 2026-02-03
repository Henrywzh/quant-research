from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandas as pd

TRADING_DAYS: int = 252
DEFAULT_DDOF: int = 0


def icir(ic_series: pd.Series, ddof: int = DEFAULT_DDOF) -> float:
    ic_series = ic_series.dropna()
    if len(ic_series) < 2:
        return float("nan")
    sd = ic_series.std(ddof=ddof)
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float(ic_series.mean() / sd)


def annualised_return_geo(r: pd.Series, freq: int = TRADING_DAYS) -> float:
    r = r.dropna()
    if len(r) == 0:
        return float("nan")
    eq = (1.0 + r).prod()
    return float(eq ** (freq / len(r)) - 1.0)


def annualised_vol(r: pd.Series, freq: int = TRADING_DAYS, ddof: int = DEFAULT_DDOF) -> float:
    r = r.dropna()
    if len(r) == 0:
        return float("nan")
    return float(r.std(ddof=ddof) * np.sqrt(freq))


def _rf_per_period(rf_annual: float, freq: int) -> float:
    # stable conversion for annual rf to per-period rf
    return float((1.0 + rf_annual) ** (1.0 / freq) - 1.0)


def sharpe_ratio(
    r: pd.Series,
    freq: int = TRADING_DAYS,
    rf_annual: float = 0.0,
    ddof: int = DEFAULT_DDOF,
) -> float:
    r = r.dropna()
    if len(r) == 0:
        return float("nan")

    rf_per = _rf_per_period(rf_annual, freq)
    ex = r - rf_per
    vol = ex.std(ddof=ddof) * np.sqrt(freq)

    if vol == 0 or not np.isfinite(vol):
        return float("nan")

    return float((ex.mean() * freq) / vol)


def rolling_sharpe(
    r: pd.Series,
    window: int = TRADING_DAYS,
    freq: int = TRADING_DAYS,
    rf_annual: float = 0.0,
    ddof: int = DEFAULT_DDOF,
) -> pd.Series:
    rf_per = _rf_per_period(rf_annual, freq)
    ex = r - rf_per
    mu = ex.rolling(window).mean() * freq
    vol = ex.rolling(window).std(ddof=ddof) * np.sqrt(freq)
    return mu / vol


def equity_curve(r: pd.Series) -> pd.Series:
    r = r.dropna()
    if len(r) == 0:
        return pd.Series(dtype=float)
    return (1.0 + r).cumprod()


def drawdown_series_from_equity(eq: pd.Series) -> pd.Series:
    if eq is None or len(eq) == 0:
        return pd.Series(dtype=float)
    peak = eq.cummax()
    return eq / peak - 1.0


def equity_curve(ret: pd.Series) -> pd.Series:
    ret = ret.fillna(0.0)
    return (1.0 + ret).cumprod()


def max_drawdown_from_returns(r: pd.Series) -> float:
    r = r.fillna(0.0)
    eq = (1.0 + r).cumprod()
    dd = eq / eq.cummax() - 1.0
    return float(dd.min())


def yearly_returns(r: pd.Series) -> pd.Series:
    r = r.dropna()
    if len(r) == 0:
        return pd.Series(dtype=float)
    return (1.0 + r).groupby(r.index.year).prod() - 1.0


def vol_target_returns(
    ret: pd.Series,
    target_vol: float = 0.10,
    vol_lookback: int = 20,
    ann_factor: int = TRADING_DAYS,
    max_leverage: float = 2.0,
) -> pd.DataFrame:
    rv = ret.rolling(vol_lookback, min_periods=vol_lookback).std(ddof=DEFAULT_DDOF) * np.sqrt(ann_factor)
    scale = (target_vol / rv).clip(0.0, max_leverage)
    scale = scale.shift(1).fillna(0.0)  # no lookahead
    ret_vt = scale * ret
    eq_vt = (1.0 + ret_vt).cumprod()
    return pd.DataFrame({"scale": scale, "ret_vt": ret_vt, "eq_vt": eq_vt})


def perf_summary(r: pd.Series, freq: int) -> Dict[str, float]:
    return {
        "ann_return_geo": annualised_return_geo(r, freq=freq),
        "ann_vol": annualised_vol(r, freq=freq),
        "sharpe": sharpe_ratio(r, freq=freq),
        "max_dd": max_drawdown_from_returns(r),
        "n_obs": int(r.dropna().shape[0]),
    }
