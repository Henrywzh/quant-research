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


def equity_curve(ret: pd.Series) -> pd.Series:
    """
    Canonical equity-curve helper for return series.

    Missing returns are treated as flat periods so callers preserve the original
    date grid instead of silently dropping rows.
    """
    if ret is None or len(ret) == 0:
        return pd.Series(dtype=float)
    ret = ret.fillna(0.0)
    return (1.0 + ret).cumprod()


def drawdown_series_from_equity(eq: pd.Series) -> pd.Series:
    if eq is None or len(eq) == 0:
        return pd.Series(dtype=float)
    peak = eq.cummax()
    return eq / peak - 1.0
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


def ic_corr_matrix(
    IC: pd.DataFrame,
    method: str = "pearson",     # "pearson" or "spearman"
    min_obs: int = 60,           # minimum overlapping days per pair
) -> pd.DataFrame:
    ic = IC.copy()
    ic.index = pd.to_datetime(ic.index)
    ic = ic.sort_index()

    cols = list(ic.columns)
    out = pd.DataFrame(index=cols, columns=cols, dtype=float)

    for i, c1 in enumerate(cols):
        s1 = ic[c1]
        for j, c2 in enumerate(cols):
            if j < i:
                continue
            s2 = ic[c2]
            xy = pd.concat([s1, s2], axis=1).dropna()
            if len(xy) < min_obs:
                val = np.nan
            else:
                val = xy.iloc[:, 0].corr(xy.iloc[:, 1], method=method)
            out.loc[c1, c2] = val
            out.loc[c2, c1] = val

    np.fill_diagonal(out.values, 1.0)
    return out


def ic_corr_top_pairs(
    corr_mat: pd.DataFrame,
    top_n: int = 30,
    abs_sort: bool = True,
) -> pd.DataFrame:
    c = corr_mat.copy()

    # upper triangle only
    mask = np.triu(np.ones(c.shape, dtype=bool), k=1)
    pairs = (
        c.where(mask)
         .stack()
         .rename("corr")
         .reset_index()
         .rename(columns={"level_0": "f1", "level_1": "f2"})
    )

    if abs_sort:
        pairs["abs_corr"] = pairs["corr"].abs()
        pairs = pairs.sort_values("abs_corr", ascending=False).drop(columns=["abs_corr"])
    else:
        pairs = pairs.sort_values("corr", ascending=False)

    return pairs.head(top_n).reset_index(drop=True)


def build_ic_df_from_rep_map(rep_map: dict[str, dict], ic_series_key: str = "ic_series") -> pd.DataFrame:
    """
    rep_map: {signal_key: rep}
    rep[ic_series_key] should be a pd.Series indexed by date (daily IC)
    """
    series = {}
    for k, rep in rep_map.items():
        s = rep.get(ic_series_key)
        if s is None:
            raise KeyError(f"rep for {k} missing '{ic_series_key}'. Available keys: {list(rep.keys())[:20]}")
        series[k] = pd.Series(s).rename(k)

    IC = pd.concat(series.values(), axis=1)
    IC.index = pd.to_datetime(IC.index)
    return IC.sort_index()
