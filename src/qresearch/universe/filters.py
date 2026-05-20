from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class UniverseFilterConfig:
    """
    Generic investability filters (market-agnostic). Everything is optional.

    All filters are evaluated in **matrix form**: (dates x tickers) aligned to close.

    Notes:
      - min_ma_price is useful for HK "penny stock" filtering; you can leave it None for A-shares.
      - Market cap and turnover_value must be supplied as matrices if the corresponding thresholds are enabled.
      - IPO seasoning uses availability of close data as a proxy for "listed age".
    """

    # Delisted stocks
    delisted_filter: bool = False

    # Price floor on rolling mean close
    ma_window: int = 21
    min_ma_price: Optional[float] = None

    # Market cap rolling mean bounds
    mcap_window: int = 21
    min_mean_mcap: Optional[float] = None
    max_mean_mcap: Optional[float] = None

    # Liquidity filter: rolling mean turnover value (e.g., JQ money/成交额)
    liq_window: int = 20
    min_mean_turnover_value: Optional[float] = None

    # IPO seasoning (trading-day count based on close availability)
    min_ipo_trading_days: int = 63


def _align_bool(mask: pd.DataFrame, like: pd.DataFrame) -> pd.DataFrame:
    """Reindex mask to match `like` and fill missing with False."""
    return mask.reindex(index=like.index, columns=like.columns).fillna(False).astype(bool)


def compute_ma_price_eligibility(close: pd.DataFrame, window: int, min_ma_price: float) -> pd.DataFrame:
    """
    eligible[t,i] = True iff rolling_mean(close, window)[t,i] >= min_ma_price
    No lookahead: uses close up to t (inclusive).
    """
    ma = close.rolling(window, min_periods=window).mean()
    return ma >= float(min_ma_price)


def compute_mean_mcap_eligibility(mcap: pd.DataFrame, window: int, min_mean_mcap: float | None, max_mean_mcap: float | None) -> pd.DataFrame:
    mc = mcap.replace([np.inf, -np.inf], np.nan)
    mean_mc = mc.rolling(window, min_periods=window).mean()

    ok = pd.DataFrame(True, index=mcap.index, columns=mcap.columns)
    if min_mean_mcap is not None:
        ok &= (mean_mc >= float(min_mean_mcap))
    if max_mean_mcap is not None:
        ok &= (mean_mc <= float(max_mean_mcap))
    return ok


def compute_mean_turnover_value_eligibility(turnover_value: pd.DataFrame, window: int, min_mean_turnover_value: float) -> pd.DataFrame:
    tv = turnover_value.replace([np.inf, -np.inf], np.nan)
    mean_tv = tv.rolling(window, min_periods=window).mean()
    return mean_tv >= float(min_mean_turnover_value)


def compute_ipo_age_eligibility(close: pd.DataFrame, min_trading_days: int) -> pd.DataFrame:
    """
    age[t,i] = number of trading days up to t with non-NaN close.
    eligible = age >= min_trading_days
    """
    if min_trading_days <= 0:
        return pd.DataFrame(True, index=close.index, columns=close.columns)
    age = close.notna().cumsum()
    return age >= int(min_trading_days)


def compute_life_span_eligibility(
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
    life_span_df: pd.DataFrame,
) -> pd.DataFrame:
    df = life_span_df.copy()
    df["code"] = df["code"].astype("string").str.strip()
    df["start_date"] = pd.to_datetime(df["start_date"]).dt.normalize()
    df["end_date"] = pd.to_datetime(df["end_date"]).dt.normalize()

    # map to tickers order
    meta = df.set_index("code")[["start_date", "end_date"]].reindex(tickers)

    start = meta["start_date"]
    end = meta["end_date"]

    # broadcast to (dates x tickers)
    alive = (dates[:, None] >= start.values[None, :]) & (dates[:, None] <= end.values[None, :])
    return pd.DataFrame(alive, index=dates, columns=tickers).fillna(False)


def build_universe_eligible(
    *,
    close: pd.DataFrame,
    cfg: UniverseFilterConfig,
    mcap: Optional[pd.DataFrame] = None,
    turnover_value: Optional[pd.DataFrame] = None,
    life_span_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Investability filters only (NOT membership gating).
    Returns boolean mask aligned to close.

    enabled filters:
      - rolling MA price floor (optional)
      - rolling mean market cap bounds (optional)
      - rolling mean turnover value floor (optional)
      - IPO seasoning (min trading days)
    """
    eligible = pd.DataFrame(True, index=close.index, columns=close.columns)

    # NEW: lifespan gate (A-share only)
    if life_span_df is not None:
        alive = compute_life_span_eligibility(close.index, close.columns, life_span_df)
        eligible &= alive

    # 1) Price floor
    if cfg.min_ma_price is not None:
        eligible &= _align_bool(
            compute_ma_price_eligibility(close, cfg.ma_window, cfg.min_ma_price),
            close,
        )

    # 2) Market cap bounds
    if (cfg.min_mean_mcap is not None) or (cfg.max_mean_mcap is not None):
        if mcap is None:
            raise ValueError("mcap matrix required when min_mean_mcap/max_mean_mcap is enabled.")
        eligible &= _align_bool(
            compute_mean_mcap_eligibility(mcap, cfg.mcap_window, cfg.min_mean_mcap, cfg.max_mean_mcap),
            close,
        )

    # 3) Liquidity (turnover value)
    if cfg.min_mean_turnover_value is not None:
        if turnover_value is None:
            raise ValueError("turnover_value matrix required when min_mean_turnover_value is enabled.")
        eligible &= _align_bool(
            compute_mean_turnover_value_eligibility(turnover_value, cfg.liq_window, cfg.min_mean_turnover_value),
            close,
        )

    # 4) IPO seasoning
    if cfg.min_ipo_trading_days > 0:
        eligible &= _align_bool(
            compute_ipo_age_eligibility(close, cfg.min_ipo_trading_days),
            close,
        )

    return eligible.fillna(False).astype(bool)

