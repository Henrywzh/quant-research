from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class UniverseFilterConfig:
    """
    Universe filter based on rolling MA price floor.

    Example:
      - window=21
      - min_ma_price=1.0  (exclude penny-ish HK stocks)
    """
    ma_window: int = 21
    min_ma_price: Optional[float] = 1.0

    # Market cap floor (HKD or USD — must be consistent)
    mcap_window: int = 21
    min_mean_mcap: Optional[float] = 1e9  # e.g., 1e9

    # IPO seasoning
    min_ipo_trading_days: int = 63  # e.g., 252 means exclude first 252 days


def build_universe_eligible(close, cfg, mcap=None):
    eligible = compute_ma_price_eligibility(close, cfg)

    if cfg.min_mean_mcap is not None:
        if mcap is None:
            raise ValueError("mcap required")
        eligible &= compute_mean_mcap_eligibility(mcap, cfg.mcap_window, cfg.min_mean_mcap)

    if cfg.min_ipo_trading_days > 0:
        eligible &= compute_ipo_age_eligibility(close, cfg.min_ipo_trading_days)

    return eligible.fillna(False)


def compute_ma_price_eligibility(
    close: pd.DataFrame,
    cfg: UniverseFilterConfig,
) -> pd.DataFrame:
    """
    Returns a boolean DataFrame (date x ticker):
      eligible[t, i] = True if MA_window(close)[t, i] >= min_ma_price

    No lookahead:
    - MA at date t uses close up to t (inclusive).
    """
    ma = close.rolling(cfg.ma_window, min_periods=cfg.ma_window).mean()
    eligible = ma >= cfg.min_ma_price
    return eligible


def compute_mean_mcap_eligibility(mcap: pd.DataFrame, window: int, min_mean_mcap: float) -> pd.DataFrame:
    mc = mcap.replace([np.inf, -np.inf], np.nan)
    mean_mc = mc.rolling(window, min_periods=window).mean()

    return mean_mc >= min_mean_mcap


def compute_ipo_age_eligibility(close: pd.DataFrame, min_trading_days: int) -> pd.DataFrame:
    if min_trading_days <= 0:
        return pd.DataFrame(True, index=close.index, columns=close.columns)

    # counts trading days with available close per ticker up to each date
    age = close.notna().cumsum()
    return age >= min_trading_days