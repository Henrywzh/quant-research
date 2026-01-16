from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


RebalanceMode = Literal["calendar", "fixed_h"]


@dataclass(frozen=True)
class TopKConfig:
    # --- schedule selection ---
    mode: RebalanceMode = "calendar"

    # Calendar mode
    rebalance: str = "W-FRI"          # e.g. "D", "W-FRI", "M"

    # Fixed-H mode (rebalance every H trading days)
    H: int = 5                        # e.g. 5, 10, 20
    offset: int = 0                   # start offset into trading-day index: 0 means first day

    # --- selection rules ---
    top_k: int = 1
    use_absolute_filter: bool = True
    fill_missing_scores: bool = False  # if True, treat NaN scores as -inf


def _rebalance_dates(index: pd.DatetimeIndex, cfg: TopKConfig) -> pd.DatetimeIndex:
    """
    Produce rebalance dates from the TRADING calendar.

    calendar mode:
      - rebalance on the last trading date inside each calendar bin given by cfg.rebalance
        (e.g. for W-FRI, use Thu if Fri is a holiday)

    fixed_h mode:
      - rebalance every H trading days starting from cfg.offset
    """
    index = pd.DatetimeIndex(index).sort_values()

    if cfg.mode == "calendar":
        # Group trading dates into calendar bins and take the last timestamp in each bin.
        # This yields actual tradable dates, not bin labels.
        grouped_last = (
            pd.Series(index=index, data=index)
            .groupby(pd.Grouper(freq=cfg.rebalance))
            .max()
            .dropna()
        )
        return pd.DatetimeIndex(grouped_last.values)

    if cfg.mode == "fixed_h":
        if cfg.H <= 0:
            raise ValueError("TopKConfig.H must be > 0 for mode='fixed_h'")
        start = int(cfg.offset)
        if start < 0 or start >= len(index):
            raise ValueError("TopKConfig.offset out of range")
        return pd.DatetimeIndex(index[start::cfg.H])

    raise ValueError(f"Unknown mode: {cfg.mode}")


def build_topk_weights(
    prices: pd.DataFrame,
    scores: pd.DataFrame,
    cfg: TopKConfig,
) -> pd.DataFrame:
    """
    Build decision-time weights (UNSHIFTED).

    Behavior:
    - Rebalance dates are determined from prices.index (calendar bins or fixed-H grid).
    - On each rebalance date:
        - rank assets by score
        - take top_k
        - optionally apply absolute filter (score > 0)
        - if none pass: hold cash (all zeros)
    - Forward-fill weights between rebalances to obtain daily decision weights.
    - No-lookahead must be enforced in the backtest engine via weights.shift(1).
    """
    # Align scores to prices grid (ensures dt lookup exists)
    scores = scores.reindex(index=prices.index, columns=prices.columns)

    # Determine rebalance dates from trading calendar
    rebal_dates = _rebalance_dates(prices.index, cfg)

    # Rebalance weights container
    w_reb = pd.DataFrame(0.0, index=rebal_dates, columns=prices.columns)

    for dt in rebal_dates:
        row = scores.loc[dt].replace([np.inf, -np.inf], np.nan)

        if cfg.fill_missing_scores:
            # Missing scores are treated as -inf (never selected)
            row = row.fillna(-np.inf)

        # Keep only finite scores for ranking
        row_valid = row.dropna()
        if row_valid.empty:
            continue  # hold cash

        # Rank and pick top-K
        top = row_valid.sort_values(ascending=False).head(cfg.top_k)

        # Absolute filter if requested
        if cfg.use_absolute_filter:
            top = top[top > 0]

        if top.empty:
            continue  # hold cash

        # Allocate equally across survivors (may be < top_k)
        w_reb.loc[dt, top.index] = 1.0 / len(top)

    # Expand to daily by forward-filling between rebalances
    weights = w_reb.reindex(prices.index).ffill().fillna(0.0)
    return weights