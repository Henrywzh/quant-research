from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple
import numpy as np
import pandas as pd

TRADING_DAYS = 252
FillPolicy = Literal["zero", "ffill", "error"]


@dataclass(frozen=True)
class PortfolioBacktestResult:
    gross_ret: pd.Series
    net_ret: pd.Series
    equity_gross: pd.Series
    equity_net: pd.Series
    turnover: pd.Series          # turnover on execution day (aligned with fee)
    fee: pd.Series               # fee on execution day
    exposure: pd.Series          # gross exposure of weights_used
    cash_weight: pd.Series       # residual cash (>=0 in this model)
    weights_used: pd.DataFrame   # weights actually applied to rets on each day


def _align_to_prices(prices: pd.DataFrame, weights: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align to the prices calendar (prices is the master calendar).
    - Keep all price dates/columns.
    - Reindex weights to match prices index/columns.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices.index must be a DatetimeIndex")
    if not isinstance(weights.index, pd.DatetimeIndex):
        raise TypeError("weights.index must be a DatetimeIndex")

    prices = prices.sort_index()
    weights = weights.sort_index()

    # Reindex weights to prices (master calendar + columns)
    weights = weights.reindex(index=prices.index, columns=prices.columns)

    return prices, weights


def backtest_weights(
    prices: pd.DataFrame,
    weights: pd.DataFrame,                 # decision-time weights as-of close t (UNSHIFTED)
    fee_bps: float = 0.0,
    rf_annual: float = 0.0,
    trading_days: int = TRADING_DAYS,
    long_only: bool = True,
    max_gross: float = 1.01,               # tolerance for gross exposure
    fill_weights: FillPolicy = "zero",
    allow_leverage: bool = False,
) -> PortfolioBacktestResult:
    """
    Weights-based backtest with:
      - No-lookahead: weights_used[t] = weights[t-1]
      - Correct fee timing: fees paid on execution day (when new weights start applying)

    Timeline
    --------
    - weights[t]: decision formed at close of day t.
    - weights_used[t]: weights applied to day t close-to-close return, equals weights[t-1].
    - turnover_trade[t] = sum_i |weights[t,i] - weights[t-1,i]|
      represents trading to move from yesterday's decision weights to today's decision weights.
    - That trade executes at start of day (t+1) under this convention, so fees should align with
      when weights become active. Therefore fee[t] uses turnover_trade[t-1].

    Assumptions
    -----------
    - prices are close prices, returns are close-to-close.
    - residual (1 - exposure) is held as cash earning rf_annual.
    - if allow_leverage=True and exposure>1, cash is clipped at 0 and no borrowing cost is modeled.
    """
    # 0) Align to prices calendar
    prices, weights = _align_to_prices(prices, weights)

    # 1) Fill decision-time weights (policy)
    if weights.isna().any().any():
        if fill_weights == "zero":
            weights = weights.fillna(0.0)
        elif fill_weights == "ffill":
            weights = weights.ffill().fillna(0.0)
        elif fill_weights == "error":
            raise ValueError("weights contains NaNs and fill_weights='error'")
        else:
            raise ValueError(f"Unknown fill_weights policy: {fill_weights}")

    # 2) Decision-weight constraints
    if long_only and (weights < -1e-12).any().any():
        raise ValueError("long-only violated in decision weights (negative weights found)")

    gross_decision = weights.sum(axis=1)
    if not allow_leverage and (gross_decision > max_gross).any():
        raise ValueError("gross exposure exceeds max_gross in decision weights and allow_leverage=False")

    # 3) Returns (close-to-close)
    rets = prices.pct_change(fill_method=None)
    rets.iloc[0] = 0.0
    rets = rets.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 4) Apply lag (no-lookahead)
    w_used = weights.shift(1).fillna(0.0)

    # 5) Execution-weight constraints (debugging safety)
    if long_only and (w_used < -1e-12).any().any():
        raise ValueError("long-only violated in weights_used")

    exposure = w_used.sum(axis=1)
    if not allow_leverage and (exposure > max_gross).any():
        raise ValueError("gross exposure exceeds max_gross in weights_used and allow_leverage=False")

    # 6) Cash return
    cash_weight = (1.0 - exposure).clip(lower=0.0)
    daily_rf = (1.0 + rf_annual) ** (1.0 / trading_days) - 1.0
    cash_ret = cash_weight * daily_rf

    # 7) Gross portfolio return
    port_gross = (w_used * rets).sum(axis=1) + cash_ret

    # 8) Turnover + fee timing
    turnover_trade = (weights - weights.shift(1)).abs().sum(axis=1).fillna(0.0)
    turnover_exec = turnover_trade.shift(1).fillna(0.0)  # align fee to day weights become active
    fee = (fee_bps / 10000.0) * turnover_exec

    port_net = port_gross - fee

    # 9) Equity
    equity_gross = (1.0 + port_gross).cumprod()
    equity_net = (1.0 + port_net).cumprod()

    return PortfolioBacktestResult(
        gross_ret=port_gross,
        net_ret=port_net,
        equity_gross=equity_gross,
        equity_net=equity_net,
        turnover=turnover_exec,
        fee=fee,
        exposure=exposure,
        cash_weight=cash_weight,
        weights_used=w_used,
    )
