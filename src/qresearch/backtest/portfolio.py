from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple, Optional
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from qresearch.backtest.metrics import TRADING_DAYS, perf_summary, yearly_returns, drawdown_series_from_equity
from qresearch.backtest.config import EntryMode, ExperimentConfig, _align_to_prices, _get_rets_from_md
from qresearch.data.types import MarketData
from qresearch.portfolio.weights import build_strategy_weights

FillPolicy = Literal["zero", "ffill", "error"]


@dataclass(frozen=True)
class PortfolioBacktestResult:
    gross_ret: pd.Series
    net_ret: pd.Series
    equity_gross: pd.Series
    equity_net: pd.Series
    turnover: pd.Series
    fee: pd.Series
    exposure: pd.Series          # now = GROSS exposure = sum(|w_used|)
    cash_weight: pd.Series       # residual cash = max(0, 1-gross_exposure)
    weights_used: pd.DataFrame


def backtest_weights(
    md: MarketData,
    weights: pd.DataFrame,                 # decision-time weights as-of close t (UNSHIFTED)
    *,
    entry_mode: EntryMode = "next_close",
    fee_bps: float = 0.0,
    rf_annual: float = 0.0,
    trading_days: int = TRADING_DAYS,
    long_only: bool = True,
    max_gross: float = 1.01,               # cap on sum(|w|)
    fill_weights: FillPolicy = "zero",
    allow_leverage: bool = False,
) -> PortfolioBacktestResult:
    """
    Supports long-only and long/short.

    Definitions:
      - gross_exposure[t] = sum_i |w_used[t,i]|
      - cash_weight[t] = max(0, 1 - gross_exposure[t]) in this simple cash-funded model

    Timing:
      - weights[t] formed at close[t]
      - w_used[t] = weights[t-1] applied to returns on day t
      - returns grid depends on entry_mode:
          next_close     : close-to-close
          next_open      : open-to-open
          open_to_close  : open-to-close (intraday)
    """
    close = md.close.sort_index()

    # Align weights to close calendar
    close, weights = _align_to_prices(close, weights)

    # Fill decision weights
    if weights.isna().any().any():
        if fill_weights == "zero":
            weights = weights.fillna(0.0)
        elif fill_weights == "ffill":
            weights = weights.ffill().fillna(0.0)
        elif fill_weights == "error":
            raise ValueError("weights contains NaNs and fill_weights='error'")
        else:
            raise ValueError(f"Unknown fill_weights policy: {fill_weights}")

    # Long-only constraint (decision-time)
    if long_only and (weights < -1e-12).any().any():
        raise ValueError("long-only violated in decision weights (negative weights found)")

    # Gross constraint (decision-time)
    gross_decision = weights.abs().sum(axis=1)
    if not allow_leverage and (gross_decision > max_gross).any():
        raise ValueError("gross exposure exceeds max_gross in decision weights and allow_leverage=False")

    # Returns (mode-aware), aligned to close grid
    rets = _get_rets_from_md(md, entry_mode=entry_mode).reindex(index=close.index, columns=close.columns)

    # No-lookahead
    w_used = weights.shift(1).fillna(0.0)

    # Long-only constraint (used weights)
    if long_only and (w_used < -1e-12).any().any():
        raise ValueError("long-only violated in weights_used")

    # Gross exposure (used weights)
    gross_exposure = w_used.abs().sum(axis=1)
    if not allow_leverage and (gross_exposure > max_gross).any():
        raise ValueError("gross exposure exceeds max_gross in weights_used and allow_leverage=False")

    # Cash return (cash-funded; no borrow cost modeled)
    cash_weight = (1.0 - gross_exposure).clip(lower=0.0)
    daily_rf = (1.0 + rf_annual) ** (1.0 / trading_days) - 1.0
    cash_ret = cash_weight * daily_rf

    # Portfolio return
    port_gross = (w_used * rets).sum(axis=1) + cash_ret

    # Turnover + fee timing (fees paid when weights become active)
    turnover_trade = (weights - weights.shift(1)).abs().sum(axis=1).fillna(0.0)
    turnover_exec = turnover_trade.shift(1).fillna(0.0)
    fee = (fee_bps / 10000.0) * turnover_exec

    port_net = port_gross - fee

    equity_gross = (1.0 + port_gross).cumprod()
    equity_net = (1.0 + port_net).cumprod()

    return PortfolioBacktestResult(
        gross_ret=port_gross,
        net_ret=port_net,
        equity_gross=equity_gross,
        equity_net=equity_net,
        turnover=turnover_exec,
        fee=fee,
        exposure=gross_exposure,  # IMPORTANT: gross exposure
        cash_weight=cash_weight,
        weights_used=w_used,
    )


def run_one(
    md: MarketData,
    scores: pd.DataFrame,
    cfg: ExperimentConfig,
    *,
    universe_eligible: Optional[pd.DataFrame] = None,
) -> dict:
    start = pd.to_datetime(cfg.start)
    end = pd.to_datetime(cfg.end) if cfg.end is not None else None

    close_df = md.close.sort_index().loc[start:end]
    open_df = None
    if getattr(md, "open", None) is not None:
        open_df = md.open.sort_index().reindex(index=close_df.index, columns=close_df.columns)

    md_s = md
    md_s.close = close_df
    md_s.open = open_df

    scores = scores.sort_index().reindex(index=close_df.index, columns=close_df.columns)

    ue = None
    if universe_eligible is not None:
        ue = universe_eligible.sort_index().reindex(index=close_df.index, columns=close_df.columns).fillna(False)

    # strategy weights via new StrategySpec
    w_strat, w_diag = build_strategy_weights(md_s, scores, cfg, universe_eligible=ue)

    strat = backtest_weights(
        md=md_s,
        weights=w_strat,
        entry_mode=cfg.entry_mode,
        fee_bps=cfg.fee_bps,
        rf_annual=cfg.rf_annual,
        long_only=bool(cfg.strategy.selector.long_only),
        allow_leverage=False,
        max_gross=1.01,
    )

    w_bench = build_benchmark_weights(close_df, cfg)
    bench = backtest_weights(
        md=md_s,
        weights=w_bench,
        entry_mode=cfg.entry_mode,
        fee_bps=0.0,
        rf_annual=cfg.rf_annual,
        long_only=True,
        allow_leverage=False,
        max_gross=1.01,
    )

    stats_s = pd.Series(perf_summary(strat.net_ret, freq=TRADING_DAYS), name="Strategy")
    stats_b = pd.Series(perf_summary(bench.net_ret, freq=TRADING_DAYS), name="Benchmark")

    return {
        "cfg": cfg,
        "scores": scores,
        "weights_strategy": w_strat,
        "weights_diag": w_diag,
        "weights_benchmark": w_bench,
        "strat": strat,
        "bench": bench,
        "stats": pd.concat([stats_s, stats_b], axis=1),
    }

def build_benchmark_weights(prices: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    cols = prices.columns.tolist()

    if cfg.benchmark_mode == "equal_weight_all":
        w = pd.Series(1.0 / len(cols), index=cols)
        return pd.DataFrame(np.tile(w.values, (len(prices), 1)), index=prices.index, columns=cols)

    if cfg.benchmark_mode == "single_ticker":
        if not cfg.benchmark_ticker or cfg.benchmark_ticker not in cols:
            raise ValueError("benchmark_ticker must be in prices.columns")
        w = pd.Series(0.0, index=cols)
        w[cfg.benchmark_ticker] = 1.0
        return pd.DataFrame(np.tile(w.values, (len(prices), 1)), index=prices.index, columns=cols)

    raise ValueError(f"Unknown benchmark_mode: {cfg.benchmark_mode}")


def plot_compare(strat: PortfolioBacktestResult, bench: PortfolioBacktestResult, title: str) -> None:
    # equity
    eq_s, ret_s = strat.equity_net, strat.net_ret
    eq_b, ret_b = bench.equity_net, bench.net_ret

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(eq_s.index, eq_s.values, label="Strategy (Net)")
    ax.plot(eq_b.index, eq_b.values, label="Benchmark")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    plt.show()

    # drawdown
    dd_s = drawdown_series_from_equity(eq_s)
    dd_b = drawdown_series_from_equity(eq_b)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(dd_s.index, dd_s.values, label="Strategy")
    ax.plot(dd_b.index, dd_b.values, label="Benchmark")
    ax.set_title("Drawdown")
    ax.legend()
    fig.tight_layout()
    plt.show()

    # yearly returns
    yr_s = yearly_returns(ret_s).rename("Strategy")
    yr_b = yearly_returns(ret_b).rename("Benchmark")
    yr_tbl = pd.concat([yr_s, yr_b], axis=1).sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    yr_tbl.plot(kind="bar", ax=ax)
    ax.set_title("Calendar-Year Returns")
    ax.set_ylabel("Return")
    ax.set_xlabel("Year")
    fig.tight_layout()
    plt.show()

