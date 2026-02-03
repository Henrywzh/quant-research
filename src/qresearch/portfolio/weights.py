from __future__ import annotations


from typing import Literal, Optional, Any, Dict

import numpy as np
import pandas as pd

from qresearch.backtest.metrics import DEFAULT_DDOF
from qresearch.backtest.config import (
    TopKBookConfig, AllocationConfig, AllocationOverlay, ExperimentConfig
)
from qresearch.data.types import MarketData


def build_topk_book_weights(
    md: MarketData,
    scores: pd.DataFrame,
    cfg: TopKBookConfig,
    *,
    universe_eligible: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Decision-time weights (UNSHIFTED). Backtest must do weights.shift(1).

    long_only:
      - choose top long_k
      - assign +long_budget equally
      - cash is implicit (1 - gross_exposure), handled in backtest

    long_short:
      - choose top long_k and bottom short_k
      - assign +long_budget to longs, -short_budget to shorts
    """
    prices = md.close.sort_index()
    scores = scores.reindex(index=prices.index, columns=prices.columns)

    if universe_eligible is not None:
        ue = universe_eligible.reindex(index=prices.index, columns=prices.columns).fillna(False)
    else:
        ue = pd.DataFrame(True, index=prices.index, columns=prices.columns)

    # validate budgets
    if cfg.long_budget < 0 or cfg.short_budget < 0:
        raise ValueError("long_budget and short_budget must be >= 0")

    if cfg.long_only and cfg.short_budget != 0:
        raise ValueError("For long_only, short_budget must be 0")

    if cfg.long_only and cfg.long_budget <= 0:
        raise ValueError("For long_only, long_budget must be > 0")

    if not cfg.long_only:
        if cfg.long_budget <= 0 or cfg.short_budget <= 0:
            raise ValueError("For long_short, long_budget and short_budget must be > 0")
        if cfg.short_k <= 0:
            raise ValueError("For long_short, short_k must be > 0")

    rebal_dates = _rebalance_dates(prices.index, cfg)  # assumed existing
    w_reb = pd.DataFrame(0.0, index=rebal_dates, columns=prices.columns)
    prev_top1 = None

    for dt in rebal_dates:
        row = scores.loc[dt].replace([np.inf, -np.inf], np.nan)

        # eligibility gate
        row = row.where(ue.loc[dt])

        # missing score handling
        if cfg.fill_missing_scores:
            row_long = row.fillna(-np.inf)
            row_short = row.fillna(+np.inf)
        else:
            row_long = row.copy()
            row_short = row.copy()

        # ---- COOLDOWN: ban previous rebalance's top1 for this rebalance ----
        if prev_top1 is not None and prev_top1 in row_long.index:
            row_long.loc[prev_top1] = -np.inf

        # valid masks
        valid_long = row_long.replace([np.inf, -np.inf], np.nan).notna()
        valid_short = row_short.replace([np.inf, -np.inf], np.nan).notna()

        if cfg.long_filter is not None:
            m = cfg.long_filter(md, scores, dt).reindex(prices.columns).fillna(False)
            valid_long &= m

        if cfg.short_filter is not None:
            m = cfg.short_filter(md, scores, dt).reindex(prices.columns).fillna(False)
            valid_short &= m

        # select longs
        long_candidates = row_long[valid_long].dropna()
        long_sel = (
            long_candidates.sort_values(ascending=False).head(cfg.long_k)
            if cfg.long_k > 0 else long_candidates.iloc[:0]
        )
        nL = len(long_sel)

        # select shorts (only if long_short)
        if not cfg.long_only:
            short_candidates = row_short[valid_short].dropna()
            short_sel = (
                short_candidates.sort_values(ascending=True).head(cfg.short_k)
                if cfg.short_k > 0 else short_candidates.iloc[:0]
            )
            nS = len(short_sel)
        else:
            short_sel = long_candidates.iloc[:0]
            nS = 0

        w = pd.Series(0.0, index=prices.columns)

        if cfg.long_only:
            if nL == 0:
                w_reb.loc[dt] = w.values
                continue
            w.loc[long_sel.index] = +cfg.long_budget / nL
            w_reb.loc[dt] = w.values
            continue

        # long_short
        if cfg.require_both_sides and (nL == 0 or nS == 0):
            w_reb.loc[dt] = w.values
            continue

        if nL > 0:
            w.loc[long_sel.index] = +cfg.long_budget / nL
        if nS > 0:
            w.loc[short_sel.index] = -cfg.short_budget / nS

        w_reb.loc[dt] = w.values

        # ---- update cooldown state ----
        prev_top1 = long_sel.index[0]  # for K=1; if K>1 you can choose which to ban

    return w_reb.reindex(prices.index).ffill().fillna(0.0)


def _rebalance_dates(index: pd.DatetimeIndex, cfg: TopKBookConfig) -> pd.DatetimeIndex:
    """
    Produce rebalance dates from the TRADING calendar.

    calendar mode:
      - rebalance on the last trading date inside each calendar bin given by cfg.rebalance
        (e.g. for W-FRI, use Thu if Fri is a holiday)

    fixed_h mode:
      - rebalance every H trading days starting from cfg.offset
    """
    index = pd.DatetimeIndex(index).sort_values()

    if cfg.rebalance_mode == "calendar":
        # Group trading dates into calendar bins and take the last timestamp in each bin.
        # This yields actual tradable dates, not bin labels.
        grouped_last = (
            pd.Series(index=index, data=index)
            .groupby(pd.Grouper(freq=cfg.rebalance))
            .max()
            .dropna()
        )
        return pd.DatetimeIndex(grouped_last.values)

    if cfg.rebalance_mode == "fixed_h":
        if cfg.H <= 0:
            raise ValueError("TopKConfig.H must be > 0 for mode='fixed_h'")
        start = int(cfg.offset)
        if start < 0 or start >= len(index):
            raise ValueError("TopKConfig.offset out of range")
        return pd.DatetimeIndex(index[start::cfg.H])

    raise ValueError(f"Unknown mode: {cfg.rebalance_mode}")


def build_strategy_weights(
    md: Any,  # MarketData (already sliced)
    scores: pd.DataFrame,
    cfg: ExperimentConfig,
    *,
    universe_eligible: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Decision-time weights (UNSHIFTED), compatible with backtest_weights().

    Workflow:
      1) w_base from build_topk_book_weights (selection + schedule + base equal within leg)
      2) optional allocation override on rebalance dates (equal / inv-vol / score-prop)
      # 3) optional stop-loss overlay (path-dependent on prices, long-only initially)
      # 4) optional vol targeting overlay (portfolio-level exposure scaling, no leverage)
      5) return final daily decision weights + diagnostics
    """
    """
    Decision-time weights (UNSHIFTED), compatible with backtest_weights().
    """

    topk = cfg.strategy.selector
    alloc = cfg.strategy.allocator
    overlays = cfg.strategy.overlays

    prices = md.close.sort_index()
    scores = scores.reindex(index=prices.index, columns=prices.columns)

    ue = None
    if universe_eligible is not None:
        ue = universe_eligible.reindex(index=prices.index, columns=prices.columns).fillna(False)

    # 1) base book (selection + schedule)
    w = build_topk_book_weights(md=md, scores=scores, cfg=topk, universe_eligible=ue)

    # 2) shared context (everything overlays might need)
    rebal_dates = _rebalance_dates(prices.index, topk)
    ctx: Dict[str, Any] = {
        "exp_cfg": cfg,
        "topk_cfg": topk,
        "alloc_cfg": alloc,
        "scores": scores,
        "rebal_dates": rebal_dates,
        "universe_eligible": ue,
    }

    diag: Dict[str, Any] = {"w_base": w}

    # 3) allocation step (always run; it may be no-op)
    w, alloc_diag = AllocationOverlay(alloc).apply(md, w, ctx)
    diag["allocation"] = alloc_diag
    diag["w_alloc"] = w

    # 4) overlays pipeline (explicit order)
    for ov in overlays:
        w, d = ov.apply(md, w, ctx)
        diag[ov.name] = d

    diag["w_final_pre_align"] = w

    # 5) final alignment safety
    w = w.reindex(index=prices.index, columns=prices.columns).fillna(0.0)

    # long-only safety
    if topk.long_only and (w < -1e-12).any().any():
        raise ValueError("build_strategy_weights produced negative weights under long_only=True")

    # decision-time gross safety if you want (optional; backtest_weights also checks)
    gross = w.abs().sum(axis=1)
    if (gross > 1.01).any():
        raise ValueError("build_strategy_weights produced gross > 1.01")

    diag["w_final"] = w

    return w, diag

