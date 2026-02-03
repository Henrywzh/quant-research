from dataclasses import dataclass, field
from typing import Literal, Optional, Callable, Any, Dict, Tuple, Protocol

import numpy as np
import pandas as pd

from qresearch.backtest.metrics import TRADING_DAYS, DEFAULT_DDOF
from qresearch.data.types import MarketData

EntryMode = Literal["next_close", "next_open", "open_to_close"]


RebalanceMode = Literal["calendar", "fixed_h"]
Rebalance = Literal["D", "W-FRI", 'ME']


@dataclass(frozen=True)
class TopKBookConfig:
    # schedule
    rebalance_mode: RebalanceMode = "calendar"
    rebalance: Rebalance = "W-FRI"           # e.g. "D", "W-FRI", "ME"
    H: int = 5
    offset: int = 0

    # book mode
    long_only: bool = True

    # selection
    long_k: int = 1
    short_k: int = 0
    fill_missing_scores: bool = False  # NaN -> -inf for long, +inf for short

    # filters (plug-in functions returning boolean masks aligned to tickers)
    long_filter: Optional[Callable[[MarketData, Any, pd.Timestamp], pd.Series]] = None
    short_filter: Optional[Callable[[MarketData, Any, pd.Timestamp], pd.Series]] = None

    # explicit leg budgets
    long_budget: float = 1.0
    short_budget: float = 0.0     # absolute notional on shorts

    require_both_sides: bool = False  # if either side empty after filters -> go flat


@dataclass(frozen=True)
class AllocationConfig:
    """
    Optional allocation override applied AFTER base selection.
    It redistributes within the selected set on rebalance dates,
    while preserving leg budgets: sum(longs)=long_budget, sum(|shorts|)=short_budget.
    """
    method: Literal["base_equal", "equal", "inv_vol", "score_prop"] = "base_equal"
    # common
    w_cap: Optional[float] = None  # cap per-asset absolute weight (within the leg), e.g. 0.8
    # inv_vol params
    vol_window: int = 20
    ann_factor: int = TRADING_DAYS
    # score_prop params
    score_clip_floor: Optional[float] = 0.0  # set to 0.0 for long-only gate in allocation; None to allow negatives


# -------------------------
# Overlay Protocol (interface)
# -------------------------
class Overlay(Protocol):
    name: str

    def apply(
        self,
        md: Any,
        w: pd.DataFrame,
        ctx: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Take decision-time weights (UNSHIFTED) and return new decision-time weights + diagnostics.
        ctx can carry useful shared info (scores, rebal_dates, cfg, etc.).
        """
        ...

# -------------------------
# StrategySpec (one container for strategy logic)
# -------------------------
@dataclass(frozen=True)
class StrategySpec:
    selector: TopKBookConfig = field(default_factory=TopKBookConfig)
    allocator: AllocationConfig = field(default_factory=lambda: AllocationConfig(method="base_equal"))
    overlays: Tuple[Overlay, ...] = ()  # ordered pipeline


@dataclass(frozen=True)
class VolTargetConfig:
    """
    Portfolio-level vol targeting overlay (no leverage by default).
    exposure(t) is computed on rebalance dates, then ffilled until next rebalance.
    """
    enabled: bool = True
    sigma_target: float = 0.20
    vol_window: int = 60
    ann_factor: int = TRADING_DAYS
    exposure_floor: float = 0.0
    exposure_cap: float = 1.0   # keep 1.0 for no leverage
    use_cov: bool = True        # True: full covariance; False: diagonal approximation


@dataclass(frozen=True)
class AssetVolTargetConfig:
    enabled: bool = True
    sigma_target: float = 0.20
    vol_window: int = 20
    ann_factor: int = TRADING_DAYS
    w_cap: float = 1.0          # no leverage; cap <= 1.0
    vol_floor: float = 1e-6     # avoid blow-ups

# -------------------------
# 1) Config: 描述 stop 规则（参数）
# -------------------------

@dataclass(frozen=True)
class PortfolioDDSwitchConfig:
    enabled: bool = True
    dd_limit: float = 0.10              # 10%
    lockout_days: int = 10              # stay in cash for N trading days after trigger
    eval_on: str = "close"              # "close" only (fits your decision time)

@dataclass(frozen=True)
class TradeDrawdownStopConfig:
    enabled: bool = True
    dd_limit: float = 0.10          # 10% trade drawdown
    cooldown_days: int = 1          # optional: block re-entry after stop
    use_close_for_dd: bool = True   # you have OHLCV; close-based is simplest


class TradeDrawdownStopOverlay:
    """
    Trade-level drawdown stop (trailing stop) applied to a Top1-style strategy.

    - Detect stop at close[t] using close price vs peak close since entry.
    - Enforce exit by setting decision weights[t]=0 (so exit occurs at open[t+1] in backtest).
    """

    def __init__(self, cfg: TradeDrawdownStopConfig):
        self.cfg = cfg

    def apply(
        self,
        md,
        w_target: pd.DataFrame,   # decision weights at close[t] (UNSHIFTED)
    ) -> Tuple[pd.DataFrame, dict]:

        if not self.cfg.enabled:
            return w_target, {"note": "dd stop disabled"}

        close = md.close.sort_index()
        w_target = w_target.reindex(index=close.index, columns=close.columns).fillna(0.0)

        dates = close.index
        cols = close.columns

        # state per ticker (works for Top1; also ok for sparse portfolios)
        in_pos: Dict[str, bool] = {t: False for t in cols}
        peak_px: Dict[str, float] = {t: np.nan for t in cols}
        cooldown_end_pos: Dict[str, int] = {t: -10**9 for t in cols}

        date_pos = pd.Series(np.arange(len(dates)), index=dates)

        w_after = w_target.copy()

        exit_flag = pd.DataFrame(False, index=dates, columns=cols)
        dd_now_df = pd.DataFrame(np.nan, index=dates, columns=cols)
        peak_df = pd.DataFrame(np.nan, index=dates, columns=cols)

        for dt in dates:
            p = int(date_pos.loc[dt])
            trade_pos = p + 1  # execution at next open in your convention

            # read desired holdings (Top1 => one-hot, but not required)
            w_row = w_target.loc[dt]

            for tkr in cols:
                desired = float(w_row.get(tkr, 0.0))
                want_hold = desired > 1e-12

                # cooldown blocks re-entry
                if trade_pos <= cooldown_end_pos[tkr] and want_hold:
                    w_after.loc[dt, tkr] = 0.0
                    want_hold = False

                px = float(close.loc[dt, tkr])

                # entry detection
                if want_hold and not in_pos[tkr]:
                    in_pos[tkr] = True
                    peak_px[tkr] = px

                # if strategy exits normally
                if (not want_hold) and in_pos[tkr]:
                    in_pos[tkr] = False
                    peak_px[tkr] = np.nan
                    continue

                # update peak and check drawdown
                if in_pos[tkr]:
                    peak_px[tkr] = max(float(peak_px[tkr]), px)
                    dd_now = px / float(peak_px[tkr]) - 1.0
                    dd_now_df.loc[dt, tkr] = dd_now
                    peak_df.loc[dt, tkr] = peak_px[tkr]

                    if dd_now <= -self.cfg.dd_limit:
                        # force exit: decision weight=0 -> exit at next open
                        w_after.loc[dt, tkr] = 0.0
                        exit_flag.loc[dt, tkr] = True

                        # set cooldown (in trade steps)
                        if self.cfg.cooldown_days > 0:
                            cooldown_end_pos[tkr] = trade_pos + self.cfg.cooldown_days

                        # reset state (we decided to exit)
                        in_pos[tkr] = False
                        peak_px[tkr] = np.nan

        diag = {"exit_flag": exit_flag, "dd": dd_now_df, "peak_px": peak_df}
        return w_after, diag


@dataclass(frozen=True)
class ExperimentConfig:
    start: str = "2015-01-01"
    end: Optional[str] = None

    # execution/return convention
    entry_mode: EntryMode = "next_close"

    fee_bps: float = 1.0
    rf_annual: float = 0.015

    benchmark_mode: Literal["equal_weight_all", "single_ticker"] = "single_ticker"
    benchmark_ticker: Optional[str] = None

    signal_name: str = "mom_ret"
    signal_params: Dict[str, Any] = field(default_factory=lambda: {"lookback": 21, "skip": 0})

    # NEW: one container for strategy logic
    strategy: StrategySpec = field(default_factory=StrategySpec)


@dataclass(frozen=True)
class AllocationOverlay:
    cfg: AllocationConfig
    name: str = "allocation"

    def apply(self, md: Any, w: pd.DataFrame, ctx: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        # If "base_equal", do nothing
        if self.cfg is None or self.cfg.method == "base_equal":
            return w, {"note": "allocator=base_equal (no-op)"}

        scores: pd.DataFrame = ctx["scores"]
        rebal_dates = ctx["rebal_dates"]
        topk_cfg: TopKBookConfig = ctx["topk_cfg"]


        w2 = _apply_allocation_override(
            md=md,
            scores=scores,
            w_base=w,
            rebal_dates=rebal_dates,
            topk_cfg=topk_cfg,
            alloc_cfg=self.cfg,
        )
        return w2, {"method": self.cfg.method}


@dataclass(frozen=True)
class AssetVolTargetOverlay:
    cfg: AssetVolTargetConfig
    name: str = "asset_vol_target"

    def apply(self, md: Any, w: pd.DataFrame, ctx: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if self.cfg is None or not self.cfg.enabled:
            return w, {"note": "asset_vol_target disabled (no-op)"}

        w2, vt_diag = _apply_asset_vol_targeting(md=md, w_target=w, cfg=self.cfg)
        return w2, vt_diag


@dataclass(frozen=True)
class PortfolioDDSwitchOverlay:
    cfg: PortfolioDDSwitchConfig
    name: str = "portfolio_dd_switch"

    def apply(self, md: Any, w: pd.DataFrame, ctx: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if self.cfg is None or not self.cfg.enabled:
            return w, {"note": "portfolio_dd_switch disabled (no-op)"}

        # we need experiment-level execution params (fee/rf/entry_mode)
        exp: ExperimentConfig = ctx["exp_cfg"]

        w2, dd_diag = _apply_portfolio_dd_kill_switch(
            md,
            w,
            entry_mode=exp.entry_mode,
            fee_bps=exp.fee_bps,
            rf_annual=exp.rf_annual,
            cfg=self.cfg,
        )
        return w2, dd_diag


@dataclass(frozen=True)
class TradeDrawdownStopOverlayWrapper:
    cfg: TradeDrawdownStopConfig
    name: str = "trade_dd_stop"

    def apply(self, md: Any, w: pd.DataFrame, ctx: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if self.cfg is None or not self.cfg.enabled:
            return w, {"note": "trade_dd_stop disabled (no-op)"}

        w2, diag = TradeDrawdownStopOverlay(self.cfg).apply(md, w)
        return w2, diag


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


def _get_rets_from_md(md: "MarketData", entry_mode: EntryMode) -> pd.DataFrame:
    """
    Daily return grid aligned to md.close.index/columns.

    Conventions (consistent with your "signal at close[t], enter at t+1"):
    - next_close: close-to-close returns for day t (close[t]/close[t-1]-1), weights_used[t]=weights[t-1]
    - next_open:  open-to-open returns  for day t (open[t+1]/open[t]-1),  weights_used[t]=weights[t-1]
    - open_to_close: intraday returns   for day t (close[t]/open[t]-1),     weights_used[t]=weights[t-1]
    """
    close = md.close.sort_index()
    cols = close.columns

    if entry_mode == "next_close":
        rets = close.pct_change(fill_method=None)

    elif entry_mode == "next_open":
        if md.open is None:
            raise ValueError("md.open is required for entry_mode='next_open'")
        open_ = md.open.sort_index().reindex(index=close.index, columns=cols)
        rets = open_.pct_change(fill_method=None).shift(-1)

    elif entry_mode == "open_to_close":
        if md.open is None:
            raise ValueError("md.open is required for entry_mode='open_to_close'")
        open_ = md.open.sort_index().reindex(index=close.index, columns=cols)
        rets = close / open_ - 1.0

    else:
        raise ValueError("entry_mode must be one of: next_close, next_open, open_to_close")

    rets = rets.replace([np.inf, -np.inf], np.nan)
    rets.iloc[0] = 0.0
    return rets.fillna(0.0)


def _apply_asset_vol_targeting(
    md,
    w_target: pd.DataFrame,          # decision weights (UNSHIFTED)
    cfg: AssetVolTargetConfig,
) -> tuple[pd.DataFrame, dict]:

    if not cfg.enabled:
        return w_target, {"note": "asset vol targeting disabled"}

    close = md.close.sort_index()
    w = w_target.reindex(index=close.index, columns=close.columns).fillna(0.0)

    # estimate per-asset vol from close-to-close returns (stable, independent of execution mode)
    ret = close.pct_change()
    vol = ret.rolling(cfg.vol_window, min_periods=cfg.vol_window).std(ddof=DEFAULT_DDOF) * np.sqrt(cfg.ann_factor)

    # scale factor per asset
    scale = (cfg.sigma_target / vol.clip(lower=cfg.vol_floor)).clip(lower=0.0, upper=cfg.w_cap)

    # no lookahead: decision at close[t] uses vol[t], trade next bar uses weights.shift(1) in backtest
    # scaling should be applied at decision-time, so no extra shift here.
    w_scaled = w * scale

    # IMPORTANT: keep cash implicit in backtest via gross exposure < 1
    # if you have long_budget != 1.0, you may want to cap gross too (optional)
    gross = w_scaled.abs().sum(axis=1)
    if (gross > 1.000001).any():
        # normalize down if needed (shouldn't happen with w_cap<=1 for Top1)
        w_scaled = w_scaled.div(gross, axis=0).fillna(0.0)

    diag = {"vol": vol, "scale": scale, "gross_scaled": gross}
    return w_scaled, diag


def _apply_vol_target_overlay(
    md: MarketData,
    w: pd.DataFrame,
    rebal_dates: pd.DatetimeIndex,
    vt_cfg: VolTargetConfig,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compute exposure on rebalance dates based on portfolio vol of w(dt),
    then ffill exposure and scale daily decision weights.
    """
    prices = md.close.sort_index()
    rets = prices.pct_change()

    exposure_reb = pd.Series(0.0, index=rebal_dates, dtype=float)
    est_vol_reb = pd.Series(np.nan, index=rebal_dates, dtype=float)

    for dt in rebal_dates:
        w_dt = w.loc[dt].fillna(0.0)
        gross = float(w_dt.abs().sum())

        if gross <= 0:
            exposure_reb.loc[dt] = 0.0
            continue

        hist = rets.loc[:dt].tail(vt_cfg.vol_window)
        if hist.dropna(how="all").shape[0] < vt_cfg.vol_window:
            exposure_reb.loc[dt] = vt_cfg.exposure_floor
            continue

        names = w_dt[w_dt != 0].index
        if len(names) == 0:
            exposure_reb.loc[dt] = 0.0
            continue

        w_vec = w_dt.loc[names].values.reshape(-1, 1).astype(float)

        if vt_cfg.use_cov:
            cov = hist.loc[:, names].cov(ddof=DEFAULT_DDOF).values
            var = float(w_vec.T @ cov @ w_vec)
        else:
            # diagonal approximation
            v = hist.loc[:, names].var(ddof=DEFAULT_DDOF).values.reshape(-1, 1)
            var = float((w_vec ** 2 * v).sum())

        if not np.isfinite(var) or var <= 0:
            exposure_reb.loc[dt] = vt_cfg.exposure_floor
            continue

        vol_ann = float(np.sqrt(var) * np.sqrt(vt_cfg.ann_factor))
        est_vol_reb.loc[dt] = vol_ann

        raw = vt_cfg.sigma_target / vol_ann if vol_ann > 0 else 0.0
        exposure_reb.loc[dt] = float(np.clip(raw, vt_cfg.exposure_floor, vt_cfg.exposure_cap))

    # ffill exposure to all days (exposure only changes at rebalance by design)
    exposure = exposure_reb.reindex(prices.index).ffill().fillna(vt_cfg.exposure_floor)

    w_scaled = w.mul(exposure, axis=0)

    return w_scaled, {"exposure": exposure, "est_vol_on_reb": est_vol_reb}


def _apply_portfolio_dd_kill_switch(
    md,
    weights: pd.DataFrame,              # decision-time weights (UNSHIFTED)
    *,
    entry_mode: str,
    fee_bps: float = 0.0,
    rf_annual: float = 0.0,
    trading_days: int = TRADING_DAYS,
    cfg: PortfolioDDSwitchConfig = PortfolioDDSwitchConfig(),
):
    """
    Realistic kill-switch: if end-of-day equity drawdown <= -dd_limit,
    set future decision weights = 0 (cash) for lockout_days.
    """
    close = md.close.sort_index()
    close, weights = _align_to_prices(close, weights)
    weights = weights.fillna(0.0)

    rets = _get_rets_from_md(md, entry_mode=entry_mode).reindex_like(close)
    w_used = weights.shift(1).fillna(0.0)

    # cash return
    daily_rf = (1.0 + rf_annual) ** (1.0 / trading_days) - 1.0
    gross_exposure = w_used.abs().sum(axis=1)
    cash_weight = (1.0 - gross_exposure).clip(lower=0.0)
    cash_ret = cash_weight * daily_rf

    port_gross = (w_used * rets).sum(axis=1) + cash_ret

    # fees (same as your backtest)
    turnover_trade = (weights - weights.shift(1)).abs().sum(axis=1).fillna(0.0)
    turnover_exec = turnover_trade.shift(1).fillna(0.0)
    fee = (fee_bps / 10000.0) * turnover_exec
    port_net = port_gross - fee

    # simulate with switch affecting future decision weights
    eq = 1.0
    peak = 1.0
    lockout_left = 0

    weights_after = weights.copy()
    eq_series = []

    idx = close.index
    for t, dt in enumerate(idx):
        # apply lockout to decision weights at dt (affects t+1 returns via shift)
        if lockout_left > 0:
            weights_after.loc[dt] = 0.0
            lockout_left -= 1

        eq *= (1.0 + port_net.loc[dt])
        peak = max(peak, eq)
        dd = eq / peak - 1.0
        eq_series.append(eq)

        # trigger at decision time: after observing dd at dt, enforce cash for next days
        if dd <= -cfg.dd_limit and lockout_left == 0:
            lockout_left = int(cfg.lockout_days)

    return weights_after, pd.Series(eq_series, index=idx, name="equity_net_killswitch")


def _apply_allocation_override(
    md: Any,
    scores: pd.DataFrame,
    w_base: pd.DataFrame,
    rebal_dates: pd.DatetimeIndex,
    topk_cfg: TopKBookConfig,
    alloc_cfg: AllocationConfig,
) -> pd.DataFrame:
    """
    Recompute weights on rebalance dates using membership implied by w_base,
    preserving leg budgets.
    """
    prices = md.close.sort_index()
    rets = prices.pct_change()

    w_reb = pd.DataFrame(0.0, index=rebal_dates, columns=prices.columns)

    for dt in rebal_dates:
        wb = w_base.loc[dt].fillna(0.0)

        long_names = wb[wb > 0].index
        short_names = wb[wb < 0].index  # negative weights

        # If selection empty, stay flat
        if topk_cfg.long_only:
            if len(long_names) == 0:
                continue
        else:
            if topk_cfg.require_both_sides and (len(long_names) == 0 or len(short_names) == 0):
                continue

        w_dt = pd.Series(0.0, index=prices.columns)

        # ---- longs ----
        if len(long_names) > 0 and topk_cfg.long_budget > 0:
            w_long = _alloc_leg(
                leg="long",
                names=list(long_names),
                dt=dt,
                scores=scores,
                rets=rets,
                alloc_cfg=alloc_cfg,
            )
            w_long = _rescale_and_cap_leg(w_long, target_gross=topk_cfg.long_budget, w_cap=alloc_cfg.w_cap)
            w_dt.loc[w_long.index] = w_long.values

        # ---- shorts ----
        if (not topk_cfg.long_only) and len(short_names) > 0 and topk_cfg.short_budget > 0:
            w_short = _alloc_leg(
                leg="short",
                names=list(short_names),
                dt=dt,
                scores=scores,
                rets=rets,
                alloc_cfg=alloc_cfg,
            )
            w_short = _rescale_and_cap_leg(w_short, target_gross=topk_cfg.short_budget, w_cap=alloc_cfg.w_cap)
            w_dt.loc[w_short.index] = -w_short.values  # negative sign for short leg

        w_reb.loc[dt] = w_dt.values

    return w_reb.reindex(prices.index).ffill().fillna(0.0)


def _alloc_leg(
    leg: Literal["long", "short"],
    names: list[str],
    dt: pd.Timestamp,
    scores: pd.DataFrame,
    rets: pd.DataFrame,
    alloc_cfg: AllocationConfig,
) -> pd.Series:
    """
    Output non-negative raw weights over `names` that sum to 1 (before leg rescale).
    """
    if alloc_cfg.method in ("equal",):
        raw = pd.Series(1.0, index=names)

    elif alloc_cfg.method == "inv_vol":
        hist = rets.loc[:dt, names].tail(alloc_cfg.vol_window)
        # if insufficient history or all-NaN, fall back to equal
        if hist.dropna(how="all").shape[0] < alloc_cfg.vol_window:
            raw = pd.Series(1.0, index=names)
        else:
            vol = hist.std(ddof=DEFAULT_DDOF)
            inv = 1.0 / vol.replace(0.0, np.nan)
            raw = inv.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if raw.sum() <= 0:
                raw = pd.Series(1.0, index=names)

    elif alloc_cfg.method == "score_prop":
        s = scores.loc[dt, names].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if leg == "long":
            if alloc_cfg.score_clip_floor is not None:
                s = s.clip(lower=float(alloc_cfg.score_clip_floor))
            raw = s
        else:
            # for shorts, allocate by "how strongly short" -> use -score, clipped >=0
            raw = (-s).clip(lower=0.0)

        raw = raw.fillna(0.0)
        if raw.sum() <= 0:
            raw = pd.Series(1.0, index=names)

    else:
        # base_equal or unknown -> equal
        raw = pd.Series(1.0, index=names)

    # normalize to sum=1
    raw = raw.astype(float)
    raw_sum = float(raw.sum())
    if raw_sum <= 0:
        raw = pd.Series(1.0, index=names)
        raw_sum = float(raw.sum())
    return raw / raw_sum


def _rescale_and_cap_leg(w_unit: pd.Series, target_gross: float, w_cap: Optional[float]) -> pd.Series:
    """
    Scale unit-sum leg weights to target_gross, then optionally cap each name to w_cap and renormalize.
    """
    w = w_unit * float(target_gross)

    if w_cap is None:
        return w

    cap = float(w_cap)
    if cap <= 0:
        raise ValueError("w_cap must be > 0")

    # cap, then redistribute the remainder proportionally among non-capped names
    w = w.clip(upper=cap)

    s = float(w.sum())
    if s == 0:
        return w

    # If total is below target due to capping, renormalize to target_gross (no leverage; just within leg)
    # This increases weights of non-capped names but respects cap already applied.
    # If everything is capped and sum < target, we cannot reach target -> keep as-is.
    if s < target_gross:
        slack_names = w[w < cap - 1e-12].index
        if len(slack_names) == 0:
            return w
        slack_sum = float(w.loc[slack_names].sum())
        if slack_sum <= 0:
            return w
        scale = (target_gross - (s - slack_sum)) / slack_sum
        w.loc[slack_names] *= scale
        w = w.clip(upper=cap)

    # Final small renorm to exactly target_gross if possible (numerical)
    final_sum = float(w.sum())
    if final_sum > 0:
        w *= (target_gross / final_sum)

    return w

