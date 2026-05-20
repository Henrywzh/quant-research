# ============================================================
# qresearch/universe/gating/ashare_connect.py
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal

import numpy as np
import pandas as pd

from qresearch.data.io import load_limit_daily_with_open
from qresearch.universe.gating.base import MembershipGate


@dataclass(frozen=True)
class AShareConnectMembershipConfig:
    effective_on_close: bool = True


@dataclass(frozen=True)
class AShareConnectExecutionConfig:
    lock_abs_tol: float = 0.01
    nan_limit_policy: Literal["allow", "block"] = "allow"


def _normalize_dates_index(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # Ensure normalized midnight and sorted unique
    d = pd.to_datetime(dates).normalize()
    if not d.is_monotonic_increasing:
        d = d.sort_values()
    return pd.DatetimeIndex(d)


def _map_effective_date(
    trade_dates: pd.DatetimeIndex,
    d: pd.Timestamp,
    *,
    effective_on_close: bool,
) -> int | None:
    """
    Map an event calendar date to an index position in trade_dates.

    Convention:
      - effective_on_close=True:
          event on date d affects membership at end-of-day d
          => mask is True for d and onward (pos = first >= d)
      - effective_on_close=False:
          conservative: event on date d affects membership from next trading day
          => mask changes from the first trading day strictly after d (pos = first > d)

    Returns:
      position (0..n-1) or None if after last date.
    """
    d = pd.Timestamp(d).normalize()
    if effective_on_close:
        pos = trade_dates.searchsorted(d, side="left")
    else:
        pos = trade_dates.searchsorted(d, side="right")
    if pos >= len(trade_dates):
        return None
    return int(pos)


def build_ashare_connect_member_mask(
    *,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
    events: pd.DataFrame,
    code_col: str = "code",
    date_col: str = "change_date",
    dir_col: str = "direction",
    effective_on_close: bool = True,
    seed_members: set[str] | None = None,
    membership_config: AShareConnectMembershipConfig | None = None,
    return_audit: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, object]]:
    cfg = membership_config or AShareConnectMembershipConfig(effective_on_close=effective_on_close)
    trade_dates = _normalize_dates_index(dates)
    tickers = pd.Index(tickers)

    nT, nN = len(trade_dates), len(tickers)
    out = np.zeros((nT, nN), dtype=bool)

    seed_members = set(seed_members or set())

    # -------- FIX 1: initialize seed for ALL tickers (not only those with events) --------
    if seed_members:
        # normalize ticker strings to match
        seed_members_norm = {str(x).strip() for x in seed_members}
        tkr_arr = np.asarray([str(x).strip() for x in tickers], dtype=object)
        seed_mask = np.isin(tkr_arr, list(seed_members_norm))  # ndarray[bool], shape (nN,)
        out[:, seed_mask] = True

    # If no events, just return seed-based membership (or all True if you prefer)
    if events is None or len(events) == 0:
        return pd.DataFrame(out, index=trade_dates, columns=tickers)

    # -------- parse events --------
    ev = events[[code_col, date_col, dir_col]].copy()
    ev[code_col] = ev[code_col].astype("string").str.strip()
    ev[date_col] = pd.to_datetime(ev[date_col], errors="coerce").dt.normalize()
    ev = ev.dropna(subset=[code_col, date_col, dir_col])

    def _dir_to_delta(x: str) -> int | None:
        s = str(x).strip().lower()
        if s in {"in", "add", "added", "加入", "进入", "entry"}:
            return +1
        if s in {"out", "remove", "removed", "删除", "剔除", "exit"}:
            return -1
        return None

    ev["_delta"] = ev[dir_col].map(_dir_to_delta)
    ev = ev.dropna(subset=["_delta"]).copy()
    ev["_delta"] = ev["_delta"].astype(int)

    g = ev.groupby(code_col, sort=False)
    ticker_to_col = {t: j for j, t in enumerate(tickers)}
    events_applied = 0
    skipped_unknown_tickers: list[str] = []

    # -------- replay events only for tickers with events (overlay on top of seed) --------
    for tkr, df_t in g:
        if tkr not in ticker_to_col:
            skipped_unknown_tickers.append(str(tkr))
            continue
        j = ticker_to_col[tkr]

        diff = np.zeros(nT, dtype=np.int16)
        for d, delta in zip(df_t[date_col].values, df_t["_delta"].values):
            pos = _map_effective_date(trade_dates, d, effective_on_close=cfg.effective_on_close)
            if pos is None:
                continue
            diff[pos] += int(delta)
            events_applied += 1

        start = 1 if (str(tkr).strip() in seed_members) else 0
        mem = (start + np.cumsum(diff)) > 0
        out[:, j] = mem  # overwrite this column with event-replayed membership

    mask = pd.DataFrame(out, index=trade_dates, columns=tickers)
    if not return_audit:
        return mask

    audit = {
        "events_total": int(len(ev)),
        "events_applied": int(events_applied),
        "skipped_unknown_tickers": skipped_unknown_tickers,
        "effective_on_close": cfg.effective_on_close,
    }
    return mask, audit


def compute_execution_masks(
    *,
    open_: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    high_limit: pd.DataFrame,
    low_limit: pd.DataFrame,
    lock_abs_tol: float = 0.01,
    config: AShareConnectExecutionConfig | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Build daily execution feasibility masks for A-shares using limit-up/limit-down info.

    All inputs must be matrices aligned to the same grid: (dates x tickers).

    Conventions (daily-only, no intraday simulation):
      - Decision time: close of day t
      - Execution time: open of day t+1

    Definitions:
      tradeable[t,i] = (open[t,i] notna) & (close[t,i] notna) & (volume[t,i] > 0)

      limit_up_locked[t,i]   = tradeable[t,i] & |close[t,i] - high_limit[t,i]| <= lock_abs_tol
      limit_down_locked[t,i] = tradeable[t,i] & |close[t,i] - low_limit[t,i]|  <= lock_abs_tol

    Execution feasibility for orders decided at t and executed at t+1 open:
      tradeable_next_open[t,i] = tradeable[t+1,i]
      can_buy_next_open[t,i]  = tradeable_next_open[t,i] & ~limit_up_locked[t,i]
      can_sell_next_open[t,i] = tradeable_next_open[t,i] & ~limit_down_locked[t,i]

    Notes:
      - NaN limits => not treated as locked (permissive). If you want conservative blocking, change the NaN handling.
      - Last date has no t+1, so tradeable_next_open is False on the last row.
    """
    # ---- align to a common grid (intersection) ----
    # In practice you probably already aligned upstream; this is just safety.
    idx = open_.index
    cols = open_.columns
    for x in (close, volume, high_limit, low_limit):
        if not idx.equals(x.index) or not cols.equals(x.columns):
            raise ValueError("All inputs must share the same index/columns alignment.")
    cfg = config or AShareConnectExecutionConfig(lock_abs_tol=lock_abs_tol)

    # ---- base tradeability (same-day) ----
    tradeable = open_.notna() & close.notna() & (volume > 0)

    # ---- limit-locked states at close (t) with absolute tolerance ----
    limit_up_locked = close.sub(high_limit).abs() <= float(cfg.lock_abs_tol)
    limit_down_locked = close.sub(low_limit).abs() <= float(cfg.lock_abs_tol)
    if cfg.nan_limit_policy == "block":
        limit_up_locked = limit_up_locked | high_limit.isna()
        limit_down_locked = limit_down_locked | low_limit.isna()
    limit_up_locked = tradeable & limit_up_locked
    limit_down_locked = tradeable & limit_down_locked

    # ---- tradeability at next open (t+1) expressed on decision day t ----
    tradeable_next_open = tradeable.shift(-1, fill_value=False).astype(bool)

    # ---- feasibility for orders placed at t and executed at t+1 open ----
    can_buy_next_open = (tradeable_next_open & ~limit_up_locked).astype(bool)
    can_sell_next_open = (tradeable_next_open & ~limit_down_locked).astype(bool)

    return {
        "tradeable": tradeable.astype(bool),
        "tradeable_next_open": tradeable_next_open,
        "limit_up_locked": limit_up_locked.astype(bool),
        "limit_down_locked": limit_down_locked.astype(bool),
        "can_buy_next_open": can_buy_next_open,
        "can_sell_next_open": can_sell_next_open,
    }


def build_execution_masks_from_limit_daily(
    *,
    limit_with_open_parquet: Path | str,
    close_grid: pd.DataFrame,      # use md.close as the canonical grid
    lock_abs_tol: float = 0.01,
    config: AShareConnectExecutionConfig | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    End-to-end Step 4:
      - load limit_daily_with_open parquet
      - pivot to matrices aligned to close_grid
      - compute execution masks (decision at t, execute t+1 open)
    """
    mats = load_limit_daily_with_open(
        file_path=limit_with_open_parquet,
        dates=close_grid.index,
        tickers=close_grid.columns,
    )

    exec_masks = compute_execution_masks(
        open_=mats["open"],
        close=mats["close"],
        volume=mats["volume"],
        high_limit=mats["high_limit"],
        low_limit=mats["low_limit"],
        lock_abs_tol=lock_abs_tol,
        config=config,
    )
    return exec_masks


@dataclass(frozen=True)
class AShareConnectMembershipGate:
    """
    A股通 membership gate driven by event log replay.
    """
    events: pd.DataFrame
    code_col: str = "code"
    date_col: str = "change_date"
    dir_col: str = "direction"
    seed_members: set[str] | None = None
    membership_config: AShareConnectMembershipConfig = AShareConnectMembershipConfig()
    name: str = "ashare_connect"

    def build_mask(self, *, dates: pd.DatetimeIndex, tickers: pd.Index, effective_on_close: bool = True) -> pd.DataFrame:
        return build_ashare_connect_member_mask(
            dates=dates,
            tickers=tickers,
            events=self.events,
            code_col=self.code_col,
            date_col=self.date_col,
            dir_col=self.dir_col,
            effective_on_close=effective_on_close,
            seed_members=self.seed_members,
            membership_config=self.membership_config,
        )
