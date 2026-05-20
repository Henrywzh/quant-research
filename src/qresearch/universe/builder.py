# ============================================================
# qresearch/universe/build.py
# ============================================================

from __future__ import annotations

from typing import Dict
import pandas as pd
from qresearch.data.types import MarketData
from qresearch.universe.gating.ashare_connect import AShareConnectMembershipGate
from qresearch.universe.filters import UniverseFilterConfig, build_universe_eligible, _align_bool
from qresearch.universe.gating.base import MembershipGate


def build_ashare_connect_universe_eligible(
    *,
    md: "MarketData",
    connect_events_df: pd.DataFrame,
    cfg: "UniverseFilterConfig",
    mcap: pd.DataFrame | None = None,           # matrix (dates x tickers), optional
    effective_on_close: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Step 6:
      - Build membership mask from A股通 events
      - Build investable mask from cfg (price floor / mcap / ipo seasoning / liquidity)
      - Combine to final_eligible

    Note:
      turnover_value uses md.turnover (JQ money mapped to Turnover in MarketData)
    """
    gate = AShareConnectMembershipGate(
        events=connect_events_df,
        code_col="code",
        date_col="change_date",
        dir_col="direction",
        seed_members=None,
    )

    bundle = prepare_universe_bundle(
        close=md.close,
        gate=gate,
        cfg=cfg,
        mcap=mcap,
        turnover_value=md.turnover,          # used if cfg.min_mean_turnover_value is set
        effective_on_close=effective_on_close,
    )
    return bundle



def build_final_universe_eligible(
    close: pd.DataFrame,
    *,
    gate: MembershipGate | None,
    cfg: UniverseFilterConfig,
    mcap: pd.DataFrame | None = None,
    turnover_value: pd.DataFrame | None = None,
    life_span_df: pd.DataFrame | None = None,
    effective_on_close: bool = True,
) -> pd.DataFrame:
    """
    final_eligible[t,i] = member_mask[t,i] & exists_mask[t,i] & investable_mask[t,i]

    - gate=None => membership gate is all True (no membership restriction)
    - exists_mask defaults to close.notna() (proxy for "listed/tradable in dataset")
    """
    dates, tickers = close.index, close.columns

    exists = close.notna()

    if gate is None:
        member = pd.DataFrame(True, index=dates, columns=tickers)
    else:
        member = gate.build_mask(dates=dates, tickers=tickers, effective_on_close=effective_on_close)
    member = _align_bool(member, close)

    investable = build_universe_eligible(
        close=close,
        cfg=cfg,
        mcap=mcap,
        turnover_value=turnover_value,
        life_span_df=life_span_df
    )
    investable = _align_bool(investable, close)

    return _align_bool(member & exists & investable, close)


def prepare_universe_bundle(
    close: pd.DataFrame,
    *,
    gate: MembershipGate | None,
    cfg: UniverseFilterConfig,
    mcap: pd.DataFrame | None = None,
    turnover_value: pd.DataFrame | None = None,
    life_span_df: pd.DataFrame | None = None,
    effective_on_close: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Convenience: return intermediate masks for debugging/plots.
    """
    dates, tickers = close.index, close.columns
    exists = close.notna()

    if gate is None:
        member = pd.DataFrame(True, index=dates, columns=tickers)
    else:
        member = gate.build_mask(dates=dates, tickers=tickers, effective_on_close=effective_on_close)
    member = _align_bool(member, close)

    # --- FIX: No membership history before first event date ---
    if gate is not None and hasattr(gate, "events"):
        # assumes gate.events is the same dataframe you passed in
        d0 = pd.to_datetime(gate.events[cfg.date_col if hasattr(cfg, "date_col") else "change_date"]).min()
        member.loc[member.index < d0, :] = False


    investable = build_universe_eligible(close=close, cfg=cfg, mcap=mcap, turnover_value=turnover_value, life_span_df=life_span_df)
    investable = _align_bool(investable, close)

    final_eligible = _align_bool(member & exists & investable, close)

    return {
        "member": member,
        "exists": exists.reindex_like(close).fillna(False).astype(bool),
        "investable": investable,
        "final_eligible": final_eligible,
    }
