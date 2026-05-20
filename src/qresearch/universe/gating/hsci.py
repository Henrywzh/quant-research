# ============================================================
# qresearch/universe/gating/hsci.py
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from qresearch.universe.gating.base import MembershipGate
from qresearch.universe.hsci import build_hsci_member_mask


@dataclass(frozen=True)
class HSCIMembershipGate:
    """
    Wrapper around existing HSCI membership logic.
    """
    events: pd.DataFrame
    seed_members: set[str] | None = None
    name: str = "hsci"

    def build_mask(self, *, dates: pd.DatetimeIndex, tickers: pd.Index, effective_on_close: bool = True) -> pd.DataFrame:
        return build_hsci_member_mask(
            dates=dates,
            tickers=tickers,
            events=self.events,
            seed_members=self.seed_members,
            effective_on_close=effective_on_close,
        )
