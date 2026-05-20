from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


class MembershipGate(Protocol):
    """
    Universe membership gating interface:
      returns boolean mask (dates x tickers): True if ticker is in-universe on date.
    """
    name: str

    def build_mask(
        self,
        *,
        dates: pd.DatetimeIndex,
        tickers: pd.Index,
        effective_on_close: bool = True,
    ) -> pd.DataFrame:
        ...


