from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=False)
class MarketData:
    close: pd.DataFrame
    open: pd.DataFrame | None = None
    high: pd.DataFrame | None = None
    low: pd.DataFrame | None = None
    volume: pd.DataFrame | None = None
    turnover: pd.DataFrame | None = None
    pct_change: pd.DataFrame | None = None

    shares_outstanding: pd.DataFrame | None = None
    mkt_cap: pd.DataFrame | None = None

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def columns(self):
        return self.close.columns
