from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
import pandas as pd


@dataclass
class FactorStore:
    """
    Disk-backed factor matrices store.

    Layout:
      root/
        valuation__pe_ratio.parquet
        valuation__pb_ratio.parquet
        indicator__roe.parquet
        income__net_profit.parquet
        ...
    """
    root: Path

    def path(self, group: str, field: str) -> Path:
        return self.root / f"{group}__{field}.parquet"

    def exists(self, group: str, field: str) -> bool:
        return self.path(group, field).exists()

    def get(
        self,
        group: str,
        field: str,
        *,
        dates: Optional[pd.DatetimeIndex] = None,
        tickers: Optional[pd.Index] = None,
        fillna: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Load a factor matrix (date x ticker), optionally align to dates/tickers.
        """
        p = self.path(group, field)
        if not p.exists():
            raise FileNotFoundError(f"Missing factor file: {p}")

        mat = pd.read_parquet(p)

        # Defensive index normalization
        if not isinstance(mat.index, pd.DatetimeIndex):
            mat.index = pd.to_datetime(mat.index).normalize()

        if dates is not None:
            mat = mat.reindex(index=dates)
        if tickers is not None:
            mat = mat.reindex(columns=tickers)

        if fillna is not None:
            mat = mat.fillna(fillna)

        return mat

    def list(self) -> pd.DataFrame:
        """
        List available factors on disk.
        """
        rows = []
        for p in sorted(self.root.glob("*.parquet")):
            name = p.stem
            if "__" not in name:
                continue
            group, field = name.split("__", 1)
            rows.append({"group": group, "field": field, "path": str(p)})
        return pd.DataFrame(rows)

    def bulk_get(
        self,
        items: Iterable[Tuple[str, str]],
        *,
        dates: Optional[pd.DatetimeIndex] = None,
        tickers: Optional[pd.Index] = None,
    ) -> Dict[Tuple[str, str], pd.DataFrame]:
        """
        Load multiple matrices at once.
        """
        out: Dict[Tuple[str, str], pd.DataFrame] = {}
        for group, field in items:
            out[(group, field)] = self.get(group, field, dates=dates, tickers=tickers)
        return out


def load_quality_value_bundle(store: FactorStore, dates, tickers) -> dict[str, pd.DataFrame]:
    return {
        "pe": store.get("valuation", "pe_ratio", dates=dates, tickers=tickers),
        "pb": store.get("valuation", "pb_ratio", dates=dates, tickers=tickers),
        "circulating_cap": store.get("valuation", "circulating_cap", dates=dates, tickers=tickers),
        "turnover_ratio": store.get("valuation", "turnover_ratio", dates=dates, tickers=tickers),
        "mcap": store.get("valuation", "market_cap", dates=dates, tickers=tickers),
        "roe": store.get("indicator", "roe", dates=dates, tickers=tickers),
        "roa": store.get("indicator", "roa", dates=dates, tickers=tickers),
        "net_profit": store.get("income", "net_profit", dates=dates, tickers=tickers),
        "operating_revenue": store.get("income", "operating_revenue", dates=dates, tickers=tickers),
    }
