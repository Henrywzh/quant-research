from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Union, List
import numpy as np
import pandas as pd
import yfinance as yf
from qresearch.data.types import MarketData

Tickers = Union[str, Sequence[str]]


def download_market_data(
        tickers: Union[str, List[str]],  # Adjusted type hint for clarity
        start: str,
        end: Optional[str] = None,
        *,
        auto_adjust_close: bool = True,
        ffill: bool = True,
        drop_all_nan_rows: bool = True,
) -> MarketData:
    """
    Download MarketData bundle with progress tracking.
    """
    tick_list = _as_list(tickers)
    n_tickers = len(tick_list)

    print(f"[-] Initializing download for {n_tickers} tickers...")
    print(f"    Date Range: {start} -> {end if end else 'Now'}")

    # 1) OHLCV raw
    # NOTE: Ensure your download_ohlc function passes `progress=True` to yf.download internally
    # so you see the bar like: [*********************100%***********************]  2000 of 2000 completed
    try:
        ohlc = download_ohlc(
            tick_list,
            start=start,
            end=end,
            auto_adjust=True,
            ffill=ffill,
            drop_all_nan_rows=drop_all_nan_rows,
        )
        print(f"[✓] Download complete. Shape: {ohlc.shape}")
    except Exception as e:
        print(f"[!] Critical Error during download: {e}")
        raise e

    # Helper to extract field->DataFrame
    def _field_df(field: str) -> Optional[pd.DataFrame]:
        print(f"    Processing field: {field}...", end="\r")  # \r overwrites the line for a cleaner look

        if not isinstance(ohlc.columns, pd.MultiIndex):
            return None
        # Check Level 0 (Price Type)
        if field not in ohlc.columns.get_level_values(0):
            print(f"    [!] Warning: Field '{field}' not found in data.")
            return None

        df = ohlc[field].copy()
        return df

    # 2) Extract Fields
    print("[-] Extracting and aligning components...")

    # FIX: Changed "close" to "Close" (yfinance returns Title Case)
    close = _field_df("Close")
    open_ = _field_df("Open")
    high = _field_df("High")
    low = _field_df("Low")
    volume = _field_df("Volume")


    print("[✓] MarketData object created.")

    return MarketData(
        close=close,
        open=open_,
        high=high,
        low=low,
        volume=volume,
    )



def _as_list(tickers: Tickers) -> list[str]:
    if isinstance(tickers, str):
        return [tickers]
    return list(tickers)


def _standardize_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    # enforce tz-naive (research standard)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    return df


def _extract_field_from_yf(raw: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    """
    yfinance outputs vary:
    - MultiIndex columns like (PriceField, Ticker) OR (Ticker, PriceField)
    - SingleIndex columns for single ticker, like ['Open','High','Low','Close','Adj Close','Volume']
    This function extracts a DataFrame indexed by date with columns=tickers for a given field.
    """
    if raw is None or raw.empty:
        raise ValueError("yfinance returned empty data")

    # Case 1: single ticker + flat columns
    if not isinstance(raw.columns, pd.MultiIndex):
        # For single ticker, raw columns likely include field directly.
        if field not in raw.columns:
            raise ValueError(f"yfinance data missing field='{field}' (columns={list(raw.columns)[:8]}...)")
        out = raw[[field]].copy()
        out.columns = [tickers[0]]
        return out

    # Case 2: MultiIndex columns
    lvl0 = raw.columns.get_level_values(0)
    lvl1 = raw.columns.get_level_values(1)

    # Format A: (Field, Ticker)
    if field in set(lvl0):
        # raw[field] -> columns are tickers (possibly plus extras)
        out = raw[field].copy()
        # Ensure we only keep requested tickers and preserve order
        keep = [t for t in tickers if t in out.columns]
        out = out.reindex(columns=keep)
        return out

    # Format B: (Ticker, Field)
    if field in set(lvl1):
        # select all columns where lvl1 == field
        sub = raw.loc[:, (slice(None), field)]
        # drop second level
        sub.columns = sub.columns.get_level_values(0)
        keep = [t for t in tickers if t in sub.columns]
        sub = sub.reindex(columns=keep)
        return sub

    raise ValueError(f"Unable to locate field='{field}' in yfinance MultiIndex columns.")


def download_close(
    tickers: Tickers,
    start: str,
    end: Optional[str] = None,
    *,
    auto_adjust: bool = True,
    use_adj_close_if_present: bool = False,
    ffill: bool = True,
    drop_all_nan_rows: bool = True,
) -> pd.DataFrame:
    """
    Download daily close prices from Yahoo Finance via yfinance.

    Contract:
    - returns DataFrame, DatetimeIndex increasing, tz-naive
    - columns are tickers, floats
    - missing allowed; by default forward-filled

    Notes:
    - If auto_adjust=True, yfinance 'Close' is adjusted.
    - If you want explicit 'Adj Close', set use_adj_close_if_present=True and auto_adjust=False.
    """
    tick_list = _as_list(tickers)

    raw = yf.download(
        tick_list,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=True,
        group_by="column",  # most stable; we handle both orientations anyway
        threads=True,
    )

    raw = _standardize_index(raw)

    field = "Adj Close" if use_adj_close_if_present and not auto_adjust else "Close"
    px = _extract_field_from_yf(raw, field=field, tickers=tick_list)

    px = _standardize_index(px)
    px = px.astype(float)

    # Basic cleaning
    px = px.replace([np.inf, -np.inf], np.nan)
    if ffill:
        px = px.ffill()

    if drop_all_nan_rows:
        px = px.dropna(how="all")

    # Keep only requested tickers (and preserve original order)
    px = px.reindex(columns=tick_list)

    return px


def download_ohlc(
    tickers: Tickers,
    start: str,
    end: Optional[str] = None,
    *,
    auto_adjust: bool = False,
    ffill: bool = True,
    drop_all_nan_rows: bool = True,
) -> pd.DataFrame:
    """
    Download OHLC (and Volume) as a DataFrame with MultiIndex columns:
        level0: ["Open","High","Low","Close","Volume"]
        level1: tickers

    This is the most compatible format for your bucket framework (price_df["Close"] etc.).
    """
    tick_list = _as_list(tickers)

    raw = yf.download(
        tick_list,
        start=start,
        end=end,
        auto_adjust=auto_adjust,  # keep False if you need raw Open/Close
        progress=True,
        group_by="column",
        threads=True,
    )
    raw = _standardize_index(raw)

    # If single ticker, raw is flat columns -> lift to MultiIndex (Field, Ticker)
    if not isinstance(raw.columns, pd.MultiIndex):
        fields = list(raw.columns)
        raw.columns = pd.MultiIndex.from_product([fields, [tick_list[0]]])

    # Ensure (Field, Ticker) orientation
    # If currently (Ticker, Field), swap
    lvl0 = raw.columns.get_level_values(0)
    if all(t in tick_list for t in set(lvl0)):
        raw = raw.swaplevel(0, 1, axis=1)

    raw = raw.sort_index(axis=1)

    # Keep canonical fields if present
    fields_keep = [f for f in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if f in raw.columns.get_level_values(0)]
    out = raw.loc[:, (fields_keep, tick_list)].copy()

    out = out.replace([np.inf, -np.inf], np.nan)
    if ffill:
        # forward fill within each field/ticker column
        out = out.ffill()

    if drop_all_nan_rows:
        out = out.dropna(how="all")

    return out
