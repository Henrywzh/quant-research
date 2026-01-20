from typing import Optional

import numpy as np
import pandas as pd
import re
from openpyxl import load_workbook
from pathlib import Path
from qresearch.data.types import MarketData
from qresearch.data.utils import get_raw_dir, get_processed_dir


SECTOR_RE = re.compile(r"恒生綜合行業指數\s*-\s*(.*?)\s*之成份股變動")

GROUP_COL_CANDIDATES = [
    "Effective Date 生效日期",
    "No. of Constituents",
    "Change 變動",
    "Count 數目",
]

ID_COL = "Stock Code 股份代號"  # used to identify real data rows


def save_market_data_to_parquet(md: MarketData, file_path: Path | str) -> None:
    """
    Save MarketData as a single Parquet file in yfinance-style MultiIndex columns:
      level0 = Field (Close/Open/High/Low/Volume)
      level1 = Ticker
    Includes only fields that exist (not None).
    """
    parts = {"Close": md.close}
    if md.open is not None:   parts["Open"] = md.open
    if md.high is not None:   parts["High"] = md.high
    if md.low is not None:    parts["Low"] = md.low
    if md.volume is not None: parts["Volume"] = md.volume

    panel = pd.concat(parts, axis=1).sort_index(axis=1)

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    panel.to_parquet(file_path)  # requires pyarrow or fastparquet installed

def load_market_data_from_parquet(file_path: Path | str) -> MarketData:
    """
    Load a yfinance-style MultiIndex Parquet back into MarketData.
    Missing fields become None.
    """
    file_path = Path(file_path)
    panel = pd.read_parquet(file_path)

    # Ensure DatetimeIndex (parquet sometimes returns object index if saved oddly)
    if not isinstance(panel.index, pd.DatetimeIndex):
        panel.index = pd.to_datetime(panel.index)

    # Expect MultiIndex columns: (Field, Ticker)
    if not isinstance(panel.columns, pd.MultiIndex) or panel.columns.nlevels != 2:
        raise ValueError("Parquet must contain MultiIndex columns (Field, Ticker).")

    lvl0 = panel.columns.get_level_values(0)

    def _get(field: str):
        return panel[field].copy() if field in lvl0 else None

    return MarketData(
        close=_get("Close"),
        open=_get("Open"),
        high=_get("High"),
        low=_get("Low"),
        volume=_get("Volume"),
    )

def save_historical_mcap_parquet(
    mcap: pd.DataFrame,
    file_path: str | Path,
    *,
    sort_index: bool = True,
    sort_columns: bool = True,
    coerce_float32: bool = False,
    compression: str = "zstd",
) -> Path:
    """
    Save historical market cap panel (date x ticker) to Parquet.

    Assumptions:
      - index is dates (or convertible to DatetimeIndex)
      - columns are tickers (strings)
      - values are numeric (floats), NaNs allowed

    Args:
      mcap: DataFrame indexed by date with ticker columns.
      file_path: destination .parquet path.
      sort_index: sort dates ascending before saving.
      sort_columns: sort tickers lexicographically before saving.
      coerce_float32: optionally downcast to float32 to reduce size (may lose precision).
      compression: parquet compression ("zstd", "snappy", "gzip", etc.).

    Returns:
      Path to saved file.
    """
    fp = Path(file_path)
    fp.parent.mkdir(parents=True, exist_ok=True)

    df = mcap.copy()

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="raise")

    # Optional sorting for stable diffs / reproducibility
    if sort_index:
        df = df.sort_index()
    if sort_columns:
        df = df.reindex(sorted(df.columns), axis=1)

    # Ensure numeric (keep NaNs)
    df = df.apply(pd.to_numeric, errors="coerce")

    # Optional downcast
    if coerce_float32:
        df = df.astype("float32")

    # Save
    df.to_parquet(fp, compression=compression)

    return fp

def load_historical_mcap_parquet(
    file_path: str | Path,
    *,
    tickers: list[str] | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    sort_index: bool = True,
    sort_columns: bool = False,
) -> pd.DataFrame:
    """
    Load historical market cap panel (date x ticker) from Parquet with standard normalization.

    Args:
      file_path: path to parquet file.
      tickers: optional list of tickers to keep (reduces memory).
      start/end: optional date filtering.
      sort_index: sort dates ascending.
      sort_columns: sort ticker columns.

    Returns:
      DataFrame indexed by DatetimeIndex, columns=tickers.
    """
    fp = Path(file_path)
    if not fp.exists():
        raise FileNotFoundError(fp)

    df = pd.read_parquet(fp)

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="raise")

    # Optional date filtering
    if start is not None:
        df = df.loc[pd.Timestamp(start):]
    if end is not None:
        df = df.loc[:pd.Timestamp(end)]

    # Optional ticker filtering
    if tickers is not None:
        tickers = [str(t) for t in tickers]
        df = df.reindex(columns=tickers)

    if sort_index:
        df = df.sort_index()
    if sort_columns:
        df = df.reindex(sorted(df.columns), axis=1)

    return df


def save_estimated_shares(
    estimated_shares: pd.Series,
    file_path: str | Path,
    *,
    asof_date: str | pd.Timestamp | None = None,
    px_asof_date: str | pd.Timestamp | None = None,
    source: str = "yfinance_marketCap / latest_close",
    compression: str = "zstd",
) -> Path:
    """
    Save estimated shares outstanding as a small reference table.

    Args:
      estimated_shares: Series indexed by ticker, values = shares outstanding estimate.
      file_path: destination .parquet or .csv.
      asof_date: market cap snapshot date (metadata).
      px_asof_date: date of latest close used to compute shares (metadata).
      source: free-text provenance string.
      compression: parquet compression if saving parquet.

    Returns:
      Path to saved file.
    """
    fp = Path(file_path)
    fp.parent.mkdir(parents=True, exist_ok=True)

    s = estimated_shares.copy()
    s.index = s.index.astype(str)
    s.name = "estimated_shares"

    df = s.reset_index().rename(columns={"index": "ticker"})

    if asof_date is not None:
        df["asof_date"] = pd.Timestamp(asof_date).date().isoformat()
    if px_asof_date is not None:
        df["px_asof_date"] = pd.Timestamp(px_asof_date).date().isoformat()

    df["source"] = source

    if fp.suffix.lower() == ".csv":
        df.to_csv(fp, index=False)
    elif fp.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(fp, index=False, compression=compression)
    else:
        raise ValueError("Unsupported file type. Use .csv or .parquet")

    return fp


def load_estimated_shares(file_path: str | Path) -> pd.Series:
    fp = Path(file_path)
    if fp.suffix.lower() == ".csv":
        df = pd.read_csv(fp)
    elif fp.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(fp)
    else:
        raise ValueError("Unsupported file type. Use .csv or .parquet")

    if "ticker" not in df.columns or "estimated_shares" not in df.columns:
        raise KeyError("Expected columns: ticker, estimated_shares")

    s = df.set_index("ticker")["estimated_shares"].astype(float)
    s.name = "estimated_shares"
    return s


def finalize_mcap_snapshot(
        mkt_cp_df: pd.DataFrame,
        *,
        ticker_col: str = "Ticker",
        mcap_col: str = "Market Cap",
        asof_date: str | pd.Timestamp,
) -> pd.DataFrame:
    df = mkt_cp_df[[ticker_col, mcap_col]].copy()
    df.columns = ["ticker", "market_cap"]
    df["ticker"] = df["ticker"].astype(str)
    df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
    df["asof_date"] = pd.Timestamp(asof_date).date().isoformat()

    # drop missing / non-positive
    df = df.dropna(subset=["market_cap"])
    df = df[df["market_cap"] > 0]

    # de-dup if yfinance returned duplicates
    df = df.drop_duplicates(subset=["ticker"], keep="last").reset_index(drop=True)
    return df


def save_mcap_snapshot(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")

    return path


def compute_estimated_shares(
        close: pd.DataFrame,
        mktcap_snapshot: pd.DataFrame,
        *,
        px_asof: str | pd.Timestamp | None = None,
) -> pd.Series:
    if px_asof is None:
        px_asof = close.index[-1]
    px_asof = pd.Timestamp(px_asof)

    latest_px = close.loc[px_asof]
    snap = mktcap_snapshot.set_index("ticker")["market_cap"]

    shares = (snap / latest_px).rename("estimated_shares")
    shares = shares.replace([np.inf, -np.inf], np.nan)
    shares = shares[(shares > 0) & shares.notna()]
    return shares


def build_historical_mcap(
        close: pd.DataFrame,
        estimated_shares: pd.Series,
) -> pd.DataFrame:
    sh = estimated_shares.reindex(close.columns.astype(str))
    return close.mul(sh, axis=1)


def save_hsci_components_history(
        fname: str = 'history_hsci.xlsx',
        exclude: Optional[dict] = None,
        header_row_excel: int = 5
):
    """
        Main pipeline function to process the HSCI components history Excel file.

        It iterates through all non-excluded sheets in the raw Excel file, parses them
        using `read_constituents_sheet`, concatenates the results into a single DataFrame,
        and saves the final dataset to the processed directory as a CSV.

        Args:
            fname (str): Filename of the raw Excel file.
            exclude (Optional[dict]): A set or dict of sheet names to exclude from processing.
                                      Defaults to {'HSCI'} if None.
            header_row_excel (int): The 1-based row index in Excel where the header is located.
    """

    if exclude is None:
        exclude = {'HSCI'}
    _path = get_raw_dir() / fname

    print(f"--- Starting Build: {fname} ---")

    xls = pd.ExcelFile(_path, engine="openpyxl")

    sheets = [sh for sh in xls.sheet_names if sh not in exclude]

    all_df = pd.concat(
        [read_constituents_sheet(_path, sh, header_row_excel=header_row_excel).assign(sheet=sh) for sh in sheets],
        ignore_index=True,
    )
    print(f"Merge complete. Total shape: {all_df.shape}")
    print(f"Unique sheets successfully read: {all_df['sheet'].nunique()} / {len(sheets)}")

    all_df.to_csv(get_processed_dir() / 'hsci_components_history.csv')
    print(f"Saved to: {_path}")


def load_hsci_components_history(fname: str = 'hsci_components_history.csv'):
    """
        Utility function to load the processed HSCI components history data.

        Args:
            fname (str): The filename to look for in the processed directory.

        Returns:
            pd.DataFrame: The loaded components history data.
    """

    return pd.read_csv(get_processed_dir() / fname)


def load_hsci_2008_components(fname: str = "hsci_components_2008.csv") -> pd.DataFrame:
    """
    Load the 2008 HSCI snapshot from raw/.
    """

    return pd.read_csv(get_raw_dir() / fname)


def _extract_sector_from_a2(path: str, sheet_name: str) -> str | None:
    wb = load_workbook(path, data_only=True)
    ws = wb[sheet_name]
    v = ws["A2"].value
    if v is None:
        return None
    v = str(v).strip()
    m = SECTOR_RE.search(v)
    return m.group(1).strip() if m else None


def read_constituents_sheet(path: str, sheet_name: str, header_row_excel: int = 5) -> pd.DataFrame:
    header_idx = header_row_excel - 1
    df = pd.read_excel(path, sheet_name=sheet_name, header=header_idx, engine="openpyxl")

    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.columns = [str(c).split("\n")[0].strip() for c in df.columns]

    group_cols = [c for c in GROUP_COL_CANDIDATES if c in df.columns]
    if group_cols:
        df[group_cols] = df[group_cols].ffill()

    if ID_COL in df.columns:
        df = df[df[ID_COL].notna()].copy()
        df[ID_COL] = (
            df[ID_COL].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        )
        mask = df[ID_COL].str.fullmatch(r"\d+")
        df.loc[mask, ID_COL] = df.loc[mask, ID_COL].str.zfill(4)

    if "Count" in df.columns:
        df["Count"] = pd.to_numeric(
            df["Count"].astype(str).str.replace("+", "", regex=False),
            errors="coerce",
        )

    if "Change" in df.columns:
        df["Action"] = df["Change"].astype(str).str.extract(r"^(Add|Delete)", expand=False)

    # ---- sector from A2 ----
    sector = _extract_sector_from_a2(path, sheet_name)
    df["Sector"] = sector

    return df
