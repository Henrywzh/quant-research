import numpy as np
import pandas as pd
import re

from qresearch.data.io import load_hsci_components_history, load_hsci_2008_components
from qresearch.data.utils import get_processed_dir


def build_hsci_ticker_master(
    all_df: pd.DataFrame,
    snapshot_2008: pd.DataFrame,
    *,
    history_code_col: str = "Stock Code 股份代號",
    snapshot_code_col: str = "Stock Code",
    current_components: pd.DataFrame | None = None,
    current_code_col: str = "Stock Code",
    save: bool = True,
    out_fname: str = "hsci_ticker_master.csv",
    return_audit: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, object]]:
    """
    Build survivorship-safe ticker master:
        tickers_ever = (2008 snapshot tickers) U (all tickers appearing in history events)

    Returns a DataFrame with:
        ticker, in_seed_2008, in_history_events
    """
    # --- extract tickers from history events ---
    hist_codes = all_df.get(history_code_col)
    if hist_codes is None:
        raise KeyError(f"Missing column in all_df: {history_code_col}")
    hist_tickers = (
        hist_codes.map(canon_hk_ticker)
        .dropna()
        .astype(str)
        .unique()
    )
    hist_set = set(hist_tickers)

    # --- extract tickers from 2008 snapshot ---
    snap_codes = snapshot_2008.get(snapshot_code_col)
    if snap_codes is None:
        raise KeyError(f"Missing column in snapshot_2008: {snapshot_code_col}")
    seed_tickers = (
        snap_codes.map(canon_hk_ticker)
        .dropna()
        .astype(str)
        .unique()
    )
    seed_set = set(seed_tickers)

    current_set: set[str] = set()
    unresolved_current: list[str] = []
    if current_components is not None:
        current_codes = current_components.get(current_code_col)
        if current_codes is None:
            raise KeyError(f"Missing column in current_components: {current_code_col}")
        for raw_code in current_codes.tolist():
            normalized = canon_hk_ticker(raw_code)
            if pd.isna(normalized):
                unresolved_current.append(str(raw_code))
                continue
            current_set.add(str(normalized))

    # --- union ---
    master = sorted(seed_set | hist_set | current_set)

    out = pd.DataFrame({
        "ticker": master,
        "in_seed_2008": [t in seed_set for t in master],
        "in_history_events": [t in hist_set for t in master],
        "in_current_components": [t in current_set for t in master],
    })

    if save:
        out_path = get_processed_dir() / out_fname
        out.to_csv(out_path, index=False)

    if not return_audit:
        return out

    audit = {
        "history_rows": int(len(all_df)),
        "snapshot_rows": int(len(snapshot_2008)),
        "current_components_rows": int(len(current_components)) if current_components is not None else 0,
        "current_components_unresolved": unresolved_current,
        "master_count": int(len(out)),
    }
    return out, audit


def load_or_build_hsci_ticker_master(
    *,
    all_df_fname: str = "hsci_components_history.csv",
    snapshot_2008_fname: str = "hsci_components_2008.csv",
    out_fname: str = "hsci_ticker_master.csv",
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    Convenience wrapper:
      - if processed/hsci_ticker_master.csv exists and not force_rebuild -> load it
      - else build from (processed all_df) and (raw 2008 snapshot), save, return
    """
    out_path = get_processed_dir() / out_fname
    if out_path.exists() and not force_rebuild:
        return pd.read_csv(out_path)

    all_df = load_hsci_components_history(all_df_fname)
    snap_2008 = load_hsci_2008_components(snapshot_2008_fname)
    current_components_path = get_processed_dir() / "hsci_components.csv"
    current_components = pd.read_csv(current_components_path) if current_components_path.exists() else None

    return build_hsci_ticker_master(
        all_df,
        snap_2008,
        current_components=current_components,
        save=True,
        out_fname=out_fname,
    )


def canon_hk_ticker(code) -> str:
    """
    Convert a stock code into canonical 'xxxx.HK' format.
    Handles ints, strings, leading zeros, and mixed text.
    """
    if pd.isna(code):
        return np.nan
    s = str(code).strip()
    # keep digits only
    digits = re.sub(r"\D+", "", s)
    if digits == "":
        return np.nan
    return digits.zfill(4) + ".HK"


def _map_to_next_trading_day(dates: pd.DatetimeIndex, d: pd.Timestamp) -> pd.Timestamp | None:
    """
    Map a calendar date to a trading day in `dates`.
    If d is in dates -> d. Else -> next trading day after d.
    Returns None if d is after the last available trading day.
    """
    d = pd.Timestamp(d).normalize()
    if d in dates:
        return d
    pos = dates.searchsorted(d, side="left")
    if pos >= len(dates):
        return None
    return dates[pos]


def build_hsci_events_from_history(
    history_df: pd.DataFrame,
    *,
    effective_col: str = "Effective Date 生效日期",
    change_col: str = "Change 變動",
    code_col: str = "Stock Code 股份代號",
    sheet_col: str = "sheet",
    include_sheets: list[str] | None = None,
) -> pd.DataFrame:
    """
    Convert your history_hsci 'all_df' into a clean events table:
      effective_date | ticker | action | sheet | sector(optional)
    """
    df = history_df.copy()

    if include_sheets is not None and sheet_col in df.columns:
        df = df[df[sheet_col].isin(include_sheets)].copy()

    # Parse effective date
    df["effective_date"] = pd.to_datetime(df[effective_col], errors="coerce").dt.normalize()

    # Canonical ticker
    df["ticker"] = df[code_col].apply(canon_hk_ticker)

    # Action mapping
    m = {
        "Add 加入": "ADD",
        "Delete 刪除": "REMOVE",
        "Delete": "REMOVE",
        "Add": "ADD",
    }
    df["action"] = df[change_col].map(m)

    out_cols = ["effective_date", "ticker", "action"]
    if sheet_col in df.columns:
        out_cols.append(sheet_col)
    if "sector" in df.columns:
        out_cols.append("sector")

    events = df[out_cols].dropna(subset=["effective_date", "ticker", "action"]).copy()

    # Remove duplicates and sort
    events = events.drop_duplicates(subset=["effective_date", "ticker", "action"])
    events = events.sort_values(["effective_date", "ticker", "action"]).reset_index(drop=True)

    return events


def build_hsci_member_mask(
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
    events: pd.DataFrame,
    seed_members: set[str] | None = None,
    effective_on_close: bool = True,  # your chosen semantics
) -> pd.DataFrame:
    """
    Build date×ticker boolean membership mask with no lookahead.

    effective_on_close=True means:
      - membership flips starting on the effective trading date (>= effective_date).
    """
    dates = pd.DatetimeIndex(dates)
    tickers = pd.Index(tickers)

    member = pd.DataFrame(False, index=dates, columns=tickers)

    # Seed initial members
    if seed_members:
        seed = [t for t in seed_members if t in tickers]
        if seed:
            member.loc[:, seed] = True

    # Apply events forward
    for row in events.itertuples(index=False):
        d = getattr(row, "effective_date")
        t = getattr(row, "ticker")
        a = getattr(row, "action")

        if t not in tickers:
            continue

        d_trade = _map_to_next_trading_day(dates, d)
        if d_trade is None:
            continue

        if a == "ADD":
            member.loc[d_trade:, t] = True
        elif a == "REMOVE":
            member.loc[d_trade:, t] = False

    return member
