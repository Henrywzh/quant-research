from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence

import pandas as pd
from qresearch.data.utils import get_raw_dir, get_processed_dir


JQ_TO_PANEL_FIELDS = {
    "open": "Open",
    "close": "Close",
    "high": "High",
    "low": "Low",
    "volume": "Volume",
    "money": "Turnover",   # <- your requirement
}


def build_jq_panel_parquet(
    *,
    jq_prices_parquet: Path | str,          # long format: date, code, open, close, ...
    out_panel_parquet: Path | str,          # yfinance-style panel parquet
    fields: dict[str, str] = None,          # mapping jq_col -> panel_field
    start: str | None = None,
    end: str | None = None,
    universe: list[str] | None = None,      # optional restrict codes
) -> Path:
    """
    Convert JQ prices (long format) -> yfinance-style panel parquet:
      index: DatetimeIndex (date)
      columns: MultiIndex (Field, Ticker)
    """
    fields = fields or {
        "open": "Open",
        "close": "Close",
        "high": "High",
        "low": "Low",
        "volume": "Volume",
        'money': 'Turnover',
    }

    jq_prices_parquet = Path(jq_prices_parquet)
    out_panel_parquet = Path(out_panel_parquet)
    out_panel_parquet.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(jq_prices_parquet)

    # --- normalize columns ---
    # Expect at least: date, code
    if "date" not in df.columns or "code" not in df.columns:
        raise ValueError(f"Expected columns ['date','code'] in {jq_prices_parquet.name}, got: {list(df.columns)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["code"] = df["code"].astype("string").str.strip()

    # --- optional filters ---
    if start is not None:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end is not None:
        df = df[df["date"] <= pd.to_datetime(end)]
    if universe is not None:
        u = pd.Series(universe, dtype="string")
        df = df[df["code"].isin(u)]

    # --- build wide matrices per field ---
    panels = []
    for jq_col, panel_field in fields.items():
        if jq_col not in df.columns:
            continue  # missing field is fine; will be absent in final panel

        wide = df.pivot(index="date", columns="code", values=jq_col)
        wide.columns = pd.Index(wide.columns, name="Ticker")
        wide = wide.sort_index(axis=0).sort_index(axis=1)

        # attach MultiIndex (Field, Ticker)
        wide.columns = pd.MultiIndex.from_product([[panel_field], wide.columns], names=["Field", "Ticker"])
        panels.append(wide)

    if not panels:
        raise ValueError(f"No requested fields found in {jq_prices_parquet.name}. Available: {list(df.columns)}")

    panel = pd.concat(panels, axis=1).sort_index(axis=1)  # sort columns by (Field, Ticker)
    panel.index.name = "Date"

    panel.to_parquet(out_panel_parquet, engine="pyarrow")
    return out_panel_parquet



def csv_years_to_parquet(
    *,
    prefix: str,                         # e.g. "prices" or "valuation_daily"
    years: Sequence[int],
    raw_dir: Path,
    out_path: Path,
    date_col: str,                        # "date" for prices, "day" for fundamentals/valuation
    required_cols: Sequence[str],         # minimal columns you require
    dtype_map: Optional[Dict[str, str]] = None,  # pandas dtype strings
    chunksize: int = 2_000_000,           # tune by RAM/IO
    postprocess: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> Path:
    """
    Concatenate yearly CSVs into one Parquet by streaming chunks.

    - Uses pyarrow engine if installed (recommended).
    - Enforces required columns.
    - Normalizes date column to 'date' and code to string.
    - Sorts & de-dupes at the end is expensive; we instead de-dupe per chunk
      and rely on no cross-chunk duplicates under normal JQ exports. If you
      suspect cross-chunk duplicates, we can do a second pass later.
    """
    _ensure_dir(out_path.parent)
    files = _list_year_files(raw_dir, prefix, years)

    # Parquet writer (pyarrow) via pandas
    writer = None
    wrote_any = False

    # Minimal dtype defaults
    dtype_map = dict(dtype_map or {})
    # Ensure code is read as string if present
    if "code" in required_cols:
        dtype_map.setdefault("code", "string")

    try:
        for fp in files:
            # Stream read
            for chunk in pd.read_csv(fp, chunksize=chunksize, dtype=dtype_map):
                # Column sanity
                missing = [c for c in required_cols if c not in chunk.columns]
                if missing:
                    raise KeyError(f"{fp.name}: missing required cols {missing}. Columns: {list(chunk.columns)}")

                # Normalize code/date
                if "code" in chunk.columns:
                    chunk["code"] = _normalize_code_series(chunk["code"])
                chunk = _normalize_date_col(chunk, date_col, out_col="date")

                # Optional custom postprocessing
                if postprocess is not None:
                    chunk = postprocess(chunk)

                # Keep only needed columns + any extras you want to preserve
                # (Here we keep all columns but ensure required are present.)
                # If you prefer strict column selection, uncomment below:
                # keep_cols = sorted(set(["date"] + list(required_cols)))
                # chunk = chunk[keep_cols]

                # Basic per-chunk de-dupe
                if "code" in chunk.columns:
                    chunk = _sort_and_dedupe(chunk, keys=["date", "code"])
                else:
                    chunk = _sort_and_dedupe(chunk, keys=["date"])

                # Write
                if not wrote_any:
                    chunk.to_parquet(out_path, index=False, engine="pyarrow")
                    wrote_any = True
                else:
                    # Append by writing to dataset-style directory is safer,
                    # but if you want a *single* parquet file, we can use pyarrow writer.
                    import pyarrow as pa
                    import pyarrow.parquet as pq

                    table = pa.Table.from_pandas(chunk, preserve_index=False)

                    if writer is None:
                        writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
                        # If we already wrote the first chunk via pandas, we need to rewrite everything
                        # to use ParquetWriter consistently. Easiest: delete and start from scratch.
                        # So: start with ParquetWriter from the beginning instead.
                        raise RuntimeError(
                            "ParquetWriter requires consistent writing from the first chunk. "
                            "Use the 'single_file_streaming' implementation below."
                        )

        if not wrote_any:
            raise RuntimeError(f"No rows written for prefix='{prefix}' (empty input files?)")

    finally:
        if writer is not None:
            writer.close()

    return out_path


def csv_years_to_single_parquet_streaming(
    *,
    prefix: str,
    years: Sequence[int],
    raw_dir: Path,
    out_path: Path,
    date_col: str,
    required_cols: Sequence[str],
    dtype_map: Optional[Dict[str, str]] = None,
    chunksize: int = 2_000_000,
    postprocess: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    compression: str = "zstd",
) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    _ensure_dir(out_path.parent)
    files = _list_year_files(raw_dir, prefix, years)

    dtype_map = dict(dtype_map or {})
    if "code" in required_cols:
        dtype_map.setdefault("code", "string")

    writer: Optional[pq.ParquetWriter] = None
    wrote_any = False

    try:
        for fp in files:
            for chunk in pd.read_csv(fp, chunksize=chunksize, dtype=dtype_map):
                missing = [c for c in required_cols if c not in chunk.columns]
                if missing:
                    raise KeyError(f"{fp.name}: missing required cols {missing}. Columns: {list(chunk.columns)}")

                if "code" in chunk.columns:
                    chunk["code"] = _normalize_code_series(chunk["code"])
                chunk = _normalize_date_col(chunk, date_col, out_col="date")

                if postprocess is not None:
                    chunk = postprocess(chunk)

                if "code" in chunk.columns:
                    chunk = _sort_and_dedupe(chunk, keys=["date", "code"])
                else:
                    chunk = _sort_and_dedupe(chunk, keys=["date"])

                table = pa.Table.from_pandas(chunk, preserve_index=False)

                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema, compression=compression)
                writer.write_table(table)
                wrote_any = True

        if not wrote_any:
            raise RuntimeError(f"No rows written for prefix='{prefix}'")

    finally:
        if writer is not None:
            writer.close()

    return out_path


def build_all_jqdata_parquets(
    years=range(2016, 2026),
) -> dict[str, Path]:
    raw_dir = get_raw_dir() / "jqdata"
    out_dir = get_processed_dir() / "jqdata"
    _ensure_dir(out_dir)

    outputs: dict[str, Path] = {}

    # # -----------------
    # # prices (date)
    # # -----------------
    # outputs["prices"] = csv_years_to_single_parquet_streaming(
    #     prefix="prices",
    #     years=list(years),
    #     raw_dir=raw_dir,
    #     out_path=out_dir / "prices_2016_2025.parquet",
    #     date_col="date",
    #     required_cols=["date", "code", "open", "close"],
    #     dtype_map={
    #         "open": "float64",
    #         "close": "float64",
    #         # add optional columns if you want strict types:
    #         "high": "float64", "low": "float64", "volume": "float64", "money": "float64",
    #     },
    #     chunksize=2_000_000,
    # )

    # -----------------
    # valuation_daily (day -> date)
    # -----------------
    outputs["valuation_daily"] = csv_years_to_single_parquet_streaming(
        prefix="valuation_daily",
        years=list(years),
        raw_dir=raw_dir,
        out_path=out_dir / "valuation_daily_2016_2025.parquet",
        date_col="day",
        required_cols=["day", "code", "pe_ratio", "pb_ratio", "market_cap", "circulating_cap", "turnover_ratio"],
        dtype_map={
            "pe_ratio": "float64",
            "pb_ratio": "float64",
            "market_cap": "float64",
            "circulating_cap": "float64",
            "turnover_ratio": "float64",
        },
        chunksize=2_000_000,
    )

    # -----------------
    # income_daily (day -> date)
    # -----------------
    outputs["income_daily"] = csv_years_to_single_parquet_streaming(
        prefix="income_daily",
        years=list(years),
        raw_dir=raw_dir,
        out_path=out_dir / "income_daily_2016_2025.parquet",
        date_col="day",
        required_cols=["day", "code", "net_profit", "operating_revenue"],
        dtype_map={
            "net_profit": "float64",
            "operating_revenue": "float64",
        },
        chunksize=2_000_000,
    )

    # -----------------
    # indicator_daily (day -> date)
    # -----------------
    outputs["indicator_daily"] = csv_years_to_single_parquet_streaming(
        prefix="indicator_daily",
        years=list(years),
        raw_dir=raw_dir,
        out_path=out_dir / "indicator_daily_2016_2025.parquet",
        date_col="day",
        required_cols=["day", "code", "roe", "roa"],
        dtype_map={
            "roe": "float64",
            "roa": "float64",
        },
        chunksize=2_000_000,
    )
    #
    # # -----------------
    # # limit_daily (day -> date)
    # # -----------------
    # outputs["limit_daily"] = csv_years_to_single_parquet_streaming(
    #     prefix="limit_daily",
    #     years=list(years),  # range(2016, 2026)
    #     raw_dir=raw_dir,
    #     out_path=out_dir / "limit_daily_2016_2025.parquet",
    #     date_col="day",
    #     required_cols=["day", "code", "high_limit", "low_limit", "close", "volume"],
    #     dtype_map={
    #         "high_limit": "float64",
    #         "low_limit": "float64",
    #         "close": "float64",
    #         "volume": "float64",
    #     },
    #     chunksize=2_000_000,
    # )

    # # -----------------
    # # single-file CSVs -> parquet (optional but nice)
    # # -----------------
    # # trade_days
    # trade_days_csv = raw_dir / "trade_days_2016-01-01_2025-12-31.csv"
    # if trade_days_csv.exists():
    #     df = pd.read_csv(trade_days_csv)
    #     df = _normalize_date_col(df, "date", out_col="date").sort_values("date").drop_duplicates(["date"])
    #     out = out_dir / "trade_days_2016_2025.parquet"
    #     df.to_parquet(out, index=False, engine="pyarrow")
    #     outputs["trade_days"] = out
    #
    # # membership events + union codes
    # events_csv = raw_dir / "ashare_connect_events_2016_2025.csv"
    # if events_csv.exists():
    #     ev = pd.read_csv(events_csv, dtype={"code": "string", "link_id": "string", "direction": "string"})
    #     # normalize dates
    #     if "change_date" in ev.columns:
    #         ev = _normalize_date_col(ev, "change_date", out_col="change_date")
    #     ev["code"] = _normalize_code_series(ev["code"])
    #     out = out_dir / "ashare_connect_events_2016_2025.parquet"
    #     ev.to_parquet(out, index=False, engine="pyarrow")
    #     outputs["ashare_connect_events"] = out
    #
    # codes_csv = raw_dir / "ashare_connect_codes_2016_2025.csv"
    # if codes_csv.exists():
    #     cd = pd.read_csv(codes_csv, dtype={"code": "string"})
    #     cd["code"] = _normalize_code_series(cd["code"])
    #     cd = cd.drop_duplicates(["code"]).sort_values("code")
    #     out = out_dir / "ashare_connect_codes_2016_2025.parquet"
    #     cd.to_parquet(out, index=False, engine="pyarrow")
    #     outputs["ashare_connect_codes"] = out

    return outputs


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _list_year_files(raw_dir: Path, prefix: str, years: Sequence[int], ext: str = ".csv") -> list[Path]:
    files = [raw_dir / f"{prefix}_{y}{ext}" for y in years]
    missing = [f for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} files for prefix='{prefix}':\n" + "\n".join(map(str, missing)))
    return files


def _normalize_code_series(s: pd.Series) -> pd.Series:
    # Keep as string; avoid categorical until later if you pivot.
    return s.astype("string").str.strip()


def _normalize_date_col(df: pd.DataFrame, col: str, out_col: str = "date") -> pd.DataFrame:
    if col not in df.columns:
        raise KeyError(f"Expected date column '{col}' not found. Columns: {list(df.columns)}")
    df[out_col] = pd.to_datetime(df[col], errors="coerce")
    if df[out_col].isna().any():
        bad = df.loc[df[out_col].isna(), [col]].head(10)
        raise ValueError(f"Found unparsable dates in column '{col}'. Examples:\n{bad}")
    if out_col != col:
        df = df.drop(columns=[col])
    # Normalize to midnight (date-level); keep datetime64[ns]
    df[out_col] = df[out_col].dt.normalize()
    return df


def _sort_and_dedupe(df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    df = df.sort_values(list(keys), kind="mergesort")
    # Keep last if duplicates exist (typical: later rows overwrite earlier)
    df = df.drop_duplicates(subset=list(keys), keep="last")
    return df


if __name__ == "__main__":
    outs = build_all_jqdata_parquets()
    print(outs)
    pass


