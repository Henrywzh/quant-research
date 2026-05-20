from __future__ import annotations

import pandas as pd

from qresearch.data.catalog import DatasetArtifactMetadata, load_dataset_metadata
from qresearch.data.yfinance import download_close

from .config import DEFAULT_ISSUERS, ensure_directories, default_paths


def price_cache_path() -> Path:
    paths = ensure_directories(default_paths())
    p = paths.model_root / "prices_daily.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
def fetch_or_load_price_cache(
    index_df: pd.DataFrame,
    force: bool = False,
    *,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, object]]:
    cache = price_cache_path()
    if cache.exists() and not force:
        px = pd.read_parquet(cache)
        metadata = load_dataset_metadata(cache)
        payload = metadata.to_dict() if metadata is not None else {}
        return (px, payload) if return_metadata else px
    if index_df.empty:
        empty = pd.DataFrame()
        return (empty, {}) if return_metadata else empty
    tickers = []
    for sc in sorted(index_df["stock_code"].astype(str).unique()):
        cfg = DEFAULT_ISSUERS.get(sc.zfill(5))
        if cfg:
            tickers.append(cfg.ticker_for_price)
    if not tickers:
        empty = pd.DataFrame()
        return (empty, {}) if return_metadata else empty
    start = (pd.to_datetime(index_df["publish_datetime_hk"]).min() - pd.Timedelta(days=30)).date().isoformat()
    end = (pd.to_datetime(index_df["publish_datetime_hk"]).max() + pd.Timedelta(days=30)).date().isoformat()
    px = download_close(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        fill_method="none",
    )
    px.to_parquet(cache)
    try:
        relative_path = str(cache.relative_to(default_paths().processed_root))
    except ValueError:
        relative_path = cache.name
    metadata = DatasetArtifactMetadata(
        dataset_name="hkex_nav_prices_daily",
        path=relative_path,
        source_system="yahoo_finance",
        schema_version=1,
        format="parquet",
        build_timestamp=pd.Timestamp.utcnow().isoformat(),
        upstream_raw_inputs=["processed/hkex_nav/curated/filings_index"],
        source_quality="exploratory",
        parameters={
            "tickers": tickers,
            "start": start,
            "end": end,
            "auto_adjust": True,
            "fill_method": "none",
        },
    )
    metadata.write(root=default_paths().processed_root.parent, artifact_path=cache)
    payload = metadata.to_dict()
    return (px, payload) if return_metadata else px


def next_trading_close_after_publish(publish_datetime_hk: str, ticker_col: str, price_df: pd.DataFrame) -> tuple[str | None, float | None]:
    if price_df is None or price_df.empty or ticker_col not in price_df.columns:
        return None, None
    ts = pd.to_datetime(publish_datetime_hk)
    d = ts.normalize()
    candidate_idx = price_df.index[price_df.index > d]
    if len(candidate_idx) == 0:
        return None, None
    px_date = candidate_idx[0]
    px = price_df.at[px_date, ticker_col]
    if pd.isna(px):
        future = price_df.loc[price_df.index > px_date, ticker_col].dropna()
        if future.empty:
            return None, None
        px_date = future.index[0]
        px = float(future.iloc[0])
    return px_date.date().isoformat(), float(px)
