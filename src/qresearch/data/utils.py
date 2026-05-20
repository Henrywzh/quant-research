from pathlib import Path
import pandas as pd
from qresearch.data.catalog import dataset_root, processed_dir, raw_dir
from qresearch.data.types import MarketData


def save_market_data_to_csv(md: MarketData, file_path: Path | str) -> None:
    """
    Legacy convenience wrapper around the canonical CSV serializer.
    """
    from qresearch.data.io import save_market_data_to_csv as _save_market_data_to_csv

    _save_market_data_to_csv(md, file_path)


def load_market_data_from_csv(file_path: Path | str) -> MarketData:
    """
    Legacy convenience wrapper around the canonical CSV loader.
    """
    from qresearch.data.io import load_market_data_from_csv as _load_market_data_from_csv

    return _load_market_data_from_csv(file_path)


def marketdata_to_yfinance(md: MarketData) -> pd.DataFrame:
    """
    Converts a MarketData object (separate DF per field) into a single
    yfinance-style DataFrame (MultiIndex columns: Field -> Ticker).
    """
    from qresearch.data.io import marketdata_to_panel

    return marketdata_to_panel(md)


def save_yf_data(df: pd.DataFrame, file_path: Path | str) -> None:
    """
    Saves a yfinance MultiIndex DataFrame to CSV, preserving the hierarchy.
    """
    df.to_csv(file_path)


def load_yf_data(file_path: Path | str) -> pd.DataFrame:
    """
    Loads a CSV saved from yfinance, reconstructing the MultiIndex columns
    and DatetimeIndex exactly as they were.
    """
    return pd.read_csv(
        file_path,
        header=[0, 1],  # Reconstructs Level 0 (Price Type) and Level 1 (Ticker)
        index_col=0,  # Sets the Date column as the index
        parse_dates=True  # Converts the index strings back to Datetime objects
    )


def yfinance_to_marketdata(df: pd.DataFrame) -> MarketData:
    """
    Converts a yfinance-style MultiIndex DataFrame back into a MarketData object.
    Handles missing fields gracefully by creating empty DataFrames if needed.
    """
    from qresearch.data.io import panel_to_marketdata

    return panel_to_marketdata(df)

def _is_under_src(p: Path) -> bool:
    parts = [x.lower() for x in p.parts]
    return "src" in parts  # simple, effective


def _find_dataset_data_dir() -> Path:
    return dataset_root()


def get_processed_dir() -> Path:
    return processed_dir()


def get_raw_dir() -> Path:
    return raw_dir()
