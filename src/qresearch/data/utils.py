from pathlib import Path
import pandas as pd
from qresearch.data.types import MarketData


def save_market_data_to_csv(md: MarketData, file_path: Path | str) -> None:
    """
    Combines MarketData components back into a single yfinance-style
    MultiIndex CSV and saves it.
    """
    # 1. Recombine the separated DataFrames into one MultiIndex DataFrame
    # Using a dictionary automatically creates the top-level column (Price Type)
    combined_df = pd.concat({
        'Close': md.close,
        'Open': md.open,
        'High': md.high,
        'Low': md.low,
        'Volume': md.volume
    }, axis=1)

    # 2. Save
    print(f"Saving MarketData to {file_path}...")
    combined_df.to_csv(file_path)
    print("Save complete.")


def load_market_data_from_csv(file_path: Path | str) -> MarketData:
    """
    Loads the single CSV and splits it back into your MarketData object.
    """
    print(f"Loading MarketData from {file_path}...")

    # 1. Read with MultiIndex header
    df = pd.read_csv(
        file_path,
        header=[0, 1],
        index_col=0,
        parse_dates=True
    )

    # 2. Split back into components using xs (Cross Section) or direct indexing
    # We use .copy() to ensure they are distinct objects
    return MarketData(
        close=df['Close'].copy(),
        open=df['Open'].copy(),
        high=df['High'].copy(),
        low=df['Low'].copy(),
        volume=df['Volume'].copy()
    )


def marketdata_to_yfinance(md: MarketData) -> pd.DataFrame:
    """
    Converts a MarketData object (separate DF per field) into a single
    yfinance-style DataFrame (MultiIndex columns: Field -> Ticker).
    """
    # Keys become Level 0 (e.g., 'Close', 'Open')
    # The columns of the inner DataFrames become Level 1 (Tickers)
    yf_df = pd.concat(
        {
            'Close': md.close,
            'Open': md.open,
            'High': md.high,
            'Low': md.low,
            'Volume': md.volume
        },
        axis=1
    )

    # Optional: Sort columns to ensure they look like standard yfinance (Level 0, then Level 1)
    return yf_df.sort_index(axis=1)


def save_yf_data(df: pd.DataFrame, file_path: Path | str) -> None:
    """
    Saves a yfinance MultiIndex DataFrame to CSV, preserving the hierarchy.
    """
    print(f"Saving data to {file_path}...")
    df.to_csv(file_path)
    print("Save complete.")


def load_yf_data(file_path: Path | str) -> pd.DataFrame:
    """
    Loads a CSV saved from yfinance, reconstructing the MultiIndex columns
    and DatetimeIndex exactly as they were.
    """
    print(f"Loading data from {file_path}...")

    df = pd.read_csv(
        file_path,
        header=[0, 1],  # Reconstructs Level 0 (Price Type) and Level 1 (Ticker)
        index_col=0,  # Sets the Date column as the index
        parse_dates=True  # Converts the index strings back to Datetime objects
    )

    print("Load complete.")
    return df


def yfinance_to_marketdata(df: pd.DataFrame) -> MarketData:
    """
    Converts a yfinance-style MultiIndex DataFrame back into a MarketData object.
    Handles missing fields gracefully by creating empty DataFrames if needed.
    """

    # Helper to safely extract a field or return an empty DF
    def _extract(field: str):
        # Handle case sensitivity (yfinance uses Title case: 'Close', 'Volume')
        if field in df.columns.get_level_values(0):
            return df[field].copy()
        else:
            print(f"Warning: '{field}' not found in DataFrame.")
            return pd.DataFrame()

    return MarketData(
        close=_extract('Close'),
        open=_extract('Open'),
        high=_extract('High'),
        low=_extract('Low'),
        volume=_extract('Volume')
    )


def get_processed_dir() -> Path:
    """
    Finds the 'data/processed' directory by searching up the directory tree.
    Automatically creates the folder if it doesn't exist.
    """
    # Start from the current working directory
    current_path = Path.cwd()

    # Check current folder and all parent folders
    for path in [current_path] + list(current_path.parents):
        # Look for the 'data' folder in this path
        data_dir = path / 'data'
        if data_dir.exists():
            processed_dir = data_dir / 'processed'

            # Create 'processed' if it is missing (good safety habit)
            processed_dir.mkdir(parents=True, exist_ok=True)

            return processed_dir

    raise FileNotFoundError("Could not locate the 'data' folder in any parent directory.")


def get_raw_dir() -> Path:
    """
    Finds the 'data/processed' directory by searching up the directory tree.
    Automatically creates the folder if it doesn't exist.
    """
    # Start from the current working directory
    current_path = Path.cwd()

    # Check current folder and all parent folders
    for path in [current_path] + list(current_path.parents):
        # Look for the 'data' folder in this path
        data_dir = path / 'data'
        if data_dir.exists():
            processed_dir = data_dir / 'raw'

            # Create 'processed' if it is missing (good safety habit)
            processed_dir.mkdir(parents=True, exist_ok=True)

            return processed_dir

    raise FileNotFoundError("Could not locate the 'data' folder in any parent directory.")
