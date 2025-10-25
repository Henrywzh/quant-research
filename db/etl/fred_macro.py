import os
import pandas as pd
from pandas_datareader import data as pdr
from sqlalchemy.orm import Session
from db.core import SessionLocal
from db.repo import get_or_create_series, upsert_observations

SERIES = [
    ("DFF", "Effective Federal Funds Rate",        "daily",   "%",   "US", "rates"),
    ("DGS10", "10-Year Treasury Constant",         "daily",   "%",   "US", "rates"),
    ("CPIAUCSL", "CPI All Urban Consumers",        "monthly", "Idx", "US", "inflation"),
    ("A191RL1Q225SBEA", "Real GDP QoQ SAAR",       "quarterly","%",  "US", "gdp"),
    ("M2SL", "M2 Money Stock",                     "weekly",  "Bn$", "US", "money_supply"),
]

def fetch_fred(code: str) -> pd.DataFrame:
    s = pdr.DataReader(code, "fred").dropna()
    df = s.rename_axis("ts").reset_index().rename(columns={code: "value"})
    df["ts"] = pd.to_datetime(df["ts"])
    return df

def main():
    if not os.getenv("DATABASE_URL"):
        raise SystemExit("Set DATABASE_URL before running (pooler 6543).")
    with SessionLocal() as s:  # uses DATABASE_URL
        for code, name, freq, unit, country, cat in SERIES:
            print(f"[FRED] {code} — {name}")
            df = fetch_fred(code)
            series = get_or_create_series(
                s,
                code=code, country=country, name=name,
                provider="FRED", provider_code=code,
                frequency=freq, unit=unit, category=cat
            )
            rows = list(df.itertuples(index=False, name=None))  # (ts, value)
            n = upsert_observations(s, macro_id=series.macro_id, rows=rows)
            print(f"  upserted {n} rows")
        s.commit()
        print("Done.")

if __name__ == "__main__":
    main()
