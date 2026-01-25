from typing import Optional

import pandas as pd
import numpy as np


# ============================================================
# Cross-sectional preprocessing
# ============================================================
def cs_winsorize_zscore(
        sig: pd.DataFrame,
        lower_q: float = 0.01,
        upper_q: float = 0.99,
) -> pd.DataFrame:
    """
    Cross-sectional winsorize + z-score per date.
    """

    def _proc_row(x: pd.Series) -> pd.Series:
        x = x.astype(float)
        m = x.notna()
        if m.sum() < 5:
            return x  # too few values
        lo = x[m].quantile(lower_q)
        hi = x[m].quantile(upper_q)
        x2 = x.copy()
        x2[m] = x[m].clip(lo, hi)
        mu = x2[m].mean()
        sd = x2[m].std(ddof=1)
        if sd == 0 or not np.isfinite(sd):
            return x2
        x2[m] = (x2[m] - mu) / sd

        return x2

    return sig.apply(_proc_row, axis=1)


def sector_series_to_df(sector_map, dates, tickers) -> pd.DataFrame:
    """
    sector_map: pd.Series (ticker->sector) OR 1-col pd.DataFrame (index=ticker).
    returns: date×ticker DataFrame of sector labels.
    """
    if isinstance(sector_map, pd.DataFrame):
        if sector_map.shape[1] != 1:
            raise ValueError("sector_map DataFrame must have exactly 1 column (sector).")
        sector_map = sector_map.iloc[:, 0]  # convert to Series

    if not isinstance(sector_map, pd.Series):
        raise TypeError("sector_map must be a pd.Series or 1-col pd.DataFrame.")

    # Align to tickers in scores
    sec_row = sector_map.reindex(pd.Index(tickers)).astype("object")  # Series length = len(tickers)

    # Broadcast to all dates efficiently
    out = pd.DataFrame(
        [sec_row.to_numpy()] * len(dates),
        index=dates,
        columns=pd.Index(tickers),
    )
    return out


def to_log_mcap(mcap: pd.DataFrame) -> pd.DataFrame:
    mc = mcap.apply(pd.to_numeric, errors="coerce")
    mc = mc.where(mc > 0)  # non-positive -> NaN
    return np.log(mc)

def neutralize_sector(scores: pd.DataFrame, sector_df: pd.DataFrame) -> pd.DataFrame:
    s = _stack(scores, "s")
    sec = _stack(sector_df, "sec")
    df = pd.concat([s, sec], axis=1).dropna(subset=["s", "sec"])

    date = df.index.get_level_values(0)
    g = df.groupby([date, "sec"], sort=False)["s"]
    df["s_sec"] = df["s"] - g.transform("mean")

    out = df["s_sec"].unstack()
    return out.reindex(index=scores.index, columns=scores.columns)

def neutralize_sector_and_mcap_fwl(
    scores: pd.DataFrame,
    sector: pd.Series,                # ticker -> sector
    mcap: pd.DataFrame,               # date x ticker (NOT logged)
    universe_eligible: Optional[pd.DataFrame] = None,
    clip_lnmcap_q: Optional[float] = 0.01,
) -> pd.DataFrame:
    dates, tickers = scores.index, scores.columns

    # --- align inputs ---
    y = scores.reindex(index=dates, columns=tickers)
    sec_df = sector_series_to_df(sector, dates, tickers)
    lnmcap = to_log_mcap(mcap.reindex(index=dates, columns=tickers))

    if universe_eligible is None:
        elig = pd.DataFrame(True, index=dates, columns=tickers)
    else:
        elig = universe_eligible.reindex(index=dates, columns=tickers).fillna(False).astype(bool)

    # optional clip lnmcap (ok)
    if clip_lnmcap_q is not None and 0.0 < clip_lnmcap_q < 0.5:
        lo = lnmcap.quantile(clip_lnmcap_q, axis=1)
        hi = lnmcap.quantile(1.0 - clip_lnmcap_q, axis=1)
        lnmcap = lnmcap.clip(lower=lo, upper=hi, axis=0)

    # IMPORTANT: enforce the final sample BEFORE neutralisation
    mask = elig & y.notna() & lnmcap.notna() & sec_df.notna()
    y = y.where(mask)
    lnmcap = lnmcap.where(mask)
    sec_df = sec_df.where(mask)

    # sector residualize on the same sample
    y_tilde = neutralize_sector(y, sec_df)
    x_tilde = neutralize_sector(lnmcap, sec_df)

    # stack (no dropna needed now if mask is correct, but harmless)
    df = pd.concat([_stack(y_tilde, "y"), _stack(x_tilde, "x")], axis=1).dropna()

    date = df.index.get_level_values(0)
    g = df.groupby(date, sort=False)

    # No extra date demeaning needed (FWL already handled intercept via sector projection)
    num = (df["x"] * df["y"]).groupby(date, sort=False).sum()
    den = (df["x"] * df["x"]).groupby(date, sort=False).sum().replace(0.0, np.nan)
    beta = num / den

    beta_row = beta.reindex(date).to_numpy(dtype=float)
    resid = df["y"].to_numpy(dtype=float) - beta_row * df["x"].to_numpy(dtype=float)

    out = pd.Series(resid, index=df.index, name="resid").unstack()
    return out.reindex(index=dates, columns=tickers)


def _stack(df: pd.DataFrame, name: str) -> pd.Series:
    # pandas>=2.1: use future_stack=True, and do NOT pass dropna=
    return df.stack(future_stack=True).rename(name)
