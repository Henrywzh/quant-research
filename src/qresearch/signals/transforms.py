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


def sector_series_to_df(
    sector: pd.Series,
    dates: pd.Index,
    tickers: pd.Index,
) -> pd.DataFrame:
    """
    sector: pd.Series index=ticker, value=sector_code
    returns: date×ticker DataFrame with repeated sector labels
    """
    sec = sector.reindex(tickers)
    sec_df = pd.DataFrame(np.tile(sec.to_numpy(), (len(dates), 1)),
                          index=dates, columns=tickers)
    return sec_df


def to_log_mcap(mcap: pd.DataFrame) -> pd.DataFrame:
    """
    mcap: date×ticker market cap (must be >0)
    returns: date×ticker ln(mcap)
    """
    mc = mcap.replace([np.inf, -np.inf], np.nan)
    mc = mc.where(mc > 0)
    return np.log(mc)


def neutralize_sector(scores: pd.DataFrame, sector_df: pd.DataFrame) -> pd.DataFrame:
    s = scores.stack(dropna=False).rename("s")
    sec = sector_df.stack(dropna=False).rename("sec")
    df = pd.concat([s, sec], axis=1).dropna(subset=["s", "sec"])

    date = df.index.get_level_values(0)
    g = df.groupby([date, "sec"])["s"]
    df["s_sec"] = df["s"] - g.transform("mean")

    out = df["s_sec"].unstack()
    return out.reindex(index=scores.index, columns=scores.columns)


def neutralize_lnmcap(scores: pd.DataFrame, lnmcap: pd.DataFrame) -> pd.DataFrame:
    y = scores.stack(dropna=False).rename("y")
    x = lnmcap.stack(dropna=False).rename("x")
    df = pd.concat([y, x], axis=1).dropna()

    date = df.index.get_level_values(0)
    g = df.groupby(date)

    x_bar = g["x"].transform("mean")
    y_bar = g["y"].transform("mean")

    xd = df["x"] - x_bar
    yd = df["y"] - y_bar

    num = (xd * yd).groupby(date).sum()
    den = (xd * xd).groupby(date).sum().replace(0.0, np.nan)
    b = num / den

    b_row = date.map(b)
    resid = yd - b_row * xd  # intercept already removed by demeaning

    out = resid.unstack()
    return out.reindex(index=scores.index, columns=scores.columns)


def neutralize_sector_and_mcap(
    scores: pd.DataFrame,
    sector: pd.Series,            # ticker -> sector
    mcap: pd.DataFrame,            # date×ticker (not logged)
    universe_eligible: pd.DataFrame | None = None,
    clip_lnmcap_q: float | None = 0.01,  # optional winsorize per date
) -> pd.DataFrame:
    dates, tickers = scores.index, scores.columns

    # Align everything once (critical)
    sec_df = sector_series_to_df(sector, dates, tickers)
    mc = mcap.reindex(index=dates, columns=tickers)
    lnmcap = to_log_mcap(mc)

    s = scores.reindex(index=dates, columns=tickers)

    # Apply eligibility BEFORE neutralisation (recommended)
    if universe_eligible is not None:
        elig = universe_eligible.reindex(index=dates, columns=tickers).fillna(False)
        s = s.where(elig)
        lnmcap = lnmcap.where(elig)
        sec_df = sec_df.where(elig)

    # Optional: per-date winsorize ln(mcap) to reduce extreme influence
    if clip_lnmcap_q is not None and 0.0 < clip_lnmcap_q < 0.5:
        lo = lnmcap.quantile(clip_lnmcap_q, axis=1)
        hi = lnmcap.quantile(1.0 - clip_lnmcap_q, axis=1)
        lnmcap = lnmcap.clip(lower=lo, upper=hi, axis=0)

    s1 = neutralize_sector(s, sec_df)
    s2 = neutralize_lnmcap(s1, lnmcap)
    return s2
